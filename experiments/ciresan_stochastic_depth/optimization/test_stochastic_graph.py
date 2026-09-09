"""CPU semantics plus opt-in CUDA tests; never provisions hardware."""
import copy
from pathlib import Path
import sys
import unittest

import torch
from torch.nn import functional as F

try:
    from ..model_data import CiresanMLP
except ImportError:
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
    from model_data import CiresanMLP
try:
    from .stochastic_graph import (ResidualGraphStep,active_parameter_indices,
                                 forward_with_mask,inverse_survival,update_active_sgd,validate_mask)
except ImportError:
    from stochastic_graph import (ResidualGraphStep,active_parameter_indices,
                                  forward_with_mask,inverse_survival,update_active_sgd,validate_mask)

WIDTHS=(4,9,8,7,6,5,3)
MASKS=[(True,)*4,(False,True,False,True),(False,)*4,(True,False,True,False),(True,)*4]
PROBS=[(.1,.2,.3,.4),(.1,.2,.3,.4),(.05,.1,.15,.2),(0.,)*4,(0.,)*4]


def explicit_step(model,buffers,x,y,mask,probs,lr=.003,shrinkage=2e-5):
    params=list(model.parameters())
    indices=active_parameter_indices(model,mask)
    for i in indices:
        if params[i].grad is None: params[i].grad=torch.zeros_like(params[i])
        else: params[i].grad.zero_()
    scales=torch.tensor(inverse_survival(probs)[1],device=x.device)
    out=forward_with_mask(model,x,mask,scales)
    loss=F.cross_entropy(out,y)
    loss.backward()
    update_active_sgd(params,buffers,indices,lr=lr,momentum=.9,shrinkage=shrinkage)
    return loss.detach()


class Semantics(unittest.TestCase):
    def test_cpu_sequence_matches_source_sgd_and_freezes_dropped_momentum(self):
        torch.manual_seed(37)
        a=CiresanMLP("sd_annealed",widths=WIDTHS,input_scale=1)
        b=copy.deepcopy(a)
        opt=torch.optim.SGD(b.parameters(),lr=.003,momentum=.9,foreach=True)
        params=list(a.parameters()); buffers=[torch.zeros_like(p) for p in params]
        x=torch.randn(7,4); y=torch.arange(7)%3
        for mask,probs in zip(MASKS,PROBS):
            previous=[v.clone() for v in buffers]
            weights=[p.detach().clone() for p in params]
            actual=explicit_step(a,buffers,x,y,mask,probs)
            opt.zero_grad(set_to_none=True)
            expected=F.cross_entropy(b(x,active=mask,drop_probs=probs),y)
            expected.backward(); opt.step()
            with torch.no_grad():
                for p in b.parameters(): p.mul_(1-2e-5)
            torch.testing.assert_close(actual,expected.detach(),rtol=1e-5,atol=1e-6)
            active_indices=active_parameter_indices(a,mask)
            for i,(p,q) in enumerate(zip(params,b.parameters())):
                torch.testing.assert_close(p,q,rtol=1e-5,atol=1e-6)
                target=opt.state.get(q,{}).get("momentum_buffer",torch.zeros_like(q))
                torch.testing.assert_close(buffers[i],target,rtol=1e-5,atol=1e-6)
                if i not in active_indices:
                    self.assertTrue(torch.equal(previous[i],buffers[i]))
                    torch.testing.assert_close(p,weights[i]*(1-2e-5),rtol=0,atol=0)

    def test_plain_residual_and_unit_dropout_forward_parity(self):
        x=torch.randn(8,4)
        for recipe in ("plain","residual","unit_dropout","residual_unit_dropout"):
            model=CiresanMLP(recipe,pmax=.5,widths=WIDTHS,input_scale=1,output_relu=True)
            torch.manual_seed(41)
            actual=forward_with_mask(model,x,(True,)*4,torch.ones(4))
            torch.manual_seed(41)
            expected=model(x)
            torch.testing.assert_close(actual,expected,rtol=0,atol=0)

    def test_zero_branch_does_not_execute(self):
        model=CiresanMLP("sd_constant",widths=WIDTHS)
        def forbidden(*args): raise AssertionError("Dropped branch evaluated")
        for layer in model.layers[1:-1]: layer.register_forward_hook(forbidden)
        result=forward_with_mask(model,torch.ones(3,4),(False,)*4,torch.ones(4))
        self.assertTrue(torch.isfinite(result).all())

    def test_validation(self):
        with self.assertRaises(ValueError): validate_mask([1,0,1,0])
        with self.assertRaises(ValueError): inverse_survival([1,0,0,0])
        with self.assertRaises(ValueError): inverse_survival([float("nan")]*4)
        model=CiresanMLP("sd_constant",widths=WIDTHS)
        with self.assertRaises(ValueError): ResidualGraphStep(model,.001,torch.ones(3,4),torch.zeros(3,dtype=torch.long))

    @unittest.skipUnless(torch.cuda.is_available(),"Needs existing CUDA hardware; no resource provisioning")
    def test_cuda_unit_dropout_draws_fresh_masks_and_reproduces_seed(self):
        def sequence():
            torch.manual_seed(2037); torch.cuda.manual_seed_all(2037)
            model=CiresanMLP("residual_unit_dropout",pmax=.2,widths=WIDTHS,input_scale=1).cuda()
            initial=[p.detach().clone() for p in model.parameters()]
            x=torch.ones(17,4,device="cuda"); y=torch.arange(17,device="cuda")%3
            rng=torch.cuda.get_rng_state().clone(); cpu_rng=torch.get_rng_state().clone()
            helper=ResidualGraphStep(model,1e-9,x,y)
            self.assertEqual(len(helper.graphs),1)
            self.assertTrue(torch.equal(rng,torch.cuda.get_rng_state()))
            self.assertTrue(torch.equal(cpu_rng,torch.get_rng_state()))
            for p,q in zip(model.parameters(),initial): self.assertTrue(torch.equal(p,q))
            losses=[]
            for _ in range(6):
                # Fix weights AND momentum before each replay. Loss variation
                # must arise from fresh masks, not changes to the predictor.
                with torch.no_grad():
                    for p,q in zip(model.parameters(),initial): p.copy_(q)
                    for buffer in helper.momentum_buffers: buffer.zero_()
                losses.append(float(helper(x,y).clone().cpu()))
            self.assertGreater(len(set(losses)),1,"Unit dropout mask is frozen across graph replays")
            return losses
        self.assertEqual(sequence(),sequence(),"Fresh same-seed captures must reproduce the dropout sequence")

    @unittest.skipUnless(torch.cuda.is_available(),"Needs existing CUDA hardware; no resource provisioning")
    def test_cuda_masks_annealing_and_restoration(self):
        torch.manual_seed(101); torch.cuda.manual_seed_all(102)
        a=CiresanMLP("sd_annealed",widths=WIDTHS,input_scale=1).cuda()
        b=copy.deepcopy(a)
        x=torch.randn(7,4,device="cuda"); y=torch.arange(7,device="cuda")%3
        initial_buffers=[torch.randn_like(p)*.01 for p in a.parameters()]
        initial_saved=[p.clone() for p in initial_buffers]
        rng=torch.cuda.get_rng_state().clone(); cpu_rng=torch.get_rng_state().clone()
        helper=ResidualGraphStep(a,.003,x,y,shrinkage=2e-5,momentum_buffers=initial_buffers)
        self.assertEqual(len(helper.graphs),16)
        self.assertTrue(torch.equal(rng,torch.cuda.get_rng_state()))
        self.assertTrue(torch.equal(cpu_rng,torch.get_rng_state()))
        for p,q in zip(a.parameters(),b.parameters()): self.assertTrue(torch.equal(p,q))
        for p,q in zip(helper.momentum_buffers,initial_saved): self.assertTrue(torch.equal(p,q))
        opt=torch.optim.SGD(b.parameters(),lr=.003,momentum=.9,foreach=True)
        for p,buf in zip(b.parameters(),initial_saved): opt.state[p]["momentum_buffer"]=buf.clone()
        for mask,probs in zip(MASKS,PROBS):
            helper.update_scales(probs)
            previous=[v.clone() for v in helper.momentum_buffers]
            actual=helper(x,y,active=mask).clone()
            opt.zero_grad(set_to_none=True)
            expected=F.cross_entropy(b(x,active=mask,drop_probs=probs),y)
            expected.backward(); opt.step()
            with torch.no_grad():
                for p in b.parameters(): p.mul_(1-2e-5)
            torch.testing.assert_close(actual,expected.detach(),rtol=1e-5,atol=1e-6)
            chosen=active_parameter_indices(a,mask)
            for i,(p,q) in enumerate(zip(a.parameters(),b.parameters())):
                torch.testing.assert_close(p,q,rtol=1e-5,atol=1e-6)
                torch.testing.assert_close(helper.momentum_buffers[i],opt.state[q]["momentum_buffer"],rtol=1e-5,atol=1e-6)
                if i not in chosen: self.assertTrue(torch.equal(previous[i],helper.momentum_buffers[i]))
        self.assertEqual(helper.last_mask,MASKS[-1])


if __name__=="__main__":
    torch.set_num_threads(1)
    unittest.main()
