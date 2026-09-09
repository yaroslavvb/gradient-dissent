import copy
import unittest

import torch
from torch import nn
try:
    from .kernels import BaselineStep, make_sgd
except ImportError:
    from kernels import BaselineStep, make_sgd


def model(device="cpu"):
    return nn.Sequential(nn.Linear(4,9),nn.ReLU(),nn.Linear(9,3),nn.ReLU()).to(device)


class StepTests(unittest.TestCase):
    def test_eager_matches_reference_sgd_and_historical_shrinkage(self):
        torch.manual_seed(3)
        a=model(); b=copy.deepcopy(a)
        oa=make_sgd(a,.003); ob=make_sgd(b,.003)
        x=torch.randn(7,4); y=torch.arange(7)%3
        helper=BaselineStep(a,oa,x,y,shrinkage=2e-5)
        for _ in range(4):
            actual=helper(x,y)
            ob.zero_grad(set_to_none=True)
            expected=nn.functional.cross_entropy(b(x),y)
            expected.backward(); ob.step()
            with torch.no_grad():
                for p in b.parameters(): p.mul_(1-2e-5)
            torch.testing.assert_close(actual,expected.detach(),rtol=0,atol=0)
            for p,q in zip(a.parameters(),b.parameters()):
                torch.testing.assert_close(p,q,rtol=0,atol=0)

    def test_static_shape_and_cpu_graph_guard(self):
        a=model(); opt=make_sgd(a,.001); x=torch.ones(5,4); y=torch.zeros(5,dtype=torch.long)
        helper=BaselineStep(a,opt,x,y)
        with self.assertRaises(ValueError): helper(x[:3],y[:3])
        with self.assertRaises(ValueError): BaselineStep(a,opt,x,y,mode="cuda_graph")
        with self.assertRaises(ValueError): make_sgd(a,.001,fused=True)

    def test_bf16_cpu_autocast_finite(self):
        a=model(); x=torch.randn(5,4); y=torch.arange(5)%3
        helper=BaselineStep(a,make_sgd(a,.01),x,y,precision="bf16")
        self.assertTrue(torch.isfinite(helper(x,y)))
        self.assertTrue(all(p.dtype==torch.float32 for p in a.parameters()))

    @unittest.skipUnless(torch.cuda.is_available(),"CUDA hardware required; never launch paid resources for tests")
    def test_graph_unit_dropout_draws_fresh_masks(self):
        torch.manual_seed(99)
        net=nn.Sequential(nn.Linear(4,32),nn.ReLU(),nn.Dropout(.5),nn.Linear(32,3)).cuda()
        x=torch.arange(32,device='cuda').float().reshape(8,4)/32
        y=torch.arange(8,device='cuda')%3
        before=[p.detach().clone() for p in net.parameters()]
        rng=torch.cuda.get_rng_state().clone()
        step=BaselineStep(net,make_sgd(net,1e-12),x,y,mode='cuda_graph')
        self.assertTrue(torch.equal(rng,torch.cuda.get_rng_state()))
        for p,q in zip(net.parameters(),before):self.assertTrue(torch.equal(p,q))
        losses=torch.stack([step(x,y).clone() for _ in range(12)])
        self.assertGreater(losses.std().item(),1e-4)

    @unittest.skipUnless(torch.cuda.is_available(),"CUDA hardware required; never launch paid resources for tests")
    def test_cuda_capture_restores_state_and_matches_eager(self):
        for fused in (False,True):
            torch.manual_seed(27); torch.cuda.manual_seed_all(27)
            a=model("cuda"); b=copy.deepcopy(a)
            oa=make_sgd(a,.003,fused=fused); ob=make_sgd(b,.003,fused=fused)
            x=torch.randn(7,4,device="cuda"); y=torch.arange(7,device="cuda")%3
            rng=torch.cuda.get_rng_state().clone()
            cpu_rng=torch.get_rng_state().clone()
            helper=BaselineStep(a,oa,x,y,mode="cuda_graph",shrinkage=2e-5)
            self.assertTrue(torch.equal(rng,torch.cuda.get_rng_state()))
            self.assertTrue(torch.equal(cpu_rng,torch.get_rng_state()))
            for p,q in zip(a.parameters(),b.parameters()): self.assertTrue(torch.equal(p,q))
            eager=BaselineStep(b,ob,x,y,shrinkage=2e-5)
            for i in range(5):
                batch=x+i*.01
                actual=helper(batch,y).clone()
                expected=eager(batch,y)
                torch.testing.assert_close(actual,expected,rtol=1e-5,atol=1e-6)
                for p,q in zip(a.parameters(),b.parameters()):
                    torch.testing.assert_close(p,q,rtol=1e-5,atol=1e-6)
            oa.param_groups[0]["lr"]*=.5
            with self.assertRaises(ValueError): helper(x,y)


if __name__=="__main__":
    torch.set_num_threads(1)
    unittest.main()
