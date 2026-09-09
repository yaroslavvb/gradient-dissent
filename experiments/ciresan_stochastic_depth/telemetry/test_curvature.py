"""CPU scientific checks against explicit tiny parameter-space matrices."""
import json
import unittest

import torch
from torch import nn
from torch.nn import functional as F

from .curvature import analyze_curvature, _stable_output_statistics, _spectrum
from ..model_data import CiresanMLP


class TinyMLP(nn.Module):
    def __init__(self, bias=True):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(3,4,bias=bias), nn.Linear(4,3,bias=bias)])

    def forward(self,x):
        return self.layers[1](self.layers[0](x).tanh())


class CurvatureTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(719)
        self.model = TinyMLP().double()
        self.x = torch.randn(5,3,dtype=torch.float64)
        self.y = torch.tensor([0,1,2,0,2])
        self.result = analyze_curvature(self.model,self.x,self.y,mode="exact")

    def explicit_blocks(self):
        """Independent direct logit-to-W Jacobians, not the factor sampler."""
        model,x,y=self.model,self.x,self.y
        logits=model(x)
        probabilities=logits.softmax(-1).detach()
        output_gradient=probabilities-F.one_hot(y,3)
        h=torch.stack([torch.autograd.functional.hessian(
            lambda z:F.cross_entropy(z[None],y[i:i+1]),logits[i])
            for i in range(len(y))])
        blocks=[]
        for layer in model.layers:
            jacobian=torch.stack([torch.stack([
                torch.autograd.grad(logits[i,c],layer.weight,retain_graph=True)[0].flatten()
                for c in range(3)]) for i in range(len(y))])
            gradients=torch.einsum("ncp,nc->np",jacobian,output_gradient)
            blocks.append({
                "empirical_fisher":gradients.T@gradients/len(y),
                "ce_ggn":torch.einsum("ncp,ncd,ndq->pq",jacobian,h,jacobian)/len(y),
                "jacobian_gram":torch.einsum("ncp,ncq->pq",jacobian,jacobian)/len(y),
                "gradients":gradients,
            })
        return blocks,h,output_gradient

    def test_exact_diagonals_match_explicit_parameter_matrices(self):
        blocks,_,_=self.explicit_blocks()
        for record,block in zip(self.result["layers"],blocks):
            for family in ["empirical_fisher","ce_ggn","jacobian_gram"]:
                diagonal=block[family].diagonal()
                got=record["families"][family]["weight_diagonal"]
                self.assertAlmostEqual(got["diag_trace"],diagonal.sum().item(),places=11)
                self.assertAlmostEqual(got["diag_frobenius_norm"],diagonal.norm().item(),places=11)
                self.assertAlmostEqual(got["diag_max"],diagonal.max().item(),places=11)
            gradients=block["gradients"]
            g=gradients.mean(0)
            moments=record["gradient_moments"]
            self.assertAlmostEqual(moments["mean_gradient_l2"],g.norm().item(),places=11)
            self.assertAlmostEqual(moments["per_example_gradient_squared_norm_mean"],gradients.square().sum(-1).mean().item(),places=11)
            self.assertAlmostEqual(moments["gradient_covariance_trace"],(gradients-g).square().sum(-1).mean().item(),places=11)

    def test_kfac_norms_and_directional_noise_match_explicit_kron(self):
        blocks,_,_=self.explicit_blocks()
        logits=self.model(self.x)
        p=logits.softmax(-1).detach()
        inputs=[self.x,self.model.layers[0](self.x).tanh().detach()]
        preactivations=[self.model.layers[0](self.x),None]
        # Analytic Jacobian of the tanh suffix, and identity at the output.
        derivatives=[self.model.layers[1].weight.detach()[None]*
                     (1-preactivations[0].tanh().square())[:,None,:],
                     torch.eye(3,dtype=torch.float64)[None].expand(5,3,3)]
        for index,record in enumerate(self.result["layers"]):
            a=inputs[index]; aa=a.T@a/5
            j=derivatives[index]
            output_grad=p-F.one_hot(self.y,3)
            f=torch.einsum("nco,nc->no",j,output_grad)
            h=torch.diag_embed(p)-p[:,:,None]*p[:,None,:]
            bb={"empirical_fisher":f.T@f/5,
                "jacobian_gram":torch.einsum("nco,ncp->op",j,j)/5,
                "ce_ggn":torch.einsum("nco,ncd,ndp->op",j,h,j)/5}
            gradients=blocks[index]["gradients"]; g=gradients.mean(0)
            for family,b in bb.items():
                k=torch.kron(b.contiguous(),aa.contiguous())
                got=record["families"][family]
                self.assertAlmostEqual(got["kfac"]["trace"],k.trace().item(),places=10)
                self.assertAlmostEqual(got["kfac"]["frobenius_norm"],k.norm().item(),places=10)
                top=torch.linalg.eigvalsh(k)[-1]
                self.assertAlmostEqual(got["kfac"]["top_eigenvalue"],top.item(),places=10)
                self.assertAlmostEqual(got["kfac"]["effective_rank"],(k.trace()/top).item(),places=10)
                self.assertAlmostEqual(got["kfac"]["stable_rank"],(k.square().sum()/top.square()).item(),places=10)
                direct_diag=blocks[index][family].diagonal()
                error=(k.diagonal()-direct_diag).abs().sum()/direct_diag.abs().sum()
                self.assertAlmostEqual(got["kfac"]["diagonal_relative_l1_error"],error.item(),places=10)
                directional=got["gradient_directions"]
                self.assertAlmostEqual(directional["mean_gradient_quadratic_form"],(g@k@g).item(),places=10)
                self.assertAlmostEqual(directional["mean_gradient_rayleigh_quotient"],(g@k@g/g.square().sum()).item(),places=10)
                centered=gradients-g
                self.assertAlmostEqual(directional["empirical_gradient_covariance_curvature_trace"],torch.einsum("np,pq,nq->",centered,k,centered).item()/5,places=10)

    def test_output_hessian_and_support_lyapunov(self):
        _,h,g=self.explicit_blocks()
        got=self.result["output_mismatch"]
        torch.testing.assert_close(torch.tensor(got["mean_logit_hessian"],dtype=torch.float64),h.mean(0))
        torch.testing.assert_close(torch.tensor(got["empirical_logit_fisher"],dtype=torch.float64),g.T@g/5)
        self.assertEqual(got["retained_hessian_rank"],2)
        self.assertLess(got["lyapunov_residual_fro"],1e-12)
        self.assertLess(got["fisher_outside_support_relative_fro"],1e-12)
        self.assertTrue(all(x>=0 for x in got["lyapunov_eigenvalues"]))

    def test_sampled_normalization_converges_to_exact(self):
        sampled=analyze_curvature(self.model,self.x,self.y,mode="sampled",samples=512,seed=312)
        for exact,approx in zip(self.result["layers"],sampled["layers"]):
            for family in ["ce_ggn","jacobian_gram"]:
                a=exact["families"][family]["weight_diagonal"]["diag_trace"]
                b=approx["families"][family]["weight_diagonal"]["diag_trace"]
                self.assertLess(abs(a-b)/a,.08)
            self.assertEqual(exact["families"]["empirical_fisher"],approx["families"]["empirical_fisher"])
        again=analyze_curvature(self.model,self.x,self.y,mode="sampled",samples=3,seed=12)
        repeat=analyze_curvature(self.model,self.x,self.y,mode="sampled",samples=3,seed=12)
        self.assertEqual(again["layers"],repeat["layers"])

    def test_state_grad_storage_rng_and_modes_unchanged(self):
        self.model.train();self.model.layers[1].eval()
        for p in self.model.parameters():p.grad=torch.randn_like(p)
        parameters=[p.detach().clone() for p in self.model.parameters()]
        gradients=[(p.grad,p.grad.clone(),p.grad.data_ptr()) for p in self.model.parameters()]
        modes=[m.training for m in self.model.modules()]
        rng=torch.get_rng_state().clone();precision=torch.get_float32_matmul_precision()
        analyze_curvature(self.model,self.x,self.y,mode="sampled",samples=2)
        self.assertEqual(modes,[m.training for m in self.model.modules()])
        self.assertEqual(precision,torch.get_float32_matmul_precision())
        torch.testing.assert_close(rng,torch.get_rng_state(),rtol=0,atol=0)
        for p,value,(old,grad,pointer) in zip(self.model.parameters(),parameters,gradients):
            torch.testing.assert_close(p,value,rtol=0,atol=0)
            self.assertIs(p.grad,old);self.assertEqual(p.grad.data_ptr(),pointer)
            torch.testing.assert_close(p.grad,grad,rtol=0,atol=0)

    def test_cancellation_resistant_output_directions(self):
        logits=torch.tensor([[80.,0.,-1.]],dtype=torch.float64)
        p,g,r=_stable_output_statistics(logits,torch.tensor([0]))
        self.assertEqual(float(p[0,0]),1.)
        self.assertLess(float(g[0,0]),0.)
        self.assertAlmostEqual(float(g.sum()),0.,places=45)
        self.assertGreater(float((r.transpose(1,2)@r)[0,0,0]),0.)

    def test_zero_and_rank_deficient_factors_json_valid(self):
        model=TinyMLP().double()
        with torch.no_grad():
            for p in model.parameters():p.zero_()
        result=analyze_curvature(model,self.x,self.y)
        first=result["layers"][0]
        self.assertIsNone(first["gradient_moments"]["gradient_diversity"])
        self.assertIsNone(first["families"]["ce_ggn"]["kfac"]["effective_rank"])
        self.assertEqual(first["families"]["ce_ggn"]["gradient_directions"]["per_example_rayleigh_quotient"]["valid_n"],0)
        json.dumps(result,allow_nan=False)

    def test_spectrum_limit_is_explicit_estimate(self):
        factor=torch.diag(torch.tensor([5.,3.,2.,1.],dtype=torch.float64))
        exact=_spectrum(factor,eig_max_dimension=4)
        approximate=_spectrum(factor,eig_max_dimension=2)
        self.assertEqual(exact["method"],"exact_smaller_gram_eigvalsh")
        self.assertIsNone(approximate["eigenvalues"])
        self.assertIn("approximate",approximate["method"])
        self.assertAlmostEqual(approximate["top_eigenvalue"],25.,places=10)
        self.assertEqual(exact["trace"],approximate["trace"])

    def test_failure_cleans_hooks_and_modes(self):
        self.model.train()
        with self.assertRaises(FloatingPointError):
            analyze_curvature(self.model,self.x*float("nan"),self.y)
        self.assertTrue(self.model.training)
        self.assertTrue(all(not m._forward_hooks for m in self.model.modules()))

    def test_no_bias_does_not_claim_bias_parameters(self):
        result=analyze_curvature(TinyMLP(bias=False).double(),self.x,self.y)
        for layer in result["layers"]:
            self.assertFalse(layer["bias_present"])
            for family in layer["families"].values():
                self.assertIsNone(family["bias_diagonal"])

    def test_real_crop_residual_class_dense_probe(self):
        model=CiresanMLP(recipe="sd_annealed",pmax=.8,input_scale=1.,
                         widths=(3,10,8,6,4,3,3)).float()
        model.train()
        result=analyze_curvature(model,self.x.float(),self.y)
        self.assertTrue(model.training)
        self.assertEqual(len(result["layers"]),6)
        self.assertEqual(result["probe_n"],5)
        self.assertEqual(result["output_classes"],3)
        self.assertTrue(all(f["kfac"]["spectrum_exact"] for l in result["layers"] for f in l["families"].values()))
        json.dumps(result,allow_nan=False)


if __name__=="__main__":unittest.main()
