"""Small CPU numerical telemetry tests; no data downloads or paid resources."""
import copy
import json
from types import SimpleNamespace
import unittest

import torch
from torch.nn import functional as F

try:
    from . import metrics as t
    from ..model_data import CiresanMLP
except ImportError:
    import metrics as t
    import sys
    from pathlib import Path
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
    from model_data import CiresanMLP


def model(recipe='residual',output_relu=False):
    torch.manual_seed(37)
    return CiresanMLP(recipe,pmax=.2 if recipe=='residual_unit_dropout' else 0,
                      input_scale=1,output_relu=output_relu,widths=(4,7,6,5,4,3,2))


class TelemetryChecks(unittest.TestCase):
    def test_probe_matches_dense_loss_and_preserves_rng_mode_params_and_grad_storage(self):
        m=model('residual_unit_dropout');m.train();x=torch.randn(9,4);y=torch.arange(9)%2
        for parameter in m.parameters():parameter.grad=torch.full_like(parameter,.123)
        parameters=[p.detach().clone() for p in m.parameters()]
        gradients=[p.grad.clone() for p in m.parameters()];pointers=[p.grad.data_ptr() for p in m.parameters()]
        rng=torch.get_rng_state().clone();reference=copy.deepcopy(m).eval()
        expected=F.cross_entropy(reference(x),y)
        r=t.probe_statistics(m,x,y,compute_gradients=True)
        self.assertAlmostEqual(r['loss'],expected.item(),places=6)
        self.assertTrue(m.training);torch.testing.assert_close(torch.get_rng_state(),rng,atol=0,rtol=0)
        for p,value,gradient,pointer in zip(m.parameters(),parameters,gradients,pointers):
            torch.testing.assert_close(p,value,atol=0,rtol=0);torch.testing.assert_close(p.grad,gradient,atol=0,rtol=0)
            self.assertEqual(p.grad.data_ptr(),pointer)
        json.dumps(r,allow_nan=False)

    def test_factorized_per_example_gradient_moments_and_neighbor_cosine(self):
        m=model().eval();x=torch.tensor([[.2,.3,-.1,.8],[1.,-.4,.1,.2],[-.2,1.,.3,.7],[.5,.6,.7,.8]])
        y=torch.tensor([0,1,1,0]);r=t.probe_statistics(m,x,y,compute_gradients=True)
        losses=F.cross_entropy(m(x),y,reduction='none')
        explicit=[[] for _ in m.layers]
        for loss in losses:
            grads=torch.autograd.grad(loss,[layer.weight for layer in m.layers],retain_graph=True)
            for group,grad in zip(explicit,grads):group.append(grad.flatten())
        for index,gradients in enumerate(explicit):
            g=torch.stack(gradients);norm=torch.linalg.vector_norm(g,dim=1);row=r['layers'][index]
            mean=g.mean(0);second=(g*g).sum(1).mean();denom=(mean*mean).sum()
            self.assertAlmostEqual(row['grad_l2'],float(mean.norm()),delta=2e-6)
            self.assertAlmostEqual(row['weight_grad_second_moment'],float(second),delta=2e-6)
            self.assertAlmostEqual(row['weight_grad_centered_noise'],float(second-denom),delta=2e-6)
            for name,value in [('mean',norm.mean()),('median',torch.quantile(norm,.5)),('min',norm.min()),('max',norm.max())]:
                self.assertAlmostEqual(row['per_example_weight_grad_norm_'+name],float(value),delta=2e-6)
            if denom>0:self.assertAlmostEqual(row['weight_grad_diversity'],float(second/denom),delta=max(2e-5,float(second/denom)*2e-5))
            valid=norm*norm.roll(1)>0
            if valid.any():
                cosine=(g*g.roll(1,0)).sum(1)[valid]/(norm*norm.roll(1))[valid]
                self.assertAlmostEqual(row['neighbor_weight_grad_cosine_mean'],float(cosine.mean()),delta=2e-6)

    def test_backprop_uses_sum_ce_not_mean_and_distinguishes_zero_from_negative(self):
        m=model('plain').eval();x=torch.randn(5,4);y=torch.arange(5)%2
        r=t.probe_statistics(m,x,y,compute_gradients=True);logits=m(x)
        b=logits.softmax(-1)-F.one_hot(y,2)
        head=r['layers'][-1]
        self.assertAlmostEqual(head['msr_backprop'],float((b*b).mean().sqrt().detach()),delta=1e-7)
        self.assertEqual(head['backprop_preactivation_zero_fraction'],0)
        self.assertEqual(head['backprop_preactivation_nonpositive_fraction'],.5)
        self.assertEqual(head['b_sparsity'],.5)

    def test_parameter_deltas_are_measured_and_stale_dropped_gradients_excluded(self):
        m=model();params=list(m.parameters())
        for p in params:p.grad=torch.full_like(p,3.)
        before=t.parameter_snapshot(m,step=0)
        with torch.no_grad():
            for p in params:p.mul_(.9).add_(.02)
        helper=SimpleNamespace(last_mask=(False,True,False,True),parameters=params,momentum_buffers=[torch.ones_like(p) for p in params])
        r=t.layer_statistics(m,previous_parameters=before,helper=helper,interval_steps=781,update_interval='one_epoch')
        self.assertEqual(r['interval_steps'],781)
        for i,row in enumerate(r['layers']):
            active=i in [0,2,4,5]
            self.assertEqual(row['training_gradient_available'],active)
            self.assertEqual('weight_last_training_gradient_l2' in row,active)
            p=m.layers[i].weight;delta=p-before['values'][f'layers.{i}.weight']
            self.assertAlmostEqual(row['weight_measured_update_l2'],float(delta.norm().detach()),places=6)
            self.assertAlmostEqual(row['weight_parameter_mean'],float(p.mean().detach()),places=6)
            self.assertIn('weight_momentum_buffer_l2',row)
        missing=t.layer_statistics(m)
        self.assertEqual(missing['layers'][1]['gradient_status'],'unknown_last_mask')
        self.assertNotIn('weight_last_training_gradient_l2',missing['layers'][1])

    def test_dead_relu_head_json_safe_no_false_diversity(self):
        m=model(output_relu=True)
        with torch.no_grad():m.layers[-1].weight.zero_();m.layers[-1].bias.fill_(-1)
        x=torch.randn(6,4);y=torch.arange(6)%2;r=t.probe_statistics(m,x,y,compute_gradients=True)
        head=r['layers'][-1]
        self.assertEqual(head['post_activation_relu_inactive_fraction'],1)
        self.assertEqual(head['post_activation_all_zero_units_on_probe_fraction'],1)
        self.assertEqual(head['grad_l2'],0)
        self.assertIsNone(head['weight_grad_diversity'])
        self.assertIsNone(head['neighbor_weight_grad_cosine_mean'])
        self.assertEqual(head['neighbor_weight_grad_cosine_valid_pairs'],0)
        json.dumps(r,allow_nan=False)

    def test_evaluation_confusion_perclass_and_state_preserved(self):
        m=model('residual_unit_dropout');m.train();x=torch.randn(11,4);y=torch.arange(11)%2
        rng=torch.get_rng_state().clone();ref=copy.deepcopy(m).eval();logits=ref(x);pred=logits.argmax(-1)
        r=t.evaluate(m,x,y,batch_size=4,num_classes=2)
        expected=[[int(((y==i)&(pred==j)).sum()) for j in range(2)] for i in range(2)]
        self.assertEqual(r['confusion'],expected)
        self.assertEqual(r['errors'],int((pred!=y).sum()))
        self.assertAlmostEqual(r['loss'],float(F.cross_entropy(logits,y).detach()),places=6)
        self.assertEqual(sum(row['n'] for row in r['per_class']),11)
        self.assertTrue(m.training);torch.testing.assert_close(torch.get_rng_state(),rng,atol=0,rtol=0)
        self.assertGreaterEqual(r['elapsed_seconds'],0)

    def test_eval_nonfinite_restores_state_and_parameter_stats_sanitize(self):
        m=model();m.train()
        with torch.no_grad():m.layers[-1].bias.fill_(float('nan'))
        with self.assertRaises(FloatingPointError):t.evaluate(m,torch.ones(3,4),torch.tensor([0,1,0]),num_classes=2)
        self.assertTrue(m.training)
        r=t.layer_statistics(m)
        self.assertIsNone(r['layers'][-1]['bias_parameter_mean'])
        self.assertEqual(r['layers'][-1]['bias_parameter_nonfinite_count'],2)
        json.dumps(r,allow_nan=False)


if __name__=='__main__':unittest.main()
