import unittest
import tempfile
from pathlib import Path
import numpy as np
import torch
from model_data import CiresanMLP
from layerdrop_hypotheses.audit import masked_forward, enumerate_logits, parity_check, summarize

class AuditTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(31)
        self.x=torch.rand(12,8)*255
    def model(self,recipe):
        return CiresanMLP(recipe=recipe,pmax=.4,widths=(8,9,8,7,6,5,3)).eval()
    def test_full_parity_all_recipes(self):
        for recipe in ('plain','residual','sd_constant','sd_annealed','residual_unit_dropout'):
            m=self.model(recipe)
            self.assertTrue(torch.equal(m(self.x),masked_forward(m,self.x,15)))
    def test_zero_mask_never_calls_body(self):
        m=self.model('residual')
        def forbidden(*_): raise AssertionError('Skipped affine executed')
        hs=[l.register_forward_hook(forbidden) for l in m.layers[1:-1]]
        z=masked_forward(m,self.x,0)
        h=torch.relu(m.layers[0](self.x*m.input_scale))[:,:5]
        self.assertTrue(torch.equal(z,m.layers[-1](h)))
        for h in hs:h.remove()
    def test_plain_surgery_not_added_shortcut(self):
        m=self.model('plain'); z=masked_forward(m,self.x,14)
        h=torch.relu(m.layers[0](self.x*m.input_scale))[:,:8]
        for l in m.layers[2:-1]:h=torch.relu(l(h))
        self.assertTrue(torch.equal(z,m.layers[-1](h)))
    def test_enumeration_batch_and_endpoints(self):
        m=self.model('sd_constant')
        z=enumerate_logits(m,self.x,batch_size=5)
        self.assertEqual(z.shape,(16,12,3))
        torch.testing.assert_close(z[0],masked_forward(m,self.x,[0]*4))
        torch.testing.assert_close(z[15],masked_forward(m,self.x,[1]*4))
        self.assertFalse(torch.equal(masked_forward(m,self.x,[2]*4),z[15]))
    def test_source_parity_rejects_predictions(self):
        z=np.array([[2.,0],[0,2.]])
        ce=float(np.log1p(np.exp(-2)))
        ref={'accuracy':1.,'loss':ce,'wrong_indices':[]}
        self.assertTrue(parity_check(z,np.array([0,1]),ref)['passed'])
        with self.assertRaises(AssertionError):parity_check(z,np.array([1,0]),ref)
    def test_summary_sidecar(self):
        z=enumerate_logits(self.model('residual'),self.x).numpy()
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'test.npz';s=summarize(z,np.arange(12)%3,p)
            d=np.load(p)
            self.assertEqual(d['energy_by_degree'].shape,(5,12))
            self.assertEqual(d['pred'].shape,(16,12))
            self.assertNotIn('per_example',s['classification'])
            self.assertTrue(s['uniform_cardinality']['both_basis_reconstructions_verified'])

if __name__=='__main__':unittest.main()
