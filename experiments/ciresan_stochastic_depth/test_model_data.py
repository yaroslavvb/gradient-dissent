"""Small CPU numerical checks; no network downloads or paid computation."""
import gzip
import struct
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch import nn
from torch.nn import functional as F

try:
    from . import model_data as md
    from .model_data import CiresanMLP, split_indices, _decode_idx, load_mnist
except ImportError:
    import model_data as md
    from model_data import CiresanMLP, split_indices, _decode_idx, load_mnist

W = (4, 7, 6, 5, 4, 3, 2)


class ModelChecks(unittest.TestCase):
    def test_reference_initialization_and_plain_operations(self):
        torch.manual_seed(173)
        actual = CiresanMLP("plain", input_scale=1, output_relu=True, widths=W)
        torch.manual_seed(173)
        reference = nn.Sequential(*(module for a,b in zip(W[:-1], W[1:]) for module in (nn.Linear(a,b), nn.ReLU())))
        for a,b in zip(actual.parameters(), reference.parameters()):
            self.assertTrue(torch.equal(a,b))
        x = torch.randn(6,4, requires_grad=True)
        y = x.detach().clone().requires_grad_()
        self.assertTrue(torch.equal(actual(x), reference(y)))
        actual(x).sum().backward(); reference(y).sum().backward()
        self.assertTrue(torch.equal(x.grad, y.grad))
        for a,b in zip(actual.parameters(), reference.parameters()):
            self.assertTrue(torch.equal(a.grad,b.grad))

    def test_zero_drop_and_shared_parameter_count(self):
        torch.manual_seed(8)
        dense = CiresanMLP("residual", widths=W)
        for recipe in ("sd_constant", "sd_annealed", "residual_unit_dropout"):
            torch.manual_seed(8)
            model = CiresanMLP(recipe, widths=W)
            self.assertEqual(sum(p.numel() for p in model.parameters()), sum(p.numel() for p in dense.parameters()))
            x = torch.randn(11,4)
            kwargs = {"drop_probs": (0,)*4} if recipe == "sd_annealed" else {}
            self.assertTrue(torch.equal(model(x, **kwargs), dense(x)))

    def test_dropped_branch_never_executes_or_receives_gradient(self):
        model = CiresanMLP("sd_constant", pmax=.5, input_scale=1, widths=W)
        x = torch.rand(5,4, requires_grad=True)
        with patch.object(model.layers[2], "forward", side_effect=AssertionError("dropped Linear executed")):
            out = model(x, active=(True,False,True,True))
            out.square().sum().backward()
        self.assertIsNone(model.layers[2].weight.grad)
        self.assertIsNotNone(model.layers[1].weight.grad)
        self.assertTrue(torch.isfinite(x.grad).all())

    def test_all_dropped_prefix_composition(self):
        model = CiresanMLP("sd_constant", input_scale=1, widths=W)
        x = torch.randn(3,4)
        expected = model.layers[-1](F.relu(model.layers[0](x))[:, :W[-2]])
        for layer in model.layers[1:-1]:
            layer.register_forward_hook(lambda *_: self.fail("A dropped branch executed"))
        self.assertTrue(torch.equal(model(x, drop_probs=(1,)*4, active=(False,)*4), expected))

    def test_eval_all_active_unscaled_no_rng(self):
        model = CiresanMLP("sd_annealed", pmax=.8, widths=W).eval()
        dense = CiresanMLP("residual", widths=W).eval()
        dense.load_state_dict(model.state_dict())
        rng = torch.Generator().manual_seed(5)
        state = rng.get_state().clone()
        x = torch.randn(4,4)
        self.assertTrue(torch.equal(model(x, active=(False,)*4, drop_probs=(1,)*4, generator=rng), dense(x)))
        self.assertTrue(torch.equal(state, rng.get_state()))

    def test_single_branch_expected_preactivation_not_relu_claim(self):
        model = CiresanMLP("sd_constant", widths=(2,2,2,1), input_scale=1)
        # Positive preactivations make ReLU identity in this one-branch test.
        with torch.no_grad():
            for layer in model.layers:
                layer.weight.fill_(.2); layer.bias.fill_(.3)
        x = torch.tensor([[1.,2.], [3.,4.]])
        p = .6
        dropped = model(x, drop_probs=(p,), active=(False,))
        kept = model(x, drop_probs=(p,), active=(True,))
        model.eval()
        torch.testing.assert_close(p*dropped+(1-p)*kept, model(x))
        # No assertion of unbiasedness through arbitrary nonlinear/multiple blocks.

    def test_seeded_masks_and_unit_dropout(self):
        x = torch.ones(12,4)
        for recipe in ("sd_constant", "unit_dropout", "residual_unit_dropout"):
            model = CiresanMLP(recipe, pmax=.5, widths=W)
            a = model(x, generator=torch.Generator().manual_seed(42))
            b = model(x, generator=torch.Generator().manual_seed(42))
            self.assertTrue(torch.equal(a,b))
            model.eval()
            self.assertTrue(torch.equal(model(x),model(x)))

    def test_plain_controls_ignore_runner_gates(self):
        x = torch.ones(12,4)
        for recipe in ("plain", "unit_dropout"):
            model = CiresanMLP(recipe, pmax=.5, widths=W)
            a = model(x, generator=torch.Generator().manual_seed(77))
            b = model(x, drop_probs=(0,)*4, active=[False]*4,
                      generator=torch.Generator().manual_seed(77))
            self.assertTrue(torch.equal(a,b))

    def test_validation(self):
        model = CiresanMLP("sd_annealed", widths=W)
        with self.assertRaises(ValueError): model(torch.ones(2,4))
        with self.assertRaises(ValueError): model(torch.ones(2,4), drop_probs=(.1,)*3)
        with self.assertRaises(ValueError): model(torch.ones(2,4), drop_probs=(float('nan'),)*4)
        with self.assertRaises(ValueError): model(torch.ones(2,4), drop_probs=(1,)*4, active=(True,)*4)


class DataChecks(unittest.TestCase):
    def test_tuning_loader_never_requests_test_files(self):
        requested = []
        def download(root, name):
            requested.append(name)
            self.assertTrue(name.startswith("train-"))
            return b"fixture", {"sha256": "fixture"}
        def decode(blob, images, count):
            return torch.zeros(100,28,28,dtype=torch.uint8) if images else torch.arange(100)%10
        with tempfile.TemporaryDirectory() as root, patch.object(md,"_download",side_effect=download), \
             patch.object(md,"_decode_idx",side_effect=decode), \
             patch.object(md,"split_indices",return_value=(torch.arange(80),torch.arange(80,100))):
            result = load_mnist(root,include_test=False)
        self.assertEqual(len(requested),2)
        self.assertNotIn("test_x",result)
        self.assertNotIn("test_y",result)
        self.assertFalse(result["metadata"]["test_included"])
        self.assertEqual(result["train_x"].dtype,torch.float32)

    def test_nested_disjoint_complete_split(self):
        train,val = split_indices()
        small,val2 = split_indices(train_size=10000)
        self.assertTrue(torch.equal(small,train[:10000]))
        self.assertTrue(torch.equal(val,val2))
        self.assertEqual(torch.unique(torch.cat((train,val))).numel(),60000)

    def test_idx_parser_raw_units_and_labels(self):
        blob = gzip.compress(struct.pack('>IIII',2051,2,28,28)+bytes(range(256))*6+bytes(range(32)))
        x = _decode_idx(blob,True,2)
        self.assertEqual(x.shape,(2,28,28))
        self.assertEqual(x.dtype,torch.uint8)
        self.assertEqual(int(x.max()),255)
        y = _decode_idx(gzip.compress(struct.pack('>II',2049,2)+bytes((0,9))),False,2)
        self.assertEqual(y.tolist(),[0,9])
        with self.assertRaises(ValueError):
            _decode_idx(gzip.compress(struct.pack('>II',2049,2)+bytes((0,10))),False,2)
        with self.assertRaises(ValueError): _decode_idx(blob[:-2],True,2)


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main()
