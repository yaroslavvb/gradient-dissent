"""Local diagnostic of the raw-pixel residual/ReLU-head pilot failure.

Runs at most 20 SGD updates per configuration. Fixed-first-batch probes perform
backpropagation without stepping the optimizer. No test data or paid services.
CPU FP32 is deliberately recorded as distinct from the A100/TF32 pilot.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path
import platform
import time

import torch
from torch.nn import functional as F

from model_data import CiresanMLP, load_mnist


HERE = Path(__file__).resolve().parent


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tensor_hash(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def parameter_hash(model):
    digest = hashlib.sha256()
    for param in model.parameters():
        digest.update(param.detach().contiguous().numpy().tobytes())
    return digest.hexdigest()


def scalar(value):
    number = float(value)
    return number if math.isfinite(number) else None


def statistics(tensor):
    x = tensor.detach().double()
    return {"shape": list(x.shape), "rms": scalar(x.square().mean().sqrt()),
            "mean": scalar(x.mean()), "min": scalar(x.min()), "max": scalar(x.max()),
            "abs_max": scalar(x.abs().max()),
            "zero_fraction": scalar((x == 0).double().mean()),
            "positive_fraction": scalar((x > 0).double().mean()),
            "finite_fraction": scalar(torch.isfinite(x).double().mean())}


class Capture:
    """Linear inputs are post-ReLU hidden activations from the previous block.

    Linear outputs are affine branches BEFORE residual addition and/or ReLU.
    These names prevent mislabeling branch outputs as complete block outputs.
    """
    def __init__(self, model):
        self.data = {}
        self.head_affine = None
        self.handles = []
        for index, layer in enumerate(model.layers):
            self.handles.append(layer.register_forward_hook(self.hook(index, index == len(model.layers)-1)))

    def hook(self, index, is_head):
        def capture(module, args, output):
            self.data[str(index)] = {"linear_input": statistics(args[0]), "affine_branch_output": statistics(output)}
            if is_head:
                self.head_affine = output.detach()
        return capture

    def close(self):
        for handle in self.handles:
            handle.remove()


def observe(model, capture, x, y, optimizer, updates_completed, batch_role, batch_indices):
    optimizer.zero_grad(set_to_none=True)
    capture.data = {}
    logits = model(x)
    loss = F.cross_entropy(logits, y)
    finite_loss = bool(torch.isfinite(loss))
    if finite_loss:
        loss.backward()
    layer_grads = []
    total_sq = 0.
    for layer in model.layers:
        weight = layer.weight.grad
        bias = layer.bias.grad
        norms = {"weight_l2": None if weight is None else scalar(weight.double().norm()),
                 "bias_l2": None if bias is None else scalar(bias.double().norm())}
        layer_grads.append(norms)
        for grad in (weight, bias):
            if grad is not None:
                total_sq += float(grad.detach().double().square().sum())
    z = capture.head_affine
    head = model.layers[-1]
    momentum = optimizer.state.get(head.weight, {}).get("momentum_buffer")
    result = {"updates_completed": updates_completed, "batch_role": batch_role,
              "batch_indices_sha256": tensor_hash(batch_indices),
              "loss": scalar(loss.detach()), "finite_loss": finite_loss,
              "accuracy": scalar((logits.argmax(1) == y).double().mean()),
              "layers": capture.data, "logits": statistics(logits),
              "logit_nonzero_fraction_per_class": (logits.detach() != 0).double().mean(0).tolist(),
              "head_affine_positive_fraction_per_class": (z > 0).double().mean(0).tolist(),
              "all_logits_zero": bool((logits.detach() == 0).all()),
              "all_head_preactivations_strictly_negative": bool((z < 0).all()),
              "layer_gradient_norms": layer_grads,
              "total_gradient_l2": scalar(math.sqrt(total_sq)),
              "head_weight": statistics(head.weight), "head_bias": statistics(head.bias),
              "head_momentum_l2": None if momentum is None else scalar(momentum.double().norm())}
    return result, finite_loss and all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in model.parameters())


def algebraic_example():
    result = {}
    for relu in (True, False):
        z = torch.tensor([[-1., -2., -3.]], dtype=torch.float64, requires_grad=True)
        logits = F.relu(z) if relu else z
        loss = F.cross_entropy(logits, torch.tensor([1]))
        loss.backward()
        result["relu_head" if relu else "linear_head"] = {
            "preactivations": z.detach().tolist(), "logits": logits.detach().tolist(),
            "target": 1, "ce": float(loss), "gradient_wrt_preactivations": z.grad.tolist()}
    assert result["relu_head"]["gradient_wrt_preactivations"] == [[0., 0., 0.]]
    assert any(v != 0 for v in result["linear_head"]["gradient_wrt_preactivations"][0])
    result["derivation"] = "dCE/dz_j=(softmax(ReLU(z))_j-1[j=y])*1[z_j>0]. All z_j<0 gives zero gradients and CE=log(C). Without ReLU the final indicator is absent."
    result["qualification"] = "Zero CE gradients alone do not freeze momentum SGD: a nonzero existing momentum buffer or another parameter update can move preactivations out of this region. Permanence is not proved."
    return result


def run(data_root, updates=20, threads=2):
    if not 1 <= updates <= 20:
        raise ValueError("This bounded diagnostic permits only 1..20 updates per case")
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True)
    started = time.perf_counter()
    data = load_mnist(data_root, include_test=False)
    assert "test_x" not in data and not data["metadata"]["test_included"]
    batches = torch.randperm(len(data["train_y"]), generator=torch.Generator().manual_seed(100001))[:updates*64].reshape(updates,64)
    first = batches[0]
    fixed_x, fixed_y = data["train_x"][first], data["train_y"][first]
    probe_after = sorted({n for n in (1,2,5,10,20,updates) if n <= updates})
    results = []
    pilot_path = HERE / "results" / "pilot-source-residual.json"
    pilot = json.loads(pilot_path.read_text()) if pilot_path.exists() else None
    for recipe, scale_name, output_relu in itertools.product(("plain", "residual"), ("raw", "normalized"), (True,False)):
        torch.manual_seed(1)
        scale, lr = (1.,.001) if scale_name == "raw" else (1/255,.03)
        model = CiresanMLP(recipe, input_scale=scale, output_relu=output_relu)
        initial_hash = parameter_hash(model)
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=.9, foreach=True)
        capture = Capture(model)
        case_started = time.perf_counter()
        case = {"id": f"{recipe}-{scale_name}-{'relu' if output_relu else 'linear'}",
                "recipe": recipe, "input_scale": scale, "output_relu": output_relu,
                "lr": lr, "seed": 1, "initial_parameters_sha256": initial_hash,
                "parameter_count": sum(p.numel() for p in model.parameters()),
                "training_batches": [], "fixed_first_batch_probes": [], "stopped_nonfinite": False,
                "updates_completed": 0}
        initial, finite = observe(model,capture,fixed_x,fixed_y,optimizer,0,"fixed_first_batch",first)
        case["fixed_first_batch_probes"].append(initial)
        for step, ids in enumerate(batches):
            record, finite = observe(model,capture,data["train_x"][ids],data["train_y"][ids],optimizer,step,"next_training_batch",ids)
            case["training_batches"].append(record)
            if not finite:
                case["stopped_nonfinite"] = True
                break
            optimizer.step()
            with torch.no_grad():
                torch._foreach_mul_(list(model.parameters()), 1-2e-5)
            case["updates_completed"] = step+1
            if step+1 in probe_after:
                probe, finite = observe(model,capture,fixed_x,fixed_y,optimizer,step+1,"fixed_first_batch",first)
                case["fixed_first_batch_probes"].append(probe)
                if not finite:
                    case["stopped_nonfinite"] = True
                    break
        capture.close()
        case["wall_seconds"] = time.perf_counter()-case_started
        case["final_parameters_sha256"] = parameter_hash(model)
        case["first_observed_zero_logit_training_batch"] = next((r["updates_completed"] for r in case["training_batches"] if r["all_logits_zero"]),None)
        results.append(case)
        print(json.dumps({"id":case["id"],"updates":case["updates_completed"],"initial_loss":initial["loss"],"last_probe_loss":case["fixed_first_batch_probes"][-1]["loss"],"last_probe_zero_fraction":case["fixed_first_batch_probes"][-1]["logits"]["zero_fraction"],"nonfinite":case["stopped_nonfinite"]}),flush=True)
        del optimizer, model
    assert len({r["initial_parameters_sha256"] for r in results}) == 1
    assert len({r["fixed_first_batch_probes"][0]["batch_indices_sha256"] for r in results}) == 1
    output = {"status":"completed", "purpose":"Local mechanistic diagnostic, not an epoch reproduction or hyperparameter selection", "configuration_count":len(results),
              "protocol":{"seed":1,"order_seed":100001,"batch_size":64,"maximum_training_updates_per_case":updates,
                          "fixed_first_batch_probe_updates":[0]+probe_after,"momentum":.9,"shrinkage_after_each_update":2e-5,
                          "raw_lr":.001,"normalized_lr":.03,"test_data_accessed":False,
                          "comparison_caveat":"Within a fixed scale, architecture and head pairs each change one flag at a time. Raw-versus-normalized pairs also change learning rate; they do not isolate input scaling alone."},
              "hardware":{"device":"cpu","torch":str(torch.__version__),"threads":threads,"architecture":platform.machine(),"dtype":"float32","tf32":False},
              "dataset":data["metadata"], "training_batch_indices_sha256":tensor_hash(batches),
              "first_batch_target_counts":torch.bincount(fixed_y,minlength=10).tolist(),
              "source_sha256":{name:file_hash(HERE/name) for name in ("diagnose_collapse.py","model_data.py","train.py","reference/train_ciresan_new.py","reference/util.py")},
              "pilot_reference":None if pilot is None else {"file":"results/pilot-source-residual.json","sha256":file_hash(pilot_path),
                   "spec":pilot["spec"],"hardware":pilot["hardware"],"initial_parameters_sha256":pilot["initial_parameters_sha256"],
                   "cpu_initial_hash_equals_pilot":results[0]["initial_parameters_sha256"]==pilot["initial_parameters_sha256"],
                   "training_dataset_hash_equals_pilot":data["metadata"]["training_data_sha256"]==pilot["dataset"]["training_data_sha256"],
                   "reported_first_epoch_zero_logit_fraction":pilot["history"][0]["zero_logit_fraction"]},
              "algebraic_example":algebraic_example(),"cases":results,"local_spend_usd":0.,"wall_seconds":time.perf_counter()-started,
              "interpretation_limits":["One seed and at most20 minibatches; no validation/test selection or inference about eventual generalization.",
                "CPU FP32 arithmetic differs from the pilot's A100 TF32; this is an independently executed diagnostic.",
                "The first batch is a training probe; zero outputs there do not prove collapse on every MNIST image.",
                "A ReLU head's negative region blocks the CE gradient; existing optimizer momentum prevents an unconditional claim of an absorbing parameter state.",
                "Architecture comparisons add deterministic crop residuals; this is an architectural adaptation, not stochastic depth on the original plain network."]}
    return output


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root",default="/tmp/ciresan-diagnostic-data")
    parser.add_argument("--output",default=str(HERE/"results"/"collapse-diagnostic.json"))
    parser.add_argument("--updates",type=int,default=20)
    parser.add_argument("--threads",type=int,default=2)
    args=parser.parse_args()
    result=run(args.data_root,args.updates,args.threads)
    destination=Path(args.output)
    destination.parent.mkdir(parents=True,exist_ok=True)
    destination.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    print(json.dumps({"output":str(destination),"wall_seconds":result["wall_seconds"]}),flush=True)


if __name__=="__main__":
    main()
