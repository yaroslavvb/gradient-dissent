"""CUDA-graph execution of the controlled Ciresan stochastic-depth study.

Preserves train.py's split, independent RNG streams, epoch schedule, validation
checkpoint selection and official-test gate. Default/main protocol: normalized
pixels, linear logits, B64, FP32 with high matmul precision, constant-rate SGD.
Graph setup and per-epoch mask/order/scaling preparation are separately timed.
No resources are provisioned here. Original train.py remains unchanged.
"""
from pathlib import Path
import argparse
import gc
import hashlib
import importlib
import io
import json
import math
import random
import sys
import time
import unittest

import numpy as np
import torch
from torch import nn

try:
    from .model_data import CiresanMLP, load_mnist
    from .optimization.stochastic_graph import ResidualGraphStep
except ImportError:
    from model_data import CiresanMLP, load_mnist
    from optimization.stochastic_graph import ResidualGraphStep


HERE = Path(__file__).resolve().parent


def synchronize(device):
    torch.cuda.synchronize(device)


@torch.no_grad()
def evaluate(model, x, y, batch_size=2048, include_errors=False):
    """Dense, unscaled inference. Nonfinite logits/loss fail explicitly."""
    model.eval()
    losses,correct,predictions,finite = [],[],[],[]
    for start in range(0,len(y),batch_size):
        logits=model(x[start:start+batch_size])
        target=y[start:start+batch_size]
        losses.append(nn.functional.cross_entropy(logits,target,reduction="sum"))
        finite.append(torch.isfinite(logits).all())
        pred=logits.argmax(1)
        correct.append((pred==target).sum())
        if include_errors: predictions.append(pred)
    loss=float(torch.stack(losses).sum().item()/len(y))
    if not math.isfinite(loss) or not bool(torch.stack(finite).all()):
        raise FloatingPointError("Nonfinite dense-evaluation logits or loss")
    result={"loss":loss,"accuracy":float(torch.stack(correct).sum().item()/len(y)),"n":len(y)}
    if include_errors:
        pred=torch.cat(predictions)
        result["wrong_indices"]=torch.where(pred!=y)[0].cpu().tolist()
        result["errors"]=len(result["wrong_indices"])
    return result


def probabilities(spec, epoch):
    pmax=spec.get("pmax",0.)
    if spec["recipe"]=="sd_annealed":
        pmax*=1-epoch/max(1,spec["epochs"]-1)
    if spec["recipe"] not in ("sd_constant","sd_annealed"):
        pmax=0.
    return [pmax*i/4 for i in range(1,5)]


def validate_kernels():
    """Runs on an already-present CUDA device, before experimental seeding."""
    if not torch.cuda.is_available():
        raise RuntimeError("Kernel validation requires an existing CUDA device")
    started=time.perf_counter()
    module_name=(__package__+".optimization.test_stochastic_graph") if __package__ else "optimization.test_stochastic_graph"
    module=importlib.import_module(module_name)
    suite=unittest.defaultTestLoader.loadTestsFromModule(module)
    stream=io.StringIO()
    result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
    report={"tests_run":result.testsRun,"failures":len(result.failures),"errors":len(result.errors),
            "skipped":len(result.skipped),"passed":result.wasSuccessful() and not result.skipped,
            "seconds":time.perf_counter()-started,"output":stream.getvalue()}
    if not report["passed"]:
        raise RuntimeError("CUDA graph kernel validation failed:\n"+report["output"])
    # Tests must not leave private graph pools in the experiment memory count.
    del suite,result,module
    gc.collect()
    torch.cuda.empty_cache()
    return report


def _source_hashes():
    names=("train_optimized.py","model_data.py","optimization/stochastic_graph.py",
           "optimization/test_stochastic_graph.py","train.py")
    return {name:hashlib.sha256((HERE/name).read_bytes()).hexdigest() for name in names}


def _check_spec(spec):
    if spec.get("stage") not in {"pilot","tune","evaluate"}:
        raise ValueError("stage must be pilot, tune or evaluate")
    if spec.get("recipe") not in {"plain","residual","sd_constant","sd_annealed","residual_unit_dropout","unit_dropout"}:
        raise ValueError("Unknown controlled-study recipe")
    if int(spec.get("epochs",0))<1 or int(spec.get("eval_every",5))<1:
        raise ValueError("Positive epochs and evaluation interval required")
    if not 0<int(spec.get("batch_size",64))<=int(spec.get("train_size",50000)):
        raise ValueError("Invalid training minibatch size")
    if spec.get("precision","fp32")!="fp32":
        raise ValueError("The controlled study uses FP32; BF16 is a separate baseline optimization")
    if not math.isfinite(spec["lr"]) or spec["lr"]<=0:
        raise ValueError("A positive constant learning rate is required")
    if spec.get("cohort")=="main" and spec["stage"] in {"tune","evaluate"}:
        expected_pmax={"plain":0.,"residual":0.,"sd_constant":.4,
                       "sd_annealed":.8,"residual_unit_dropout":.2}
        if spec["recipe"] not in expected_pmax or spec.get("pmax")!=expected_pmax[spec["recipe"]]:
            raise ValueError("Main cohort requires its frozen five recipe/probability combinations")
        if (spec["epochs"]!=100 or spec.get("batch_size",64)!=64 or
            spec.get("train_size",50000)!=50000 or spec.get("input_scale",1/255)!=1/255 or
            spec.get("output_relu",False) or spec.get("momentum",.9)!=.9 or
            spec.get("shrinkage",0.)!=0. or spec.get("lr") not in (.01,.03,.1) or
            spec.get("eval_every",5)!=5):
            raise ValueError("Main cohort must preserve the frozen normalized/linear/B64/100-epoch protocol")


def run(spec, root, progress_commit=None):
    _check_spec(spec)
    if not torch.cuda.is_available():
        raise RuntimeError("train_optimized requires an existing CUDA device")
    device=torch.device(spec.get("device","cuda"))
    if device.type!="cuda":
        raise ValueError("train_optimized is CUDA-only")
    # Resolve cuda vs cuda:0 before helper comparisons with parameter devices.
    device=torch.device("cuda",torch.cuda.current_device() if device.index is None else device.index)
    root=Path(root)
    run_dir=root/"runs"/spec["run_id"]
    run_dir.mkdir(parents=True,exist_ok=True)
    with (run_dir/"claimed.json").open("x") as handle:
        json.dump({"spec":spec,"unix":time.time()},handle)
    if progress_commit: progress_commit()
    start_time=time.perf_counter()
    torch.set_num_threads(2)
    # Use this public API exclusively: do not mix old/new backend setters.
    torch.set_float32_matmul_precision("high")
    kernel_validation=validate_kernels() if spec.get("validate_kernels",False) else None
    seed=int(spec["seed"])
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    data=load_mnist(root/"data",train_size=spec.get("train_size",50000),include_test=spec["stage"]=="evaluate")
    data_meta=data["metadata"]
    tensors={k:v.to(device) for k,v in data.items() if isinstance(v,torch.Tensor)}
    model=CiresanMLP(recipe=spec["recipe"],pmax=spec.get("pmax",0.),
                     input_scale=spec.get("input_scale",1/255),output_relu=spec.get("output_relu",False))
    initial_hash=hashlib.sha256(b"".join(p.detach().numpy().tobytes() for p in model.parameters())).hexdigest()
    model.to(device)
    order_rng=torch.Generator(device="cpu").manual_seed(seed+100000)
    mask_rng=torch.Generator(device="cpu").manual_seed(seed+200000)
    torch.manual_seed(seed+300000)
    n=len(tensors["train_y"])
    batch_size=spec.get("batch_size",64)
    steps_per_epoch=n//batch_size
    eval_every=spec.get("eval_every",5)
    bx=torch.empty((batch_size,)+tuple(tensors["train_x"].shape[1:]),device=device,dtype=tensors["train_x"].dtype)
    by=torch.empty(batch_size,device=device,dtype=tensors["train_y"].dtype)
    bootstrap=torch.arange(batch_size,device=device)
    torch.index_select(tensors["train_x"],0,bootstrap,out=bx)
    torch.index_select(tensors["train_y"],0,bootstrap,out=by)
    helper=ResidualGraphStep(model,spec["lr"],bx,by,momentum=spec.get("momentum",.9),
                             shrinkage=spec.get("shrinkage",0.),precision="fp32")
    epoch_losses=torch.empty(steps_per_epoch,device=device,dtype=torch.float32)
    history=[]
    best_val_loss=float("inf")
    best_epoch=None
    train_seconds=0.
    preparation_seconds=0.
    skipped_counts=np.zeros(4,dtype=np.int64)
    grad_observations=[]
    failed=False
    failure_reason=None
    probe_n=min(10000,n)
    sources=_source_hashes()
    order_digest=hashlib.sha256()
    mask_digest=hashlib.sha256()
    for epoch in range(spec["epochs"]):
        prep_start=time.perf_counter()
        probs=probabilities(spec,epoch)
        order_cpu=torch.randperm(n,generator=order_rng)
        order_digest.update(order_cpu.numpy().tobytes())
        order=order_cpu.to(device)
        mask_tensor=torch.rand((steps_per_epoch,4),generator=mask_rng)>=torch.tensor(probs)
        mask_digest.update(mask_tensor.numpy().tobytes())
        active_masks=mask_tensor.tolist()
        skipped_counts+=np.sum(np.logical_not(active_masks),axis=0)
        helper.update_scales(probs)
        model.train()
        synchronize(device)
        preparation_seconds+=time.perf_counter()-prep_start
        epoch_start=time.perf_counter()
        for step in range(steps_per_epoch):
            ids=order[step*batch_size:(step+1)*batch_size]
            torch.index_select(tensors["train_x"],0,ids,out=bx)
            torch.index_select(tensors["train_y"],0,ids,out=by)
            active=active_masks[step]
            loss=helper(bx,by,active=active)
            # Every graph owns a reusable scalar. Copy NOW, before another
            # replay can overwrite it; never append detached graph aliases.
            epoch_losses[step].copy_(loss)
            if step==0 and (epoch%eval_every==0 or epoch==spec["epochs"]-1):
                norms=[]
                for index,layer in enumerate(model.layers):
                    is_active=index in (0,len(model.layers)-1) or not model.residual or active[index-1]
                    grad=layer.weight.grad if is_active else None
                    value=None if grad is None else float(grad.norm().item())
                    norms.append(value if value is None or math.isfinite(value) else None)
                grad_observations.append({"epoch":epoch+1,"active_mask":active,"layer_grad_norms":norms})
        synchronize(device)
        elapsed=time.perf_counter()-epoch_start
        train_seconds+=elapsed
        stochastic_loss=float(epoch_losses.mean().item())
        if not math.isfinite(stochastic_loss):
            failed=True; failure_reason="Nonfinite stochastic training loss"
            history.append({"epoch":epoch+1,"diverged":True,"failure_reason":failure_reason,"training_seconds":train_seconds})
            break
        if epoch==0 or (epoch+1)%eval_every==0 or epoch==spec["epochs"]-1:
            try:
                val=evaluate(model,tensors["val_x"],tensors["val_y"])
                train_eval=evaluate(model,tensors["train_x"][:probe_n],tensors["train_y"][:probe_n])
                with torch.no_grad():
                    logits=model(tensors["val_x"][:2048])
                    if not bool(torch.isfinite(logits).all()):
                        raise FloatingPointError("Nonfinite validation probe logits")
                    predicted=torch.bincount(logits.argmax(1),minlength=10).cpu().tolist()
                    zero_logit_fraction=float((logits==0).float().mean().item())
            except FloatingPointError as error:
                failed=True; failure_reason=str(error)
                history.append({"epoch":epoch+1,"diverged":True,"failure_reason":failure_reason,"training_seconds":train_seconds})
                break
            history.append({"epoch":epoch+1,"stochastic_training_loss":stochastic_loss,
                            "dense_train_probe":train_eval,"validation":val,"training_seconds":train_seconds,
                            "epoch_training_seconds":elapsed,"drop_probabilities":probs,
                            "zero_logit_fraction":zero_logit_fraction,"validation_probe_prediction_counts":predicted})
            if val["loss"]<best_val_loss:
                best_val_loss,best_epoch=val["loss"],epoch+1
                torch.save(model.state_dict(),run_dir/"best.pt")
            (run_dir/"progress.json").write_text(json.dumps({"spec":spec,"history":history},indent=2,allow_nan=False))
            if epoch==0 or (epoch+1)%10==0 or epoch==spec["epochs"]-1:
                print(json.dumps({"run_id":spec["run_id"],"epoch":epoch+1,"val_accuracy":round(val["accuracy"],5),
                                  "val_loss":round(val["loss"],5),"training_seconds":round(train_seconds,2)}),flush=True)
    result={"spec":spec,"run_id":spec["run_id"],"history":history,
            "best_validation_loss":best_val_loss if math.isfinite(best_val_loss) else None,"best_epoch":best_epoch,
            "diverged":failed,"failure_reason":failure_reason,"parameter_count":sum(p.numel() for p in model.parameters()),
            "initial_parameters_sha256":initial_hash,"dataset":data_meta,"training_seconds":train_seconds,
            "epoch_preparation_seconds":preparation_seconds,"graph_setup_seconds":helper.setup_seconds,
            "kernels":helper.metadata,"kernel_validation":kernel_validation,"source_sha256":sources,
            "gradient_observations":grad_observations,"skipped_batch_counts":skipped_counts.tolist(),
            "steps_per_epoch":steps_per_epoch,"training_examples_per_epoch":steps_per_epoch*batch_size,
            "executed_epoch_order_sha256":order_digest.hexdigest(),"executed_epoch_masks_sha256":mask_digest.hexdigest(),
            "hardware":{"device":str(device),"gpu":torch.cuda.get_device_name(device),"torch":str(torch.__version__),
                        "cuda_runtime":torch.version.cuda,"tf32":True,"float32_matmul_precision":torch.get_float32_matmul_precision(),
                        "dtype":"float32","cpu_threads":torch.get_num_threads()}}
    if spec["stage"]=="evaluate" and not failed:
        try:
            result["final_train"]=evaluate(model,tensors["train_x"],tensors["train_y"])
            result["final_test"]=evaluate(model,tensors["test_x"],tensors["test_y"],include_errors=True)
            torch.save(model.state_dict(),run_dir/"final.pt")
            model.load_state_dict(torch.load(run_dir/"best.pt",weights_only=True,map_location=device))
            result["selected_train"]=evaluate(model,tensors["train_x"],tensors["train_y"])
            result["selected_test"]=evaluate(model,tensors["test_x"],tensors["test_y"],include_errors=True)
        except FloatingPointError as error:
            result["diverged"]=True
            result["failure_reason"]=str(error)
    if _source_hashes()!=sources:
        raise RuntimeError("Source files changed during this run")
    result["total_run_seconds"]=time.perf_counter()-start_time
    (run_dir/"result.json").write_text(json.dumps(result,indent=2,allow_nan=False))
    return result


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--spec",required=True)
    parser.add_argument("--root",required=True)
    args=parser.parse_args()
    print(json.dumps(run(json.loads(Path(args.spec).read_text()),Path(args.root)),indent=2))
