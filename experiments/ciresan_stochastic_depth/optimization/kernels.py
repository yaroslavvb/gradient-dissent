"""Eager/compiled-model/full CUDA-graph SGD steps for the plain Ciresan MLP.

CUDA graph setup restores parameters, model buffers, SGD momentum and RNG after
warmup/capture. New momentum/gradient buffers remain allocated and are zeroed:
this is numerically the fresh-SGD state with dampening=0, while preserving the
storage addresses captured by the graph. Graph hyperparameters and shapes are
fixed. Input copies are outside capture; forward, zero_grad, backward, SGD, and
optional historical post-step parameter shrinkage are inside capture.
"""
from __future__ import annotations

import contextlib
import inspect
import time

import torch


def make_sgd(model, lr, momentum=.9, weight_decay=0., fused=False, foreach=True):
    """Fused SGD is explicit and CUDA-only; no silent optimizer substitution."""
    if fused and "fused" not in inspect.signature(torch.optim.SGD).parameters:
        raise RuntimeError("This PyTorch does not provide fused SGD")
    if fused and any(p.device.type != "cuda" for p in model.parameters()):
        raise ValueError("fused SGD requires CUDA parameters")
    return torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum,
                           weight_decay=weight_decay, dampening=0,
                           foreach=False if fused else foreach, fused=fused)


def _configuration(optimizer):
    keys = ("lr", "momentum", "dampening", "weight_decay", "nesterov", "maximize", "foreach", "fused", "differentiable")
    result = []
    for group in optimizer.param_groups:
        values = []
        for key in keys:
            value = group.get(key)
            if isinstance(value, torch.Tensor):
                raise ValueError("This helper requires fixed scalar SGD hyperparameters")
            values.append(value)
        result.append(tuple(values))
    return tuple(result)


class BaselineStep:
    """Call with a fixed-shape batch; returns device loss without synchronizing.

    mode='compile' compiles the MODEL, not Python optimizer orchestration; any
    compiler-internal graph use is distinct from mode='cuda_graph', which
    explicitly captures the entire SGD step. All setup is timed, including a
    compiled-model warmup on CUDA. No GradScaler is used for BF16.
    """
    def __init__(self, model, optimizer, example_x, example_y, *, mode="eager",
                 precision="fp32", compile_mode="reduce-overhead", warmup_steps=3,
                 shrinkage=0.):
        started = time.perf_counter()
        if mode not in {"eager", "compile", "cuda_graph"}:
            raise ValueError("Unknown execution mode")
        if precision not in {"fp32", "bf16"}:
            raise ValueError("precision must be fp32 or bf16")
        if not 0 <= shrinkage < 1:
            raise ValueError("shrinkage must be in [0,1)")
        if not isinstance(optimizer, torch.optim.SGD):
            raise ValueError("BaselineStep supports SGD only")
        if warmup_steps < 1:
            raise ValueError("At least one warmup is required")
        self.model, self.optimizer = model, optimizer
        self.mode, self.precision = mode, precision
        self.shrinkage = float(shrinkage)
        self.params = list(model.parameters())
        self.device = example_x.device
        if not self.params or any(p.device != self.device for p in self.params) or example_y.device != self.device:
            raise ValueError("Model and examples must share a device")
        if any(p.dtype != torch.float32 for p in self.params):
            raise ValueError("Parameters remain FP32; BF16 is autocast only")
        if {id(p) for g in optimizer.param_groups for p in g["params"]} != {id(p) for p in self.params}:
            raise ValueError("Optimizer must own exactly the model parameters")
        self._config = _configuration(optimizer)
        if any(g.get("differentiable") or g.get("dampening", 0) != 0 for g in optimizer.param_groups):
            raise ValueError("Non-differentiable SGD with dampening=0 is required")
        self.x_shape, self.y_shape = tuple(example_x.shape), tuple(example_y.shape)
        self.x_dtype, self.y_dtype = example_x.dtype, example_y.dtype
        self.train_model = model
        model.train()
        self.graph = None
        self.static_x = self.static_y = self.static_loss = None
        if mode in {"cuda_graph", "compile"}:
            if mode == "cuda_graph" and self.device.type != "cuda":
                raise ValueError("Explicit CUDA graphs require a CUDA device")
            if mode == "compile":
                # Forward autocast exits before backward in _step. PyTorch 2.14
                # needs this explicit AOTAutograd setting for that AMP pattern.
                import torch._functorch.config as aot_config
                aot_config.backward_pass_autocast = "off"
                self.train_model = torch.compile(model, mode=compile_mode)
            self._prepare(example_x, example_y, warmup_steps)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        self.setup_seconds = time.perf_counter()-started

    def _autocast(self):
        return (torch.autocast(device_type=self.device.type, dtype=torch.bfloat16)
                if self.precision == "bf16" else contextlib.nullcontext())

    def _step(self, x, y):
        self.optimizer.zero_grad(set_to_none=False)
        with self._autocast():
            logits = self.train_model(x)
            loss = torch.nn.functional.cross_entropy(logits, y)
        loss.backward()
        self.optimizer.step()
        if self.shrinkage:
            with torch.no_grad():
                torch._foreach_mul_(self.params, 1-self.shrinkage)
        return loss.detach()

    def _snapshot(self):
        state = {"params": [p.detach().clone() for p in self.params],
                 "buffers": [(b,b.detach().clone()) for b in self.model.buffers()],
                 "grads": [None if p.grad is None else p.grad.detach().clone() for p in self.params],
                 "momentum": [], "cpu_rng": torch.get_rng_state().clone(),
                 "cuda_rng": None}
        if self.device.type == "cuda":
            state["cuda_rng"] = torch.cuda.get_rng_state(self.device).clone()
        for p in self.params:
            values = self.optimizer.state.get(p,{})
            if set(values)-{"momentum_buffer"}:
                raise ValueError("Unsupported optimizer state beyond SGD momentum")
            buffer = values.get("momentum_buffer")
            state["momentum"].append(None if buffer is None else buffer.detach().clone())
        return state

    @torch.no_grad()
    def _restore(self, state):
        for p, saved, grad, momentum in zip(self.params,state["params"],state["grads"],state["momentum"]):
            p.copy_(saved)
            if p.grad is not None:
                p.grad.zero_() if grad is None else p.grad.copy_(grad)
            buffer = self.optimizer.state.get(p,{}).get("momentum_buffer")
            if buffer is not None:
                buffer.zero_() if momentum is None else buffer.copy_(momentum)
        for buffer, saved in state["buffers"]:
            buffer.copy_(saved)
        torch.set_rng_state(state["cpu_rng"])
        if state["cuda_rng"] is not None:
            torch.cuda.set_rng_state(state["cuda_rng"],self.device)

    def _prepare(self, example_x, example_y, warmup_steps):
        state = self._snapshot()
        self.static_x, self.static_y = example_x.clone(), example_y.clone()
        # Preallocate gradient storage; capture never replaces these tensors.
        for p in self.params:
            if p.requires_grad and p.grad is None:
                p.grad = torch.zeros_like(p)
        try:
            if self.device.type == "cuda":
                current = torch.cuda.current_stream(self.device)
                side = torch.cuda.Stream(device=self.device)
                side.wait_stream(current)
                with torch.cuda.stream(side):
                    for _ in range(warmup_steps):
                        self._step(self.static_x,self.static_y)
                current.wait_stream(side)
                torch.cuda.synchronize(self.device)
                self._restore(state)
                torch.cuda.synchronize(self.device)
                if self.mode == "cuda_graph":
                    self.graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(self.graph,stream=side):
                        self.static_loss = self._step(self.static_x,self.static_y)
            else:
                for _ in range(warmup_steps):
                    self._step(self.static_x,self.static_y)
        finally:
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            self._restore(state)
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)

    def __call__(self, x, y):
        if (tuple(x.shape),tuple(y.shape),x.dtype,y.dtype,x.device,y.device) != (self.x_shape,self.y_shape,self.x_dtype,self.y_dtype,self.device,self.device):
            raise ValueError("A static step requires matching batch shapes, dtypes and devices")
        if self.mode == "cuda_graph":
            if _configuration(self.optimizer) != self._config:
                raise ValueError("Changing SGD hyperparameters after capture requires recapture")
            self.static_x.copy_(x)
            self.static_y.copy_(y)
            self.graph.replay()
            # Graph loss storage is overwritten next replay: callers collecting
            # history must copy/clone it, not append this alias repeatedly.
            return self.static_loss
        return self._step(x,y)

    @property
    def metadata(self):
        return {"mode":self.mode,"precision":self.precision,"setup_seconds":self.setup_seconds,
                "shrinkage":self.shrinkage,"input_copy_inside_graph":False,
                "full_step_explicit_graph":self.mode=="cuda_graph",
                "loss_aliases_static_storage":self.mode=="cuda_graph"}
