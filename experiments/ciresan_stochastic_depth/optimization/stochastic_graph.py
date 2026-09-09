"""Full-step CUDA graphs for the four crop-residual Ciresan transitions.

Implementation preparation only: this module launches no jobs or experiments.
Each SD mask gets its own graph. Parameters, gradients, momentum, inputs, and
inverse-survival scales have persistent shared storage. Dropped parameters are
absent from both backward and momentum SGD; optional historical shrinkage still
applies to ALL parameters, exactly as in the source trainer.

Gradients for dropped layers may contain old values in their persistent storage;
only active gradients enter an update. Filter diagnostic gradients by last_mask.
Returned losses alias graph storage: clone before collecting a history.
"""
from __future__ import annotations

import contextlib
import math
import time

import torch
from torch.nn import functional as F


def _cpu_sequence(values, name):
    if isinstance(values, torch.Tensor):
        if values.device.type != "cpu" or values.ndim != 1:
            raise ValueError(f"{name} requires a Python sequence or 1D CPU tensor")
        return values.tolist()
    return list(values)


def validate_mask(active):
    result = tuple(_cpu_sequence(active, "active"))
    if len(result) != 4 or any(type(v) is not bool for v in result):
        raise ValueError("active requires exactly four booleans")
    return result


def inverse_survival(drop_probs):
    probabilities = tuple(float(p) for p in _cpu_sequence(drop_probs, "drop_probs"))
    if len(probabilities) != 4 or any(not math.isfinite(p) or not 0 <= p < 1 for p in probabilities):
        raise ValueError("drop_probs requires four finite probabilities in [0,1)")
    return probabilities, tuple(1/(1-p) for p in probabilities)


def forward_with_mask(model, x, active, survivor_scales):
    """Same crop/add/ReLU operator as model_data; scales are device tensors.

    Active linear branches use multiplication by inverse survival, which can
    differ by FP rounding from source division by a Python survival scalar.
    Ordinary dropout remains per-unit, after each of the five hidden ReLUs.
    """
    h = x.reshape(-1, model.widths[0]) * model.input_scale
    h = F.relu(model.layers[0](h))
    if model.unit_dropout:
        h = F.dropout(h,p=model.pmax,training=True)
    for index, layer in enumerate(model.layers[1:-1]):
        if model.residual:
            skip = h[..., :layer.out_features]
            h = F.relu(skip + layer(h)*survivor_scales[index]) if active[index] else F.relu(skip)
        else:
            h = F.relu(layer(h))
        if model.unit_dropout:
            h = F.dropout(h,p=model.pmax,training=True)
    h = model.layers[-1](h)
    return F.relu(h) if model.output_relu else h


def active_parameter_indices(model, active):
    """Indices in model.parameters(), excluding the skipped affine branches."""
    kept = {id(p) for index, layer in enumerate(model.layers)
            if index in (0,len(model.layers)-1) or not model.residual or active[index-1]
            for p in layer.parameters()}
    return tuple(i for i,p in enumerate(model.parameters()) if id(p) in kept)


@torch.no_grad()
def update_active_sgd(parameters, momentum_buffers, indices, *, lr, momentum, shrinkage):
    """Dampening-zero, non-Nesterov SGD; absent branches freeze momentum.

    Zero initial momentum is mathematically/numerically the fresh SGD state
    for dampening=0. Parameter shrinkage follows the historical decoupled step.
    """
    chosen = [parameters[i] for i in indices]
    gradients = [parameters[i].grad for i in indices]
    if momentum:
        buffers = [momentum_buffers[i] for i in indices]
        torch._foreach_mul_(buffers,momentum)
        torch._foreach_add_(buffers,gradients)
        torch._foreach_add_(chosen,buffers,alpha=-lr)
    else:
        torch._foreach_add_(chosen,gradients,alpha=-lr)
    if shrinkage:
        torch._foreach_mul_(parameters,1-shrinkage)


class ResidualGraphStep:
    def __init__(self, model, lr, example_x, example_y, *, momentum=.9,
                 shrinkage=0., precision="fp32", warmup_steps=1,
                 momentum_buffers=None):
        started = time.perf_counter()
        if example_x.device.type != "cuda" or example_y.device != example_x.device:
            raise ValueError("ResidualGraphStep requires CUDA-resident example batches")
        if len(model.layers) != 6:
            raise ValueError("Expected stem, four body transitions, and head")
        if precision not in {"fp32","bf16"}:
            raise ValueError("precision must be fp32 or bf16")
        if not all(math.isfinite(v) for v in (lr,momentum,shrinkage)) or lr <= 0 or not 0 <= momentum < 1 or not 0 <= shrinkage < 1:
            raise ValueError("Invalid SGD or shrinkage hyperparameter")
        if warmup_steps < 1:
            raise ValueError("At least one warmup per mask is required")
        self.model = model
        self.lr, self.momentum, self.shrinkage = float(lr),float(momentum),float(shrinkage)
        self._fixed_configuration = (self.lr,self.momentum,self.shrinkage,model.input_scale,model.output_relu,model.pmax,model.recipe)
        self.precision, self.device = precision,example_x.device
        self.parameters = list(model.parameters())
        if any(p.device != self.device or p.dtype != torch.float32 or not p.requires_grad for p in self.parameters):
            raise ValueError("All parameters must be trainable FP32 tensors on the input CUDA device")
        if {id(p) for layer in model.layers for p in layer.parameters()} != {id(p) for p in self.parameters}:
            raise ValueError("All trainable parameters must belong to the six affine layers")
        self.momentum_buffers = ([torch.zeros_like(p) for p in self.parameters] if momentum_buffers is None else list(momentum_buffers))
        if len(self.momentum_buffers) != len(self.parameters) or any(b.shape != p.shape or b.dtype != p.dtype or b.device != p.device for b,p in zip(self.momentum_buffers,self.parameters)):
            raise ValueError("Momentum buffers must match all parameters")
        self.static_x,self.static_y = example_x.clone(),example_y.clone()
        self.survivor_scales = torch.ones(4,device=self.device,dtype=torch.float32)
        self.drop_probs = (0.,)*4
        self.graphs,self.losses,self.indices = {},{},{}
        self.last_mask = None
        self._shapes = (tuple(example_x.shape),tuple(example_y.shape),example_x.dtype,example_y.dtype)
        self.model.train()
        self._memory_before = torch.cuda.memory_allocated(self.device)
        masks = [tuple(bool(i & (1<<j)) for j in range(4)) for i in range(16)] if model.stochastic_depth else [(True,)*4]
        snapshot = self._snapshot()
        for p in self.parameters:
            if p.grad is None:
                p.grad = torch.zeros_like(p)
        current = torch.cuda.current_stream(self.device)
        side = torch.cuda.Stream(device=self.device)
        try:
            for active in masks:
                self.indices[active] = active_parameter_indices(model,active)
                self._restore(snapshot)
                side.wait_stream(current)
                with torch.cuda.stream(side):
                    for _ in range(warmup_steps):
                        self._step(active)
                current.wait_stream(side)
                torch.cuda.synchronize(self.device)
                self._restore(snapshot)
                torch.cuda.synchronize(self.device)
                graph = torch.cuda.CUDAGraph()
                # Deliberately separate pools; never assume concurrent/ordered
                # replay properties that would justify sharing private pools.
                with torch.cuda.graph(graph,stream=side):
                    loss = self._step(active)
                self.graphs[active],self.losses[active] = graph,loss
                current.wait_stream(side)
                torch.cuda.synchronize(self.device)
        finally:
            torch.cuda.synchronize(self.device)
            self._restore(snapshot)
            torch.cuda.synchronize(self.device)
        if model.stochastic_depth:
            self.update_scales(model.default_drop_probs)
        del snapshot
        torch.cuda.synchronize(self.device)
        self.setup_seconds = time.perf_counter()-started
        self.allocated_after_setup = torch.cuda.memory_allocated(self.device)
        self.reserved_after_setup = torch.cuda.memory_reserved(self.device)

    def _snapshot(self):
        return {"parameters":[p.detach().clone() for p in self.parameters],
                "momentum":[b.clone() for b in self.momentum_buffers],
                "grads":[None if p.grad is None else p.grad.detach().clone() for p in self.parameters],
                "buffers":[(b,b.clone()) for b in self.model.buffers()],
                "cpu_rng":torch.get_rng_state().clone(),
                "cuda_rng":torch.cuda.get_rng_state(self.device).clone()}

    @torch.no_grad()
    def _restore(self, state):
        for p,saved,grad in zip(self.parameters,state["parameters"],state["grads"]):
            p.copy_(saved)
            if p.grad is not None:
                p.grad.zero_() if grad is None else p.grad.copy_(grad)
        for buffer,saved in zip(self.momentum_buffers,state["momentum"]):
            buffer.copy_(saved)
        for buffer,saved in state["buffers"]:
            buffer.copy_(saved)
        torch.set_rng_state(state["cpu_rng"])
        torch.cuda.set_rng_state(state["cuda_rng"],self.device)

    def _step(self, active):
        indices = self.indices[active]
        torch._foreach_zero_([self.parameters[i].grad for i in indices])
        context = torch.autocast("cuda",dtype=torch.bfloat16) if self.precision=="bf16" else contextlib.nullcontext()
        with context:
            logits = forward_with_mask(self.model,self.static_x,active,self.survivor_scales)
            loss = F.cross_entropy(logits,self.static_y)
        loss.backward()
        update_active_sgd(self.parameters,self.momentum_buffers,indices,lr=self.lr,
                          momentum=self.momentum,shrinkage=self.shrinkage)
        return loss.detach()

    def update_scales(self, drop_probs):
        """Call once per epoch for an annealed schedule; no graph recapture."""
        probabilities,scales = inverse_survival(drop_probs)
        if not self.model.stochastic_depth and any(probabilities):
            raise ValueError("Only stochastic-depth recipes accept positive branch drop rates")
        self.drop_probs = probabilities
        self.survivor_scales.copy_(torch.tensor(scales,dtype=torch.float32,device="cpu"))

    def __call__(self, x, y, active=None, drop_probs=None, *, generator=None):
        if (tuple(x.shape),tuple(y.shape),x.dtype,y.dtype) != self._shapes or x.device != self.device or y.device != self.device:
            raise ValueError("Batch shapes/dtypes/devices must match graph capture")
        configuration = (self.lr,self.momentum,self.shrinkage,self.model.input_scale,self.model.output_relu,self.model.pmax,self.model.recipe)
        if configuration != self._fixed_configuration:
            raise ValueError("Optimizer/model scalar settings changed; recapture required")
        if drop_probs is not None:
            self.update_scales(drop_probs)
        if active is None:
            if self.model.stochastic_depth and any(self.drop_probs):
                if generator is not None and generator.device.type != "cpu":
                    raise ValueError("Mask sampling requires a CPU generator")
                draws = torch.rand(4,device="cpu",generator=generator).tolist()
                active = tuple(u>=p for u,p in zip(draws,self.drop_probs))
            else:
                active = (True,)*4
        active = validate_mask(active)
        if active not in self.graphs:
            raise ValueError("This dense/ordinary-dropout recipe has only an all-active graph")
        self.static_x.copy_(x)
        self.static_y.copy_(y)
        self.graphs[active].replay()
        self.last_mask = active
        return self.losses[active]

    @property
    def metadata(self):
        return {"graph_count":len(self.graphs),"setup_seconds":self.setup_seconds,"precision":self.precision,
                "allocated_before_capture_bytes":self._memory_before,
                "allocated_after_setup_bytes":self.allocated_after_setup,
                "reserved_after_setup_bytes":self.reserved_after_setup,
                "parameter_count":sum(p.numel() for p in self.parameters),
                "shared_parameter_gradient_momentum_storage":True,"separate_private_graph_pools":True,
                "dropout_draw":"one CPU Bernoulli per body transition per minibatch, or externally supplied mask",
                "dropped_parameter_momentum_updated":False,"shrinkage_applies_to_dropped_parameters":bool(self.shrinkage),
                "loss_aliases_static_storage":True,"input_copies_inside_capture":False,
                "scale_buffer_dtype":"float32","scale_update_requires_recapture":False}
