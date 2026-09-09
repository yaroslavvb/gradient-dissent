"""Epoch/probe telemetry for CiresanMLP; never add hooks to captured training.

All entry points are called outside the training-step timer. Device reductions
are packed before a host read; no per-minibatch host synchronization is added.
Probe gradients use autograd.grad of SUM CE with respect to affine outputs,
preserving persistent parameter .grad storage and all model/RNG state.
"""
import math
import time

import torch
from torch.nn import functional as F


def _device(model):return next(model.parameters()).device


def _start(device):
    if device.type=='cuda':torch.cuda.synchronize(device)
    return time.perf_counter()


class _Collector:
    """Scalar reductions stay on device until one packed transfer at finish."""
    def __init__(self,device):self.device=device;self.keys=[];self.values=[]
    def add(self,key,value):
        self.keys.append(key)
        self.values.append(value.detach().to(device=self.device,dtype=torch.float64).reshape(()))
    def summary(self,prefix,tensor):
        value=tensor.detach().float()
        self.add(prefix+'/mean',value.mean())
        norm=torch.linalg.vector_norm(value)
        self.add(prefix+'/l2',norm);self.add(prefix+'/rms',norm/math.sqrt(value.numel()))
        self.add(prefix+'/nonfinite_count',(~torch.isfinite(value)).sum())
    def finish(self):
        if not self.values:return {}
        values=torch.stack(self.values).cpu().tolist()
        return {key:float(value) if math.isfinite(value) else None for key,value in zip(self.keys,values)}


def parameter_snapshot(model,*,step=None):
    """GPU-resident clone; asynchronous enqueue time is NOT GPU elapsed time.

    Take at an epoch boundary or one explicitly designated step, not every
    minibatch. The trainer's separate telemetry clock should cover completion.
    """
    started=time.perf_counter()
    return {'values':{name:p.detach().clone() for name,p in model.named_parameters()},
            'step':step,'enqueue_seconds':time.perf_counter()-started,
            'timing_note':'Asynchronous clone enqueue; not synchronized GPU duration'}


def layer_statistics(model,*,previous_parameters=None,last_active=None,helper=None,
                     interval_steps=None,update_interval='since_parameter_snapshot'):
    """Parameter and LAST training-gradient statistics, plus measured deltas.

    Dropped residual-branch gradient buffers may be stale after graph replay.
    They are absent from metrics when last_active marks a drop. Without a
    supplied last mask/helper, residual training gradients are withheld.
    """
    device=_device(model);started=_start(device);collector=_Collector(device)
    if helper is not None and last_active is None:last_active=helper.last_mask
    if last_active is not None:
        last_active=tuple(last_active)
        if len(last_active)!=len(model.layers)-2 or any(type(v) is not bool for v in last_active):
            raise ValueError('last_active must be Python booleans for body transitions')
    previous=None if previous_parameters is None else previous_parameters.get('values',previous_parameters)
    named=dict(model.named_parameters());name_by_id={id(p):name for name,p in named.items()}
    if previous is not None and set(previous)!=set(named):raise ValueError('Parameter snapshot names changed')
    momentum={id(p):b for p,b in zip(helper.parameters,helper.momentum_buffers)} if helper is not None else {}
    descriptions=[]
    with torch.no_grad():
        for index,layer in enumerate(model.layers):
            body=0<index<len(model.layers)-1
            known=not(model.residual and body) or last_active is not None
            active=known and (not(model.residual and body) or last_active[index-1])
            description={'layer':index,'name':f'layer_{index}','training_gradient_available':bool(active),
                         'gradient_status':'eligible_last_step' if active else 'unknown_last_mask' if not known else 'dropped_last_step'}
            for short,p in layer.named_parameters(recurse=False):
                prefix=f'layer_{index}/{short}';collector.summary(prefix+'/parameter',p)
                if active and p.grad is not None:collector.summary(prefix+'/last_training_gradient',p.grad)
                elif active:description['gradient_status']='missing_gradient'
                if previous is not None:
                    before=previous[name_by_id[id(p)]]
                    if before.shape!=p.shape or before.device!=p.device:raise ValueError('Snapshot shape/device mismatch')
                    delta=p.detach()-before
                    collector.summary(prefix+'/measured_update',delta)
                    norm=torch.linalg.vector_norm(before.float())
                    collector.add(prefix+'/measured_update/relative_l2',torch.where(norm>0,torch.linalg.vector_norm(delta.float())/norm,torch.full_like(norm,float('nan'))))
                if id(p) in momentum:collector.summary(prefix+'/momentum_buffer',momentum[id(p)])
            descriptions.append(description)
    scalars=collector.finish()
    for layer in descriptions:
        prefix=f"layer_{layer['layer']}/"
        layer.update({key[len(prefix):].replace('/','_'):value for key,value in scalars.items() if key.startswith(prefix)})
        if layer['gradient_status']=='missing_gradient':layer['training_gradient_available']=False
    elapsed=time.perf_counter()-started
    return {'layers':descriptions,'scalars':scalars,'interval_steps':interval_steps if previous is not None else None,
            'update_interval':update_interval if previous is not None else None,
            'last_active':list(last_active) if last_active is not None else None,
            'definitions':{'gradient':'Last training-step parameter gradient, not a dense probe mean; stale skipped branches omitted.',
                           'update':'Actual current parameter minus snapshot, including momentum and direct shrinkage; not lr*gradient.',
                           'relative_update':'L2 update divided by snapshot parameter L2; null for zero denominator.',
                           'reduction':'FP32 mean/L2 reductions; scalar transfer packed once. Nonfinite scalar outputs are null.'},
            'overhead_seconds':elapsed,'elapsed_seconds':elapsed}


def evaluate(model,x,y,*,batch_size=2048,num_classes=10):
    """Dense eval CE/accuracy/confusion/per-class metrics, preserving mode/RNG.

    Confusion rows are true classes; columns are predictions. One packed host
    read after all minibatches; diagnostics do not enter training-step graphs.
    """
    if len(x)!=len(y) or not len(y) or batch_size<1:raise ValueError('Invalid evaluation data/batch size')
    device=_device(model);started=_start(device)
    modes={module:module.training for module in model.modules()}
    confusion=torch.zeros((num_classes,num_classes),dtype=torch.int64,device=device)
    class_ce=torch.zeros(num_classes,dtype=torch.float64,device=device)
    nonfinite=torch.zeros((),dtype=torch.int64,device=device);zero=torch.zeros((),dtype=torch.int64,device=device)
    try:
        model.eval()
        with torch.no_grad():
            for start in range(0,len(y),batch_size):
                labels=y[start:start+batch_size].to(device);logits=model(x[start:start+batch_size].to(device))
                if logits.ndim!=2 or logits.shape[1]!=num_classes:raise ValueError('Wrong classifier width')
                nonfinite+=(~torch.isfinite(logits)).sum();zero+=(logits==0).sum()
                losses=F.cross_entropy(logits,labels,reduction='none')
                confusion+=torch.bincount(labels*num_classes+logits.argmax(-1),minlength=num_classes**2).reshape(num_classes,num_classes)
                class_ce.scatter_add_(0,labels,losses.double())
    finally:
        for module,mode in modes.items():module.training=mode
    packed=torch.cat((confusion.reshape(-1).double(),class_ce,torch.stack((nonfinite,zero)).double())).cpu().tolist()
    if packed[-2]!=0 or any(not math.isfinite(v) for v in packed):raise FloatingPointError('Nonfinite evaluation logits/loss')
    matrix=[[int(v) for v in packed[i*num_classes:(i+1)*num_classes]] for i in range(num_classes)]
    sums=packed[num_classes**2:num_classes**2+num_classes];classes=[]
    for cls,row in enumerate(matrix):
        n=sum(row);correct=row[cls];predicted=sum(r[cls] for r in matrix)
        classes.append({'class_id':cls,'n':n,'correct':correct,'errors':n-correct,'accuracy':correct/n if n else None,
                        'ce':sums[cls]/n if n else None,'predicted_count':predicted,
                        'precision':correct/predicted if predicted else None,'recall':correct/n if n else None})
    total=len(y);correct=sum(matrix[i][i] for i in range(num_classes))
    elapsed=time.perf_counter()-started
    return {'n':total,'loss':sum(sums)/total,'accuracy':correct/total,'errors':total-correct,
            'confusion':matrix,'confusion_axes':'rows=true digit, columns=predicted digit','per_class':classes,
            'zero_logit_fraction':packed[-1]/(total*num_classes),'overhead_seconds':elapsed,'elapsed_seconds':elapsed,
            'mode':'Deterministic dense evaluation; unit dropout and SD disabled; previous train/eval flags restored'}


def _activation(collector,prefix,value,*,relu_input=None):
    collector.summary(prefix,value)
    collector.add(prefix+'/zero_fraction',(value==0).float().mean())
    collector.add(prefix+'/nonpositive_fraction',(value<=0).float().mean())
    collector.add(prefix+'/all_zero_units_on_probe_fraction',(value==0).all(dim=0).float().mean())
    if relu_input is not None:collector.add(prefix+'/relu_inactive_fraction',(relu_input<=0).float().mean())


def probe_statistics(model,probe_x,probe_y=None,*,compute_gradients=False):
    """Fixed dense Ciresan probe; no RNG draws, parameter writes, or .grad writes.

    A_i is each affine layer's input. B_i=d(sum CE)/d(affine_output_i), so each
    row is that example's unaveraged CE derivative. Weight gradient g_i=B_i A_iᵀ.
    Moment calculations exploit rank1 factorization and never materialize one
    full parameter gradient per example or covariance/Hessian matrices.
    """
    if not len(probe_x):raise ValueError('Empty probe')
    if compute_gradients and (probe_y is None or len(probe_y)!=len(probe_x)):raise ValueError('Probe labels required for gradients')
    if not all(isinstance(layer,torch.nn.Linear) for layer in model.layers):raise ValueError('Probe expects Ciresan Linear layers')
    device=_device(model);started=_start(device);collector=_Collector(device);n=len(probe_x)
    inputs=[];affines=[];post=[];relu_inputs=[]
    with torch.inference_mode(False),torch.enable_grad() if compute_gradients else torch.no_grad():
        h=probe_x.to(device).reshape(n,model.widths[0])*model.input_scale
        for i,layer in enumerate(model.layers):
            inputs.append(h);affine=layer(h);affines.append(affine)
            if i==len(model.layers)-1:
                pre=affine;h=F.relu(pre) if model.output_relu else pre
                relu_inputs.append(pre if model.output_relu else None)
            else:
                pre=h[:,:layer.out_features]+affine if i>0 and model.residual else affine
                h=F.relu(pre);relu_inputs.append(pre)
            post.append(h)
        loss=None;backprop=None
        if probe_y is not None:
            loss=F.cross_entropy(h,probe_y.to(device),reduction='sum')
            collector.add('probe/dense/ce',loss/n)
            collector.add('probe/dense/accuracy',(h.argmax(-1)==probe_y.to(device)).float().mean())
        if compute_gradients:backprop=torch.autograd.grad(loss,affines,retain_graph=False,create_graph=False)
    for i,(a,z,output) in enumerate(zip(inputs,affines,post)):
        prefix=f'probe/dense/layer_{i}'
        _activation(collector,prefix+'/activation_input',a)
        _activation(collector,prefix+'/affine_output',z)
        _activation(collector,prefix+'/post_activation',output,relu_input=relu_inputs[i])
        # Explicit legacy aliases: mean_activation refers to A, not post-ReLU h.
        collector.add(prefix+'/mean_activation',a.detach().float().mean())
        collector.add(prefix+'/msr_activation',torch.linalg.vector_norm(a.detach().float())/math.sqrt(a.numel()))
        collector.add(prefix+'/a_sparsity',(a.detach()<=0).float().mean())
        if backprop is not None:
            a=a.detach().float();b=backprop[i].detach().float()
            _activation(collector,prefix+'/backprop_preactivation',b)
            collector.add(prefix+'/mean_backprop',b.mean())
            collector.add(prefix+'/msr_backprop',torch.linalg.vector_norm(b)/math.sqrt(b.numel()))
            collector.add(prefix+'/b_sparsity',(b<=0).float().mean())
            grad=b.T@a/n;collector.summary(prefix+'/weight_gradient_mean',grad)
            grad_l2=torch.linalg.vector_norm(grad);collector.add(prefix+'/grad_l2',grad_l2);collector.add(prefix+'/grad_fro',grad_l2)
            if model.layers[i].bias is not None:collector.summary(prefix+'/bias_gradient_mean',b.mean(dim=0))
            norm2=(a*a).sum(1)*(b*b).sum(1);norm=torch.sqrt(norm2)
            for label,value in [('mean',norm.mean()),('median',torch.quantile(norm,.5)),('min',norm.min()),('max',norm.max())]:
                collector.add(prefix+'/per_example_weight_grad_norm/'+label,value)
            second=norm2.mean();mean_norm2=(grad*grad).sum()
            collector.add(prefix+'/weight_grad_second_moment',second)
            collector.add(prefix+'/weight_grad_mean_norm_squared',mean_norm2)
            collector.add(prefix+'/weight_grad_centered_noise',second-mean_norm2)
            collector.add(prefix+'/weight_grad_diversity',torch.where(mean_norm2>0,second/mean_norm2,torch.full_like(mean_norm2,float('nan'))))
            # Cyclic neighboring examples; zero-gradient pairs are undefined.
            inner=(a*a.roll(1,0)).sum(1)*(b*b.roll(1,0)).sum(1)
            denominator=norm*norm.roll(1,0);valid=denominator>0
            cos=torch.where(valid,inner/denominator,torch.zeros_like(inner))
            collector.add(prefix+'/neighbor_weight_grad_cosine_mean',torch.where(valid.any(),cos.sum()/valid.sum(),torch.full_like(cos.sum(),float('nan'))))
            collector.add(prefix+'/neighbor_weight_grad_cosine_valid_pairs',valid.sum())
    scalars=collector.finish()
    layers=[]
    for index in range(len(model.layers)):
        prefix=f'probe/dense/layer_{index}/'
        layers.append({'layer':index,**{key[len(prefix):].replace('/','_'):value for key,value in scalars.items() if key.startswith(prefix)}})
    elapsed=time.perf_counter()-started
    return {'n':n,'compute_gradients':bool(compute_gradients),'layers':layers,'scalars':scalars,
            'loss':scalars.get('probe/dense/ce'),'accuracy':scalars.get('probe/dense/accuracy'),
            'overhead_seconds':elapsed,'elapsed_seconds':elapsed,
            'definitions':{'activation_input':'A: input to affine layer; mean_activation/msr_activation and a_sparsity refer to A.',
                           'backprop_preactivation':'B=d(sum probe CE)/d(affine output), unaveraged per-example derivative; mean_backprop/msr_backprop use B.',
                           'zero_vs_nonpositive':'zero_fraction counts exact0; a_sparsity/b_sparsity count <=0. They are different for signed inputs/derivatives.',
                           'dead_units':'all_zero_units_on_probe_fraction is inactivity on this finite probe, not permanent neuron death.',
                           'grad_l2':'L2/Frobenius norm of mean per-example WEIGHT gradient B.T@A/n; bias gradients separate. grad_fro is the same explicit legacy alias.',
                           'gradient_noise':'E||g_i||² - ||E g_i||², weight-only, population moment; no Hessian or covariance materialized. Tiny negative values from FP32 cancellation are retained.',
                           'diversity':'E||g_i||² / ||E g_i||²; null if denominator0.',
                           'neighbor_cosine':'Cyclic adjacent fixed-probe weight gradients, factored inner product; zero-gradient pairs excluded; not IID gradient-noise inference.',
                           'median':'Interpolated .5 quantile of per-example weight-gradient L2 norms.',
                           'mode':'Manual dense eval operator, no SD/unit dropout; model flags, RNG and persistent parameter.grad buffers untouched.'}}
