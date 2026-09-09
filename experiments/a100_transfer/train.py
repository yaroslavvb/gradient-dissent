"""Training and fixed-mask evaluation shared by the three A100 model families."""
from __future__ import annotations
import hashlib
import dataclasses
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import time
import numpy as np
import torch
from torch.nn import functional as F

RECIPES = ('dense', 'constant_ild', 'decreasing_ild')

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def probabilities(recipe, depth, step, steps, device='cpu'):
    if recipe not in RECIPES: raise ValueError(recipe)
    if recipe == 'dense': return torch.zeros(depth, device=device)
    p = torch.arange(depth, device=device, dtype=torch.float32) / (depth-1)
    return p * (.4 if recipe == 'constant_ild' else .8 * (1-step/max(1, steps-1)))

def panel(model, task):
    n = model.prunable_count
    items = {'full': tuple(range(n)), 'primary_two_thirds': tuple(model.primary_keep()) if hasattr(model, 'primary_keep') else tuple(range(2*n//3))}
    if task == 'convnext_cifar100':
        for title, counts in [('one_third', (1,1,3,1)), ('half', (2,1,4,2)), ('five_sixths', (3,2,8,2))]:
            offsets = (0,3,6,15)
            items[title] = tuple(j for start, count in zip(offsets, counts) for j in range(start,start+count))
    else:
        items.update({f'prefix_{k}': tuple(range(k)) for k in (n//3,n//2,5*n//6)})
    rng = np.random.default_rng(913079)
    for i in range(3):
        items[f'random_keeps_first_{i}'] = tuple(sorted([0]+rng.choice(np.arange(1,n),2*n//3-1,replace=False).tolist()))
    items['delete_first_only'] = tuple(range(1,n))
    assert len(items['primary_two_thirds']) == 2*n//3
    return items

class Batches:
    def __init__(self, spec, root):
        self.spec = spec
        self.task = spec['task']
        self.stream = hashlib.sha256()
        self.data_rng = np.random.default_rng(10000+spec['seed'])
        self.augment_rng = torch.Generator(device='cuda').manual_seed(30000+spec['seed'])
        if self.task == 'gpt_wikitext103':
            from language import TokenWindowLoader
            self.train = TokenWindowLoader(root/'data/language', 'train', spec['context'], 10000+spec['seed'])
            self.val = TokenWindowLoader(root/'data/language', 'val', spec['context'])
            self.test = None  # Test windows are opened only in final evaluation.
            self.language_root = root/'data/language'
            self.train_size = int(self.train.tokens.size)
        else:
            from vision import load_cifar100
            data = load_cifar100(root/'data/vision', download=False)
            self.vision = {k: tuple(vv.cuda() for vv in v) for k,v in data.items() if k != 'metadata'}
            self.train_size = len(self.vision['train'][1])
            self.metadata = data['metadata']
    def sample(self):
        s = self.spec
        if self.task == 'gpt_wikitext103':
            x,y = self.train.sample(s['batch_size'])
            self.stream.update(x.numpy().tobytes())
            return x.cuda(), y.cuda()
        from vision import prepare_images
        raw, labels = self.vision['train']
        ids = self.data_rng.integers(len(labels), size=s['batch_size'], dtype=np.int64)
        self.stream.update(ids.tobytes())
        indices = torch.from_numpy(ids).cuda()
        return prepare_images(raw[indices], training=True, generator=self.augment_rng, image_size=s['image_size']), labels[indices]
    def heldout(self, split):
        s = self.spec
        if self.task == 'gpt_wikitext103':
            from language import TokenWindowLoader
            loader = self.val if split == 'validation' else TokenWindowLoader(self.language_root, 'test', s['context'])
            windows = s.get('validation_windows',64) if split == 'validation' else s.get('test_windows',128)
            yield from loader.fixed_batches(s.get('eval_batch_size',4), device='cuda', max_windows=windows)
        else:
            from vision import prepare_images
            x,y = self.vision[split]
            maximum = s.get('validation_examples',len(y)) if split == 'validation' else len(y)
            # Fixed subset for pilots only. Final manifests can request all validation images.
            ids = torch.linspace(0,len(y)-1,maximum,device='cuda').long() if maximum < len(y) else torch.arange(len(y),device='cuda')
            for ix in ids.split(s.get('eval_batch_size',256)):
                yield prepare_images(x[ix], image_size=s['image_size']), y[ix]
    def evaluation_metadata(self, split):
        if self.task == 'gpt_wikitext103':
            from language import TokenWindowLoader
            loader = self.val if split == 'validation' else TokenWindowLoader(self.language_root, 'test', self.spec['context'])
            return loader.evaluation_manifest(self.spec.get('validation_windows',64) if split=='validation' else self.spec.get('test_windows',128))
        count = len(self.vision[split][1])
        return {'split':split, 'examples':min(count,self.spec.get('validation_examples',count)) if split=='validation' else count,
                'source_split':'official CIFAR100 test' if split=='test' else 'stratified subset of official CIFAR100 train'}

@torch.no_grad()
def evaluate(model, batches, split='validation', keep=None):
    model.eval()
    ce = torch.zeros((),device='cuda',dtype=torch.float64)
    correct = torch.zeros((),device='cuda',dtype=torch.int64)
    count = 0
    for x,y in batches.heldout(split):
        with torch.autocast('cuda', dtype=torch.bfloat16):
            logits = model(x, keep=keep)
            loss = F.cross_entropy(logits.reshape(-1,logits.shape[-1]),y.reshape(-1),reduction='sum')
        ce += loss.double()
        correct += (logits.argmax(-1)==y).sum()
        count += y.numel()
    torch.cuda.synchronize()
    return {'ce':float(ce/count),'accuracy':float(correct/count),'targets':count}

def build(spec):
    if spec['task'] == 'gpt_wikitext103':
        from language import GPT,GPTConfig
        return GPT(GPTConfig(block_size=spec['context']))
    from vision import build_model
    return build_model(spec['task'],image_size=spec['image_size'])

def layerscale(model):
    return {name:{'mean_abs':float(p.detach().abs().mean()),'max_abs':float(p.detach().abs().max())}
            for name,p in model.named_parameters() if name.endswith('.gamma')}

def run(spec, root, progress_commit=lambda:None):
    started = time.monotonic()
    root = Path(root)
    out = root/'results'/spec['run_id']; out.mkdir(parents=True,exist_ok=True)
    if (out/'result.json').exists():
        raise RuntimeError('Output already exists; refuse duplicate billable execution.')
    # Persist a claim before training. Modal retry=0; refuse a transparent restart.
    if (out/'claim.json').exists():
        raise RuntimeError('Prior attempt claimed this run; inspect it and reserve any retry explicitly.')
    (out/'claim.json').write_text(json.dumps({'spec':spec,'claimed_unix':time.time()}))
    progress_commit()
    assert torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(spec['seed'])
    torch.cuda.manual_seed_all(spec['seed'])
    model = build(spec).cuda()
    n = model.prunable_count
    source_dir = Path(__file__).parent
    source_hashes = {name:digest(source_dir/name) for name in ('train.py','language.py','vision.py')}
    initial_hash = hashlib.sha256()
    for name,p in model.state_dict().items():
        initial_hash.update(name.encode()); initial_hash.update(p.cpu().numpy().tobytes())
    batches = Batches(spec,root)
    mask_rng = torch.Generator(device='cuda').manual_seed(20000+spec['seed'])
    groups = [{'params':[p for p in model.parameters() if p.ndim>=2], 'weight_decay':spec['weight_decay']},
              {'params':[p for p in model.parameters() if p.ndim<2], 'weight_decay':0.}]
    opt = torch.optim.AdamW(groups, lr=spec['lr'],betas=tuple(spec['betas']),fused=True)
    metadata = {'torch':torch.__version__,'numpy':np.__version__,'python':platform.python_version(),
                'gpu':torch.cuda.get_device_name(), 'gpu_memory_total_bytes':torch.cuda.get_device_properties(0).total_memory,
                'nvidia_smi':subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.total,driver_version','--format=csv,noheader'],text=True).strip(),
                'parameters':sum(p.numel() for p in model.parameters()),'model_config':dataclasses.asdict(model.config) if dataclasses.is_dataclass(model.config) else model.config,'precision':'BF16 autocast; FP32 master parameters/AdamW states; TF32 allowed',
                'source_sha256':source_hashes,'initial_state_sha256':initial_hash.hexdigest(),
                'prunable_count':n,'prunable_ids':getattr(model,'prunable_ids',list(range(n))),
                'training_examples_or_corpus_tokens':batches.train_size}
    metadata['dataset_manifest_sha256'] = (digest(root/'data/language/manifest.json') if spec['task']=='gpt_wikitext103'
        else hashlib.sha256(json.dumps(batches.metadata,sort_keys=True).encode()).hexdigest())
    model.eval()
    check_x, _ = next(batches.heldout('validation'))
    with torch.no_grad(), torch.autocast('cuda',dtype=torch.bfloat16):
        reference=model(check_x)
        all_kept=model(check_x,keep=tuple(range(n)))
        assert torch.equal(reference,all_kept), 'GPU full/all-kept mismatch'
    correctness={'gpu_all_kept_equals_default':True,'checks_use_validation_only':True}
    del check_x, reference, all_kept
    initial_gamma = layerscale(model)
    before = evaluate(model,batches)
    steps = spec['steps']; warmup = max(1,min(steps//10,spec.get('warmup_steps',100)))
    interval = max(1,spec.get('log_every',max(1,steps//10)))
    curve=[];active=torch.zeros((),device='cuda',dtype=torch.int64);clipped=0
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize(); training_start=time.monotonic();step_times=[]
    for step in range(steps):
        if time.monotonic()-started > spec['timeout_seconds']-45:
            raise RuntimeError(f'Wall-time guard before completing planned steps: {step}/{steps}; no shortened-run substitution.')
        tick=time.monotonic()
        model.train();x,y=batches.sample()
        p=probabilities(spec['recipe'],n,step,steps,'cuda')
        masks=torch.rand((x.shape[0],n),device='cuda',generator=mask_rng)>=p
        active+=masks.sum()
        scales=masks.float()/(1-p)
        progress=max(0.,(step-warmup)/max(1,steps-1-warmup))
        lr=spec['lr']*min(1.,(step+1)/warmup)*(.1+.9*.5*(1+math.cos(math.pi*progress)))
        for group in opt.param_groups:group['lr']=lr
        opt.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            logits=model(x,scales=scales)
            loss=F.cross_entropy(logits.reshape(-1,logits.shape[-1]),y.reshape(-1))
        if not torch.isfinite(loss).item(): raise FloatingPointError(f'Non-finite loss at step{step}')
        loss.backward()
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        if not torch.isfinite(norm).item(): raise FloatingPointError(f'Non-finite gradient at step{step}')
        clipped+=int(norm>1.)
        opt.step()
        torch.cuda.synchronize()
        step_times.append(time.monotonic()-tick)
        if step==0 or (step+1)%interval==0 or step+1==steps:
            point={'step':step+1,'train_ce':float(loss.detach()),'lr':lr,'nominal_mean_dropout':float(p.mean()),
                   'active_example_blocks':int(active),'elapsed_training_seconds':time.monotonic()-training_start,
                   'step_seconds':step_times[-1]}
            curve.append(point)
            (out/'progress.json').write_text(json.dumps({'spec':spec,'metadata':metadata,'curve':curve},indent=2))
            print(spec['run_id'],'step',step+1,'loss',round(float(loss.detach()),4),'seconds',round(time.monotonic()-training_start,1),flush=True)
        del logits, loss
    train_seconds=time.monotonic()-training_start
    peak_allocated=torch.cuda.max_memory_allocated();peak_reserved=torch.cuda.max_memory_reserved()
    validation=evaluate(model,batches)
    rows=[]
    if spec['stage']=='evaluate':
        for name,keep in panel(model,spec['task']).items():
            if time.monotonic()-started > spec['timeout_seconds']-15:
                raise RuntimeError('Wall-time guard during final pruning evaluation; incomplete panel is not a complete run.')
            metric=evaluate(model,batches,'test',keep)
            rows.append({'mask_name':name,'kept_indices_zero_based':list(keep),'retained':len(keep),**metric})
        full=next(r for r in rows if r['mask_name']=='full')['ce']
        for row in rows:row['excess_ce']=row['ce']-full
        ckpt=root/'checkpoints';ckpt.mkdir(exist_ok=True)
        torch.save({'spec':spec,'state_dict':model.state_dict(),'metadata':metadata},ckpt/(spec['run_id']+'.pt'))
    result={'run_id':spec['run_id'],'spec':spec,'metadata':metadata,'correctness':correctness,'validation_before':before,'validation':validation,
            'validation_panel':batches.evaluation_metadata('validation'),'test_panel':batches.evaluation_metadata('test') if rows else None,
            'rows':rows,'curve':curve,'train_seconds':train_seconds,'container_function_seconds':time.monotonic()-started,
            'mean_step_seconds_after_warmup':float(np.mean(step_times[min(10,len(step_times)-1):])),
            'peak_allocated_bytes':peak_allocated,'peak_reserved_bytes':peak_reserved,'clipped_steps':clipped,
            'nominal_mean_dropout':float(torch.stack([probabilities(spec['recipe'],n,i,steps) for i in range(steps)]).mean()),
            'realized_active_example_blocks':int(active),'dense_example_blocks':steps*spec['batch_size']*n,
            'training_data_stream_sha256':batches.stream.hexdigest(),'layerscale_before':initial_gamma,'layerscale_after':layerscale(model),
            'training_targets_or_images':steps*spec['batch_size']*(spec['context'] if spec['task']=='gpt_wikitext103' else 1),
            'cost_note':'Use Modal billing plus invocation reservation ledger; runtime alone excludes startup/build/idle/egress.'}
    (out/'result.json').write_text(json.dumps(result,indent=2,allow_nan=False))
    return result
