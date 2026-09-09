"""Epoch-boundary telemetry; never called from the CUDA graph replay loop.

No network logging. Scientific dictionaries are persisted after GPU work, and
expensive curvature runs later from sparse frozen checkpoint files.
"""
from pathlib import Path
import hashlib,json,time
import torch
from .metrics import evaluate, parameter_snapshot, layer_statistics, probe_statistics


def flatten(prefix,value,out):
    if isinstance(value,dict):
        for k,v in value.items():flatten(prefix+'/'+str(k) if prefix else str(k),v,out)
    elif isinstance(value,(int,float)) and not isinstance(value,bool):out[prefix]=value


class Recorder:
    def __init__(self,model,spec,run_dir,tensors,started,helper=None):
        self.model=model;self.spec=spec;self.directory=Path(run_dir);self.tensors=tensors
        self.started=started;self.helper=helper;self.enabled=spec.get('telemetry',True)
        self.rows=[];self.seconds=0.;self.snapshot_seconds=0.;self.snapshots=[]
        self.previous=None;self.previous_epoch=None
        self.every=spec.get('telemetry_every',5);self.probe_n=spec.get('telemetry_probe_n',256)
        self.snapshot_epochs=set(spec.get('snapshot_epochs',[]))
        self.probe_hash=hashlib.sha256(tensors['train_x'][:self.probe_n].detach().cpu().numpy().tobytes()+tensors['train_y'][:self.probe_n].detach().cpu().numpy().tobytes()).hexdigest() if self.enabled else None

    def capture(self,epoch,training_seconds,epoch_seconds,legacy_row=None,force=False):
        if not self.enabled:return
        detailed=force or epoch==1 or epoch%self.every==0 or epoch==self.spec.get('epochs',self.spec.get('max_epochs',100))
        if not detailed and legacy_row is None:return
        torch.cuda.synchronize();started=time.perf_counter()
        was_training=self.model.training
        row={'epoch':epoch,'training_seconds':training_seconds,'metrics':{},'diagnostics':{}}
        m=row['metrics'];m.update({'train/epoch_seconds':epoch_seconds,'optimizer/lr':self.spec['lr']*(self.spec.get('lr_schedule',{}).get('21',1) if epoch>=21 else 1),
                                 'optimizer/momentum':self.spec.get('momentum',.9),'train/examples_seen':epoch*(len(self.tensors['train_y'])//self.spec['batch_size'])*self.spec['batch_size']})
        if legacy_row:
            for key,prefix in [('validation','validation'),('dense_train_probe','train_probe'),('test','test')]:
                if key in legacy_row:
                    z=legacy_row[key];m[prefix+'/loss']=z['loss'];m[prefix+'/accuracy_pct']=100*z['accuracy']
            if 'stochastic_training_loss' in legacy_row:m['train/stochastic_minibatch_loss']=legacy_row['stochastic_training_loss']
            if 'last_minibatch_loss' in legacy_row:m['train/last_minibatch_loss']=legacy_row['last_minibatch_loss']
            if 'drop_probabilities' in legacy_row:
                for i,p in enumerate(legacy_row['drop_probabilities']):m[f'layer-{i+1}/drop_probability']=p
        if detailed:
            for prefix in ['train','val','test']:
                if prefix+'_x' not in self.tensors:continue
                score=evaluate(self.model,self.tensors[prefix+'_x'],self.tensors[prefix+'_y'])
                label='validation' if prefix=='val' else prefix
                row['diagnostics'][label]=score
                m[label+'/loss']=score['loss'];m[label+'/accuracy_pct']=100*score['accuracy'];m[label+'/errors']=score['errors']
                for c in score['per_class']:
                    ident=c.get('class_id',c.get('digit'))
                    flatten(f'{label}/class-{ident}',c,m)
            layers=layer_statistics(self.model,previous_parameters=self.previous,last_active=self.helper.last_mask if self.helper else None,helper=self.helper)
            row['diagnostics']['parameters']=layers
            for i,z in enumerate(layers['layers']):flatten(f'layer-{i}/training_state',z,m)
            row['diagnostics']['update_interval']={'from_epoch':self.previous_epoch,'to_epoch':epoch,'meaning':'Net displacement between telemetry checkpoints, not one optimizer step'}
            probe=probe_statistics(self.model,self.tensors['train_x'][:self.probe_n],self.tensors['train_y'][:self.probe_n],compute_gradients=True)
            row['diagnostics']['probe']=probe
            for i,z in enumerate(probe['layers']):flatten(f'layer-{i}/probe',z,m)
            self.previous=parameter_snapshot(self.model);self.previous_epoch=epoch
        self.model.train(was_training)
        if epoch in self.snapshot_epochs:
            t=time.perf_counter();path=self.directory/f'snapshot-{epoch:03d}.pt'
            torch.save(self.model.state_dict(),path);self.snapshot_seconds+=time.perf_counter()-t
            self.snapshots.append({'epoch':epoch,'path':path.name,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
        torch.cuda.synchronize();self.seconds+=time.perf_counter()-started
        row['run_elapsed_seconds']=time.perf_counter()-self.started
        m['time/telemetry_cumulative_seconds']=self.seconds;m['time/checkpoint_cumulative_seconds']=self.snapshot_seconds
        row['telemetry_cumulative_seconds']=self.seconds
        self.rows.append(row)

    def finalize(self,result):
        result['telemetry_history']=self.rows
        result['telemetry']={'enabled':self.enabled,'seconds':self.seconds,'checkpoint_seconds_within_telemetry':self.snapshot_seconds,
             'cadence_epochs':self.every,'probe_n':self.probe_n,'probe_sha256':self.probe_hash,
             'probe_definition':'First fixed training-pool examples, raw pixel+label hash; dense eval semantics, no sampled masks or unit dropout',
             'network_logging_during_training':False,'snapshots':self.snapshots,
             'time_definition':'GPU-synchronized epoch training excludes evaluation, telemetry and checkpoint I/O; run elapsed includes them; upload later',
             'metrics_definition':'Corrected modern statistics retain explicit namespaces; no historical bugs or expensive Hessian callbacks in training'}
        (self.directory/'telemetry.json').write_text(json.dumps({'run_id':self.spec['run_id'],'spec':self.spec,'telemetry_history':self.rows,'telemetry':result['telemetry']},allow_nan=False)+'\n')
