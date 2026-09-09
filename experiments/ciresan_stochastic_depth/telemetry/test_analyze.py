"""Cheap CPU checks for honest curve export and strict replay verification."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

try:
    from . import analyze as a
except ImportError:
    import analyze as a


H='a'*64
SOURCES={'model_data.py':H,'optimization/stochastic_graph.py':H}


def score():return {'loss':.1,'accuracy':1.,'n':10,'errors':0,'wrong_indices':[]}


def fixture():
    spec={'run_id':'telemetry-main-residual-s101-v1','runner':'controlled','recipe':'residual','seed':101,
          'epochs':100,'batch_size':64,'lr':.01,'pmax':0.,'input_scale':1/255,'output_relu':False,
          'shrinkage':0.,'eval_every':5,'train_size':50000,'precision':'fp32','source_sha256':SOURCES,
          'telemetry_probe_n':128,'telemetry_every':10,'telemetry':True}
    history=[{'epoch':epoch,'validation':score(),'dense_train_probe':score(),
              'stochastic_training_loss':.1,'training_seconds':float(epoch)} for epoch in (1,100)]
    raw={'spec':spec,'run_id':spec['run_id'],'history':history,'initial_parameters_sha256':H,
         'trained_final_parameters_sha256':'b'*64,'final_parameters_sha256':'c'*64,
         'executed_epoch_order_sha256':H,'executed_epoch_masks_sha256':H,'source_sha256':SOURCES,
         'telemetry_source_sha256':SOURCES,'parameter_count':100,'dataset':{'split':'fixed'},
         'steps_per_epoch':781,'training_examples_per_epoch':49984,'skipped_batch_counts':[0]*4,
         'best_epoch':100,'best_validation_loss':.1,'training_seconds':100.,'total_run_seconds':105.,
         'telemetry':{'enabled':True,'seconds':2.,'checkpoint_seconds_within_telemetry':0.,'probe_n':128,
                      'probe_sha256':H,'network_logging_during_training':False},
         'telemetry_history':[{'epoch':e,'training_seconds':float(e),'run_elapsed_seconds':e+1.,
                              'metrics':{'validation/accuracy_pct':100.,'layer-0/probe/grad_l2':.3},
                              'diagnostics':{}} for e in (0,1,100)]}
    raw.update({k:score() for k in ('selected_test','final_test','selected_train','final_train')})
    original=copy.deepcopy(raw)
    for k in ('trained_final_parameters_sha256','final_parameters_sha256','telemetry_history','telemetry'):
        original.pop(k,None)
    original['history'][0]['training_seconds']=.3
    return raw,original,spec


class AnalyzeTests(unittest.TestCase):
    def test_exact_scientific_curve_different_clock_and_missing_old_end_hash(self):
        raw,original,spec=fixture();run=a.analyze_run(raw,original,spec)
        self.assertEqual(run['status'],'verified',run['issues'])
        self.assertEqual(run['verification']['trained_final_parameters']['status'],'reference_unavailable')
        self.assertNotEqual(run['parameter_hashes']['trained_final'],run['parameter_hashes']['reloaded_selected'])
        self.assertEqual(run['telemetry_rows'][1]['metrics']['layer-0/probe/grad_l2'],.3)

    def test_exact_equality_does_not_round_scientific_changes(self):
        raw,original,spec=fixture();raw['history'][0]['validation']['loss']+=1e-15
        result=a.analyze_run(raw,original,spec)
        self.assertEqual(result['status'],'mismatch')
        self.assertIn('scientific_history:mismatch',result['issues'])

    def test_source_data_initialization_and_mask_tampering_rejected(self):
        for field,value,issue in [('initial_parameters_sha256','z'*64,'initial_parameters:missing_or_invalid'),
                                  ('executed_epoch_masks_sha256','b'*64,'executed_epoch_masks_sha256:mismatch'),
                                  ('dataset',{'split':'other'},'dataset:mismatch'),
                                  ('telemetry_source_sha256',{},'frozen_source/model_data.py:missing_or_invalid')]:
            with self.subTest(field=field):
                raw,original,spec=fixture();raw[field]=value
                self.assertIn(issue,a.analyze_run(raw,original,spec)['issues'])

    def test_full_history_length_checked(self):
        raw,original,spec=fixture();raw['history'].pop()
        result=a.analyze_run(raw,original,spec)
        self.assertEqual(result['status'],'incomplete')
        self.assertIn('/length',result['verification']['scientific_history']['different_paths'])

    def test_nonfinite_scalar_rejected_and_confusion_checked(self):
        raw,_,_=fixture();raw['telemetry_history'][0]['metrics']['bad']=float('nan');issues=[]
        a.telemetry_rows(raw,issues);self.assertIn('telemetry/0:nonfinite_metric',issues)
        matrix=[[0]*10 for _ in range(10)]
        s=score();s['confusion']=matrix
        issues=[];a.check_eval(s,'test',issues);self.assertIn('test:confusion_totals',issues)

    def test_historical_explicit_complete_curve_and_hash(self):
        history=[{'epoch':i,'_step':i*2,'_runtime':i+.5,'val_accuracy':97.+i,'val_loss':.1,
                  'train_accuracy':99.,'train_loss':.02} for i in range(3)]
        digest=hashlib.sha256(json.dumps(history,sort_keys=True,separators=(',',':')).encode()).hexdigest()
        source={'history':history,'provenance':{'accuracy_rows_expected':3,'curve_sha256':digest}}
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'historical.json';path.write_text(json.dumps(source))
            out=a.historical_series(path)
            self.assertTrue(out['complete_series']);self.assertEqual(out['rows'][0]['metrics']['test/accuracy_pct'],97.)
            self.assertIsNone(out['rows'][0]['training_seconds']);self.assertEqual(out['rows'][0]['wandb_runtime_seconds'],.5)
            source['history'][0]['val_accuracy']=96.;path.write_text(json.dumps(source))
            with self.assertRaisesRegex(ValueError,'content hash mismatch'):a.historical_series(path)

    def test_sparse_historical_anchors_never_become_complete(self):
        row={'epoch':1,'_runtime':2.,'val_accuracy':98.,'val_loss':.1}
        source={'runs':[{'id':'ts4k9n55','accuracy_rows_expected':91,
                        'best_observed_accuracy':row,'last_observed_evaluation':row}]}
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'h.json';path.write_text(json.dumps(source));out=a.historical_series(path)
        self.assertFalse(out['complete_series']);self.assertEqual(out['retained_rows'],1)

    def test_qualification_end_hash_and_full_history_required(self):
        runs=[]
        for seed in (101,102,103):
            for enabled in (False,True):
                raw,_,_=fixture();raw['spec'].update(qualification_pair=str(seed),seed=seed,telemetry=enabled)
                raw['run_id']=raw['spec']['run_id']=str(seed)+'-'+str(enabled)
                runs.append(raw)
        spec={'run_id':'qualification','base_specs':[r['spec'] for r in runs]}
        raw={'run_id':'qualification','runs':runs,'spec':spec,'passed':True}
        out=a.verify_qualification(raw,spec);self.assertEqual(out['status'],'verified',out['issues'])
        self.assertTrue(out['overhead_gate']['passed'])
        runs[1]['telemetry']['seconds']=11.
        out=a.verify_qualification(raw,spec)
        self.assertEqual(out['status'],'verified');self.assertFalse(out['overhead_gate']['passed'])
        runs[1]['history'].pop();out=a.verify_qualification(raw,spec)
        self.assertEqual(out['status'],'mismatch')

    def test_all_pending_declared_runs_are_exported(self):
        specs=[]
        for recipe in ('plain','residual','sd_constant','sd_annealed','residual_unit_dropout'):
            for seed in (101,102,103):specs.append({'run_id':recipe+str(seed),'recipe':recipe,'seed':seed,'runner':'controlled'})
        for seed in (101,102,103):specs.append({'run_id':'baseline'+str(seed),'recipe':'unit_dropout','seed':seed,'runner':'baseline'})
        with tempfile.TemporaryDirectory() as directory:
            directory=Path(directory);manifest=directory/'manifest.json';manifest.write_text(json.dumps(specs))
            qual=directory/'qualification.json';qual.write_text('[]')
            historical=directory/'historical.json';historical.write_text(json.dumps([{'_runtime':1.,'val_accuracy':98.,'epoch':1}]))
            doc,plot=a.build(directory,manifest,directory,historical,qual)
            self.assertEqual(doc['counts']['pending'],18);self.assertFalse(doc['complete']);self.assertEqual(doc['status'],'partial')
            self.assertEqual(len(plot['runs']),18);self.assertTrue(all('telemetry_rows' in r for r in plot['runs']))
            a.write_outputs(directory,doc,plot)
            self.assertEqual(len(json.loads((directory/'plot-data.json').read_text())['runs']),18)

    def test_curvature_manifest_and_checkpoint_provenance(self):
        raw,original,spec=fixture();source=a.analyze_run(raw,original,spec)
        source['telemetry']['snapshots']=[{'epoch':e,'path':f'snapshot-{e:03d}.pt','sha256':H} for e in (0,100)]
        cs={'run_id':'curvature-fixture','source_run_id':source['id'],'snapshot_epochs':[0,100],
            'curvature_source_sha256':{'curvature.py':H}}
        result={'run_id':cs['run_id'],'spec':cs,'source_run_id':source['id'],'curvature_source_sha256':cs['curvature_source_sha256'],
                'source_training_source_sha256':SOURCES,'probe':{'source_recorder_probe_sha256':H},
                'complete':True,'passed':True,'status':'completed','snapshots':[]}
        for e,ph in ((0,H),(100,'b'*64)):
            result['snapshots'].append({'epoch':e,'checkpoint_sha256':H,'checkpoint_file':f'snapshot-{e:03d}.pt',
                                       'source_fp32_parameters_sha256':ph,'parameters_unchanged':True,
                                       'curvature':{'probe_n':128,'output_classes':10,'derivative_dtype':'torch.float64','layers':[{}]*6}})
        with tempfile.TemporaryDirectory() as directory:
            directory=Path(directory);manifest=directory/'curvature-manifest.json';manifest.write_text(json.dumps([cs]))
            path=directory/'telemetry-curvature-fixture.json';path.write_text(json.dumps(result))
            entries,status=a.curvature_results(directory,(),[source],manifest)
            self.assertTrue(status['complete']);self.assertEqual(entries[0]['verification']['status'],'verified')
            result['snapshots'][1]['checkpoint_sha256']='b'*64;path.write_text(json.dumps(result))
            entries,status=a.curvature_results(directory,(),[source],manifest)
            self.assertFalse(status['complete']);self.assertIn('checkpoint_manifest_mismatch',entries[0]['verification']['issues'])


if __name__=='__main__':unittest.main()
