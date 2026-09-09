"""Synthetic exporter/schema tests; never read actual study outcomes."""
import base64
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

try:
    from . import export_report as e
except ImportError:
    import export_report as e


def synthetic_panel(n_per_digit=10):
    labels=np.repeat(np.arange(10),n_per_digit).astype(np.uint8);n=len(labels)
    pred=np.broadcast_to(labels,(16,n)).copy()
    for digit in range(10):
        indices=np.flatnonzero(labels==digit)
        for index in indices[3:6]:pred[list(e.TWO_MASKS[:3]),index]=(digit+1)%10
        for index in indices[6:9]:pred[list(e.TWO_MASKS),index]=(digit+1)%10
        pred[15,indices[9]]=(digit+1)%10
    ce=np.broadcast_to(.1+np.arange(n)/100000,(16,n)).copy()+np.arange(16)[:,None]/100
    energy=np.ones((5,n));higher=np.full(n,.75)
    panel={'pred':pred,'ce':ce,'labels':labels,'dense_margin':np.ones(n),'energy_by_degree':energy,'higher_fraction':higher}
    return panel,labels


def record(recipe='residual',state='selected',seed=101,panel=None,labels=None):
    result={'recipe':recipe,'state':state,'seed':seed,'epoch':10 if state=='selected' else 100,
            'checkpoint_sha256':'a'*64,'direct_forward_parity':{'bitwise_equal':True},
            'test_parity':{'passed':True}}
    if panel is not None:
        result['test_parity'].update(accuracy=float(np.mean(panel['pred'][15]==labels)),loss=float(panel['ce'][15].mean()))
    return result


class ExportChecks(unittest.TestCase):
    def test_categories_lowest_indices_and_dense_wrong_exclusion(self):
        panel,labels=synthetic_panel();model=e.build_model(record(),panel,labels)
        examples=[r for r in model['examples'] if r['label']==0]
        self.assertEqual([(r['index'],r['category'],r['retained_two_correct_count']) for r in examples],
                         [(0,'preserved',6),(1,'preserved',6),(3,'mixed',3),(4,'mixed',3),(6,'lost',0),(7,'lost',0)])
        self.assertEqual(len(model['examples']),60)
        self.assertEqual(model['representative_category_counts'][0],{'digit':0,'preserved':3,'mixed':3,'lost':3})
        self.assertTrue(all(panel['pred'][15,r['index']]==r['label'] for r in model['examples']))

    def test_mask_class_rates_and_cardinality_aggregation(self):
        panel,labels=synthetic_panel();model=e.build_model(record(),panel,labels)
        self.assertEqual(len(model['masks']),16);self.assertEqual(len(model['by_k']),5)
        row=model['masks'][3]
        self.assertAlmostEqual(row['accuracy'],.4);self.assertAlmostEqual(row['harm'],.6);self.assertAlmostEqual(row['repair'],.1)
        self.assertAlmostEqual(model['dense_accuracy']-row['harm']+row['repair'],row['accuracy'])
        self.assertEqual(row['per_class'][0]['harm_count'],6)
        self.assertEqual(row['per_class'][0]['repair_count'],1)
        self.assertAlmostEqual(model['by_k'][2]['accuracy'],np.mean([model['masks'][i]['accuracy'] for i in e.TWO_MASKS]))
        self.assertEqual(model['masks'][15]['cost'],1)
        self.assertEqual(model['masks'][0]['raw_macs'],1965000)
        self.assertEqual(model['higher_fraction'],.75)
        json.dumps(model,allow_nan=False)

    def test_no_qualifying_examples_and_undefined_spectrum_stay_empty(self):
        panel,labels=synthetic_panel();panel['pred'][15]=(labels+1)%10;panel['higher_fraction'][:]=np.nan
        model=e.build_model(record('plain'),panel,labels)
        self.assertEqual(model['examples'],[]);self.assertIsNone(model['higher_fraction'])
        self.assertEqual(model['higher_fraction_undefined_count'],len(labels))
        self.assertIn('forced crop surgery',model['intervention'])

    def test_gate_blocks_all_outcome_reads_without_ready_and_matching_freeze(self):
        with tempfile.TemporaryDirectory() as directory,patch.object(e.np,'load',side_effect=AssertionError('Outcome read forbidden')):
            root=Path(directory)
            with self.assertRaises(RuntimeError):e.export_report(root,root/'output')
            with self.assertRaises(RuntimeError):e.export_report(root,root/'output',ready=True)
            manifest={'policies':{f'{seed}/{recipe}-{state}':{} for seed in e.SEEDS for recipe in e.RECIPES for state in e.STATES},
                      'test_arrays_opened_during_fit':False}
            path=root/'policy-manifest.json';path.write_text(json.dumps(manifest));(root/'policy-manifest.sha256').write_text(e.sha(path))
            self.assertEqual(e.policy_gate(root,True)[1],e.sha(path))
            path.write_text(path.read_text()+' ')
            with self.assertRaises(RuntimeError):e.policy_gate(root,True)

    def test_full_synthetic30_state_export_images_and_js(self):
        # Entire production-shaped pipeline in a temporary tree, using invented
        # predictions/images only. No study path or publication directory used.
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);output=root/'website';panel,labels=synthetic_panel(1000)
            pixels=np.zeros((10000,28,28),dtype=np.uint8);pixels[:]=np.arange(10000,dtype=np.uint16)[:,None,None]%256
            manifest={'policies':{f'{s}/{r}-{state}':{} for s in e.SEEDS for r in e.RECIPES for state in e.STATES},'test_arrays_opened_during_fit':False}
            path=root/'policy-manifest.json';path.write_text(json.dumps(manifest));(root/'policy-manifest.sha256').write_text(e.sha(path))
            for seed in e.SEEDS:
                job=root/f'hypothesis-audit-s{seed}-v1';job.mkdir()
                np.savez_compressed(job/'data.npz',test_labels=labels,test_images=pixels)
                artifacts={'data.npz':{'sha256':e.sha(job/'data.npz')}};records=[]
                for recipe in e.RECIPES:
                    for state in e.STATES:
                        r=record(recipe,state,seed,panel,labels);name=f'{recipe}-{state}'
                        (job/(name+'.json')).write_text(json.dumps(r))
                        np.savez_compressed(job/(name+'-test.npz'),**panel)
                        for filename in [name+'.json',name+'-test.npz']:artifacts[filename]={'sha256':e.sha(job/filename)}
                        records.append({'recipe':recipe,'state':state})
                (job/'result.json').write_text(json.dumps({'run_id':job.name,'records':records,'artifacts':artifacts}))
            status=e.export_report(root,output,ready=True)
            data=json.loads((output/'data.json').read_text())
            self.assertEqual(status['models'],30);self.assertEqual(len({m['id'] for m in data['models']}),30)
            self.assertEqual(len(data['images']),60) # Shared across every state, no30x duplication.
            for index,encoded in data['images'].items():
                self.assertEqual(base64.b64decode(encoded),pixels[int(index)].tobytes())
                self.assertEqual(len(base64.b64decode(encoded)),784)
            js=(output/'data.js').read_text()
            self.assertTrue(js.startswith('window.HYPOTHESIS_DATA = '))
            self.assertEqual(json.loads(js[len('window.HYPOTHESIS_DATA = '):-2]),data)
            self.assertEqual(data['primary_panel']['masks'],list(e.TWO_MASKS))


if __name__=='__main__':unittest.main()
