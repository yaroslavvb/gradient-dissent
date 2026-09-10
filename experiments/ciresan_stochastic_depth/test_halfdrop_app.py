"""Offline bounds and artifact-transport tests; no cloud calls."""
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import halfdrop_app as launcher


def main_jobs():
    return [{'run_id': f'halfdrop-m{m:02d}-s{s}-v1', 'stage': 'evaluate',
             'seed': s, 'drop_mask': m, 'epochs': 100, 'lr': .01,
             'batch_size': 64, 'momentum': .9, 'gpu': 'A100-40GB',
             'timeout_seconds': 180, 'source_sha256': {'fixture.py': 'a'*64}}
            for m in range(16) for s in (201,202,203)]


class HalfdropLauncherTests(unittest.TestCase):
    def test_exact_main_scope_and_resources(self):
        launcher.validate_jobs(main_jobs(), 'main')
        for mutate in (lambda j: j.pop(), lambda j: j[0].update(drop_mask=1),
                       lambda j: j[0].update(lr=.03), lambda j: j[0].update(epochs=101),
                       lambda j: j[0].update(timeout_seconds=181),
                       lambda j: j[0].update(gpu='A100-80GB')):
            jobs=main_jobs(); mutate(jobs)
            with self.assertRaises(ValueError): launcher.validate_jobs(jobs, 'main')

    def test_qualification_has_no_test_stage(self):
        parent={'run_id':'halfdrop-qualification-v1','stage':'pilot','kind':'qualification',
                'gpu':'A100-40GB','timeout_seconds':180,'source_sha256':{'x':'a'*64},
                'subjobs':[{'stage':'pilot','epochs':2,'seed':201,'drop_mask':m} for m in (0,15)]}
        launcher.validate_jobs([parent], 'qualification')
        parent['subjobs'][0]['stage']='evaluate'
        with self.assertRaises(ValueError): launcher.validate_jobs([parent], 'qualification')

    def test_artifacts_skip_weights_and_verify_download_bytes(self):
        blob=b'raw experimental bytes\n'; digest=hashlib.sha256(blob).hexdigest()
        result={'run_id':'fixture', 'artifacts':{'raw.npz':{'sha256':digest,'bytes':len(blob)},
                'best.pt':{'sha256':'a'*64,'bytes':48000000}}}
        class Remote:
            def read_file_into_fileobj(self,path,stream):
                if path!='/runs/fixture/raw.npz': raise AssertionError('Unexpected remote path')
                stream.write(blob)
        with tempfile.TemporaryDirectory() as td, patch.object(launcher,'OUT',Path(td)), patch.object(launcher.modal.Volume,'from_name',return_value=Remote()):
            record=launcher.download_artifacts(result)
            self.assertEqual([r['status'] for r in record['files']],['downloaded_verified','retained_on_volume'])
            self.assertFalse((Path(td)/'raw/fixture/best.pt').exists())
            (Path(td)/'raw/fixture/raw.npz').write_bytes(b'corrupted')
            with self.assertRaises(AssertionError):launcher.download_artifacts(result)


if __name__=='__main__': unittest.main()
