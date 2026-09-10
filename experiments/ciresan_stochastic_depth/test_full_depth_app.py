"""Offline resource/source guards; importing the app provisions no resources."""
import copy
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import full_depth_app as launcher


def jobs():
    return [{'run_id':f'full-depth-s{s}-v2', 'seed':s, 'gpu':'A100-40GB',
             'stage':'inference', 'timeout_seconds':180, 'batch_size':2048,
             'source_sha256':{'fixture.py':'a'*64},
             'models':[{'id':f'{r}-{state}-s{s}'} for r in
                       ('plain','residual','sd_constant','sd_annealed','residual_unit_dropout')
                       for state in ('selected','final')]} for s in (101,102,103)]


class LauncherTests(unittest.TestCase):
    def test_exact_bounded_cohort(self):
        launcher.validate_jobs(jobs())
        for mutate in (lambda j:j.pop(), lambda j:j[0].update(gpu='A100-80GB'),
                       lambda j:j[0].update(timeout_seconds=181),
                       lambda j:j[0].update(batch_size=1024),
                       lambda j:j[0]['models'].pop(),
                       lambda j:j[0].update(source_sha256={})): 
            j=jobs();mutate(j)
            with self.assertRaises(ValueError):launcher.validate_jobs(j)

    def test_source_integrity_and_path_boundary(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'fixture.py';p.write_text('frozen\n')
            spec={'source_sha256':{'fixture.py':hashlib.sha256(p.read_bytes()).hexdigest()}}
            launcher.verify_sources(spec,td)
            p.write_text('changed\n')
            with self.assertRaises(ValueError):launcher.verify_sources(spec,td)
            with self.assertRaises(ValueError):launcher.verify_sources({'source_sha256':{'../outside':'a'*64}},td)

    def test_nonfinite_billing_is_not_accepted(self):
        class Bad:
            metered_cost=float('nan')
        with patch.object(launcher.modal.Environment, 'from_name') as env:
            env.return_value.billing.summary.return_value=Bad()
            with self.assertRaises(ValueError):launcher.billing_snapshot()


if __name__=='__main__':unittest.main()
