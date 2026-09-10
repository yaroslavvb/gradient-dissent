"""Offline transport-integrity tests; no service credentials or GPU calls."""
from pathlib import Path
import hashlib
import importlib.util
import json
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('halfdrop_verify_transport',
    Path(__file__).with_name('verify_transport.py'))
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class TransportTest(unittest.TestCase):
    def test_streaming_sha(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'blob'
            value = b'counterexample\x00' * 100001
            path.write_bytes(value)
            self.assertEqual(module.sha(path), hashlib.sha256(value).hexdigest())

    def fixture(self, root, remote_value):
        run_id = 'halfdrop-fixture'
        (root / 'raw' / run_id).mkdir(parents=True)
        (root / (run_id + '.json')).write_text(json.dumps({
            'run_id': run_id, 'complete': True, 'local_dispatch_elapsed_seconds': 4.5}))
        calls = []

        class FakeVolume:
            def read_file_into_fileobj(self, name, stream):
                calls.append(name)
                stream.write(json.dumps(remote_value).encode())

        fake = types.SimpleNamespace(Volume=types.SimpleNamespace(
            from_name=lambda *args, **kwargs: FakeVolume()))
        return run_id, fake, calls

    def test_authoritative_only_dispatch_difference_allowed_and_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_id, fake, calls = self.fixture(root, {'run_id': 'halfdrop-fixture', 'complete': True})
            with patch.dict(sys.modules, {'modal': fake}):
                module.authoritative_download(root, [{'run_id': run_id}])
                module.authoritative_download(root, [{'run_id': run_id}])
            self.assertEqual(calls, ['/runs/halfdrop-fixture/result.json'])
            receipt = module.read(root / 'authoritative-result-download.json')
            self.assertEqual(receipt['gpu_calls'], 0)
            self.assertTrue(receipt['complete'])

    def test_different_remote_payload_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_id, fake, _ = self.fixture(root, {'run_id': 'halfdrop-fixture', 'complete': False})
            with patch.dict(sys.modules, {'modal': fake}), self.assertRaises(AssertionError):
                module.authoritative_download(root, [{'run_id': run_id}])
            self.assertFalse((root / 'authoritative-result-download.json').exists())


if __name__ == '__main__':
    unittest.main()
