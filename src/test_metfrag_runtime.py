"""Regression for the observed Java17 / class65 failure; no model or spectra."""
import ast
import os
import re
import shutil
import subprocess
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

source = ast.parse(Path(__file__).with_name('metfrag_stage.py').read_text(encoding='utf-8'))
helper = next(n for n in source.body if isinstance(n, ast.FunctionDef) and n.name == '_runtime_preflight')
scope = dict(os=os, re=re, shutil=shutil, subprocess=subprocess, zipfile=zipfile)
exec(compile(ast.Module(body=[helper], type_ignores=[]), '<runtime helper>', 'exec'), scope)


class RuntimeTests(unittest.TestCase):
    def check_runtime(self, major):
        with tempfile.TemporaryDirectory() as tmp:
            jar = Path(tmp) / 'class65.jar'
            with zipfile.ZipFile(jar, 'w') as z:
                z.writestr('META-INF/MANIFEST.MF', 'Main-Class: example.Main\n')
                z.writestr('example/Main.class', b'\xca\xfe\xba\xbe\0\0\0A')
            completed = subprocess.CompletedProcess([], 0, '', f'openjdk version "{major}.0.2"')
            with patch.dict(os.environ, {'METFRAG_JAVA': '/official/java'}), patch.object(subprocess, 'run', return_value=completed):
                return scope['_runtime_preflight'](jar)

    def test_actual_class65_rejects_java17(self):
        java, info = self.check_runtime(17)
        self.assertIsNone(java)
        self.assertEqual(info['required_java_major'], 21)
        self.assertEqual(info['status'], 'incompatible Java version')

    def test_class65_accepts_java21(self):
        java, info = self.check_runtime(21)
        self.assertEqual(java, '/official/java')
        self.assertEqual(info['status'], 'compatible')


if __name__ == '__main__':
    unittest.main()
