# Pinned, official vendor JRE; no package-manager or system-Java changes.
from pathlib import Path as _MFPath
import hashlib as _mf_hash, tarfile as _mf_tar, os as _mf_os
_mf_candidates = list(_MFPath('/kaggle/input').rglob('OpenJDK21U-jre_x64_linux_hotspot_21.0.13_9.tar.gz.bin'))
if len(_mf_candidates) != 1:
    raise FileNotFoundError('Attach pinned Temurin21 dataset version 2, with one hash-verifiable .tar.gz.bin archive')
_mf_archive, = _mf_candidates
assert _mf_archive.stat().st_size == 52181297
assert _mf_hash.sha256(_mf_archive.read_bytes()).hexdigest() == 'a535a1df9c6284d9618872984e026ae0352d1f7e9e0605425c0e6114f47351dd'
_mf_dest = _MFPath('/kaggle/working/metfrag-runtime'); _mf_dest.mkdir(exist_ok=True)
with _mf_tar.open(_mf_archive) as _mf_bundle:
    _mf_bundle.extractall(_mf_dest, filter='data')
_mf_java, = _mf_dest.glob('*/bin/java')
_mf_os.chmod(_mf_java, 0o755)
_mf_os.environ['METFRAG_JAVA'] = str(_mf_java)
print('METFRAG_RUNTIME_VENDOR_SHA256_VERIFIED', 'a535a1df9c6284d9618872984e026ae0352d1f7e9e0605425c0e6114f47351dd')
