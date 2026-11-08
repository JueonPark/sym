"""Content-address the code and binaries actually used by a measurement."""
import hashlib
from pathlib import Path


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for data in iter(lambda: stream.read(1 << 20), b''):
            value.update(data)
    return value.hexdigest()


def provenance(repo, runtime, compiler):
    source = {}
    for directory in ('libreloc/include', 'libreloc/src', 'libreloc/quant',
                      'libreloc/python', 'bench/issue267'):
        for path in (repo / directory).rglob('*'):
            if path.suffix in ('.py', '.cpp', '.h', '.cu') and '__pycache__' not in path.parts:
                source[str(path.relative_to(repo))] = sha(path)
    for relative in ('bench/issue266/isolation.py', 'bench/issue227/run.py'):
        source[relative] = sha(repo / relative)
    files = {}
    for directory in ('libreloc', 'python', 'helper'):
        for path in (runtime / directory).rglob('*'):
            if path.is_file() and path.suffix in ('.py', '.so') and '__pycache__' not in path.parts:
                files[str(path.relative_to(runtime))] = sha(path)
    return dict(source=source, runtime=files, compiler=sha(compiler))


def cache_inventory(directory):
    result = {}
    for path in directory.rglob('*'):
        if path.is_file():
            result[str(path.relative_to(directory))] = path.stat().st_size
    return dict(files=len(result), bytes=sum(result.values()),
                generated_files=sum(name.endswith(('.cpp', '.so', '.py')) for name in result),
                generated_bytes=sum(size for name, size in result.items()
                                    if name.endswith(('.cpp', '.so', '.py'))))
