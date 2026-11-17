"""Build the RTX 5090 calibration library and install this rank's measured plans."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rank', type=int, choices=range(4), required=True)
    parser.add_argument('--include', type=Path, required=True, help='CUDA include directory')
    parser.add_argument('--lib', type=Path, required=True, help='directory containing libcublasLt.so.13')
    parser.add_argument('destination', type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    output = args.destination.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent) as temp:
        directory = Path(temp) / 'plans'
        directory.mkdir()
        table = root / 'configs/full480/rtx5090-cu132' / f'rank{args.rank}.json.gz'
        for name, contents in json.loads(gzip.decompress(table.read_bytes())).items():
            (directory / name).write_text(contents)
        source = root / 'rl/search/full480/native/activated_plan_owner.cpp'
        library = args.lib.resolve()
        subprocess.run(['g++', '-std=c++17', '-O2', '-fPIC', '-shared', str(source),
                        '-I', str(args.include), '-L', str(library), '-l:libcublasLt.so.13',
                        f'-Wl,-rpath,{library}', '-o', str(directory / 'activated_plan_owner.so')], check=True)
        manifest = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in directory.iterdir()}
        (directory / 'calibration.json').write_text(json.dumps(manifest, indent=2))
        directory.rename(output)


if __name__ == '__main__':
    main()
