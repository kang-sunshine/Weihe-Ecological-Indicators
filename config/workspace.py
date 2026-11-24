"""Stage the archived relative code layout in a dedicated analysis workspace."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

REPOSITORY = Path(__file__).resolve().parents[1]

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("workspace", type=Path)
    parser.add_argument("--run", help="Public script path from config/layout.json")
    args = parser.parse_args()
    work = args.workspace.resolve()
    if work == REPOSITORY or REPOSITORY in work.parents:
        parser.error("Choose a separate analysis workspace.")
    layout = json.loads((REPOSITORY / "config/layout.json").read_text(encoding="utf-8"))
    if args.run and args.run not in layout:
        parser.error("Unknown script")
    for key, rel in layout.items():
        target = work / rel
        data = (REPOSITORY / key).read_bytes()
        if target.exists() and target.read_bytes() != data:
            parser.error(f"Existing script differs: {target}")
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            target.write_bytes(data)
    if args.run:
        env = os.environ.copy()
        env["PYTHONPATH"] = os.pathsep.join(dict.fromkeys(str((work / v).parent) for v in layout.values()))
        subprocess.run([sys.executable, str(work / layout[args.run])], cwd=work, env=env, check=True)

if __name__ == "__main__":
    main()
