"""Execute the candidate CLI in an isolated interpreter with a project-file read guard."""

import json
import os
from pathlib import Path
import runpy
import sys


def main():
    root = Path(__file__).resolve().parents[1]
    original = Path(sys.argv[1]).resolve()
    prefix = Path(sys.prefix).resolve()
    assert Path.cwd().resolve() == root
    assert root != original and sys.flags.isolated and sys.dont_write_bytecode
    assert prefix != original / ".venv"
    blocked = []
    project_files = set()

    def guard(event, args):
        if event != "open" or not isinstance(args[0], (str, bytes, os.PathLike)):
            return
        path = Path(os.fsdecode(args[0])).resolve()
        if path.is_relative_to(root):
            project_files.add(path.relative_to(root).as_posix())
        elif path.is_relative_to(original) and not path.is_relative_to(prefix):
            blocked.append(str(path))
            raise PermissionError(f"Read outside the independent package: {path}")

    sys.addaudithook(guard)
    sys.path.insert(0, str(root / "scripts"))
    sys.argv = ["go2_speed_hold.py", "--controller", "212", "--side", "right", "--kind", "zero",
                "--output", "artifacts/runs/standalone-right-zero"]
    try:
        runpy.run_path(str(root / "scripts/go2_speed_hold.py"), run_name="__main__")
    except SystemExit as exc:
        if exc.code not in (0, None):
            raise
    distributions = {name: str(Path(sys.modules[name].__file__).resolve())
                     for name in ("numpy", "torch", "mujoco", "stable_baselines3")}
    assert not blocked and all(Path(p).is_relative_to(prefix) for p in distributions.values())
    path = root / "validation/isolation.json"
    path.parent.mkdir(exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(dict(interpreter=str(sys.executable), locked_environment=str(prefix),
                       project_file_guard=True, blocked=blocked, actual_dependencies=distributions,
                       project_paths_accessed=sorted(project_files)), stream, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
