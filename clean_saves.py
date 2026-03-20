#!/usr/bin/env python3
"""Delete MegaMek saved game files to reclaim disk space."""

import argparse
import shutil
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Clean MegaMek save files")
    parser.add_argument("--megamek-dir", default="../megamek", help="Path to megamek repo (default: ../megamek)")
    parser.add_argument("--run-dirs", action="store_true", help="Also remove entire run_* directories (logs, data, etc.)")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be deleted without deleting")
    args = parser.parse_args()

    base = Path(args.megamek_dir) / "megamek"
    if not base.is_dir():
        print(f"Error: {base} not found")
        return 1

    total_bytes = 0
    total_files = 0

    # Delete .sav.gz files from shared savegames dir
    shared_dir = base / "savegames"
    if shared_dir.is_dir():
        for f in shared_dir.glob("*.sav.gz"):
            size = f.stat().st_size
            if args.dry_run:
                print(f"  would delete {f}")
            else:
                f.unlink()
            total_bytes += size
            total_files += 1

    # Handle run_* directories
    run_dirs = sorted(base.glob("run_*"))
    for run_dir in run_dirs:
        if not run_dir.is_dir():
            continue
        if args.run_dirs:
            size = sum(f.stat().st_size for f in run_dir.rglob("*") if f.is_file())
            if args.dry_run:
                print(f"  would remove {run_dir}/ ({size / 1024 / 1024:.1f} MB)")
            else:
                shutil.rmtree(run_dir)
            total_bytes += size
            total_files += 1  # count as one unit
        else:
            save_dir = run_dir / "savegames"
            if not save_dir.is_dir():
                continue
            for f in save_dir.glob("*.sav.gz"):
                size = f.stat().st_size
                if args.dry_run:
                    print(f"  would delete {f}")
                else:
                    f.unlink()
                total_bytes += size
                total_files += 1

    mb = total_bytes / 1024 / 1024
    action = "Would free" if args.dry_run else "Freed"
    print(f"{action} {mb:.1f} MB ({total_files} items)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
