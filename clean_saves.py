#!/usr/bin/env python3
"""Delete MegaMek training artifacts to reclaim disk space.

Cleans: savegames, run_* directories, Java/GC logs, and heap dumps.
By default everything is cleaned; use --keep-* flags to preserve categories.
"""

import argparse
import shutil
from pathlib import Path


def _file_size(path: Path) -> int:
    """Return file size, or 0 if stat fails (e.g. broken symlink)."""
    try:
        return path.stat().st_size
    except OSError:
        return 0


def _delete_file(path: Path, dry_run: bool) -> int:
    """Delete a single file. Returns bytes freed."""
    size = _file_size(path)
    if dry_run:
        print(f"  would delete {path}")
    else:
        path.unlink()
    return size


def _delete_dir(path: Path, dry_run: bool) -> int:
    """Delete a directory tree. Returns bytes freed."""
    size = sum(_file_size(f) for f in path.rglob("*") if f.is_file())
    if dry_run:
        print(f"  would remove {path}/ ({size / 1024 / 1024:.1f} MB)")
    else:
        shutil.rmtree(path)
    return size


def main():
    parser = argparse.ArgumentParser(
        description="Clean MegaMek training artifacts (saves, logs, run dirs, heap dumps)"
    )
    parser.add_argument("--megamek-dir", default="../megamek",
                        help="Path to megamek repo (default: ../megamek)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be deleted without deleting")
    parser.add_argument("--keep-saves", action="store_true",
                        help="Keep savegame files (*.sav.gz)")
    parser.add_argument("--keep-run-dirs", action="store_true",
                        help="Keep run_* directories (game logs, per-port data)")
    parser.add_argument("--keep-logs", action="store_true",
                        help="Keep Java logs, GC logs, and heap dumps")
    args = parser.parse_args()

    megamek_dir = Path(args.megamek_dir)
    base = megamek_dir / "megamek"
    if not base.is_dir():
        print(f"Error: {base} not found")
        return 1

    categories = {}  # name -> (bytes, count)

    # --- Java stderr logs (in megamek_dir, not megamek/megamek/) ---
    if not args.keep_logs:
        bytes_freed = 0
        count = 0
        for f in sorted(megamek_dir.glob("rl_java_*.log*")):
            if f.is_file():
                bytes_freed += _delete_file(f, args.dry_run)
                count += 1
        if count:
            categories["Java logs"] = (bytes_freed, count)

    # --- GC logs and heap dumps (in megamek/megamek/) ---
    if not args.keep_logs:
        bytes_freed = 0
        count = 0
        for f in sorted(base.glob("rl_gc.log*")):
            if f.is_file():
                bytes_freed += _delete_file(f, args.dry_run)
                count += 1
        heap = base / "rl_heapdump.hprof"
        if heap.is_file():
            bytes_freed += _delete_file(heap, args.dry_run)
            count += 1
        if count:
            categories["GC logs / heap dumps"] = (bytes_freed, count)

    # --- Run directories ---
    if not args.keep_run_dirs:
        bytes_freed = 0
        count = 0
        for run_dir in sorted(base.glob("run_*")):
            if run_dir.is_dir():
                bytes_freed += _delete_dir(run_dir, args.dry_run)
                count += 1
        if count:
            categories["run_* directories"] = (bytes_freed, count)

    # --- Shared savegames ---
    if not args.keep_saves:
        bytes_freed = 0
        count = 0
        shared_dir = base / "savegames"
        if shared_dir.is_dir():
            for f in sorted(shared_dir.glob("*.sav.gz")):
                bytes_freed += _delete_file(f, args.dry_run)
                count += 1
        if count:
            categories["Savegames"] = (bytes_freed, count)

    # --- Summary ---
    total_bytes = sum(b for b, _ in categories.values())
    total_items = sum(c for _, c in categories.values())
    action = "Would free" if args.dry_run else "Freed"

    if categories:
        print()
        for name, (b, c) in categories.items():
            print(f"  {name}: {b / 1024 / 1024:.1f} MB ({c} items)")
        print(f"\n{action} {total_bytes / 1024 / 1024:.1f} MB total ({total_items} items)")
    else:
        print("Nothing to clean.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
