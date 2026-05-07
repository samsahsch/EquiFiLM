"""Apply EquiFiLM's four-file patch to a local mace-torch==0.3.15 installation.

Run as::

    python -m equifilm.apply_patch
    python -m equifilm.apply_patch --revert        # restore upstream files
    python -m equifilm.apply_patch --dry-run       # show what would change

The four upstream files are backed up with a ``.upstream-bak`` suffix the first
time the patch is applied. ``--revert`` restores them.
"""
import argparse
import importlib.util
import os
import shutil
import sys
from pathlib import Path

# Mapping: package-relative target path -> bundled patch file in equifilm/_patches/
PATCH_FILES = {
    "modules/blocks.py":        "blocks.py",
    "modules/models.py":        "models.py",
    "tools/arg_parser.py":      "arg_parser.py",
    "tools/model_script_utils.py": "model_script_utils.py",
}


def find_mace_root() -> Path:
    """Locate the installed mace-torch package directory."""
    spec = importlib.util.find_spec("mace")
    if spec is None or spec.origin is None:
        sys.exit(
            "ERROR: 'mace' is not importable. Install it first:\n"
            "    pip install mace-torch==0.3.15"
        )
    return Path(spec.origin).parent


def check_version() -> str:
    """Warn if the installed MACE version is not 0.3.15."""
    try:
        import mace
        v = getattr(mace, "__version__", None) or "unknown"
    except Exception:
        v = "unknown"
    if v != "0.3.15":
        print(
            f"WARNING: installed mace-torch is version {v}. EquiFiLM patches were\n"
            "         developed against 0.3.15. The patch may still apply cleanly\n"
            "         but is not guaranteed to.",
            file=sys.stderr,
        )
    return v


def apply(dry_run: bool = False, force: bool = False) -> int:
    """Copy the four patch files into the local MACE installation.

    Returns the number of files actually changed.
    """
    mace_root = find_mace_root()
    patches_dir = Path(__file__).parent / "_patches"
    if not patches_dir.is_dir():
        sys.exit(f"ERROR: cannot find {patches_dir}")

    print(f"MACE root:    {mace_root}")
    print(f"Patches dir:  {patches_dir}")
    n_changed = 0

    for rel_target, patch_name in PATCH_FILES.items():
        target = mace_root / rel_target
        source = patches_dir / patch_name
        backup = target.with_suffix(target.suffix + ".upstream-bak")

        if not target.exists():
            print(f"  [skip] {rel_target}  (does not exist; MACE layout changed?)")
            continue
        if not source.exists():
            print(f"  [skip] {rel_target}  (patch file {patch_name} missing)")
            continue

        # Skip if source and target are byte-identical
        if not force and target.read_bytes() == source.read_bytes():
            print(f"  [ok ] {rel_target}  (already patched)")
            continue

        if dry_run:
            print(f"  [DRY] would copy  {patch_name}  ->  {rel_target}")
            n_changed += 1
            continue

        # Make a one-time backup of the upstream version
        if not backup.exists():
            shutil.copy2(target, backup)
            print(f"  [bak] {rel_target}  ->  {backup.name}")

        shutil.copy2(source, target)
        print(f"  [ok ] {rel_target}  patched")
        n_changed += 1

    return n_changed


def revert(dry_run: bool = False) -> int:
    """Restore the upstream-bak files."""
    mace_root = find_mace_root()
    n_changed = 0
    for rel_target in PATCH_FILES:
        target = mace_root / rel_target
        backup = target.with_suffix(target.suffix + ".upstream-bak")
        if not backup.exists():
            print(f"  [skip] {rel_target}  (no backup found)")
            continue
        if dry_run:
            print(f"  [DRY] would restore  {backup.name}  ->  {rel_target}")
            n_changed += 1
            continue
        shutil.copy2(backup, target)
        print(f"  [ok ] {rel_target}  restored from upstream-bak")
        n_changed += 1
    return n_changed


def main() -> None:
    p = argparse.ArgumentParser(
        description="Patch a local mace-torch==0.3.15 install with EquiFiLM."
    )
    p.add_argument("--revert", action="store_true",
                   help="Restore the upstream files from .upstream-bak.")
    p.add_argument("--dry-run", action="store_true",
                   help="Print what would be done without modifying anything.")
    p.add_argument("--force", action="store_true",
                   help="Re-copy patch files even if already in place.")
    args = p.parse_args()

    check_version()
    if args.revert:
        n = revert(dry_run=args.dry_run)
    else:
        n = apply(dry_run=args.dry_run, force=args.force)

    if args.dry_run:
        print(f"\nDry run: {n} file(s) would be changed.")
    else:
        print(f"\nDone: {n} file(s) changed.")


if __name__ == "__main__":
    main()
