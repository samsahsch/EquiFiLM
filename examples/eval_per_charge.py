"""Reproduce per-charge force/energy RMSE on the held-out 10% validation split.

Reads the same val_indices file that mace_run_train wrote during training
(``<name>_valid_indices_<seed>.txt``) and groups errors by ``total_charge``.
Outputs a JSON summary that matches the format used by Table 2 of the paper.

Usage::

    python examples/eval_per_charge.py \\
        --model_path checkpoints/MACE-FiLM-large_stagetwo.model \\
        --train_xyz  data/combined_all_charges.xyz \\
        --valid_indices_file checkpoints/MACE-FiLM-large_valid_indices_123.txt \\
        --out_json   eval_per_charge.json
"""
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from ase.io import read
from mace.calculators.mace import MACECalculator


def evaluate(model_path: str, all_frames, val_indices, device: str = "cuda"):
    print(f"Loading model: {model_path}")
    calc = MACECalculator(
        model_paths=model_path,
        device=device,
        default_dtype="float32",
        info_keys={"total_charge": "total_charge", "total_spin": "total_spin"},
    )

    per_charge: dict = {}
    skipped = 0
    for vi in val_indices:
        ref_atoms = all_frames[int(vi)]
        q = int(round(ref_atoms.info.get("total_charge", 0)))
        try:
            ref_E = ref_atoms.get_potential_energy()
            ref_F = ref_atoms.get_forces()
        except Exception:
            skipped += 1
            continue
        a = ref_atoms.copy()
        a.calc = calc
        try:
            pred_E = a.get_potential_energy()
            pred_F = a.get_forces()
        except Exception as e:
            print(f"  inference failed at idx {vi}: {type(e).__name__}: {e}")
            skipped += 1
            continue
        bucket = per_charge.setdefault(q, {"e_diffs": [], "f_sq": []})
        n_atoms = len(ref_atoms)
        bucket["e_diffs"].append((pred_E - ref_E) / n_atoms * 1000.0)
        bucket["f_sq"].append(((pred_F - ref_F) ** 2).mean())

    out = {"n_val_total": len(val_indices), "n_skipped": skipped, "per_charge": {}}
    for q in sorted(per_charge):
        e = np.asarray(per_charge[q]["e_diffs"])
        f = np.asarray(per_charge[q]["f_sq"])
        out["per_charge"][f"q{q}"] = {
            "n": int(len(e)),
            "F_RMSE_meV_A": float(np.sqrt(np.mean(f)) * 1000.0),
            "E_RMSE_meV_atom_raw": float(np.sqrt(np.mean(e ** 2))),
        }
        print(f"  q={q:>2}: n={len(e):>3}  F={out['per_charge'][f'q{q}']['F_RMSE_meV_A']:6.2f}"
              f"  E={out['per_charge'][f'q{q}']['E_RMSE_meV_atom_raw']:7.4f}")
    Fs = [v["F_RMSE_meV_A"] for v in out["per_charge"].values()]
    out["F_mean_meV_A"] = float(np.mean(Fs))
    print(f"\n  -> mean F across charges: {out['F_mean_meV_A']:.2f} meV/A")

    del calc
    torch.cuda.empty_cache()
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", required=True)
    p.add_argument("--train_xyz", required=True,
                   help="Combined-charges xyz file used during training.")
    p.add_argument("--valid_indices_file", required=True,
                   help="Path to <name>_valid_indices_<seed>.txt produced by mace_run_train.")
    p.add_argument("--out_json", default="eval_per_charge.json")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    print(f"Reading {args.train_xyz} ...")
    t0 = time.time()
    all_frames = read(args.train_xyz, index=":")
    print(f"  {len(all_frames)} frames in {time.time()-t0:.1f}s")

    val_indices = np.loadtxt(args.valid_indices_file, dtype=int)
    print(f"  {len(val_indices)} validation indices")

    out = evaluate(args.model_path, all_frames, val_indices, device=args.device)

    out["_meta"] = {
        "model_path": args.model_path,
        "train_xyz": args.train_xyz,
        "valid_indices_file": args.valid_indices_file,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
    }
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved -> {args.out_json}")


if __name__ == "__main__":
    main()
