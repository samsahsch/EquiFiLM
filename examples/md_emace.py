"""NVT-then-NVE MD driver for an E-MACE checkpoint.

Used to produce the energy-conservation Fig. 6 and the trajectory ensembles
underlying the Delta-g(r) and Delta-PDF analyses (Figs. 7, 8). Logs E_pot,
E_kin, E_tot every step, and dumps NVE positions every ``--xyz_interval`` steps.

Usage::

    python examples/md_emace.py \\
        --model_path  checkpoints/MACE-FiLM-large_stagetwo.model \\
        --seed_xyz    data/seed_2592atom.xyz \\
        --total_charge 12 --supercell 2 2 2 \\
        --n_nvt 1500 --n_nve 1500 --temperature 300 \\
        --xyz_interval 25 \\
        --output_dir  md_q12 --output_prefix q12

Notes
-----
* For ``--total_charge 0`` use ``--timestep_fs 0.5`` and double the ``--n_nvt``
  / ``--n_nve`` to maintain the same physical duration. The model is
  numerically unstable at q=0 with 1.0 fs timestep in 2x2x2 supercells.
* Charged states (q >= 4) are stable at 1.0 fs.
"""
import argparse
import os
import sys
import time
from pathlib import Path

import torch
import numpy as np
from ase import units
from ase.io import read, write
from ase.md.langevin import Langevin
from ase.md.verlet import VelocityVerlet
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution

# Avoid the 'weights_only=True' default that PyTorch is moving to.
os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

from mace.calculators.mace import MACECalculator


def model_uses_film(model_path: str) -> bool:
    try:
        m = torch.load(model_path, map_location="cpu", weights_only=False)
        return getattr(m, "charge_film_config", None) is not None
    except Exception:
        return False


def build_initial_config(seed_xyz: str, seed_index: int, total_charge: float, supercell):
    atoms = read(seed_xyz, index=seed_index)
    atoms.info["total_charge"] = float(total_charge)
    atoms.info["total_spin"] = 0.0
    if list(supercell) != [1, 1, 1]:
        atoms = atoms.repeat(tuple(supercell))
    return atoms


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", required=True)
    p.add_argument("--seed_xyz", required=True)
    p.add_argument("--seed_index", type=int, default=-1)
    p.add_argument("--total_charge", type=float, required=True)
    p.add_argument("--temperature", type=float, default=300.0)
    p.add_argument("--supercell", type=int, nargs=3, default=[2, 2, 2])
    p.add_argument("--n_nvt", type=int, default=1500)
    p.add_argument("--n_nve", type=int, default=1500)
    p.add_argument("--timestep_fs", type=float, default=1.0)
    p.add_argument("--friction_invps", type=float, default=2.0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="float32", choices=["float32", "float64"])
    p.add_argument("--output_dir", required=True)
    p.add_argument("--output_prefix", required=True)
    p.add_argument("--xyz_interval", type=int, default=10,
                   help="Steps between NVE-phase position dumps. Default 10.")
    args = p.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    print(f"FiLM model: {model_uses_film(args.model_path)}", flush=True)
    print(f"Building config: src={args.seed_xyz} q={args.total_charge} sc={args.supercell}", flush=True)
    atoms = build_initial_config(args.seed_xyz, args.seed_index,
                                 args.total_charge, args.supercell)
    print(f"  {len(atoms)} atoms cell={atoms.cell.diagonal()}", flush=True)

    calc = MACECalculator(
        model_paths=args.model_path, device=args.device, default_dtype=args.dtype,
        info_keys={"total_charge": "total_charge", "total_spin": "total_spin"},
    )
    atoms.calc = calc
    MaxwellBoltzmannDistribution(atoms, temperature_K=args.temperature)

    log_path = os.path.join(args.output_dir, f"{args.output_prefix}_energies.csv")
    fhandle = open(log_path, "w")
    fhandle.write("step,t_ps,phase,E_pot_eV,E_kin_eV,E_tot_eV,T_K\n")
    natoms = len(atoms)
    nve_xyz_path = os.path.join(args.output_dir, f"{args.output_prefix}_nve.xyz")

    def log_energies(step: int, phase: str) -> None:
        epot = atoms.get_potential_energy()
        ekin = atoms.get_kinetic_energy()
        etot = epot + ekin
        T_inst = ekin / (1.5 * natoms * units.kB)
        t_ps = step * args.timestep_fs / 1000.0
        fhandle.write(f"{step},{t_ps:.4f},{phase},{epot:.6f},{ekin:.6f},{etot:.6f},{T_inst:.2f}\n")
        fhandle.flush()
        if step % 200 == 0:
            print(f"  step={step:5d} t={t_ps:5.2f} ps phase={phase} "
                  f"E_pot={epot/natoms:.4f} eV/atom E_tot={etot/natoms:.4f} T={T_inst:.0f} K",
                  flush=True)

    nve_first = True

    def maybe_dump_xyz(step: int, phase: str) -> None:
        nonlocal nve_first
        if (step % args.xyz_interval) != 0:
            return
        atoms.info["step"] = step
        atoms.info["t_ps"] = step * args.timestep_fs / 1000.0
        atoms.info["phase"] = phase
        write(nve_xyz_path, atoms, format="extxyz", append=(not nve_first))
        nve_first = False

    print(f"\n=== Phase 1: NVT Langevin, {args.n_nvt} steps "
          f"({args.n_nvt * args.timestep_fs / 1000.0:.1f} ps) ===", flush=True)
    dyn = Langevin(atoms, timestep=args.timestep_fs * units.fs,
                   temperature_K=args.temperature,
                   friction=args.friction_invps / 1000.0)
    log_energies(0, "nvt")
    t0 = time.time()
    for step in range(1, args.n_nvt + 1):
        dyn.run(1)
        log_energies(step, "nvt")
    print(f"NVT done: {time.time()-t0:.1f}s", flush=True)

    print(f"\n=== Phase 2: NVE Verlet, {args.n_nve} steps "
          f"({args.n_nve * args.timestep_fs / 1000.0:.1f} ps) ===", flush=True)
    dyn = VelocityVerlet(atoms, timestep=args.timestep_fs * units.fs)
    maybe_dump_xyz(args.n_nvt, "nve_init")
    t0 = time.time()
    for step in range(args.n_nvt + 1, args.n_nvt + args.n_nve + 1):
        dyn.run(1)
        log_energies(step, "nve")
        maybe_dump_xyz(step, "nve")
    print(f"NVE done: {time.time()-t0:.1f}s", flush=True)

    fhandle.close()
    print(f"\nWrote {log_path}")
    print(f"Wrote {nve_xyz_path}")


if __name__ == "__main__":
    main()
