"""Smoothed Delta-g(r) and Delta-PDF analysis from NVE trajectory dumps.

Reads ``q*_*/q*_*_nve.xyz`` files from a directory of MD runs, computes O-O,
O-H, H-H pair correlation functions per charge with Gaussian smoothing, then
constructs the Z^2-weighted reduced PDF and writes everything to a JSON.

Run twice (once to compute, once to plot) or all at once with ``--make_plots``.

Usage::

    python examples/plot_dpdf.py \\
        --md_root  md_runs/ \\
        --out_json dpdf_drdf.json \\
        --sigma_A  0.225 --r_max 12.0 --dr 0.05 \\
        --make_plots
"""
import argparse
import glob
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from ase.io import read
from scipy.ndimage import gaussian_filter1d


# ----------------------- g(r) computation (vectorised, cubic MIC) -----------

def gr_pair(traj, pair, r_max=12.0, dr=0.05, sigma_A=0.225):
    bins = np.arange(0, r_max + dr, dr)
    centers = 0.5 * (bins[:-1] + bins[1:])
    a, b = pair
    syms0 = np.array(traj[0].get_chemical_symbols())
    ia = np.where(syms0 == a)[0]
    ib = np.where(syms0 == b)[0]
    if len(ia) == 0 or len(ib) == 0:
        return centers, np.zeros_like(centers)

    L = traj[0].cell.diagonal()
    if not np.allclose(L, L[0]):
        raise ValueError("Non-cubic cell not supported in this simple driver.")
    L = float(L[0])
    V = traj[0].get_volume()

    hist = np.zeros(len(centers))
    for atoms in traj:
        x_a = atoms.positions[ia]
        x_b = atoms.positions[ib]
        diff = x_a[:, None, :] - x_b[None, :, :]
        diff -= np.round(diff / L) * L
        d = np.sqrt(np.sum(diff * diff, axis=-1))
        if a == b:
            np.fill_diagonal(d, np.inf)
        h, _ = np.histogram(d.ravel(), bins=bins)
        hist += h

    n_frames = len(traj)
    rho = len(ib) / V
    shell_vol = 4 * np.pi * centers ** 2 * dr
    gr_raw = hist / n_frames / (len(ia) * shell_vol * rho)
    sigma_bins = sigma_A / dr
    gr_smooth = gaussian_filter1d(gr_raw, sigma_bins, mode="nearest")
    return centers, gr_smooth


def per_charge_grs(traj, r_max, dr, sigma_A):
    pairs = [("O", "O"), ("O", "H"), ("H", "H")]
    out = {}
    r = None
    for a, b in pairs:
        rr, gr = gr_pair(traj, (a, b), r_max=r_max, dr=dr, sigma_A=sigma_A)
        out[f"{a}{b}"] = gr
        r = rr
    return r, out


# ----------------------- Compute everything from MD root --------------------

def compute(args) -> dict:
    nve_xyzs = sorted(glob.glob(f"{args.md_root}/q*_*/*_nve.xyz"))
    nve_xyzs = [p for p in nve_xyzs if os.path.getsize(p) > 5 * 1024 * 1024]
    if not nve_xyzs:
        sys.exit(f"No usable nve.xyz under {args.md_root}")
    print(f"Found {len(nve_xyzs)} trajectories.", flush=True)

    out = {"_meta": {
        "md_root": args.md_root,
        "r_max": args.r_max, "dr": args.dr, "sigma_A": args.sigma_A,
        "n_frames_used": args.n_frames, "skip_initial": args.skip_initial,
        "weighting": "Z_a*Z_b * f_a*f_b * (2 if mixed) (UED-style contrast)",
        "smoothing": "Gaussian convolution on g(r) in r-space",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
    }, "per_charge": {}}

    for path in nve_xyzs:
        tag = os.path.basename(os.path.dirname(path))     # e.g. "q12_interp"
        q = int(tag.split("_")[0][1:])
        cat = "_".join(tag.split("_")[1:])
        print(f"\n=== q={q} ({cat}) — {os.path.basename(path)}", flush=True)
        t0 = time.time()
        full = read(path, index=":")
        usable = full[args.skip_initial:]
        if len(usable) > args.n_frames:
            idx = np.linspace(0, len(usable) - 1, args.n_frames).astype(int)
            traj = [usable[i] for i in idx]
        else:
            traj = usable
        r, grs = per_charge_grs(traj, args.r_max, args.dr, args.sigma_A)
        print(f"   {len(full)} frames -> {len(traj)} used; "
              f"{time.time()-t0:.1f}s; g_OO max={max(grs['OO']):.2f}", flush=True)

        existing = out["per_charge"].get(f"q{q}")
        if existing and existing.get("n_frames_in_full_traj", 0) > len(full):
            continue
        out["per_charge"][f"q{q}"] = {
            "category": cat,
            "trajectory_path": path,
            "n_frames_in_full_traj": len(full),
            "n_frames_used": len(traj),
            "r": r.tolist(),
            "g_OO": grs["OO"].tolist(),
            "g_OH": grs["OH"].tolist(),
            "g_HH": grs["HH"].tolist(),
        }

    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved -> {args.out_json}")
    return out


# ----------------------- Plotting (Z^2-contrast reduced PDF) ----------------

def make_plots(out: dict, fig_dir: str) -> None:
    import matplotlib.pyplot as plt

    Path(fig_dir).mkdir(parents=True, exist_ok=True)
    pc = out["per_charge"]
    if "q0" not in pc:
        print("No q=0 reference present; skipping plots.")
        return

    r = np.asarray(pc["q0"]["r"])
    refOO = np.asarray(pc["q0"]["g_OO"])
    refOH = np.asarray(pc["q0"]["g_OH"])
    refHH = np.asarray(pc["q0"]["g_HH"])

    sample = read(pc["q0"]["trajectory_path"], index=0)
    rho = len(sample) / sample.get_volume()

    f_O, f_H = 1/3, 2/3
    w_OO = f_O * f_O * 64
    w_OH = 2 * f_O * f_H * 8
    w_HH = f_H * f_H * 1
    w_total = w_OO + w_OH + w_HH

    def g_total(gOO, gOH, gHH):
        return (w_OO * gOO + w_OH * gOH + w_HH * gHH) / w_total

    G0 = 4 * np.pi * r * rho * (g_total(refOO, refOH, refHH) - 1)

    charges = sorted(int(k[1:]) for k in pc if k != "q0")
    colors = ["#0072B2", "#0a8043", "#E69F00", "#56B4E9", "#9c27b0", "#c0392b", "#34495e"]

    # --- Fig: Delta-g(r) per pair ---
    R_VIEW = 6.0
    mask = r <= R_VIEW
    fig, axes = __import__("matplotlib.pyplot").pyplot.subplots(1, 3, figsize=(13.5, 5.0))
    panels = [("g_OO", refOO, "O$-$O"), ("g_OH", refOH, "O$-$H"), ("g_HH", refHH, "H$-$H")]
    for ax, (key, refarr, title) in zip(axes, panels):
        ax.axhline(0, color="grey", lw=0.5, alpha=0.6)
        for i, q in enumerate(charges):
            gq = np.asarray(pc[f"q{q}"][key])
            ax.plot(r[mask], (gq - refarr)[mask], color=colors[i % len(colors)],
                    lw=1.7, label=fr"$q\!=\!{q}e$")
        ax.set_xlabel("r (A)", fontsize=14)
        ax.set_xlim(0, R_VIEW)
        ax.grid(True, ls="--", alpha=0.4)
        ax.set_title(f"$\\Delta g(r)$ for {title}", fontsize=14)
        if ax is axes[0]:
            ax.set_ylabel("$\\Delta g(r)$", fontsize=14)
        if ax is axes[1]:
            ax.legend(loc="upper right", fontsize=10, frameon=True)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "delta_gr.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(fig_dir, "delta_gr.png"), dpi=200, bbox_inches="tight")
    print(f"wrote {fig_dir}/delta_gr.{{pdf,png}}")
    plt.close(fig)

    # --- Fig: Delta-PDF (reduced form, Z^2-contrast) ---
    R_VIEW8 = 8.0
    mask8 = r <= R_VIEW8
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.axhline(0, color="grey", lw=0.5, alpha=0.6)
    for i, q in enumerate(charges):
        gOO = np.asarray(pc[f"q{q}"]["g_OO"])
        gOH = np.asarray(pc[f"q{q}"]["g_OH"])
        gHH = np.asarray(pc[f"q{q}"]["g_HH"])
        Gq = 4 * np.pi * r * rho * (g_total(gOO, gOH, gHH) - 1)
        ax.plot(r[mask8], (Gq - G0)[mask8], color=colors[i % len(colors)],
                lw=1.7, label=fr"$q\!=\!{q}e$")
    ax.set_xlabel("r (A)", fontsize=16)
    ax.set_ylabel("$\\Delta$PDF(r)", fontsize=16)
    ax.set_xlim(0, R_VIEW8)
    ax.grid(True, ls="--", alpha=0.4)
    ax.legend(loc="upper left", fontsize=12, ncol=2)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "delta_pdf.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(fig_dir, "delta_pdf.png"), dpi=200, bbox_inches="tight")
    print(f"wrote {fig_dir}/delta_pdf.{{pdf,png}}")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--md_root", required=True)
    p.add_argument("--out_json", default="dpdf_drdf.json")
    p.add_argument("--r_max", type=float, default=12.0)
    p.add_argument("--dr", type=float, default=0.05)
    p.add_argument("--sigma_A", type=float, default=0.225)
    p.add_argument("--n_frames", type=int, default=120)
    p.add_argument("--skip_initial", type=int, default=20)
    p.add_argument("--make_plots", action="store_true")
    p.add_argument("--fig_dir", default="figures")
    args = p.parse_args()

    if os.path.isfile(args.out_json):
        out = json.load(open(args.out_json))
        print(f"Reusing existing {args.out_json}")
    else:
        out = compute(args)

    if args.make_plots:
        make_plots(out, args.fig_dir)


if __name__ == "__main__":
    main()
