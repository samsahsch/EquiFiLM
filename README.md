# EquiFiLM: Charge-Conditioned Equivariant Force Fields via Feature-wise Linear Modulation

Reference implementation for the NeurIPS 2026 AI4Science paper *EquiFiLM*. This
repository contains the per-layer FiLM adapter that turns an equivariant
foundation MLFF (here, MACE) into a charge-conditioned model. The headline
deployment, **E-MACE**, is EquiFiLM applied to MACE-MATPES-r2SCAN-omat-ft and
fine-tuned on charged liquid water at integer ionizations
$q\in\{0, 6e, 10e, 16e\}$.

## What's here

| Path | What it is |
|---|---|
| `equifilm/_patches/` | Four drop-in replacement files for upstream `mace-torch==0.3.15` that introduce `ChargeFiLMBlock` and the `--charge_film` training flag. |
| `equifilm/apply_patch.py` | Helper that copies the four files into the right places of an installed `mace-torch`. Runs once per environment. |
| `examples/train.sh` | Single command that trains E-MACE end to end with the paper's hyperparameters. |
| `examples/train_config.yaml` | All hyperparameters used in the paper. |
| `examples/eval_per_charge.py` | Reproduces the per-charge val-split numbers in Table 2. |
| `examples/md_emace.py` | NVT$\to$NVE molecular-dynamics driver used for Figs. 6–8. |
| `examples/plot_dpdf.py` | Smoothed $\Delta g(r)$ and $\Delta$PDF analysis for Figs. 7–8. |
| `examples/sample_data/` | A small (~5 MB) excerpt of charged-water frames so reviewers can run train/eval end to end without downloading the full dataset. |
| `tests/` | Unit tests verifying identity-at-init and equivariance preservation of `ChargeFiLMBlock`. |
| `docs/architecture.md` | Walk-through of `ChargeFiLMBlock` design choices (multiplicative vs additive, identity-at-init, where in the message-passing pipeline it lives). |
| `docs/reproduction.md` | Step-by-step reproduction of every figure in the paper. |

## What's elsewhere

| Where | What |
|---|---|
| Zenodo (DOI in `docs/reproduction.md`) | Full training dataset (`gs.xyz`, `6e.xyz`, `10e.xyz`, `16e.xyz`; ~200 MB total) and trained `MACE-FiLM-large_stagetwo.model` weights. |

## Architecture in one paragraph

`ChargeFiLMBlock` produces per-charge gain $\gamma(q)$ and shift $\beta(q)$ from a
small two-layer MLP and applies them only to the scalar ($\ell{=}0$) channels of
each interaction layer's hidden message tensor:
$x_{\ell=0}\!\to\!\gamma(q)\!\cdot\!x_{\ell=0}+\beta(q)$.
Higher-rank channels ($\ell\!>\!0$) pass through untouched, so $E(3)$ equivariance
is preserved exactly. Both $\gamma$ and $\beta$ MLPs are zero-initialized so the
adapter is the identity at training start, recovering the foundation model
exactly. Per-layer attachment lets the model learn charge-aware corrections at
every depth of the message-passing stack.

## Install

```bash
# 1) Install upstream MACE
pip install mace-torch==0.3.15

# 2) Apply the EquiFiLM patches to your local MACE installation
python -m equifilm.apply_patch
```

`apply_patch` is idempotent and prints which files it overwrote. It does not
modify your environment in any other way.

## Reproduction

See `docs/reproduction.md` for the full walkthrough. Quick start:

```bash
# Get the data
# (download gs.xyz, 6e.xyz, 10e.xyz, 16e.xyz from the Zenodo record into ./data/)

# Train E-MACE (~24 GPU-hours on a single A100)
bash examples/train.sh ./data ./checkpoints

# Evaluate per-charge force/energy RMSE on held-out 10%
python examples/eval_per_charge.py \
    --model_path ./checkpoints/MACE-FiLM-large_stagetwo.model \
    --train_xyz  ./data/combined_all_charges.xyz

# Run a 3 ps NVT->NVE trajectory at q=12 in a 2x2x2 supercell
python examples/md_emace.py \
    --model_path ./checkpoints/MACE-FiLM-large_stagetwo.model \
    --seed_xyz   ./data/seed_2592atom.xyz \
    --total_charge 12 --supercell 2 2 2 \
    --n_nvt 1500 --n_nve 1500 --output_dir ./md_q12

# Build Figs. 7+8 (Delta-g(r), Delta-PDF) from a directory of NVE xyz files
python examples/plot_dpdf.py \
    --md_root ./md_runs/ \
    --out_json ./dpdf_drdf.json \
    --sigma_A 0.225 --r_max 12 --dr 0.05
```

## Citation

```bibtex
@inproceedings{equifilm2026,
  title  = {EquiFiLM: Charge-Conditioned Equivariant Force Fields via Feature-wise Linear Modulation},
  author = {Anonymous},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS) AI for Science track},
  year   = {2026},
  note   = {Anonymous submission}
}
```

## License

Apache License 2.0. See `LICENSE`.

The reproduced files in `equifilm/_patches/` derive from the MIT-licensed
[ACEsuit/mace](https://github.com/ACEsuit/mace) project; in those files the
upstream MIT notice is preserved at the top.
