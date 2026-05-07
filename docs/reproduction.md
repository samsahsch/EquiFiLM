# Reproducing the figures and tables in EquiFiLM

End-to-end recipe for reviewers and future researchers. Total compute budget
on a single A100-40GB:

* Training the headline E-MACE model: ~24 GPU-hours.
* All evaluation + MD figures: ~6 GPU-hours.

## 0. Install

```bash
pip install mace-torch==0.3.15
git clone <this anonymous repo URL> EquiFiLM
cd EquiFiLM
pip install -e .
python -m equifilm.apply_patch        # patches your local mace install
python -m pytest tests/               # sanity check (~30 s)
```

## 1. Get the data

The full training dataset, plus reference seed configurations and the trained
E-MACE checkpoint, are deposited on Zenodo. See the Zenodo DOI listed in the
paper appendix.

```bash
mkdir -p data
# Place these from the Zenodo record:
#   data/gs.xyz         (q=0  AIMD, ~1500 frames)
#   data/6e.xyz         (q=6e AIMD, ~2000 frames)
#   data/10e.xyz        (q=10e AIMD, ~2000 frames)
#   data/16e.xyz        (q=16e AIMD, ~970 frames)
#   examples/sample_data/seed_324atom.xyz  (one frame, 2x2x2 supercell, used as MD seed)
#   data/MACE-FiLM-large_stagetwo.model  (trained checkpoint, 12 MB)

# Optional: build the combined training xyz that mace_run_train expects
python -c "
from ase.io import read, write
frames = []
for st, q in [('gs', 0), ('6e', 6), ('10e', 10), ('16e', 16)]:
    for a in read(f'data/{st}.xyz', index=':'):
        a.info['total_charge'] = q
        a.info['total_spin']   = 0
        frames.append(a)
write('data/combined_all_charges.xyz', frames)
print(f'Wrote {len(frames)} frames')
"
```

## 2. Train E-MACE (or download the pretrained checkpoint)

```bash
bash examples/train.sh ./data ./checkpoints
```

This writes ``MACE-FiLM-large_stagetwo.model`` (the SWA-final checkpoint) and
``MACE-FiLM-large_valid_indices_123.txt`` (the held-out 10% indices).

## 3. Reproduce Table 2 (per-charge held-out RMSE)

```bash
python examples/eval_per_charge.py \
    --model_path           ./checkpoints/MACE-FiLM-large_stagetwo.model \
    --train_xyz            ./data/combined_all_charges.xyz \
    --valid_indices_file   ./checkpoints/MACE-FiLM-large_valid_indices_123.txt \
    --out_json             eval_per_charge.json
```

Expected:

```
q= 0:  F = 5.46 meV/A   E = 0.10 meV/atom
q= 6:  F = 7.08          E = 0.10
q=10:  F = 7.47          E = 0.12
q=16:  F = 7.46          E = 0.06
mean:  F = 6.87
```

## 4. Reproduce Fig 6 (energy conservation)

For each of q in {12, 20}:

```bash
# q=12
python examples/md_emace.py \
    --model_path  checkpoints/MACE-FiLM-large_stagetwo.model \
    --seed_xyz    examples/sample_data/seed_324atom.xyz \
    --total_charge 12 --supercell 2 2 2 \
    --n_nvt 1500 --n_nve 1500 --temperature 300 \
    --xyz_interval 25 \
    --output_dir md_q12 --output_prefix q12

# q=20
python examples/md_emace.py \
    --model_path  checkpoints/MACE-FiLM-large_stagetwo.model \
    --seed_xyz    examples/sample_data/seed_324atom.xyz \
    --total_charge 20 --supercell 2 2 2 \
    --n_nvt 1500 --n_nve 1500 --temperature 300 \
    --xyz_interval 25 \
    --output_dir md_q20 --output_prefix q20
```

The energy CSV files (`md_qN/qN_energies.csv`) are read by your plotting
framework of choice; the paper's NVE drift is computed from the second half
of each trajectory.

## 5. Reproduce Figs 7-8 (Delta-g(r) and Delta-PDF)

You need NVE trajectories at q=0 plus the charged states whose differences
you want to plot. q=0 requires `--timestep_fs 0.5` for stability (the model
is unstable at q=0 with 1.0 fs timestep at this supercell size).

```bash
# q=0 reference (longer trajectory, smaller timestep)
python examples/md_emace.py \
    --model_path  checkpoints/MACE-FiLM-large_stagetwo.model \
    --seed_xyz    examples/sample_data/seed_324atom.xyz \
    --total_charge 0 --supercell 2 2 2 \
    --timestep_fs 0.5 --n_nvt 2000 --n_nve 10000 --temperature 300 \
    --xyz_interval 50 \
    --output_dir md_q0_extended --output_prefix q0_extended

# Charged species (1.0 fs is fine)
for q in 4 8 12 16 18 20 ; do
  python examples/md_emace.py \
      --model_path  checkpoints/MACE-FiLM-large_stagetwo.model \
      --seed_xyz    examples/sample_data/seed_324atom.xyz \
      --total_charge $q --supercell 2 2 2 \
      --n_nvt 1000 --n_nve 5000 --temperature 300 \
      --xyz_interval 25 \
      --output_dir md_q${q}_extended --output_prefix q${q}_extended
done
```

Then aggregate:

```bash
python examples/plot_dpdf.py \
    --md_root       . \
    --out_json      dpdf_drdf.json \
    --sigma_A 0.225 --r_max 12 --dr 0.05 \
    --n_frames 120 --skip_initial 20 \
    --make_plots --fig_dir paper_figures
```

This produces `paper_figures/delta_gr.{pdf,png}` and
`paper_figures/delta_pdf.{pdf,png}`.

## 6. Tiny-data smoke test (no Zenodo download)

If you just want to verify the code runs end to end without downloading the
full dataset, the `examples/sample_data/` directory has a ~5 MB excerpt that
is large enough to fine-tune the foundation for a few epochs:

```bash
python -m mace.cli.run_train \
    --name=smoke_test \
    --train_file=examples/sample_data/sample_combined.xyz \
    --total_charge_key=total_charge \
    --energy_key=energy --forces_key=forces \
    --E0s=average \
    --valid_fraction=0.1 \
    --max_num_epochs=3 --batch_size=2 --valid_batch_size=2 \
    --charge_film=64 \
    --r_max=4.5 --num_interactions=2 --num_channels=16 \
    --MLP_irreps='8x0e' --hidden_irreps='16x0e + 16x1o' \
    --lr=1e-3 --device=cuda --default_dtype=float32 --save_cpu \
    --model_dir=./smoke_out
```

After 3 epochs this should print non-trivial loss values and write a
`smoke_test_stagetwo.model` to `./smoke_out/`. Wall time on a single A100
is ~3 minutes.

The full training (`examples/train.sh`) uses the same energy/forces keys
because the dataset's xyz fields are named `energy` and `forces` (not
the MACE 0.3.15 default `REF_energy` / `REF_forces`).
