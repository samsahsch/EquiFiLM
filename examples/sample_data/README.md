# Sample data

`sample_combined.xyz` contains 80 frames (~20 per training charge $q\in\{0,
6e, 10e, 16e\}$) sub-sampled every 50th step from the full AIMD trajectories.
The full training set is ~6,400 frames; this excerpt is for end-to-end smoke
testing only — it is not large enough to reproduce the paper's accuracy.

For the full dataset, see the Zenodo DOI listed in `docs/reproduction.md`.

| Field | Notes |
|---|---|
| `total_charge` | integer charge of the configuration (added electrons) |
| `total_spin`   | always 0 in this dataset (closed-shell DFT) |
| Energies/forces | computed at $r^2$SCAN-omat level with VASP |
| Cell           | 14.8 Å cubic for q=16; 13.8 Å cubic for the rest |

The 13.8 vs 14.8 Å mix reflects a real bias in the AIMD source data; it is
documented as a limitation in §6 of the paper.
