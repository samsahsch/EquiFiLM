# ChargeFiLMBlock — design walkthrough

This document explains the four design choices in `ChargeFiLMBlock` and how
they collectively preserve E(3) equivariance while learning a continuous
external conditioning signal.

## What it modulates

Every interaction layer in MACE produces a hidden tensor whose channels are
grouped by their irreducible-representation rank ($\ell$). Scalar channels
($\ell=0$) are invariant under rotations; higher-rank channels ($\ell\geq 1$)
transform non-trivially. `ChargeFiLMBlock` modulates **only the scalar
channels**:

```
x_out[:,    :N_scalar] = gamma(q) * x_in[:, :N_scalar] + beta(q)
x_out[:, N_scalar:    ] = x_in[:, N_scalar:]
```

Modulating an $\ell\geq 1$ channel by a scalar gain $\gamma(q)$ would still
preserve equivariance (since rotation and scaling commute), but **adding** a
charge-dependent offset $\beta(q)$ to a vector or higher-rank channel would
break equivariance. Restricting both operations to the scalar block is the
simplest E(3)-preserving choice.

## How $\gamma(q)$ and $\beta(q)$ are produced

A small two-layer MLP maps the per-graph charge to per-channel multiplier and
shift:

```
q_norm = total_charge / N_atoms                # per-graph scalar
gamma  = 1 + MLP_gamma(q_norm)                  # zero-init MLP -> gamma=1 at start
beta   = 0 + MLP_beta(q_norm)                   # zero-init MLP -> beta=0 at start
```

`q_norm` (charge per atom) is an intensive quantity, invariant to the
cell size. The training cells contain 324 atoms. When running MD in a larger
supercell (e.g., 2×2×2 = 2592 atoms), pass `total_charge = q_label × 8`
so that `q_norm = total_charge / 2592 = q_label / 324` matches the
training distribution.

## Identity at initialization

Both MLPs are zero-initialized, so at training step 0 the block is the
identity transformation. The pre-trained foundation model is therefore
recovered exactly, and training can only improve from this baseline. This
matters for fine-tuning stability: any other init leaks an arbitrary
initial perturbation into the foundation's weights.

The ablation `--charge_film_no_zero_init` removes this constraint and
replaces it with a small random init; we report a small but real degradation
in convergence stability in the paper's Table 2.

## Why per-layer

A single charge gating at the model output cannot capture how charge alters
the radial / angular features at intermediate depths (e.g., shielding length
scales). Attaching one `ChargeFiLMBlock` per interaction layer ($K=2$ for the
MACE-MATPES backbone used in the paper) gives the network capacity to learn
depth-dependent charge corrections. The two interaction layers have different
scalar-channel counts (Ns=128 for layer 0, Ns=256 for layer 1), so the
adapter parameters per block differ: 33,536 for layer 0 and 66,560 for
layer 1, totalling ~100k adapter parameters (~0.10 M) against the ~0.66 M
backbone — approximately a 15% overhead.

## Where it lives in the message-passing pipeline

In MACE, each interaction block calls `forward` on a node-feature tensor and
returns the updated node features plus optional `sc` (skip connection) and
`gating` outputs. The patched `RealAgnosticInteractionBlock.forward` and
`RealAgnosticResidualInteractionBlock.forward` accept an optional
`charge_conditioning` keyword argument and apply the FiLM modulation right
after the standard message aggregation, before the readout.

This placement was chosen so that the modulated message is what gets
combined with the residual / skip connection, rather than the raw
unconditioned message. We did not run an ablation comparing positions
within the block; the paper's results all use this single placement.
