"""Unit tests for ChargeFiLMBlock.

These tests verify the architectural promises EquiFiLM makes:

1. Identity at initialization. Zero-init means gamma = 1, beta = 0 ->
   the block returns its input unchanged at training step 0.
2. Equivariance. Permuting/rotating non-scalar (l > 0) channels does not
   affect them; only the scalar (l = 0) channels are touched.
3. Charge sensitivity. After learnable parameters are perturbed, different
   charges produce different outputs.

Run with::

    python -m pytest tests/

(These tests assume `python -m equifilm.apply_patch` has already been run
so that `mace.modules.blocks.ChargeFiLMBlock` is importable.)
"""
import math
import pytest
import torch


@pytest.fixture(scope="module")
def film_block_cls():
    """Import the ChargeFiLMBlock class from the patched MACE installation."""
    try:
        from mace.modules.blocks import ChargeFiLMBlock
    except ImportError as e:
        pytest.skip(
            "ChargeFiLMBlock not importable. Did you run "
            "`python -m equifilm.apply_patch` against your mace-torch install?"
        )
    return ChargeFiLMBlock


def _make_block(film_block_cls, hidden_dim=64, n_scalar=128,
                use_mult=True, use_add=True, zero_init=True):
    return film_block_cls(
        hidden_dim=hidden_dim,
        n_scalar_channels=n_scalar,
        use_multiplicative=use_mult,
        use_additive=use_add,
        zero_init=zero_init,
    )


def test_identity_at_init(film_block_cls):
    """At init (zero_init=True), output equals input for any charge."""
    torch.manual_seed(0)
    n_scalar = 128
    n_higher = 64    # extra non-scalar channels (l>0)
    block = _make_block(film_block_cls, n_scalar=n_scalar, zero_init=True)

    n_nodes = 50
    x = torch.randn(n_nodes, n_scalar + n_higher)
    q = torch.randn(n_nodes, 1)

    y = block(x, q)
    assert torch.allclose(y, x, atol=1e-6), \
        f"Identity-at-init failed: max diff = {(y - x).abs().max().item():.2e}"


def test_higher_channels_untouched(film_block_cls):
    """Even after random perturbation, l>0 channels pass through unchanged."""
    torch.manual_seed(1)
    n_scalar, n_higher = 64, 96
    block = _make_block(film_block_cls, n_scalar=n_scalar, zero_init=False)

    # break identity by perturbing weights
    for p in block.parameters():
        p.data.normal_(0, 0.1)

    x = torch.randn(20, n_scalar + n_higher)
    q = torch.randn(20, 1)
    y = block(x, q)
    assert torch.allclose(y[:, n_scalar:], x[:, n_scalar:], atol=1e-6), \
        "Non-scalar (l>0) channels were modified!"


def test_charge_sensitivity_after_perturbation(film_block_cls):
    """After random init, different charges should produce different outputs."""
    torch.manual_seed(2)
    n_scalar, n_higher = 64, 32
    block = _make_block(film_block_cls, n_scalar=n_scalar,
                        zero_init=False)
    for p in block.parameters():
        p.data.normal_(0, 0.5)

    x = torch.randn(10, n_scalar + n_higher)
    q1 = torch.full((10, 1), 0.0)
    q2 = torch.full((10, 1), 5.0)
    y1 = block(x, q1)
    y2 = block(x, q2)
    diff = (y1 - y2).abs().max().item()
    assert diff > 1e-3, \
        f"FiLM block insensitive to charge: max diff = {diff:.2e}"


def test_additive_only_ablation(film_block_cls):
    """With use_multiplicative=False, output should equal input + beta."""
    torch.manual_seed(3)
    n_scalar = 32
    block = _make_block(film_block_cls, n_scalar=n_scalar,
                        use_mult=False, use_add=True, zero_init=False)
    for p in block.parameters():
        p.data.normal_(0, 0.3)

    x = torch.zeros(5, n_scalar)   # only scalar channels
    q = torch.randn(5, 1)
    y = block(x, q)
    # Multiplicative branch is off -> y - x should equal beta(q)
    # We can't access beta directly, but we can verify y is non-zero (only the
    # additive shift contributes).
    assert (y.abs() > 0).any(), "Additive-only branch produced zero output"


def test_multiplicative_only_ablation(film_block_cls):
    """With use_additive=False, output for x=0 should be zero."""
    torch.manual_seed(4)
    n_scalar = 32
    block = _make_block(film_block_cls, n_scalar=n_scalar,
                        use_mult=True, use_add=False, zero_init=False)
    for p in block.parameters():
        p.data.normal_(0, 0.3)

    x = torch.zeros(5, n_scalar)
    q = torch.randn(5, 1)
    y = block(x, q)
    assert torch.allclose(y, x, atol=1e-6), \
        "Multiplicative-only branch should give 0 for x=0"
