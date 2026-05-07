"""Unit tests for ChargeFiLMBlock.

These tests verify the architectural promises EquiFiLM makes:

1. Identity at initialization. zero_init makes the FiLM block return its input
   unchanged at training step 0, so fine-tuning from a foundation MLFF starts
   from the same function and only learns charge dependence on top.
2. Higher-order (l>0) channels pass through untouched, preserving E(3)
   equivariance.
3. After parameters are perturbed, different charges produce different outputs
   (the block is actually charge-sensitive).
4. Ablation flags work: use_multiplicative=False / use_additive=False each
   disable the corresponding branch.

Run with::

    python -m equifilm.apply_patch       # one-time, patches local mace
    python -m pytest tests/

(Or run as a script with stdlib unittest::

    python tests/test_film_block.py
)
"""
import torch

try:
    import pytest
    HAVE_PYTEST = True
except ImportError:
    HAVE_PYTEST = False


def _import_block():
    """Import ChargeFiLMBlock from the patched MACE installation."""
    try:
        from mace.modules.blocks import ChargeFiLMBlock
        return ChargeFiLMBlock
    except ImportError:
        return None


if HAVE_PYTEST:
    @pytest.fixture(scope="module")
    def film_block_cls():
        cls = _import_block()
        if cls is None:
            pytest.skip(
                "ChargeFiLMBlock not importable. Did you run "
                "`python -m equifilm.apply_patch` against your mace-torch install?"
            )
        return cls


def _make_block(film_block_cls, num_scalar=128, mlp_hidden=64,
                use_mult=True, use_add=True, zero_init=True):
    return film_block_cls(
        num_scalar_channels=num_scalar,
        mlp_hidden=mlp_hidden,
        use_multiplicative=use_mult,
        use_additive=use_add,
        zero_init=zero_init,
    )


def test_identity_at_init(film_block_cls):
    """At init (zero_init=True), block output equals input for any charge."""
    torch.manual_seed(0)
    n_scalar, n_higher = 128, 64
    block = _make_block(film_block_cls, num_scalar=n_scalar, zero_init=True)

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
    block = _make_block(film_block_cls, num_scalar=n_scalar, zero_init=False)

    # break identity by perturbing the trainable weights
    for p in block.parameters():
        p.data.normal_(0, 0.1)

    x = torch.randn(20, n_scalar + n_higher)
    q = torch.randn(20, 1)
    y = block(x, q)
    assert torch.allclose(y[:, n_scalar:], x[:, n_scalar:], atol=1e-6), \
        "Non-scalar (l>0) channels were modified by FiLM!"


def test_charge_sensitivity_after_perturbation(film_block_cls):
    """After random init, different charges should produce different outputs."""
    torch.manual_seed(2)
    n_scalar, n_higher = 64, 32
    block = _make_block(film_block_cls, num_scalar=n_scalar, zero_init=False)
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
    """With use_multiplicative=False, only the additive branch contributes."""
    torch.manual_seed(3)
    n_scalar = 32
    block = _make_block(film_block_cls, num_scalar=n_scalar,
                        use_mult=False, use_add=True, zero_init=False)
    # Should not have a gamma_net at all
    assert not hasattr(block, "gamma_net") or block.use_multiplicative is False
    for p in block.parameters():
        p.data.normal_(0, 0.3)

    x = torch.zeros(5, n_scalar)
    q = torch.randn(5, 1)
    y = block(x, q)
    # x = 0; multiplicative branch off; additive branch active -> y = beta(q)
    assert (y.abs() > 0).any(), \
        "Additive-only branch produced zero output for non-zero q"


def test_multiplicative_only_ablation(film_block_cls):
    """With use_additive=False, output for x=0 should be zero (no shift)."""
    torch.manual_seed(4)
    n_scalar = 32
    block = _make_block(film_block_cls, num_scalar=n_scalar,
                        use_mult=True, use_add=False, zero_init=False)
    assert not hasattr(block, "beta_net") or block.use_additive is False
    for p in block.parameters():
        p.data.normal_(0, 0.3)

    x = torch.zeros(5, n_scalar)
    q = torch.randn(5, 1)
    y = block(x, q)
    assert torch.allclose(y, x, atol=1e-6), \
        "Multiplicative-only branch should give 0 output for x=0"


# ----------- Allow running as a plain script with stdlib unittest -----------
if __name__ == "__main__":
    import sys, unittest
    cls = _import_block()
    if cls is None:
        sys.exit("ChargeFiLMBlock not importable. Run `python -m equifilm.apply_patch` first.")

    class _T(unittest.TestCase):
        def test_identity(self):  test_identity_at_init(cls)
        def test_higher(self):    test_higher_channels_untouched(cls)
        def test_charge(self):    test_charge_sensitivity_after_perturbation(cls)
        def test_add_only(self):  test_additive_only_ablation(cls)
        def test_mult_only(self): test_multiplicative_only_ablation(cls)

    unittest.main(argv=[__file__], verbosity=2)
