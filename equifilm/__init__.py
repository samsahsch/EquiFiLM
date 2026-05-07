"""EquiFiLM: per-layer Feature-wise Linear Modulation adapter for MACE.

After running ``python -m equifilm.apply_patch``, the four files in
``equifilm/_patches/`` are copied over the corresponding files in your local
``mace-torch==0.3.15`` installation. The standard MACE training entrypoint
(``mace_run_train`` / ``python -m mace.cli.run_train``) then accepts the new
flag ``--charge_film N`` (and three ablation flags), and the trained model
will read ``total_charge`` from atoms.info during inference.

There is no Python API exposed by this package itself; everything works
through the patched MACE entrypoints. The package exists only to deliver and
apply the patch files.
"""

__version__ = "0.1.0"
