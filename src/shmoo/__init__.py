"""DRAM Shmoo Plot blob generator.

The PASS region of a shmoo over (timing, vref) is the blob. See
docs/shmoo_model.md for the canonical design.
"""

from .shmoo_eval import generate_dataset, generate_shmoo

__all__ = ["generate_dataset", "generate_shmoo"]
