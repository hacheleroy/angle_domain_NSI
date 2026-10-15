"""Compatibility imports for the acquisition-specific reconstruction scripts.

The shared definitions live in :mod:`angle_nsi._core`. New applications
should use the public :mod:`angle_nsi` package.
"""

if __package__ == "src":
    from .angle_nsi._core import (
        AngularWeightDiagnostics, angular_sign_weights, nsi_envelope,
        coherence_factor_from_moments, coherence_factor,
        coherence_weighted_das, nsi_c_sweep,
    )
else:
    from angle_nsi._core import (
        AngularWeightDiagnostics, angular_sign_weights, nsi_envelope,
        coherence_factor_from_moments, coherence_factor,
        coherence_weighted_das, nsi_c_sweep,
    )
