"""Canonical display names for the manuscript-facing beamformers.

The revision uses these labels verbatim in figures, tables, JSON summaries,
and prose.  Keeping them in one small module prevents the receive-domain
implementation details (for example ``receive_cf_das``) from leaking into the
published method names.
"""

from __future__ import annotations


DAS = "DAS"
CF_DAS = "CF-DAS"
MV = "MV"
DMAS = "DMAS"
RECEIVE_NSI = "Receive-NSI"
ANGLE_NSI = "Angle-NSI"

METHODS = (DAS, CF_DAS, MV, DMAS, RECEIVE_NSI, ANGLE_NSI)
IQ_METHODS = (DAS, CF_DAS, MV, RECEIVE_NSI, ANGLE_NSI)

# Earlier revision outputs used implementation-qualified labels.  They remain
# readable so validated GPU caches and frozen result archives can be migrated
# without recomputation; all newly written outputs use the canonical names.
LEGACY_TO_CANONICAL = {
    "Receive CF-DAS": CF_DAS,
    "F-DMAS": DMAS,
    "Conventional NSI": RECEIVE_NSI,
    "Angular NSI": ANGLE_NSI,
}
CANONICAL_TO_LEGACY = {value: key for key, value in LEGACY_TO_CANONICAL.items()}


def canonical_method_name(name: str) -> str:
    """Return the publication label for a current or legacy method name."""

    return LEGACY_TO_CANONICAL.get(name, name)
