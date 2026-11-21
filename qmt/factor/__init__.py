"""QMT symbolic-learning factor production platform.

Ten-layer pipeline (doc/techdesign/ ch.2), targeting off-exchange public-fund
cross-sectional ranking (20–60 trading-day relative return):

    raw_data -> safe_data -> base_factor_library -> factor_dsl -> auto_search
    -> factor_archive -> factor_screening -> combination -> execution_sim -> monitoring

The shared data-structure contract and the nine module Protocols live in
``qmt.factor.types``. Old black-box models (see ``qmt.pipeline``) are retained only
as baselines-to-beat (instruction §7.4), never as core methods here.
"""

from qmt.factor import types  # noqa: F401

__all__ = ["types"]
