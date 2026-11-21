"""Immutable, exploratory fund NAV ranking evaluation.

This package deliberately does not confer executable/net-alpha approval. Historical
fees, dealing status and distributions belong to ``qmt.factor.fund_execution``.
"""

from .platform import (
    DEFAULT_SEALED_FROM,
    EvaluationProtocol,
    prepare_snapshot,
    register_protocol,
    run_evaluation,
    verify_snapshot,
)

__all__ = ["DEFAULT_SEALED_FROM", "EvaluationProtocol", "prepare_snapshot",
           "register_protocol", "run_evaluation", "verify_snapshot"]
