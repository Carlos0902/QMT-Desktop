"""QMT control plane (subsystem E89) — the FastAPI backend that lets a developer
CONTROL, DEBUG and REFINE the eleven-stage pipeline.

This is a Part-B addition beyond the design doc's eleven conceptual modules (see
DECISIONS.md Part B / D-API-*). It imports the pipeline as a library and never adds a
non-point-in-time read path: every endpoint INGESTS (writes, ``available_at``-tagged)
then RUNS (reads ``get_asof`` only), keeps analysis/advice mode-isolated, and fails
closed. Importing this subpackage pulls in FastAPI, so ``import qmt`` deliberately does
**not** re-export it — import :func:`qmt.controlplane.create_app` explicitly.
"""

from __future__ import annotations

from qmt.controlplane.api import app, create_app

__all__ = ["create_app", "app"]
