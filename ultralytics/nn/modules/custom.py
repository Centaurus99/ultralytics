# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Pika custom layers — thin re-export of framework-agnostic modules from the ``pika`` package.

Algorithm bodies live in ``pika.modules`` (permissive license, zero framework
imports, installed editable); this file is the ONLY place ultralytics learns their
names. ``PIKA_CHANNELWISE`` lists channel-preserving layers that need the input
channel count (c1) injected by ``parse_model``. See the pika_holes CLAUDE.md.
"""

from pika.modules import SHSA

__all__ = ("SHSA", "PIKA_CHANNELWISE")

# Channel-preserving custom layers: parse_model prepends c1 and records c2 = ch[f].
# YAML usage, e.g.: [-1, 1, SHSA, [0.25]]
PIKA_CHANNELWISE = frozenset({SHSA})
