"""Learned components for tiptop.

Currently: the conditional flow-matching stroke model (``flow_timing``) that ``flow_blending`` samples a
human-like stroke from, plus the measured DROID pace / event-speed conditionals (``droid_timing_stats``)
that set its absolute clock. Training + DROID extraction + eval scripts live alongside them; checkpoints
default to ``tiptop/tiptop/checkpoints``.

Submodules are imported directly (e.g. ``from tiptop.networks.flow_timing import FlowModel``) so that
importing this package pulls in no torch-heavy module by itself.
"""
