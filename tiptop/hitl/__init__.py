"""Human-in-the-loop TAMP: goals TiPToP cannot express, with a human filling in the gap.

TiPToP plans over two goal predicates, ``on(movable, surface)`` and ``holding(movable)``. A task like
"place the toy on the cloth and fold it" therefore loses its second half at translation time: the
fold is not expressible, so it silently disappears from the goal and the robot reports success once
the toy is on the cloth.

This package keeps the fold in the goal. Given the workspace image and the instruction, a VLM

  1. interprets the goal into cuTAMP atoms, INVENTING a predicate where none of TiPToP's fit
     (``IsFolded(cloth)``, grounded by a natural-language description a VLM checks in an image);
  2. breaks the instruction into an ORDERED list of phases, each one either a sub-goal for TiPToP or
     an action only a human can do;
  3. hands each human phase over via the teleop hand-off this repo already has;
  4. checks, from a fresh image, that the human's phase actually had the intended effect.

Each robot phase becomes one ordinary TiPToP rollout aimed at that phase's sub-goal, and each human
phase becomes one teleop leg. Legs of one task share a trajectory_id, so collect/merge_trajectory.py
joins them into a single episode exactly as it does for a hand-off today.

Everything here is inert unless a config turns it on (see ``config.resolve_hitl_config``); with it
off, ``tiptop_run`` does not import this package at all.

Ported from the design in github.com/tomsilver/hitl-tamp, whose world, perception, TAMP system, VLM
and teleoperator are all simulated. Here each of those is the real one, so only the parts that repo
marks as "meant to transfer" are ported: proposal, VLM grounding, and the outer plan/segment loop.
"""

from tiptop.hitl.config import HITLConfig, load_hitl_config, resolve_hitl_config
from tiptop.hitl.structs import (
    HITLProposalError,
    Phase,
    SceneTypes,
    TaskSpecification,
    VLMPredicate,
)

__all__ = [
    "HITLConfig",
    "HITLProposalError",
    "Phase",
    "SceneTypes",
    "TaskSpecification",
    "VLMPredicate",
    "load_hitl_config",
    "resolve_hitl_config",
]
