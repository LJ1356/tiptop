"""Reading invented predicates off an image, and checking the human did what was asked.

An invented predicate has no code behind it -- its definition is the sentence the proposer wrote. So
it is evaluated the only way it can be: show a VLM the workspace and the sentence, and ask. The same
machinery does step 1.4, since "did the box get opened?" is that question asked about the atoms of
the phase the human was handed.
"""

import asyncio
import logging
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np
from cutamp.task_planning import Atom
from PIL import Image

from tiptop.hitl.config import HITLConfig
from tiptop.hitl.llm import query_json
from tiptop.hitl.prompts import CLASSIFIER_SCHEMA, classifier_prompt
from tiptop.hitl.structs import HITLProposalError, Phase, VLMPredicate, describe_atom

_log = logging.getLogger(__name__)

# Phrasings for the robot's own state predicates, so a phase mentioning one can still be described to
# the human. An invented predicate brings its own description; these are the fallbacks.
DEFAULT_DESCRIPTIONS: dict[str, str] = {
    "On": "{0} is resting on top of {1}",
    "Holding": "the robot's gripper is holding {0}",
    "HandEmpty": "the robot's gripper is empty",
}

# Atoms a camera can settle. HandEmpty/Holding are deliberately absent -- see verify_phase.
_CHECKABLE_ROBOT_FLUENTS = frozenset({"On"})

# The largest image sent to the classifier. Full ZED frames are far bigger than the model needs for
# "is this open", and shrinking them is the difference between a snappy check and one the operator
# waits on with the arm parked.
_MAX_IMAGE_EDGE = 1024


def to_pil(rgb: np.ndarray | Image.Image) -> Image.Image:
    """A camera frame as a PIL image the VLM can take, downscaled if it is large."""
    image = rgb if isinstance(rgb, Image.Image) else Image.fromarray(np.asarray(rgb).astype(np.uint8))
    if max(image.size) > _MAX_IMAGE_EDGE:
        scale = _MAX_IMAGE_EDGE / max(image.size)
        image = image.resize((max(1, round(image.width * scale)), max(1, round(image.height * scale))))
    return image


def descriptions_for(invented: Sequence[VLMPredicate]) -> dict[str, str]:
    """Fluent name -> natural-language template, for everything that can be described."""
    return {**DEFAULT_DESCRIPTIONS, **{p.name: p.instructions for p in invented}}


@dataclass(frozen=True)
class Verdict:
    """One VLM judgement about one atom in one image."""

    atom: Atom
    statement: str
    holds: bool
    reason: str

    def summary(self) -> dict:
        return {"atom": str(self.atom), "statement": self.statement, "holds": self.holds, "reason": self.reason}


async def classify(image: Image.Image, atom: Atom, descriptions: Mapping[str, str], cfg: HITLConfig) -> Verdict:
    """Ask whether one atom holds in one image."""
    statement = describe_atom(atom, descriptions)

    def parse(data):
        if not isinstance(data, dict) or "holds" not in data:
            raise HITLProposalError("Respond with an object containing 'holds' (a boolean) and 'reason'.")
        return bool(data["holds"]), str(data.get("reason", ""))

    holds, reason = await query_json(
        classifier_prompt(statement), parse, model=cfg.vlm_model, schema=CLASSIFIER_SCHEMA,
        image=image, max_attempts=cfg.max_attempts, label=f"classify {atom}",
    )
    _log.info(f"HITL VLM: {atom} = {holds} ({reason})")
    return Verdict(atom, statement, holds, reason)


async def classify_all(
    image: Image.Image, atoms: Iterable[Atom], descriptions: Mapping[str, str], cfg: HITLConfig
) -> list[Verdict]:
    """Classify several atoms against one image, concurrently."""
    atoms = list(atoms)
    if not atoms:
        return []
    return list(await asyncio.gather(*(classify(image, atom, descriptions, cfg) for atom in atoms)))


async def classify_initial_state(
    image: Image.Image, spec, cfg: HITLConfig
) -> frozenset[Atom]:
    """Which invented atoms named in the plan already hold before anything is done.

    Only the invented atoms the PLAN mentions are checked, rather than every grounding of every
    invented predicate over every object tuple: the reference evaluates the full cross product and
    flags it as the thing that will not scale, and nothing else reads the others.
    """
    invented_names = {p.name for p in spec.invented}
    candidates = {a for phase in spec.phases for a in phase.atoms if a.name in invented_names}
    if not candidates:
        return frozenset()
    verdicts = await classify_all(image, sorted(candidates, key=str), descriptions_for(spec.invented), cfg)
    return frozenset(v.atom for v in verdicts if v.holds)


async def verify_phase(
    image: Image.Image, phase: Phase, invented: Sequence[VLMPredicate], cfg: HITLConfig
) -> tuple[bool, list[Verdict]]:
    """Step 1.4: did the human's phase actually leave the workspace as it should have?

    Only atoms a camera can settle are put to the VLM: the invented predicates, plus ``On``, which is
    plainly visible. ``HandEmpty``/``Holding`` are excluded -- the frame used here is a third-person
    view (after a hand-off the arm is wherever the operator left it, so a wrist view points nowhere
    useful), the gripper is often out of shot, and the classifier is told to answer false when it
    cannot see the statement to be true. That would fail a phase over something the robot knows
    exactly. They are still SHOWN to the human, just not used to judge them.
    """
    checkable = {p.name for p in invented} | _CHECKABLE_ROBOT_FLUENTS
    atoms = [a for a in sorted(phase.atoms, key=str) if a.name in checkable]
    verdicts = await classify_all(image, atoms, descriptions_for(invented), cfg)
    return all(v.holds for v in verdicts), verdicts


def missing_statements(verdicts: Sequence[Verdict]) -> list[str]:
    """The statements that should hold and do not, phrased for the operator."""
    return [v.statement for v in verdicts if not v.holds]


def describe_expectations(phase: Phase, descriptions: Mapping[str, str]) -> list[str]:
    """What should be true after the human acts, for the hand-off instructions."""
    return [describe_atom(a, descriptions) for a in sorted(phase.atoms, key=str)]
