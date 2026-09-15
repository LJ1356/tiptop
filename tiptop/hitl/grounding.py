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

# Atoms a camera can settle. HandEmpty/Holding are deliberately absent -- see verify_effects.
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
    """One VLM judgement about one atom in one image, against what was expected of it.

    ``holds`` is what the classifier saw; ``expected`` is what the plan said should be there. They
    come apart for a DELETE effect, where the plan expects the atom to be false afterwards and a
    "holds: true" answer is the failure. ``satisfied`` is the one to branch on -- reading ``holds``
    as the verdict silently inverts every delete-effect check.
    """

    atom: Atom
    statement: str
    holds: bool
    reason: str
    expected: bool = True
    # What this atom was being checked AS, for the audit record: a precondition, an add effect or a
    # delete effect. Free text rather than an enum because nothing branches on it.
    role: str = "effect"

    @property
    def satisfied(self) -> bool:
        return self.holds == self.expected

    def summary(self) -> dict:
        return {
            "atom": str(self.atom),
            "statement": self.statement,
            "holds": self.holds,
            "expected": self.expected,
            "satisfied": self.satisfied,
            "role": self.role,
            "reason": self.reason,
        }


async def classify(
    image: Image.Image,
    atom: Atom,
    descriptions: Mapping[str, str],
    cfg: HITLConfig,
    *,
    expected: bool = True,
    role: str = "effect",
) -> Verdict:
    """Ask whether one atom holds in one image.

    The classifier is always asked the same, positive question -- "is this true?" -- whatever the
    plan expected. Asking it to confirm a negative ("the box is NOT open") reads as a double negative
    and got worse answers; ``expected`` is applied to the answer instead.
    """
    statement = describe_atom(atom, descriptions)

    def parse(data):
        if not isinstance(data, dict) or "holds" not in data:
            raise HITLProposalError("Respond with an object containing 'holds' (a boolean) and 'reason'.")
        return bool(data["holds"]), str(data.get("reason", ""))

    holds, reason = await query_json(
        classifier_prompt(statement), parse, model=cfg.vlm_model, schema=CLASSIFIER_SCHEMA,
        image=image, max_attempts=cfg.max_attempts, label=f"classify {atom}",
    )
    if holds != expected:
        _log.info(f"HITL VLM: {atom} = {holds}, expected {expected} [{role}] ({reason})")
    else:
        _log.info(f"HITL VLM: {atom} = {holds} [{role}] ({reason})")
    return Verdict(atom, statement, holds, reason, expected=expected, role=role)


async def classify_all(
    image: Image.Image,
    atoms: Iterable[Atom],
    descriptions: Mapping[str, str],
    cfg: HITLConfig,
    *,
    expected: bool = True,
    role: str = "effect",
) -> list[Verdict]:
    """Classify several atoms against one image, concurrently."""
    atoms = list(atoms)
    if not atoms:
        return []
    return list(await asyncio.gather(
        *(classify(image, atom, descriptions, cfg, expected=expected, role=role) for atom in atoms)
    ))


async def classify_initial_state(
    image: Image.Image, spec, cfg: HITLConfig
) -> frozenset[Atom]:
    """Which invented atoms named in the plan already hold before anything is done.

    Only the invented atoms the PLAN mentions are checked, rather than every grounding of every
    invented predicate over every object tuple: the reference evaluates the full cross product and
    flags it as the thing that will not scale, and nothing else reads the others.
    """
    invented_names = {p.name for p in spec.invented}
    # Every invented atom the plan mentions ANYWHERE, operators included. A precondition or a delete
    # effect may be the only place one is named ("the box starts open, and the human closes it"), and
    # those are precisely the ones whose starting value the plan cannot derive.
    mentioned = {
        a
        for phase in spec.phases
        for a in (set(phase.atoms) | set(phase.preconditions) | set(phase.add_effects) | set(phase.delete_effects))
    }
    candidates = {a for a in mentioned if a.name in invented_names}
    if not candidates:
        return frozenset()
    verdicts = await classify_all(image, sorted(candidates, key=str), descriptions_for(spec.invented), cfg)
    return frozenset(v.atom for v in verdicts if v.holds)


def checkable(atoms: Iterable[Atom], invented: Sequence[VLMPredicate]) -> list[Atom]:
    """The atoms a camera can settle, sorted.

    The invented predicates -- each of which is a sentence written to be looked for in an image --
    plus ``On``, which is plainly visible. ``HandEmpty``/``Holding`` are excluded for the reason
    ``verify_effects`` gives: the frame is third-person, the gripper is often out of shot, and the
    classifier answers false when it cannot see a statement to be true. The robot knows its own hand
    exactly, so putting it to a camera can only lose information.
    """
    names = {p.name for p in invented} | _CHECKABLE_ROBOT_FLUENTS
    return [a for a in sorted(atoms, key=str) if a.name in names]


async def verify_atoms(
    image: Image.Image,
    invented: Sequence[VLMPredicate],
    cfg: HITLConfig,
    *,
    expect_true: Iterable[Atom] = (),
    expect_false: Iterable[Atom] = (),
    role: str = "effect",
) -> tuple[bool, list[Verdict]]:
    """Check one image against atoms that should hold and atoms that should not.

    Returns ``(ok, verdicts)`` where ``ok`` is every checkable atom coming out the way the plan said
    it would. An atom in both sets would be a contradiction and is refused at proposal time, so the
    two are simply checked together against the one frame.
    """
    yes = checkable(expect_true, invented)
    no = checkable(expect_false, invented)
    descriptions = descriptions_for(invented)
    verdicts = await classify_all(image, yes, descriptions, cfg, expected=True, role=role)
    verdicts += await classify_all(image, no, descriptions, cfg, expected=False, role=f"{role} (deleted)")
    return all(v.satisfied for v in verdicts), verdicts


async def verify_preconditions(
    image: Image.Image, phase: Phase, invented: Sequence[VLMPredicate], cfg: HITLConfig
) -> tuple[bool, list[Verdict]]:
    """Is the workspace in a state this phase can be carried out from?

    The preconditions of the phase's operator, read off a frame taken BEFORE the hand-off. A phase
    with no operator has no preconditions and trivially passes -- there is nothing to check, which is
    exactly what every plan proposed before operators existed says.
    """
    return await verify_atoms(image, invented, cfg, expect_true=phase.preconditions, role="precondition")


async def verify_effects(
    image: Image.Image, phase: Phase, invented: Sequence[VLMPredicate], cfg: HITLConfig
) -> tuple[bool, list[Verdict]]:
    """Did this phase leave the workspace as its operator said it would?

    Add effects must now hold and delete effects must not. Only atoms a camera can settle are put to
    the VLM: the invented predicates, plus ``On``, which is plainly visible. ``HandEmpty``/``Holding``
    are excluded -- the frame used here is a third-person view (after a hand-off the arm is wherever
    the operator left it, so a wrist view points nowhere useful), the gripper is often out of shot,
    and the classifier is told to answer false when it cannot see the statement to be true. That
    would fail a phase over something the robot knows exactly. They are still SHOWN to the human,
    just not used to judge them.
    """
    return await verify_atoms(
        image, invented, cfg,
        expect_true=phase.add_effects, expect_false=phase.delete_effects, role="effect",
    )


def missing_statements(verdicts: Sequence[Verdict]) -> list[str]:
    """What is wrong with the workspace, phrased for the operator.

    Reads ``satisfied``, not ``holds``: a delete effect that is still true is just as much a reason
    the phase did not happen as an add effect that never became true, and it has to be said the
    other way round or the operator is told to do what they already did.
    """
    out = []
    for v in verdicts:
        if v.satisfied:
            continue
        out.append(v.statement if v.expected else f"{v.statement} -- and it should no longer be")
    return out


def describe_expectations(phase: Phase, descriptions: Mapping[str, str]) -> list[str]:
    """What the workspace should look like after the human acts, for the hand-off instructions.

    The operator's delete effects are included, phrased the other way round. A step whose whole point
    is that something stops being the case -- the lid is no longer on the jar -- reads as a missing
    instruction if only the add effects are shown.
    """
    out = [describe_atom(a, descriptions) for a in sorted(phase.add_effects, key=str)]
    out += [
        f"NO LONGER: {describe_atom(a, descriptions)}"
        for a in sorted(phase.delete_effects, key=str)
    ]
    return out
