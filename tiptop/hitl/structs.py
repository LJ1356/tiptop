"""Data structures for human-in-the-loop planning, built on cuTAMP's symbolic layer.

A task is an ORDERED LIST OF PHASES: each one either a sub-goal for TiPToP or an action for a human.
That is a change from the reference design, which plans over the robot's operators together with
invented "magic" operators and lets ordering fall out of their preconditions. It had to change
because that ordering only ever flows one way. A magic operator can require robot work BEFORE it
(fold the cloth once the toy is on it), but nothing can make robot work wait on a human: cuTAMP's
Pick and Place have fixed preconditions that never mention an invented predicate. "Open the box, then
put the toy in it" was therefore inexpressible, and the planner cheerfully put the toy in the closed
box instead.

Two limits go with it. A goal that is a set of atoms describes a FINAL state, so an intermediate one
-- the toy resting on the table while the box is opened -- cannot be said at all, and `On(toy, table)`
would contradict `On(toy, box)`. And `HasNotPickedUp` lets one plan pick each object at most once, so
"toy off the box ... toy into the box" could not appear in a single plan. Phases dissolve both: each
robot phase is planned by cuTAMP from its own perception pass, so it starts from a clean state.
"""

import re
from dataclasses import dataclass
from typing import Mapping, Sequence

from cutamp.tamp_domain import Movable, Surface
from cutamp.task_planning import Atom, Fluent, Parameter

# Invented fluent names carry a per-session suffix internally; `display_name` strips it. See
# `session_fluent_name` for why.
_SESSION_SUFFIX = re.compile(r"#\d+$")

_PLACEHOLDER = re.compile(r"\{(\d+)\}")


class HITLProposalError(Exception):
    """A VLM proposal could not be parsed or failed validation.

    Carries the message that is fed back to the model on the next attempt, so it has to read as an
    instruction to the proposer rather than as a stack trace.
    """


def session_fluent_name(name: str, session_tag: int) -> str:
    """Internal name for an invented fluent, unique to one HITL session.

    cuTAMP interns ground atoms in a process-global ``_ATOM_CACHE`` keyed on (fluent name, values),
    and ``Atom.__eq__``/``__hash__`` compare only that string -- the parameter TYPES are not part of
    atom identity, but they are what the search validates goal literals against. Since an object's
    cuTAMP type is decided per task (a cloth is a Surface in a task that places something on it and a
    Movable in one that does not), the same invented atom proposed in two tasks of one long-lived
    tiptop process would otherwise be served from the cache carrying the FIRST task's types. Suffixing
    the name per session gives each task its own cache keys. The suffix must never reach the model --
    see `display_atom`.
    """
    return f"{name}#{session_tag}"


def display_name(name: str) -> str:
    """The name to show a human: an invented fluent's name without its session suffix."""
    return _SESSION_SUFFIX.sub("", name)


def display_atom(atom: Atom) -> str:
    """An atom as the PROPOSER must see it: ``IsFolded(cloth)``, never ``IsFolded#2(cloth)``.

    ``str(atom)`` renders the internal, session-suffixed fluent name. Showing that to the model is
    actively harmful, not merely ugly: told the goal is ``is_unfolded#2(cloth)`` while the predicate
    menu lists ``is_unfolded``, it takes the suffixed spelling for a predicate it has not been given
    and DEFINES one under that name -- so its step establishes ``is_unfolded#2`` while the goal still
    wants ``is_unfolded``. Observed doing exactly that, twice.
    """
    return f"{display_name(atom.name)}({', '.join(atom.values)})"


def describe_atom(atom: Atom, descriptions: Mapping[str, str] | None = None) -> str:
    """One atom in natural language, from a ``{0}``-style template when there is one."""
    template = (descriptions or {}).get(atom.name)
    if template is None:
        return display_atom(atom)
    return template.format(*atom.values)


def _validate_template(template: str, arity: int, context: str) -> None:
    """Reject a template that refers to arguments the predicate does not have."""
    used = {int(i) for i in _PLACEHOLDER.findall(template)}
    allowed = set(range(arity))
    if not used.issubset(allowed):
        raise HITLProposalError(
            f"{context} uses placeholders {sorted(used)}, but only {sorted(allowed)} are available."
        )


@dataclass(frozen=True)
class SceneTypes:
    """Which cuTAMP type each perceived object carries for the whole of one task.

    ``create_tamp_environment`` infers this from the goal it is handed -- anything appearing as the
    second argument of an ``on(...)`` becomes a Surface, everything else a Movable. Under HITL the
    goal is different for every phase, so inferring per phase would make the box a Surface (and
    therefore a static obstacle) in the phase that puts the toy into it and a Movable in the phase
    that does not, changing the world geometry mid-task. The split is computed ONCE, from every
    phase's atoms together, and held here.
    """

    surfaces: frozenset[str]
    movables: frozenset[str]

    def type_of(self, name: str) -> str:
        """The cuTAMP type of ``name``, or raise if it was not perceived in this scene."""
        if name in self.surfaces:
            return Surface
        if name in self.movables:
            return Movable
        raise HITLProposalError(
            f"'{name}' is not an object in this scene. Available objects: "
            f"{', '.join(sorted(self.surfaces | self.movables))}."
        )

    @property
    def all_names(self) -> frozenset[str]:
        return self.surfaces | self.movables

    def rebind(self, mapping: Mapping[str, str]) -> "SceneTypes":
        """The same split under this pass's object labels."""
        return SceneTypes(
            surfaces=frozenset(mapping.get(n, n) for n in self.surfaces),
            movables=frozenset(mapping.get(n, n) for n in self.movables),
        )


@dataclass(frozen=True)
class VLMPredicate:
    """A predicate whose truth value a VLM reads off an image.

    ``instructions`` is what must be visible for the atom to hold, with ``{0}``, ``{1}``, ... standing
    in for the arguments -- e.g. "the cloth {0} has been folded over on itself". That text is the
    predicate's entire definition: it is the classifier at verification time AND the description the
    human is shown when they are asked to bring it about.
    """

    fluent: Fluent
    instructions: str

    def __post_init__(self) -> None:
        _validate_template(self.instructions, len(self.fluent.parameters), f"Instructions for {self.name}")

    @property
    def name(self) -> str:
        return self.fluent.name

    def describe(self, values: Sequence[str]) -> str:
        """Fill the instructions in for specific objects."""
        return self.instructions.format(*values)


@dataclass(frozen=True)
class Phase:
    """One step of the task, carried out by one agent.

    A ``robot`` phase is a sub-goal handed to TiPToP: ``atoms`` are over cuTAMP's own goal predicates
    and go straight into ``create_tamp_environment``, which plans and executes them like any ordinary
    rollout. A ``human`` phase is one thing a person does; ``atoms`` are what should be true
    afterwards, and are what the VLM is asked about to check they did it.
    """

    executor: str  # "robot" | "human"
    description: str
    atoms: frozenset[Atom]
    instructions: str = ""

    def __post_init__(self) -> None:
        assert self.executor in ("robot", "human"), self.executor

    @property
    def is_human(self) -> bool:
        return self.executor == "human"

    def rebind(self, mapping: Mapping[str, str]) -> "Phase":
        """The same phase with its objects renamed to this pass's labels."""
        atoms = frozenset(
            a.fluent.ground(*[mapping.get(v, v) for v in a.values]) for a in self.atoms
        )
        return Phase(self.executor, self.description, atoms, self.instructions)

    def summary(self) -> dict:
        record = {
            "executor": self.executor,
            "description": self.description,
            "atoms": sorted(display_atom(a) for a in self.atoms),
        }
        if self.is_human:
            record["instructions"] = self.instructions
        return record


@dataclass(frozen=True)
class TaskSpecification:
    """What the proposal stage made of the instruction: an ordered plan, and the predicates behind it.

    A specification whose phases are all ``robot`` needs no human at all, which is how a task TiPToP
    already handles degrades to exactly its previous behaviour.
    """

    instruction: str
    phases: tuple[Phase, ...]
    scene_types: SceneTypes
    invented: tuple[VLMPredicate, ...] = ()
    # Clauses of the instruction that made it into no phase, with the reason. Non-empty means the run
    # is deliberately doing LESS than it was asked to.
    unrepresented: tuple[dict[str, str], ...] = ()
    # The proposer's own clause-by-clause account of which phase carries out which part of the
    # instruction. Recorded so a plan that quietly covers less than it was asked can be read off the
    # audit trail rather than inferred from the phases.
    coverage: tuple[dict, ...] = ()

    @property
    def needs_human(self) -> bool:
        return any(p.is_human for p in self.phases)

    @property
    def descriptions(self) -> dict[str, str]:
        """Fluent name -> natural-language template, for everything that has one."""
        return {p.name: p.instructions for p in self.invented}

    def rebind(self, mapping: Mapping[str, str]) -> "TaskSpecification":
        """The same plan under this pass's object labels (see planning.match_drifted_names)."""
        if not mapping:
            return self
        return TaskSpecification(
            instruction=self.instruction,
            phases=tuple(p.rebind(mapping) for p in self.phases),
            scene_types=self.scene_types.rebind(mapping),
            invented=self.invented,
            unrepresented=self.unrepresented,
            coverage=self.coverage,
        )

    def to_json(self) -> dict:
        """The audit record written beside each rollout (see hitl.json)."""
        return {
            "instruction": self.instruction,
            "invented_predicates": [
                {
                    "name": display_name(p.name),
                    "types": [q.type for q in p.fluent.parameters],
                    "instructions": p.instructions,
                }
                for p in sorted(self.invented, key=lambda p: p.name)
            ],
            "surfaces": sorted(self.scene_types.surfaces),
            "movables": sorted(self.scene_types.movables),
            # Empty on a run that planned the whole instruction. Anything here is a clause the run
            # knowingly left out -- read it before trusting a "success" label.
            "unrepresented": [dict(u) for u in self.unrepresented],
            "coverage": [dict(c) for c in self.coverage],
        }


def make_parameter(name: str, type_name: str) -> Parameter:
    """A cuTAMP operator parameter, with the ``?`` spelling the prompts use stripped."""
    return Parameter(name.lstrip("?"), type_name)
