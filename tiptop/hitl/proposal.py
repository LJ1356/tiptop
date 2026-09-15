"""The proposal stage: one instruction becomes an ordered list of robot and human phases.

Everything produced here is validated against what the rest of the system can actually consume before
it is returned, and a rejection is phrased for the model so the reprompt loop can fix it. The
alternative is a failure several seconds later with the arm warm and an operator watching -- or,
worse, a plan that validates and then does the wrong thing, which is how "put the toy in the box"
came to be planned before "open the box".
"""

import logging
from pathlib import Path
from typing import Any, Sequence

from cutamp.tamp_domain import HandEmpty, Holding, Movable, On, Surface, all_tamp_fluents
from cutamp.task_planning import Atom, Fluent
from PIL import Image

from tiptop.hitl.cache import ProposalCache
from tiptop.hitl.config import HITLConfig
from tiptop.hitl.llm import query_json
from tiptop.hitl.planning import check_plan_effects, wasted_robot_move
from tiptop.hitl.prompts import PLAN_SCHEMA, plan_prompt
from tiptop.hitl.structs import (
    DeferredObject,
    HITLProposalError,
    HumanOperator,
    Phase,
    SceneTypes,
    TaskSpecification,
    VLMPredicate,
    display_atom,
    display_name,
    make_parameter,
    session_fluent_name,
)

_log = logging.getLogger(__name__)

# The fluents a ROBOT phase may be stated over. Everything else in the cuTAMP domain is either motion
# bookkeeping (At, CanMove, JustMoved) or a type declaration (IsMovable, IsSurface) that
# get_initial_state supplies -- neither belongs in a proposal.
_ROBOT_FLUENTS: dict[str, Fluent] = {"On": On, "Holding": Holding, "HandEmpty": HandEmpty}

# Names a proposal may not reuse: every fluent cuTAMP already has.
_RESERVED_FLUENTS = frozenset(f.name for f in all_tamp_fluents)


def proposal_cache(cfg: HITLConfig) -> ProposalCache | None:
    """The proposal response cache, when the config asks for one."""
    return ProposalCache(Path(cfg.cache_path)) if cfg.cache_path else None


def _field(entry: Any, key: str, default: Any = None) -> Any:
    """Read a key from a decoded JSON object, with an error the model can act on."""
    if not isinstance(entry, dict):
        raise HITLProposalError(f"Expected a JSON object, got {type(entry).__name__}: {entry!r}")
    if key not in entry:
        if default is not None:
            return default
        raise HITLProposalError(f"The object {entry!r} is missing the required key '{key}'.")
    return entry[key]


def _atom_entries(data: Any, key: str) -> list[tuple[str, list[str]]]:
    """Parse a list of ``{"predicate": ..., "args": [...]}`` into (name, args) pairs."""
    entries = _field(data, key, [])
    if not isinstance(entries, list):
        raise HITLProposalError(f"'{key}' must be a list, got {type(entries).__name__}.")
    out = []
    for entry in entries:
        name = str(_field(entry, "predicate"))
        args = _field(entry, "args", [])
        if not isinstance(args, list) or not all(isinstance(a, str) for a in args):
            raise HITLProposalError(f"The args of {name} must be a list of strings, got {args!r}.")
        out.append((name, [str(a) for a in args]))
    return out


def _phase_entries(data: Any) -> list[tuple[str, str, list[tuple[str, list[str]]], str, Any]]:
    """Parse ``phases`` into (executor, description, atom entries, instructions, raw operator)."""
    entries = _field(data, "phases", [])
    if not isinstance(entries, list) or not entries:
        raise HITLProposalError("The plan must contain at least one phase.")
    out = []
    for entry in entries:
        executor = str(_field(entry, "executor")).strip().lower()
        if executor not in ("robot", "human"):
            raise HITLProposalError(f"A phase's executor must be 'robot' or 'human', got {executor!r}.")
        description = str(_field(entry, "description", "(no description)"))
        atoms = _atom_entries(entry, "atoms")
        if not atoms:
            raise HITLProposalError(
                f"The phase {description!r} has no atoms, so there is no way to tell when it is done."
            )
        instructions = str(entry.get("instructions") or "")
        if executor == "human" and not instructions.strip():
            raise HITLProposalError(
                f"The human phase {description!r} needs `instructions` telling the person what to do."
            )
        operator = entry.get("operator")
        if executor == "robot" and operator:
            # cuTAMP's operators are the robot's, and they are fixed. A proposal that declares one
            # is describing a step it has misjudged -- most often a human step written as a robot
            # phase, which is the failure the `operator` field exists to make visible.
            raise HITLProposalError(
                f"The ROBOT phase {description!r} declares an `operator`. Only a human phase does "
                "that: the robot's operators are fixed (pick and place). If this step needs an "
                "operator of its own, it is a human phase."
            )
        out.append((executor, description, atoms, instructions, operator))
    return out


_OPERATOR_ATOM_KEYS = ("preconditions", "add_effects", "delete_effects")


def _operator_predicate_uses(
    phases: Sequence[tuple[str, str, list[tuple[str, list[str]]], str, Any]]
) -> list[tuple[str, list[str]]]:
    """Every (predicate, args) pair named inside an operator, so one can type a predicate.

    An invented predicate may be named ONLY in an operator -- a precondition or a delete effect
    never has to appear in any phase's `atoms` -- and ``_build_invented`` types a predicate from its
    uses. Without counting these, such a predicate has no uses at all and is rejected as unused,
    which would make "the human closes what an earlier phase opened" unstatable.
    """
    uses: list[tuple[str, list[str]]] = []
    for _, _, _, _, raw in phases:
        if not isinstance(raw, dict):
            continue
        for key in _OPERATOR_ATOM_KEYS:
            uses.extend(_atom_entries(raw, key))
    return uses


def _scene_types(
    phases: Sequence[tuple[str, str, list[tuple[str, list[str]]], str, object]],
    objects: Sequence[str],
    table_name: str,
    deferred: Sequence[str] = (),
) -> SceneTypes:
    """Split the perceived objects into surfaces and movables, from EVERY phase's atoms.

    Mirrors what create_tamp_environment infers (``on(x, y)`` makes y a surface), but computed once
    across the whole plan rather than per phase, so an object does not change type -- and with it,
    whether cuTAMP treats it as a static obstacle -- between two phases of the same task.

    Declared-but-not-yet-existing objects are typed here too, so a phase over one validates like any
    other. They are always movables: _deferred_entries refuses to let one be a surface.
    """
    surfaces = {table_name}
    for _, _, atoms, _, _ in phases:
        for name, args in atoms:
            if name == "On" and len(args) == 2:
                surfaces.add(args[1])
    known = set(objects) | {table_name} | set(deferred)
    return SceneTypes(
        surfaces=frozenset(surfaces & known),
        movables=frozenset(known - surfaces),
        deferred=frozenset(deferred),
    )


def _deferred_entries(
    data: Any,
    phases: Sequence[tuple[str, str, list[tuple[str, list[str]]], str, object]],
    objects: Sequence[str],
    table_name: str,
) -> list[tuple[str, int, str]]:
    """Parse ``new_objects`` into (name, created_by_phase, description), with everything checkable
    before the atoms are grounded.

    The anchor check needs grounded atoms, so it lives in _check_deferred_anchor; everything that can
    be settled from the raw entries is settled here, where the message can still name the entry.
    """
    entries = _field(data, "new_objects", [])
    if not isinstance(entries, list):
        raise HITLProposalError(f"'new_objects' must be a list, got {type(entries).__name__}.")
    known = set(objects) | {table_name}
    out: list[tuple[str, int, str]] = []
    for entry in entries:
        name = str(_field(entry, "name"))
        description = str(_field(entry, "description", ""))
        try:
            created = int(_field(entry, "created_by_phase", -1))
        except (TypeError, ValueError) as exc:
            raise HITLProposalError(
                f"The `created_by_phase` of the new object '{name}' must be a whole number."
            ) from exc
        if name in known:
            raise HITLProposalError(
                f"'{name}' is already an object in the scene, so it is not a new one. Use it directly."
            )
        if any(name == n for n, _, _ in out):
            raise HITLProposalError(f"You declared the new object '{name}' more than once.")
        if not 0 <= created < len(phases):
            raise HITLProposalError(
                f"The new object '{name}' says it is created by phase {created}, but there are only "
                f"{len(phases)} phase(s) (numbered 0 to {len(phases) - 1})."
            )
        if phases[created][0] != "human":
            raise HITLProposalError(
                f"The new object '{name}' is created by phase {created}, which is a ROBOT phase. The "
                "robot only picks things up and puts them down, so it cannot bring a new object into "
                "existence. A person has to, so that must be a human phase."
            )
        uses = [i for i, (_, _, atoms, _, _) in enumerate(phases) if any(name in args for _, args in atoms)]
        if any(i < created for i in uses):
            raise HITLProposalError(
                f"The new object '{name}' is used by phase {min(uses)}, which comes before phase "
                f"{created} where it is created. Nothing can refer to it until it exists."
            )
        out.append((name, created, description))
    return out


def _check_deferred_anchor(name: str, created: int, phases: Sequence[Phase], deferred: Sequence[str]) -> str:
    """Where the creating phase says the new object ends up -- the one assertion binding can use.

    Required, and required to be a real object: it is both what the camera check for that phase looks
    for and, once the human has acted, how the object is picked out of the next perception pass. See
    planning.bind_deferred_object.
    """
    anchors = [a.values[1] for a in phases[created].atoms if a.name == On.name and a.values[0] == name]
    if len(anchors) != 1:
        raise HITLProposalError(
            f"Phase {created} creates the new object '{name}', so it must say where it ends up with "
            f"exactly one On({name}, <a real object>) atom; it has {len(anchors)}."
        )
    if anchors[0] in deferred:
        raise HITLProposalError(
            f"On({name}, {anchors[0]}) puts the new object on '{anchors[0]}', which is another new "
            "object. It has to end up on something that is already in the scene, so it can be found."
        )
    return anchors[0]


def _build_invented(
    entries: Any, arg_types: dict[str, list[list[str]]], session_tag: int
) -> dict[str, VLMPredicate]:
    """Turn ``new_predicates`` entries into VLMPredicates, typed from how they are USED.

    The parameter types are not asked for and not guessed: they are read off the predicate's own uses,
    where each argument is a concrete object whose type this scene has already fixed. That matters
    because cuTAMP validates goal literals per type -- an invented predicate over the cloth is a
    predicate over a SURFACE in a task that also puts something on the cloth, and over a MOVABLE in
    one that does not. Asking the model to declare the types instead invited placeholder answers
    ("container", "cover_object") that named no real object.
    """
    if entries is None:
        return {}
    if not isinstance(entries, list):
        raise HITLProposalError(f"'new_predicates' must be a list, got {type(entries).__name__}.")
    invented: dict[str, VLMPredicate] = {}
    for entry in entries:
        raw_name = str(_field(entry, "name"))
        instructions = str(_field(entry, "instructions"))
        if raw_name in _RESERVED_FLUENTS:
            raise HITLProposalError(
                f"A predicate named '{raw_name}' already exists. Use it directly instead of inventing one."
            )
        if "#" in raw_name:
            # '#' separates the internal per-session suffix from the name. It should never reach the
            # model at all (see structs.display_atom), so this is a backstop -- but a name carrying
            # one would be double-suffixed into nonsense, so refuse it rather than build it.
            raise HITLProposalError(
                f"'{raw_name}' is not a valid predicate name: '#' is not allowed. If you meant the "
                f"existing predicate '{display_name(raw_name)}', use that name and do not define it again."
            )
        if raw_name in invented:
            raise HITLProposalError(f"You defined the predicate '{raw_name}' more than once.")
        uses = arg_types.get(raw_name, [])
        if not uses:
            raise HITLProposalError(
                f"You invented the predicate '{raw_name}' but no phase uses it. Either use it or do "
                "not define it."
            )
        if len({tuple(u) for u in uses}) > 1:
            raise HITLProposalError(
                f"'{raw_name}' is used with inconsistent arguments: {sorted({tuple(u) for u in uses})}. "
                "A predicate takes the same number of arguments, of the same kinds, everywhere."
            )
        parameters = [make_parameter(f"x{i}", type_name) for i, type_name in enumerate(uses[0])]
        invented[raw_name] = VLMPredicate(Fluent(session_fluent_name(raw_name, session_tag), parameters), instructions)
    return invented


def _build_operator(
    raw: Any,
    *,
    description: str,
    phase_atoms: frozenset,
    invented: dict[str, VLMPredicate],
    scene_types: SceneTypes,
) -> HumanOperator:
    """One human phase's ``operator`` entry -> a grounded :class:`HumanOperator`.

    Every atom is grounded through the same ``_ground_atom`` the phase's own atoms go through, with
    ``executor="human"``, so an invented predicate is legal here and an unknown one is rejected with
    the same message. What is checked beyond that is the operator's internal coherence: an add effect
    that is also a delete effect says nothing, and an operator whose add effects do not cover the
    phase's own atoms is contradicting the phase it belongs to.
    """
    if not isinstance(raw, dict):
        raise HITLProposalError(
            f"The human phase {description!r} needs an `operator` object saying what the action is "
            "(`name`, `args`) and what it does (`preconditions`, `add_effects`, `delete_effects`)."
        )
    name = str(_field(raw, "name")).strip()
    if not name:
        raise HITLProposalError(f"The operator for {description!r} needs a `name`, e.g. \"Push\".")
    if "#" in name:
        # Same backstop as _build_invented: '#' is the internal session-suffix separator.
        raise HITLProposalError(f"'{name}' is not a valid operator name: '#' is not allowed.")
    args = _field(raw, "args", [])
    if not isinstance(args, list) or not all(isinstance(a, str) for a in args):
        raise HITLProposalError(f"The args of the operator {name} must be a list of strings, got {args!r}.")
    args = [str(a) for a in args]
    # type_of raises with the available object names, which is the message the model can act on.
    parameters = tuple(make_parameter(f"x{i}", scene_types.type_of(a)) for i, a in enumerate(args))

    def atoms(key: str) -> frozenset:
        return frozenset(
            _ground_atom(n, a, invented, scene_types, "human") for n, a in _atom_entries(raw, key)
        )

    preconditions, add_effects, delete_effects = (atoms(k) for k in _OPERATOR_ATOM_KEYS)
    if not add_effects:
        raise HITLProposalError(
            f"The operator {name} has no `add_effects`, so nothing about the workspace changes when "
            "the person does it and there is no way to tell it happened."
        )
    both = add_effects & delete_effects
    if both:
        raise HITLProposalError(
            f"The operator {name} both adds and deletes "
            f"{', '.join(sorted(display_atom(a) for a in both))}. An atom can be one or the other."
        )
    uncovered = phase_atoms - add_effects
    if uncovered:
        raise HITLProposalError(
            f"The phase {description!r} says "
            f"{', '.join(sorted(display_atom(a) for a in uncovered))} must be true afterwards, but "
            f"its operator {name} does not make that true. Every atom in a phase's `atoms` must "
            "appear in its operator's `add_effects`."
        )
    contradicted = phase_atoms & delete_effects
    if contradicted:
        raise HITLProposalError(
            f"The operator {name} deletes "
            f"{', '.join(sorted(display_atom(a) for a in contradicted))}, which the phase "
            f"{description!r} says must be TRUE afterwards."
        )
    return HumanOperator(
        name=name,
        args=tuple(args),
        parameters=parameters,
        preconditions=preconditions,
        add_effects=add_effects,
        delete_effects=delete_effects,
    )


def _resolve_fluent(name: str, invented: dict[str, VLMPredicate], executor: str) -> Fluent:
    """Look up a predicate by the name a proposal calls it, for a phase with this executor."""
    if name in _ROBOT_FLUENTS:
        return _ROBOT_FLUENTS[name]
    if name in invented:
        if executor == "robot":
            # The one rule that makes a phase a HUMAN phase. cuTAMP has no operator that can make an
            # invented predicate true, so a robot phase asking for one is unachievable -- and, before
            # this check existed, produced a plan that simply never reached its goal.
            raise HITLProposalError(
                f"A robot phase cannot achieve '{name}': the robot only picks and places, and no "
                f"operator it has can make '{name}' true. Make this a human phase, or state the "
                "phase with On, Holding or HandEmpty."
            )
        return invented[name].fluent
    known = sorted(set(_ROBOT_FLUENTS) | (set(invented) if executor == "human" else set()))
    raise HITLProposalError(f"Unknown predicate '{name}'. Available predicates: {', '.join(known)}.")


def _ground_atom(
    name: str, args: Sequence[str], invented: dict[str, VLMPredicate], scene_types: SceneTypes, executor: str
) -> Atom:
    """Ground one atom, checking arity, that the objects exist, and that their types line up."""
    fluent = _resolve_fluent(name, invented, executor)
    if len(args) != len(fluent.parameters):
        raise HITLProposalError(
            f"{display_name(fluent.name)} takes {len(fluent.parameters)} argument(s), but it is "
            f"applied to {len(args)}: {list(args)}."
        )
    for arg, param in zip(args, fluent.parameters):
        actual = scene_types.type_of(arg)
        if actual != param.type:
            raise HITLProposalError(
                f"{name}({', '.join(args)}) applies the predicate to '{arg}', which is a {actual} in "
                f"this scene, but {name} expects a {param.type} there. A {Surface} is something other "
                f"things are put ON; a {Movable} is something the robot can pick up."
            )
    return fluent.ground(*args)


def check_coverage(data: Any, phase_count: int, unrepresented: Sequence[dict[str, str]]) -> tuple[dict, ...]:
    """Check the clause-by-clause mapping the proposer was asked for.

    Its real job is upstream of this check: enumerating the instruction's clauses and pinning each to
    a phase is the reasoning step whose absence produced a two-phase answer to a three-clause
    instruction, both phases covering clause one. Validating it here also catches a clause pointed at
    a phase that does not exist, and one marked as dropped without saying why.
    """
    entries = (data or {}).get("coverage") or []
    if not isinstance(entries, list):
        raise HITLProposalError(f"'coverage' must be a list, got {type(entries).__name__}.")
    dropped = {u["clause"] for u in unrepresented}
    coverage = []
    for entry in entries:
        clause = str(_field(entry, "clause"))
        try:
            index = int(_field(entry, "phase", -1))
        except (TypeError, ValueError) as exc:
            raise HITLProposalError(f"The phase for the clause {clause!r} must be a whole number.") from exc
        if index >= phase_count:
            raise HITLProposalError(
                f"The clause {clause!r} is assigned to phase {index}, but there are only {phase_count} "
                f"phase(s) (numbered 0 to {phase_count - 1}). Either add the phase that carries it "
                "out, or set its phase to -1 and list it in `unrepresented` with the reason."
            )
        if index < 0 and clause not in dropped:
            raise HITLProposalError(
                f"The clause {clause!r} has no phase, but it is not in `unrepresented` either. Either "
                "plan a phase for it, or say in `unrepresented` why it cannot be done."
            )
        coverage.append({"clause": clause, "phase": index})
    return tuple(coverage)


def parse_unrepresented(data: Any) -> tuple[dict[str, str], ...]:
    """Clauses of the instruction the proposer says it could not express.

    Almost always an object the instruction names that perception did not detect. Kept and reported
    rather than dropped: a run that silently plans two clauses of a three-clause instruction and then
    reports success is the exact failure this package exists to remove.
    """
    entries = (data or {}).get("unrepresented") or []
    if not isinstance(entries, list):
        raise HITLProposalError(f"'unrepresented' must be a list, got {type(entries).__name__}.")
    return tuple(
        {"clause": str(_field(e, "clause")), "reason": str(_field(e, "reason", "no reason given"))}
        for e in entries
    )


def parse_plan_response(
    data: Any, instruction: str, objects: Sequence[str], table_name: str, session_tag: int
) -> TaskSpecification:
    """Validate a plan response into the phases the rest of the system executes."""
    phase_entries = _phase_entries(data)
    deferred_entries = _deferred_entries(data, phase_entries, objects, table_name)
    deferred_names = [name for name, _, _ in deferred_entries]
    scene_types = _scene_types(phase_entries, objects, table_name, deferred_names)
    misused = sorted(set(deferred_names) & scene_types.surfaces)
    if misused:
        # A surface is pinned for the whole task and becomes static world geometry, so one that does
        # not exist for the first half of it is not something to guess at. Nothing has needed it.
        raise HITLProposalError(
            f"{', '.join(misused)} is a new object, but the plan puts something ON it. A new object "
            "can be picked up and moved; it cannot be used as a surface."
        )

    # An invented predicate's signature comes from its uses, where every argument is a concrete
    # object whose type this scene has already fixed.
    declared = {str(_field(e, "name")) for e in (data.get("new_predicates") or [])}
    arg_types: dict[str, list[list[str]]] = {}
    uses = [(n, a) for _, _, atoms, _, _ in phase_entries for n, a in atoms]
    # Uses inside an operator count too: a precondition or a delete effect may be the only place an
    # invented predicate is named, and an unused predicate is rejected.
    uses += _operator_predicate_uses(phase_entries)
    for name, args in uses:
        if name in declared:
            arg_types.setdefault(name, []).append([scene_types.type_of(a) for a in args])
    invented = _build_invented(data.get("new_predicates"), arg_types, session_tag)

    phases = []
    for executor, description, atoms, instructions, raw_operator in phase_entries:
        grounded = frozenset(_ground_atom(n, a, invented, scene_types, executor) for n, a in atoms)
        operator = None
        if executor == "human":
            operator = _build_operator(
                raw_operator,
                description=description,
                phase_atoms=grounded,
                invented=invented,
                scene_types=scene_types,
            )
        phases.append(Phase(
            executor=executor,
            description=description,
            atoms=grounded,
            instructions=instructions,
            operator=operator,
        ))
    phases = tuple(phases)
    deferred = tuple(
        DeferredObject(
            name=name,
            created_by_phase=created,
            anchor=_check_deferred_anchor(name, created, phases, deferred_names),
            description=description,
        )
        for name, created, description in deferred_entries
    )
    unrepresented = parse_unrepresented(data)
    return TaskSpecification(
        instruction=instruction,
        phases=phases,
        scene_types=scene_types,
        invented=tuple(invented.values()),
        deferred=deferred,
        unrepresented=unrepresented,
        coverage=check_coverage(data, len(phases), unrepresented),
    )


async def propose_plan(
    image: Image.Image,
    instruction: str,
    objects: Sequence[str],
    table_name: str,
    cfg: HITLConfig,
    session_tag: int,
) -> TaskSpecification:
    """Steps 1.1-1.2: the instruction becomes an ordered plan of robot and human phases."""
    prompt = plan_prompt(instruction, list(objects))

    def parse(data: Any) -> TaskSpecification:
        spec = parse_plan_response(data, instruction, objects, table_name, session_tag)
        if cfg.check_plan_effects:
            # Inside `parse`, so a plan whose operators contradict each other is REPROMPTED with the
            # reason rather than failing the task. The starting workspace is unknown here (nothing
            # has been classified yet), so only what the plan itself makes false is provable -- see
            # planning.check_plan_effects.
            broken = check_plan_effects(spec)
            if broken:
                raise HITLProposalError(f"This plan does not hang together: {broken}")
        return spec

    spec = await query_json(
        prompt, parse, model=cfg.proposal_model, schema=PLAN_SCHEMA,
        image=image, max_attempts=cfg.max_attempts, label="task plan",
        cache=proposal_cache(cfg),
    )
    _log.info(f"HITL plan for {instruction!r}: {len(spec.phases)} phase(s)")
    for i, phase in enumerate(spec.phases):
        atoms = ", ".join(sorted(display_atom(a) for a in phase.atoms))
        _log.info(f"HITL phase {i} [{phase.executor}] {phase.description} -> {atoms}")
        if phase.is_human:
            _log.info(f"HITL phase {i} instructions: {phase.instructions}")
    for predicate in spec.invented:
        _log.info(f"HITL invented predicate {display_name(predicate.name)}: {predicate.instructions}")
    for i, phase in enumerate(spec.phases):
        if phase.operator is None:
            continue
        op = phase.operator
        _log.info(
            f"HITL phase {i} operator {op.display}: pre "
            f"{sorted(display_atom(a) for a in op.preconditions) or 'none'} -> add "
            f"{sorted(display_atom(a) for a in op.add_effects)} del "
            f"{sorted(display_atom(a) for a in op.delete_effects) or 'none'}"
        )
    for deferred in spec.deferred:
        _log.info(
            f"HITL new object {deferred.name}: does not exist yet -- phase {deferred.created_by_phase} "
            f"creates it, on {deferred.anchor} ({deferred.description})"
        )
    # A warning, deliberately NOT a rejection. Two consecutive robot phases moving the same object are
    # a SUPPORTED shape -- `robot_run` splits exactly there and the tamp -> tamp -> teleop chain is
    # built on it (data-collection ARCHITECTURE.md §6c) -- so refusing the plan would make that path
    # unreachable. But it is also what a misclassified human step looks like, and the operator finding
    # out by watching the arm do the same pick twice is the outcome this line exists to prevent.
    wasted = wasted_robot_move(spec.phases)
    if wasted:
        _log.warning(f"HITL: this plan repeats work -- {wasted}")
    for dropped in spec.unrepresented:
        _log.warning(
            f"HITL: NOT part of the plan -- {dropped['clause']!r}: {dropped['reason']}. "
            f"Detected objects were: {', '.join(sorted(objects))}"
        )
    return spec
