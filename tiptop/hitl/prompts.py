"""The two prompts: plan the task into phases, and check one statement against an image.

The robot is described to the model in ABSTRACT terms -- "pick up an object", "place it on a surface"
-- rather than with cuTAMP's real operator signatures. Those carry motion-level parameters (``conf``,
``traj``, ``grasp``) and bookkeeping fluents (``At``, ``CanMove``, ``JustMoved``) that a proposer has
no business reasoning about, and shown them it writes goals over the alternation lock. Restricting
the vocabulary to the three state predicates a sub-goal can be phrased in keeps every phase
groundable by construction.
"""

# The predicates a ROBOT phase may use. Exactly the cuTAMP fluents a goal can be stated over and that
# some cuTAMP operator can change -- the same set create_tamp_environment reads (On / Holding), plus
# HandEmpty, which it supplies itself.
STATE_PREDICATE_DESCRIPTION = """\
- On(?obj: movable, ?surface: surface): {0} is resting on top of {1}
- Holding(?obj: movable): the robot's gripper is holding {0}
- HandEmpty(): the robot's gripper is empty"""

_PLACEHOLDER_NOTE = (
    "Write the text with {0}, {1}, ... standing in for the arguments, in the order they are declared."
)

_INVENTION_NOTE = (
    "Invent a new predicate only for something no existing predicate can express. A new predicate is "
    "evaluated by a vision-language model looking at a camera image of the workspace, so it needs "
    "`instructions`: a description of what must be VISIBLE in the image for it to be true. "
    + _PLACEHOLDER_NOTE
)

_ATOM_ITEM = {
    "type": "object",
    "properties": {
        "predicate": {"type": "string"},
        "args": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["predicate", "args"],
}

PLAN_SCHEMA = {
    "type": "object",
    "properties": {
        # No argument types are asked for: they are read off how the predicate is USED, where every
        # argument is a real object. Asking for them produced placeholder answers -- "container",
        # "cover_object" -- naming nothing in the scene.
        "new_predicates": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"name": {"type": "string"}, "instructions": {"type": "string"}},
                "required": ["name", "instructions"],
            },
        },
        "phases": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "executor": {"type": "string", "enum": ["robot", "human"]},
                    "description": {"type": "string"},
                    "atoms": {"type": "array", "items": _ATOM_ITEM},
                    "instructions": {"type": "string"},
                },
                "required": ["executor", "description", "atoms"],
            },
        },
        # Forces the instruction to be enumerated clause by clause and each clause pinned to a phase.
        # Without it the model plans the first clause, over-decomposes it, and stops -- observed
        # answering a three-clause instruction with two phases, both for clause one.
        "coverage": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "clause": {"type": "string"},
                    "phase": {"type": "integer"},
                },
                "required": ["clause", "phase"],
            },
        },
        # Where a dropped clause goes. Without somewhere to put it, the only way to answer at all is
        # to leave it out, and the run then does most of the task and reports success.
        "unrepresented": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"clause": {"type": "string"}, "reason": {"type": "string"}},
                "required": ["clause", "reason"],
            },
        },
    },
    "required": ["phases"],
}

CLASSIFIER_SCHEMA = {
    "type": "object",
    "properties": {"holds": {"type": "boolean"}, "reason": {"type": "string"}},
    "required": ["holds", "reason"],
}


def plan_prompt(instruction: str, objects: list[str]) -> str:
    """Turn an instruction into an ordered list of robot and human phases."""
    object_list = "\n".join(f"- {name}" for name in sorted(objects))
    return f"""\
A robot and a human share a workspace. Break the instruction below into an ORDERED list of phases. \
Each phase is done either by the robot or by the human, and they happen in the order you give.

THE INSTRUCTION:
{instruction}

The image shows the workspace as it is right now. Plan the INSTRUCTION -- the picture is context for \
where things are, not a task in itself.

The workspace contains exactly these objects, and no others:
{object_list}

The robot can do one thing: pick an object up and place it on a surface. That is all. It plans and \
executes each of its phases itself; you only say what must be TRUE when the phase is finished, using \
these predicates:
{STATE_PREDICATE_DESCRIPTION}

The human can do anything the robot cannot -- open, close, fold, unfold, tie, flatten, rotate, \
manipulate cloth. Give a human phase `instructions` addressed to the person, and `atoms` saying what \
should be true afterwards. That is what a camera will be used to check, so it must be visible.

How to divide the work:
- Give the robot every pick-and-place. A human phase that includes moving an object from A to B is \
taking the robot's work away from it.
- Use a human phase only for something the robot genuinely cannot do.
- ORDER MATTERS AND IS YOURS TO SET. If the box must be opened before anything can go in it, the \
human's "open the box" phase comes BEFORE the robot's "put the toy in the box" phase. If a cloth is \
folded over a toy, the robot places the toy first and the human folds afterwards. Think about what \
has to be true for the next phase to be physically possible.
- Intermediate states are fine and often necessary. "Take the toy off the box, open the box, put the \
toy back in" is three phases, and the first one ends with the toy somewhere else -- On(toy, table). \
Each robot phase is planned fresh, so an object may be picked up in more than one phase.
- Do not add phases the instruction does not ask for, and do not merge two of its steps into one.

PLAN THE WHOLE INSTRUCTION. Work through it clause by clause and give every clause a phase, in the \
order stated. Fill in `coverage` with one entry per clause, naming the phase that carries it out \
(the index in your `phases` list), or -1 if you had to leave it out. The commonest mistake is to \
plan the first clause carefully and stop; the last phase must leave the workspace as the END of the \
instruction describes.

Worked example. Objects: red_toy, cardboard_box. Instruction: "take the toy off the box, open the \
box, then put the toy inside it". That is three clauses, so three phases:
  phase 0, robot  -- "take the toy off the box"   atoms: On(red_toy, table)
  phase 1, human  -- "open the box"               atoms: IsOpen(cardboard_box)
                     instructions: "Open the cardboard_box and fold its flaps back."
  phase 2, robot  -- "put the toy inside the box" atoms: On(red_toy, cardboard_box)
  coverage: [["take the toy off the box", 0], ["open the box", 1], ["put the toy inside it", 2]]
Note the order: the box is opened BEFORE anything is put in it, the robot does both pick-and-places, \
and the toy is picked up in two different phases, which is allowed.

Rules, all of which are checked:
- A ROBOT phase's atoms may use ONLY On, Holding and HandEmpty. The robot cannot achieve a predicate \
you invent -- if a phase needs one, it is a human phase.
- Give a whole pick-and-place ONE phase, ending with On(?obj, ?surface). Do not split it into a \
"pick it up" phase and a "put it down" phase; the robot does both as one piece of work.
- Every phase needs at least one atom, and a human phase needs `instructions` too.
- Every object name must be one of the objects listed above, spelled exactly. Do not invent objects.
- {_INVENTION_NOTE}

If some clause CANNOT be expressed, put it in `unrepresented` with the reason, and leave it out of \
the phases. The usual reason is that it refers to something not in the object list -- "pick another \
toy" when only one toy was detected. Never invent an object to satisfy a clause, and never bind it \
to a different object that happens to be present. Saying you could not do it is always better than \
quietly doing something else: a human is watching, and can put the missing object on the table and \
start again."""


def classifier_prompt(statement: str) -> str:
    """Ask whether one statement holds in one image of the workspace."""
    return f"""\
You are the perception system of a robot. Look at the image of the robot's workspace and decide \
whether the following statement is true right now.

Statement: {statement}

Judge only what you can see. If the workspace does not clearly show the statement to be true, it is \
false. Give a one-sentence reason for your answer."""
