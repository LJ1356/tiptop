"""The three prompts: plan the task into phases, order one robot leg's picks and places, and check
one statement against an image.

The robot is described to the model in ABSTRACT terms -- "pick up an object", "place it on a surface"
-- rather than with cuTAMP's real operator signatures. Those carry motion-level parameters (``conf``,
``traj``, ``grasp``) and bookkeeping fluents (``At``, ``CanMove``, ``JustMoved``) that a proposer has
no business reasoning about, and shown them it writes goals over the alternation lock. Restricting
the vocabulary to the three state predicates a sub-goal can be phrased in keeps every phase
groundable by construction.

``task_plan_prompt`` strikes the same bargain one level down. It asks for the ORDER of picks and
places -- the thing that replaced cuTAMP's breadth-first search -- in exactly the vocabulary the plan
prompt already uses, and it too never mentions ``MoveFree``, ``MoveHolding`` or a symbol.
``task_plan.expand_task_plan`` supplies all of those, which is precisely why the model does not have
to: the alternation is FORCED by the domain, so there is nothing in it to decide.
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
        # An object that is not in the scene yet because a human phase is what produces it. Without
        # this the proposer had no name to bind such a thing to, so it fell back on an invented
        # predicate -- and an invented predicate forces the phase to be a human one, handing a plain
        # pick-and-place to a teleoperator. See structs.DeferredObject.
        "new_objects": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "created_by_phase": {"type": "integer"},
                    "description": {"type": "string"},
                },
                "required": ["name", "created_by_phase", "description"],
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

TASK_PLAN_SCHEMA = {
    "type": "object",
    "properties": {
        # First and required, so the order is REASONED about before it is committed to -- the same job
        # `coverage` does in PLAN_SCHEMA. It is also the most useful line in the audit trail when a
        # plan turns out to be geometrically infeasible.
        "reasoning": {"type": "string"},
        "steps": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "action": {"type": "string", "enum": ["pick", "place"]},
                    "object": {"type": "string"},
                    # Absent on a pick; the parser requires it on a place and says so.
                    "surface": {"type": "string"},
                },
                "required": ["action", "object"],
            },
        },
        # Where a goal that picks and places genuinely cannot reach goes -- one that needs an object
        # moved twice, say. Without somewhere to put that, the only way to answer at all is to invent
        # a plan that fails the goal check, and the reprompt loop then spends every attempt on a
        # correct refusal restated three times.
        "problem": {"type": "string"},
    },
    "required": ["reasoning", "steps"],
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

The workspace contains exactly these objects RIGHT NOW, and no others:
{object_list}

(If the instruction goes on to talk about something that is not there yet because the human has to \
produce it first, see NEW OBJECTS below. That is the one way to name something not on this list.)

The robot can do one thing: pick an object up and place it on a surface. That is all. It plans and \
executes each of its phases itself; you only say what must be TRUE when the phase is finished, using \
these predicates:
{STATE_PREDICATE_DESCRIPTION}

WHAT On CANNOT SAY. On({{0}}, {{1}}) means {{0}} is RESTING LOOSELY somewhere on top of {{1}}, and \
nothing more. The robot places by opening its gripper above a surface, so it cannot fit, insert, \
slot, thread, plug, screw, seat, close, or align one thing to another, and On cannot ask it to. If a \
clause needs the object to end up IN something, or in a particular position or orientation on it -- a \
puzzle piece in its matching cut-out, a lid seated on a jar, a plug in a socket, a book squared onto \
a shelf -- that is a HUMAN phase, however much it looks like a pick-and-place. Say so with an \
invented predicate, not with On.

The human can do anything the robot cannot -- open, close, fold, unfold, tie, flatten, rotate, \
manipulate cloth. Give a human phase `instructions` addressed to the person, and `atoms` saying what \
should be true afterwards. That is what a camera will be used to check, so it must be visible.

NEW OBJECTS. Sometimes the instruction is about a thing that is not a separate object yet, and only \
becomes one because of what the human does: a block still inside the tower, a card still in the \
deck, a lid still on the jar. It is not in the list above because it cannot be seen yet.

Declare it in `new_objects`, giving its `name`, the `created_by_phase` index of the HUMAN phase that \
brings it into existence, and a `description` of what it will look like once it does. That phase \
must also say where it ends up -- On(<the new object>, <something from the list above>) among its \
atoms -- because that is how it is found in the camera image afterwards. From then on it is an \
ordinary object, and a ROBOT phase can pick it up and place it like anything else.

Reach for this whenever the instruction goes on to MOVE the thing the human produced. Writing that \
step as another human phase because you had no name for the object gives the robot's own work away.

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

A second worked example, this time with a new object. Objects: jenga_tower, screwdriver, \
white_paper. Instruction: "remove a block from the jenga tower using the screwdriver onto the white \
paper, and then place the block on top of the jenga tower". The loose block is not in the object \
list -- it is still a brick inside the tower -- so declare it:
  new_objects: [{{"name": "loose_block", "created_by_phase": 0,
                 "description": "the single wooden block the human pushes out of the tower"}}]
  phase 0, human  -- "push a block out of the tower onto the paper"
                     atoms: On(loose_block, white_paper)
                     instructions: "Use the screwdriver to push one block out of the jenga_tower and
                     put it on the white_paper."
  phase 1, robot  -- "put the block on top of the tower"  atoms: On(loose_block, jenga_tower)
  coverage: [["remove a block ... onto the white paper", 0], ["place the block on top ...", 1]]
Phase 1 is a ROBOT phase. It is one pick and one place, which is exactly what the robot is for; the \
human is needed for phase 0 only because prying a block out with a screwdriver is not a pick-and-place.

A third worked example, about that last point. Objects: pink_toy, puzzle_board, yellow_cloth. \
Instruction: "place the toy on the cloth and solve the puzzle". Two clauses, so two phases:
  new_predicates: [{{"name": "IsSolved", "instructions": "every piece of {{0}} is sitting down inside
                     its own matching cut-out, flush with the board, with no gaps"}}]
  phase 0, robot  -- "place the toy on the cloth"  atoms: On(pink_toy, yellow_cloth)
  phase 1, human  -- "solve the puzzle"            atoms: IsSolved(puzzle_board)
                     instructions: "Fit each puzzle piece into its matching cut-out in the
                     puzzle_board so it sits flush."
  coverage: [["place the toy on the cloth", 0], ["solve the puzzle", 1]]
"solve the puzzle" is NOT On(pink_toy, puzzle_board). Resting the toy on the board solves nothing -- \
the piece has to go INTO its slot, which the robot cannot do. Writing it as a robot phase is worse \
than useless: the robot picks the toy straight back up and drops it on the board, undoing phase 0 to \
achieve nothing.

Rules, all of which are checked:
- A ROBOT phase's atoms may use ONLY On, Holding and HandEmpty. The robot cannot achieve a predicate \
you invent -- if a phase needs one, it is a human phase.
- Give a whole pick-and-place ONE phase, ending with On(?obj, ?surface). Do not split it into a \
"pick it up" phase and a "put it down" phase; the robot does both as one piece of work.
- Two ROBOT phases in a row must not move the same object twice. The second placement throws the \
first one away, so the first is wasted motion -- and it almost always means a step that is not really \
a pick-and-place was given to the robot. If a human phase belongs between them, put it there; if the \
second phase is the one the robot cannot do, make IT the human phase.
- Every phase needs at least one atom, and a human phase needs `instructions` too.
- Every object name must be one of the objects listed above, spelled exactly, or one you declared in \
`new_objects`. Do not name an object any other way.
- A `new_objects` entry must be created by a HUMAN phase, that phase must come before every phase \
that uses the object, and that phase must place it with On(<the new object>, <a listed object>).
- {_INVENTION_NOTE}

If some clause CANNOT be expressed, put it in `unrepresented` with the reason, and leave it out of \
the phases. The usual reason is that it refers to something not in the object list and nothing in \
your plan produces it -- "pick another toy" when only one toy was detected, and no phase makes a \
second one. (If a human phase of yours IS what produces it, that is a `new_objects` entry, not an \
unrepresented clause.) Never invent an object to satisfy a clause, and never bind it to a different \
object that happens to be present. Saying you could not do it is always better than \
quietly doing something else: a human is watching, and can put the missing object on the table and \
start again."""


def task_plan_prompt(goal: list[str], descriptions: list[str], movables: list[str], surfaces: list[str]) -> str:
    """Order the picks and places that leave the workspace satisfying one robot leg's goal.

    ``goal`` is the leg's atoms as the model must see them (``display_atom`` -- no session suffix),
    ``descriptions`` the phase descriptions it covers, and the two lists are exactly what
    ``create_tamp_environment`` typed as pickable and as surfaces for this rollout. Nothing outside
    those lists exists as far as this call is concerned, and the parser refuses anything else.
    """
    work = "\n".join(f"- {text}" for text in descriptions) or "- (the goal below)"
    goal_lines = "\n".join(f"- {text}" for text in goal)
    movable_list = "\n".join(f"- {name}" for name in movables) or "- (nothing)"
    surface_list = "\n".join(f"- {name}" for name in surfaces) or "- (none)"
    return f"""\
A robot and a person are sharing a workspace. The robot is about to carry out its next piece of the \
task, and your job is to decide the ORDER in which it picks things up and puts them down.

The image shows the workspace exactly as it is right now.

WHAT THIS PIECE OF WORK IS:
{work}

WHEN IT IS DONE, all of these must be true:
{goal_lines}

where
{STATE_PREDICATE_DESCRIPTION}

Objects the robot may pick up:
{movable_list}

Surfaces it may put things down on:
{surface_list}

The robot does exactly two things, and it does them one at a time:
  pick(object)            - close the gripper on that object and lift it
  place(object, surface)  - put down the object it is holding, onto that surface

Give `reasoning` first: one or two sentences saying why this order and not another. Then give \
`steps`: the picks and places, in the order the robot should do them.

WHY THE ORDER MATTERS. This is the whole question you are being asked, and it is a question about \
the picture. The robot cannot reach through things, and it cannot put something down where something \
else already is. Look at the image:
- If an object is sitting ON something that has to move, or is sitting where something else has to \
end up, move it out of the way FIRST.
- If two things are going onto the same surface, put down first the one the other would otherwise \
block or bury.
- If nothing is in anything's way, the order does not matter. Say so, and pick either.

Rules, all of which are checked before the arm moves:
- The gripper holds one object at a time. Every `pick` is followed by the `place` of that SAME object \
before anything else is picked up.
- The robot may pick each object up ONCE in this list. There is no way to move something and then \
move it again.
- After the last step, every goal line above must be true. A goal line `HandEmpty()` means the last \
step is a place. A goal line `Holding(x)` means the last step is a pick of x.
- Give every goal line a step, EVEN IF the picture already shows it to be true. The robot does the \
work again from where things are now, and a goal line with no step fails the check.
- Spell object and surface names exactly as they are listed above. Anything not on those two lists \
does not exist here.
- Move something the goal does not mention only when it is physically in the way, and say in \
`reasoning` which goal line it was blocking. Every extra move is another chance to fail.
- Do not add steps the goal does not need. Two goal lines about two objects mean two picks and two \
places.

Worked example. Pickable: lid, red_toy. Surfaces: table, cardboard_box. \
Goal: On(red_toy, cardboard_box), On(lid, table), HandEmpty(). \
The picture shows the lid sitting on the cardboard_box and the toy on the table.
  reasoning: "The lid is closing the box, so the toy cannot go into the box until the lid is off it. \
Take the lid off onto the table first."
  steps: pick(lid), place(lid, table), pick(red_toy), place(red_toy, cardboard_box)
The wrong answer puts the toy in the box first and then has nowhere to put the lid.

A second example, where the order is free. Goal: On(block, tray), On(cup, tray), HandEmpty(). The \
picture shows the block and the cup side by side on the table, nothing on top of either, tray empty.
  reasoning: "Neither is on top of the other and the tray has room for both, so either order works."
  steps: pick(block), place(block, tray), pick(cup), place(cup, tray)

If no order of picks and places can reach the goal -- something would have to be moved twice, or a \
goal line names something that is not on the lists above -- leave `steps` EMPTY and say why in \
`problem`. Do not offer the closest sequence you can find instead: a plan that does not reach the \
goal is rejected, and saying plainly that it cannot be done is the more useful answer."""


def classifier_prompt(statement: str) -> str:
    """Ask whether one statement holds in one image of the workspace."""
    return f"""\
You are the perception system of a robot. Look at the image of the robot's workspace and decide \
whether the following statement is true right now.

Statement: {statement}

Judge only what you can see. If the workspace does not clearly show the statement to be true, it is \
false. Give a one-sentence reason for your answer."""
