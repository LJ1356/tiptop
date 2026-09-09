"""Plan a task into phases against a real image, with no robot and no GPU.

The cheapest way to see whether the prompts work on a real workspace photo, and the one to reach for
when a rollout produces a plan that looks wrong: it prints the ordered phases, who does each, the
sub-goal handed to TiPToP, the invented predicates and their classifiers, and anything the proposer
could not express.

    python -m tiptop.hitl.demo --image workspace.png --goal "open the box and put the toy in it"

Objects default to whatever Gemini detects in the image, so this exercises the same labels a rollout
would see. Pass --objects to pin them instead, e.g. when reproducing a run from its saved perception.
"""

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

from PIL import Image

from tiptop.hitl.config import HITLConfig, load_hitl_config
from tiptop.hitl.grounding import DEFAULT_DESCRIPTIONS, classify_initial_state, descriptions_for
from tiptop.hitl.planning import check_robot_phases, goal_atoms_to_dicts, initial_state_for
from tiptop.hitl.proposal import propose_plan
from tiptop.hitl.record import set_recorder
from tiptop.hitl.structs import describe_atom, display_name
from tiptop.perception.gemini import detect_and_translate_async

_log = logging.getLogger(__name__)


async def _detect_objects(image: Image.Image, goal: str) -> list[str]:
    """The object labels perception would produce for this image and goal."""
    bboxes, _ = await detect_and_translate_async(image, goal)
    # Same sanitisation perception_wrapper applies before a label becomes a symbol.
    return sorted({str(b["label"]).strip().replace(" ", "_") for b in bboxes if b.get("label")})


async def _run(args: argparse.Namespace) -> int:
    image = Image.open(args.image).convert("RGB")
    cfg = load_hitl_config(args.hitl_config) if args.hitl_config else HITLConfig(enabled=True)
    if args.cache:
        cfg = HITLConfig(**{**cfg.__dict__, "cache_path": args.cache})
    if args.vlm_io:
        set_recorder(Path(args.vlm_io))
        print(f"Recording every VLM query to {args.vlm_io}\n")

    objects = args.objects or await _detect_objects(image, args.goal)
    if not objects:
        print("No objects detected in the image; pass --objects to name them yourself.", file=sys.stderr)
        return 1
    print(f"Objects: {', '.join(objects)}\n")

    spec = await propose_plan(image, args.goal, objects, args.table, cfg, 1)

    if spec.unrepresented:
        print("NOT REPRESENTED IN THE PLAN")
        for dropped in spec.unrepresented:
            print(f"  - {dropped['clause']}: {dropped['reason']}")
        print()
    for predicate in sorted(spec.invented, key=lambda p: p.name):
        print(f"INVENTED PREDICATE {display_name(predicate.name)}"
              f"({', '.join(p.type for p in predicate.fluent.parameters)})")
        print(f"  VLM classifier: {predicate.instructions}")
    if spec.invented:
        print()
    print(f"Surfaces: {', '.join(sorted(spec.scene_types.surfaces))}")
    print(f"Movables: {', '.join(sorted(spec.scene_types.movables))}\n")

    initially_true = frozenset()
    if cfg.classify_initial and spec.invented:
        initially_true = await classify_initial_state(image, spec, cfg)
        print(f"Already true: {sorted(str(a) for a in initially_true) or '(none)'}\n")

    initial_state = initial_state_for(spec.scene_types, initially_true)
    reason = check_robot_phases(spec, initial_state)
    if reason is not None:
        print(f"PLAN REJECTED: {reason}", file=sys.stderr)
        return 1

    print("PLAN")
    descriptions = descriptions_for(spec.invented)
    for i, phase in enumerate(spec.phases):
        who = "human, via teleop" if phase.is_human else "robot, via cuTAMP"
        print(f"  phase {i} [{who}] {phase.description}")
        if phase.is_human:
            print(f"    instructions: {phase.instructions}")
            for atom in sorted(phase.atoms, key=str):
                print(f"    will be checked: {describe_atom(atom, descriptions)}")
        else:
            print(f"    tamp goal: {goal_atoms_to_dicts(phase.atoms)}")
            for atom in sorted(phase.atoms, key=str):
                print(f"               = {describe_atom(atom, DEFAULT_DESCRIPTIONS)}")
    if not spec.needs_human:
        print("\n  (no human phases: the robot can do this whole task on its own)")

    if args.json:
        Path(args.json).write_text(json.dumps(spec.to_json(), indent=2))
        print(f"\nWrote {args.json}")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--image", required=True, type=Path, help="a photo of the workspace")
    parser.add_argument("--goal", required=True, help="the task, in natural language")
    parser.add_argument("--objects", nargs="*", help="object names; default is to detect them")
    parser.add_argument("--table", default="table", help="the fitted table's name (perception calls it 'table')")
    parser.add_argument("--hitl-config", help="the cfg/tamp `hitl` block as JSON or a path to it")
    parser.add_argument("--cache", help="SQLite path to cache proposal responses in, for prompt iteration")
    parser.add_argument("--vlm-io", help="directory to save every image sent to the VLM and its answer")
    parser.add_argument("--json", help="write the proposed plan here")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    logging.basicConfig(level=args.log_level, format="%(levelname)s: %(message)s")
    sys.exit(asyncio.run(_run(args)))


if __name__ == "__main__":
    main()
