"""The ``hitl`` config block, resolved into defaults.

Read from the per-task config in data-collection (``cfg/tamp/<name>.yml``), which is where a knob
belongs that changes what a dataset CONTAINS rather than how the arm moves. Deliberately not part of
``tamp_overrides``: that dict is a cuRobo/cuTAMP cost funnel whose reader is a hand-written if-ladder
(motion_planning.apply_cost_overrides), so an unrecognised key there is dropped without a word.
"""

import json
from dataclasses import dataclass
from pathlib import Path

# Planning the task into phases is reasoning, not spatial grounding, so it does NOT reuse the
# detection model. detect_and_translate runs gemini-robotics-er with thinking disabled because it is
# localising boxes; asking that same configuration to sequence a task and invent a predicate gets a
# worse answer than a general model with reasoning left on.
DEFAULT_PROPOSAL_MODEL = "gemini-2.5-pro"
# Grounding ("is the cloth folded?") is a visual judgement over one image. Flash is enough and is
# called once per grounding, which is the query the run pays for repeatedly.
DEFAULT_VLM_MODEL = "gemini-2.5-flash"


@dataclass(frozen=True)
class HITLConfig:
    """Resolved ``hitl`` block. ``enabled`` False means nothing in this package ever runs."""

    enabled: bool = False
    proposal_model: str = DEFAULT_PROPOSAL_MODEL
    vlm_model: str = DEFAULT_VLM_MODEL
    # Reprompts allowed when a proposal comes back unparseable or fails validation. The error message
    # is fed back to the model, which is what makes a second attempt worth making at all.
    max_attempts: int = 3
    # Classify the plan's invented predicates on the FIRST image, before anything runs. Off by
    # default: a human is being asked precisely because the predicate is false, so it costs one VLM
    # call per grounding to learn what was already assumed. Worth turning on for a scene that may
    # start already solved.
    classify_initial: bool = False
    # Extra chances the operator gets at a human phase the VLM says did not happen.
    # 1 = show what is missing and hand the arm back once more; 0 = fail on the first bad verdict.
    verify_retries: int = 1
    # Treat a failed verification as a rollout failure (the operator still labels the episode, so a
    # false negative is recoverable by answering the label prompt). False records the verdict in
    # hitl.json and carries on, which is what you want while calibrating the classifier prompts.
    verify_enforced: bool = True
    # Write every image sent to the VLM, and a rendered PNG of what it answered, into `vlm/` beside
    # each rollout (plus index.jsonl with the full prompts and replies). Rejected attempts included.
    # On by default: when a HITL run goes wrong the question is almost always "what did the model
    # actually see, and what did it say", and that is unanswerable after the fact without this.
    save_vlm_io: bool = True
    # Which registered planner carries out the ROBOT phases (planners.register_robot_planner);
    # `policy_type` below is the same choice for the human's. cuTAMP is the only robot planner this
    # build ships, and naming one it does not have raises rather than quietly planning the task with
    # a different one.
    robot_planner: str = "cutamp"
    # WHO CARRIES OUT THE PLAN'S HUMAN PHASES. "human" is the teleop hand-off this package was
    # built around and is the default; any other value names a registered policy planner
    # (planners.register_human_planner), which drives the arm through that phase itself.
    #
    # Nothing upstream of the hand-off is told. A phase is the human's because cuTAMP cannot express
    # what it asks for -- "fold the cloth", "open the box" -- and that is true however the phase is
    # then carried out, so the proposal stage, the phase's atoms and the verification afterwards are
    # all unchanged. This selects the executor, never the plan.
    policy_type: str = "human"
    # The checkpoint that policy loads. Required for every policy_type but "human"; the path is the
    # LeRobot checkpoint DIRECTORY (the one holding `pretrained_model/`), which is what
    # `checkpoints/<task>/checkpoints/last` is.
    policy_checkpoint: str | None = None
    # Steps of one predicted action chunk executed before the policy is asked for another. This is
    # LeRobot's `n_action_steps`, which is NOT baked into the model (the U-Net's output length is
    # `horizon`), so it is retuned here at deploy time rather than at training time. It must stay
    # within `horizon - n_obs_steps + 1`, which the policy server checks against the checkpoint it
    # loaded -- LeRobot itself does not, and silently executes a shorter chunk when it is exceeded.
    open_loop_horizon: int = 8
    # Reverse-diffusion steps per prediction (LeRobot's `num_inference_steps`). Another deploy-time
    # knob, and one that has to be set: LeRobot defaults it to `num_train_timesteps`, which is 100
    # for DDPM and takes ~334 ms on these checkpoints. The driver asks for a new chunk every
    # `open_loop_horizon` steps, so that default stalls the 15 Hz control loop for five periods on
    # every eighth step -- the 2026-09-09 toy-puzzle leg averaged 9.2 Hz, well off the rate the
    # policy was trained at. 10 steps costs 36 ms and, replayed over a full teleop leg of
    # 1_toy_puzzle, predicts actions indistinguishable from the 100-step ones (cosine against the
    # demonstrated action 0.867 vs 0.866, same mean magnitude). 0 keeps the checkpoint's own value.
    policy_num_inference_steps: int = 10
    # Hard stop for one policy leg, in control steps at 15 Hz (450 = 30 s). A behaviour-cloning
    # policy has no idea when it is finished -- there is no termination head and no reward -- so
    # something has to end the leg, and the phase verification that follows is what decides whether
    # what it did counts. Sized from the teleop legs these policies were trained on, whose longest is
    # 648 frames and whose median is ~300 (hitl-baseline/diffusion_policy/README.md).
    policy_max_steps: int = 450
    # Scales the policy's joint-velocity channels (not the gripper) before they reach the arm. The
    # deploy knob for a policy that moves faster or slower than the demonstrations it learned from;
    # 1.0 sends what it predicted, which is what the training data's units mean.
    policy_velocity_scale: float = 1.0
    # Interpreter for the policy SERVER, which loads the checkpoint. Its default is the venv of
    # hitl-baseline/diffusion_policy, the project that trained these checkpoints: LeRobot and torch
    # live only there, never in the DROID env that drives the arm. A machine that keeps them
    # somewhere else sets this; nothing about a task decides it.
    policy_python: str | None = None
    # SQLite cache for PROPOSAL responses only, keyed on the model, the prompt and a noise-robust
    # hash of the image (after prpl_llm_utils' SQLite3PretrainedLargeModelCache). Worth setting while
    # iterating on prompts, where the same scene and instruction are proposed over and over. Never
    # applied to grounding or verification -- see cache.ProposalCache for why that would be unsafe.
    cache_path: str | None = None


def resolve_hitl_config(raw: dict | None) -> HITLConfig:
    """Build a HITLConfig from the raw ``hitl`` block, ignoring absent keys.

    An absent or empty block resolves to the disabled default, which is what keeps a config that has
    never heard of HITL behaving exactly as before.
    """
    if not raw:
        return HITLConfig()
    if not isinstance(raw, dict):
        raise ValueError(f"the hitl config block must be a mapping, got {type(raw).__name__}")
    known = set(HITLConfig.__dataclass_fields__)
    unknown = sorted(set(raw) - known)
    if unknown:
        # Loud, unlike tamp_overrides: a misspelled key here silently disables the feature the config
        # was written to turn on.
        raise ValueError(f"unknown hitl config key(s): {', '.join(unknown)}. Known keys: {sorted(known)}")
    cfg = HITLConfig(**dict(raw))
    check_policy_config(cfg)
    return cfg


def check_policy_config(cfg: HITLConfig) -> None:
    """Reject a policy-executed block that is wrong on its own terms, at parse time.

    Everything here is a statement about the YAML and nothing about the machine, which is what lets
    it run inside ``resolve_hitl_config``: a config is parsed in places that will never run it (the
    data-collection server listing configs, a test over the shipped ones), and a check that reached
    for the filesystem there would make a portable config fail on the wrong host.

    Deliberately not a check of the planner NAME either -- planners.human_planner does that against
    the registry, which is the only thing that knows what is registered.
    """
    if not cfg.enabled or cfg.policy_type == "human":
        return
    if not cfg.policy_checkpoint:
        raise ValueError(
            f"hitl.policy_type is {cfg.policy_type!r}, so hitl.policy_checkpoint must name the "
            "checkpoint directory it runs (the one holding pretrained_model/)"
        )
    if cfg.open_loop_horizon < 1:
        raise ValueError(f"hitl.open_loop_horizon must be at least 1, got {cfg.open_loop_horizon}")
    if cfg.policy_num_inference_steps < 0:
        raise ValueError(
            "hitl.policy_num_inference_steps must be 0 (the checkpoint's own value) or more, got "
            f"{cfg.policy_num_inference_steps}"
        )
    if cfg.policy_max_steps < 1:
        raise ValueError(f"hitl.policy_max_steps must be at least 1, got {cfg.policy_max_steps}")
    if cfg.policy_velocity_scale <= 0:
        raise ValueError(f"hitl.policy_velocity_scale must be positive, got {cfg.policy_velocity_scale}")


def check_policy_checkpoint(cfg: HITLConfig) -> None:
    """Reject a checkpoint that is not on THIS machine. Called once, at session start.

    Separate from check_policy_config because it is the half that only the host running the arm can
    answer. Worth doing at startup rather than letting the policy server report it: it surfaces
    minutes and a whole warm-up earlier, before any part of a trajectory has been collected against
    a phase that was never going to run.
    """
    if not cfg.enabled or cfg.policy_type == "human" or not cfg.policy_checkpoint:
        return
    if not Path(cfg.policy_checkpoint).is_dir():
        raise ValueError(f"hitl.policy_checkpoint is not a directory on this machine: {cfg.policy_checkpoint}")


def load_hitl_config(spec: str | None) -> HITLConfig:
    """Resolve the ``hitl`` block from a JSON file path or an inline JSON string.

    Same spelling as ``--curobo-overrides``, so the data-collection server passes it the same way.
    """
    if not spec:
        return HITLConfig()
    path = Path(spec)
    text = path.read_text() if path.exists() else spec
    try:
        raw = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"the hitl config must be a JSON object or a path to one: {exc}") from exc
    return resolve_hitl_config(raw)
