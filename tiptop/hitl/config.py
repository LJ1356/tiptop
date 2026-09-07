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
    return HITLConfig(**dict(raw))


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
