import json
import logging
from functools import cache
from pathlib import Path
from typing import Sequence

from PIL import Image
from google import genai
from google.genai import types

_log = logging.getLogger(__name__)

_PROMPTS_DIR = Path(__file__).parent / "prompts"


@cache
def load_prompt(prompt_name: str) -> str:
    """Load a prompt template from the prompts directory."""
    return (_PROMPTS_DIR / f"{prompt_name}.txt").read_text().strip()


@cache
def gemini_client() -> genai.Client:
    return genai.Client()


def load_json(response_text: str) -> list | dict:
    """Extract JSON string from code fencing if present."""
    cleaned_text = response_text.strip()
    if cleaned_text.startswith("```json"):
        cleaned_text = cleaned_text.replace("```json", "").replace("```", "")
    elif cleaned_text.startswith("```"):
        cleaned_text = cleaned_text.replace("```", "")

    try:
        results = json.loads(cleaned_text)
    except json.decoder.JSONDecodeError:
        _log.error(f"Invalid JSON: {cleaned_text}")
        raise
    return results


def extra_objects_section(extra_objects: Sequence[str]) -> str:
    """The prompt block asking for objects a plan in progress is waiting for, or nothing.

    Purely additive: detection is otherwise driven by the task instruction alone. It exists because
    an object that only appears part-way through a task -- a block prised out of a jenga tower, say --
    reads as scenery to a detector working from the original instruction, and gets missed on exactly
    the pass that needs it. Naming what to look for is a hint, not an assertion that it is there; a
    detection that comes back is still checked against where it was supposed to end up before
    anything binds to it.
    """
    wanted = [o for o in extra_objects if o]
    if not wanted:
        return ""
    items = "\n".join(f"- {o}" for o in wanted)
    return (
        "\nA step of this task has already been carried out, and it should have left the following "
        "behind. Look for each one and, if you can see it, use EXACTLY the name given here. If one is "
        "not visible, leave it out -- do not label something else with its name.\n" + items + "\n"
    )

def _parse_response(response_text: str) -> tuple[list, list]:
    """Parse Gemini response text into bboxes and grounded atoms."""
    try:
        result = load_json(response_text)
    except Exception:
        raise ValueError(f"Gemini returned a non-JSON response; check for a discrepancy in your image: {response_text}")
    bboxes = result.get("bboxes", [])
    grounded_atoms = [
        {"predicate": spec["name"], "args": spec["args"]}
        for spec in result.get("predicates", [])
        if spec.get("name") and spec.get("args")
    ]
    return bboxes, grounded_atoms


def detect_and_translate(
    image: Image.Image,
    task_instruction: str,
    client: genai.Client | None = None,
    model_id: str = "gemini-robotics-er-1.6-preview",
    temperature: float | None = None,
    extra_objects: Sequence[str] = (),
) -> tuple[list[dict], list[dict]]:
    """Detect objects and translate task in a single Gemini API call.

    Args:
        image: The image to analyze.
        task_instruction: The natural language task to translate.
        client: Gemini API client. If None, a new client will be created.
        model_id: Gemini model ID to use.
        temperature: Temperature for generation.

    Returns:
        Tuple of (bboxes, grounded_atoms) where:
        - bboxes: List of detected objects with bounding boxes
        - grounded_atoms: List of predicate specifications
    """
    client = client or gemini_client()
    prompt = load_prompt("detect_and_translate").format(
        task_instruction=task_instruction, extra_objects=extra_objects_section(extra_objects)
    )
    response = client.models.generate_content(
        model=model_id,
        contents=[image, prompt],
        config=types.GenerateContentConfig(
            temperature=temperature, thinking_config=types.ThinkingConfig(thinking_budget=0)
        ),
    )
    return _parse_response(response.text)


async def detect_and_translate_async(
    image: Image.Image,
    task_instruction: str,
    client: genai.Client | None = None,
    model_id: str = "gemini-robotics-er-1.6-preview",
    temperature: float | None = None,
    extra_objects: Sequence[str] = (),
) -> tuple[list[dict], list[dict]]:
    """Asynchronously detect objects and translate task in a single Gemini API call.

    Args:
        image: The image to analyze.
        task_instruction: The natural language task to translate.
        client: Gemini API client. If None, a new client will be created.
        model_id: Gemini model ID to use.
        temperature: Temperature for generation.

    Returns:
        Tuple of (bboxes, grounded_atoms) where:
        - bboxes: List of detected objects with bounding boxes.
        - grounded_atoms: List of predicate specifications.
    """
    client = client or gemini_client()
    prompt = load_prompt("detect_and_translate").format(
        task_instruction=task_instruction, extra_objects=extra_objects_section(extra_objects)
    )
    response = await client.aio.models.generate_content(
        model=model_id,
        contents=[image, prompt],
        config=types.GenerateContentConfig(
            temperature=temperature, thinking_config=types.ThinkingConfig(thinking_budget=0)
        ),
    )
    return _parse_response(response.text)
