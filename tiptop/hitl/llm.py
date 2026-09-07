"""Gemini calls that must come back as valid, well-formed JSON.

Two things are layered over the raw SDK call. Structured output (``response_schema``) fixes the SHAPE
of the reply, so nothing here has to cope with prose around a JSON blob. A reprompt loop fixes its
CONTENT: the parser raises ``HITLProposalError`` with a message written for the model, that message
is handed back, and the model gets another go. Almost every proposal failure in practice is semantic
-- a predicate applied to the wrong number of objects, a parameter that no precondition binds -- and
those are exactly the ones a second attempt fixes.
"""

import json
import logging
from typing import Any, Callable, TypeVar

from google.genai import types
from PIL import Image

from tiptop.hitl.cache import ProposalCache
from tiptop.hitl.record import active_recorder
from tiptop.hitl.structs import HITLProposalError
from tiptop.perception.gemini import gemini_client

_log = logging.getLogger(__name__)

_T = TypeVar("_T")

_REPROMPT = """\
Your previous response was rejected.

Your response was:
{response}

The problem was:
{error}

Try again, fixing exactly that problem and keeping everything else that was correct."""


async def query_json(
    prompt: str,
    parse: Callable[[Any], _T],
    *,
    model: str,
    schema: dict,
    image: Image.Image | None = None,
    max_attempts: int = 3,
    temperature: float | None = None,
    label: str = "proposal",
    cache: ProposalCache | None = None,
) -> _T:
    """Ask Gemini for JSON matching ``schema`` and parse it, reprompting when ``parse`` objects.

    ``parse`` receives the decoded JSON and either returns a value or raises ``HITLProposalError``
    with a message aimed at the model. The last error is re-raised once the attempts run out, so the
    caller sees why the proposal could not be used rather than a bare failure.

    ``cache`` is consulted and written only for responses that PARSED and VALIDATED, so a rejected
    proposal is never replayed. See cache.ProposalCache for why grounding never passes one.
    """
    if cache is not None:
        cached = cache.get(model, prompt, image)
        if cached is not None:
            try:
                parsed = parse(json.loads(cached))
                _log.info(f"HITL {label}: reusing the cached response")
                return parsed
            except (HITLProposalError, json.JSONDecodeError) as exc:
                # The validator has changed since the entry was written; ask again rather than fail.
                _log.info(f"HITL {label}: cached response no longer validates ({exc}); re-querying")

    client = gemini_client()
    config = types.GenerateContentConfig(
        temperature=temperature,
        response_mime_type="application/json",
        response_schema=schema,
    )
    attempt_prompt = prompt
    last_error: HITLProposalError | None = None
    for attempt in range(1, max_attempts + 1):
        contents: list = [image, attempt_prompt] if image is not None else [attempt_prompt]
        response = await client.aio.models.generate_content(model=model, contents=contents, config=config)
        text = (response.text or "").strip()
        recorder = active_recorder()
        try:
            if not text:
                raise HITLProposalError("The response was empty. Respond with the JSON object that was asked for.")
            try:
                data = json.loads(text)
            except json.JSONDecodeError as exc:
                raise HITLProposalError(f"The response was not valid JSON: {exc}") from exc
            parsed = parse(data)
            if recorder is not None:
                recorder.record(
                    label=label, attempt=attempt, model=model, prompt=attempt_prompt, response=text, image=image
                )
            if attempt > 1:
                _log.info(f"HITL {label}: accepted on attempt {attempt}")
            if cache is not None:
                cache.put(model, prompt, image, text)
            return parsed
        except HITLProposalError as exc:
            last_error = exc
            _log.warning(f"HITL {label} attempt {attempt}/{max_attempts} rejected: {exc}")
            if recorder is not None:
                # Recorded too, and marked as rejected: a proposal that had to be corrected is the
                # one worth looking at afterwards, and keeping only the accepted answer hides it.
                recorder.record(
                    label=label, attempt=attempt, model=model, prompt=attempt_prompt,
                    response=text, image=image, rejected=str(exc),
                )
            attempt_prompt = f"{prompt}\n\n{_REPROMPT.format(response=text, error=exc)}"

    assert last_error is not None
    raise last_error
