"""Save every image sent to the VLM, and what it said about it, beside the rollout.

Each query writes three things: the image exactly as it was sent, a rendered PNG pairing that image
with the response, and a line in ``index.jsonl`` carrying the full prompt and reply. The rendered PNG
is the one to open -- a run's worth of them scrolls past as "here is what the model saw, here is what
it decided", which is the only practical way to tell a bad classifier wording from a bad camera view.

Every ATTEMPT is recorded, rejected ones included. A proposal that had to be reprompted is exactly
the case worth looking at, and it is invisible if only the accepted answer is kept.
"""

import json
import logging
import re
import textwrap
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

_log = logging.getLogger(__name__)

# Set per rollout by tiptop_run, so a query does not have to be handed a directory through five
# layers of call. One rollout runs at a time in this process, which is what makes that safe.
_active: "VLMRecorder | None" = None

_PANEL_WIDTH = 900
_MARGIN = 16
_LINE_SPACING = 4
_BACKGROUND = (250, 250, 250)
_INK = (24, 24, 24)
_MUTED = (110, 110, 110)
_RULE = (200, 200, 200)


def _font(size: int) -> ImageFont.ImageFont:
    """A readable font, falling back to Pillow's built-in when no TTF is installed."""
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ):
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    try:
        return ImageFont.load_default(size=size)  # Pillow >= 10.1
    except TypeError:
        return ImageFont.load_default()


def _slug(text: str) -> str:
    """A filename-safe version of a query label."""
    return re.sub(r"[^A-Za-z0-9._-]+", "-", text).strip("-")[:60] or "query"


def _pretty(text: str) -> str:
    """Pretty-print a JSON response; leave anything else alone."""
    try:
        return json.dumps(json.loads(text), indent=2)
    except (json.JSONDecodeError, TypeError):
        return text


def _wrap(text: str, columns: int = 100) -> list[str]:
    lines: list[str] = []
    for raw in text.splitlines() or [""]:
        lines.extend(textwrap.wrap(raw, columns) or [""])
    return lines


class VLMRecorder:
    """Writes the image/response pair for each VLM query into one directory."""

    def __init__(self, directory: Path) -> None:
        self._dir = Path(directory)
        self._seq = 0

    def record(
        self,
        *,
        label: str,
        attempt: int,
        model: str,
        prompt: str,
        response: str,
        image: Image.Image | None,
        rejected: str | None = None,
    ) -> None:
        """Save one query. Never raises: an audit trail is not worth failing a rollout over."""
        try:
            self._seq += 1
            self._dir.mkdir(parents=True, exist_ok=True)
            stem = f"{self._seq:03d}_{_slug(label)}"
            if attempt > 1:
                stem += f"_attempt{attempt}"

            input_path = output_path = None
            if image is not None:
                input_path = self._dir / f"{stem}_input.png"
                image.save(input_path)
            output_path = self._dir / f"{stem}_output.png"
            self._render(label, attempt, model, response, image, rejected).save(output_path)

            with (self._dir / "index.jsonl").open("a") as f:
                f.write(json.dumps({
                    "seq": self._seq,
                    "label": label,
                    "attempt": attempt,
                    "model": model,
                    "input_image": input_path.name if input_path else None,
                    "output_image": output_path.name,
                    "rejected": rejected,
                    "prompt": prompt,
                    "response": response,
                }) + "\n")
        except Exception:
            _log.exception("HITL: could not record a VLM query; continuing")

    def _render(
        self,
        label: str,
        attempt: int,
        model: str,
        response: str,
        image: Image.Image | None,
        rejected: str | None,
    ) -> Image.Image:
        """The image the model saw, above what it answered."""
        title = _font(19)
        small = _font(14)
        body = _font(14)

        thumbnail = None
        if image is not None:
            thumbnail = image.copy()
            thumbnail.thumbnail((_PANEL_WIDTH - 2 * _MARGIN, 520))

        header = f"{label}" + (f"   (attempt {attempt})" if attempt > 1 else "")
        subtitle = model + ("   REJECTED" if rejected else "")
        lines = _wrap(_pretty(response))
        if rejected:
            lines = ["Rejected: " + rejected, ""] + lines

        line_height = body.getbbox("Ay")[3] + _LINE_SPACING
        height = _MARGIN
        height += title.getbbox("Ay")[3] + 6 + small.getbbox("Ay")[3] + _MARGIN
        if thumbnail is not None:
            height += thumbnail.height + _MARGIN
        height += _MARGIN + len(lines) * line_height + _MARGIN

        canvas = Image.new("RGB", (_PANEL_WIDTH, height), _BACKGROUND)
        draw = ImageDraw.Draw(canvas)
        y = _MARGIN
        draw.text((_MARGIN, y), header, font=title, fill=_INK)
        y += title.getbbox("Ay")[3] + 6
        draw.text((_MARGIN, y), subtitle, font=small, fill=(180, 60, 60) if rejected else _MUTED)
        y += small.getbbox("Ay")[3] + _MARGIN
        if thumbnail is not None:
            canvas.paste(thumbnail, (_MARGIN, y))
            y += thumbnail.height + _MARGIN
        draw.line([(_MARGIN, y), (_PANEL_WIDTH - _MARGIN, y)], fill=_RULE)
        y += _MARGIN
        for line in lines:
            draw.text((_MARGIN, y), line, font=body, fill=_INK)
            y += line_height
        return canvas


def set_recorder(directory: Path | None) -> None:
    """Start recording VLM queries into ``directory``, or stop recording when None."""
    global _active
    _active = VLMRecorder(directory) if directory is not None else None


def active_recorder() -> "VLMRecorder | None":
    return _active
