# android_world/agents/gui_agent/utils.py
from __future__ import annotations

import json
import re
import time
from typing import Any

import numpy as np

from android_world.env import interface
from android_world.env import json_action
from android_world.env import representation_utils
from android_world.agents.gui_agent.schemas import HistoryStep


def clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def extract_json_object(text: str) -> dict[str, Any]:
    """
    Robustly extracts the first JSON object from model output.
    """

    text = text.strip()

    # Remove markdown fences if the model ignored the JSON-only instruction.
    text = re.sub(r"^```(?:json)?", "", text, flags=re.IGNORECASE).strip()
    text = re.sub(r"```$", "", text).strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    if start < 0:
        raise ValueError(f"No JSON object found in model output:\n{text}")

    depth = 0
    in_string = False
    escaped = False

    for i in range(start, len(text)):
        ch = text[i]

        if escaped:
            escaped = False
            continue

        if ch == "\\":
            escaped = True
            continue

        if ch == '"':
            in_string = not in_string
            continue

        if in_string:
            continue

        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1

            if depth == 0:
                return json.loads(text[start:i + 1])

    raise ValueError(f"Could not extract complete JSON object:\n{text}")


def ui_elements_to_text(
    ui_elements: list[representation_utils.UIElement],
) -> str:
    """
    Converts AndroidWorld UIElement objects to a compact representation.
    """

    rows = []

    for index, element in enumerate(ui_elements):
        bbox = element.bbox_pixels

        bbox_value = None
        if bbox is not None:
            bbox_value = [
                int(bbox.x_min),
                int(bbox.y_min),
                int(bbox.x_max),
                int(bbox.y_max),
            ]

        row = {
            "index": index,
            "text": element.text,
            "content_description": element.content_description,
            "hint_text": element.hint_text,
            "class_name": element.class_name,
            "bbox": bbox_value,
            "clickable": element.is_clickable,
            "editable": element.is_editable,
            "scrollable": element.is_scrollable,
            "enabled": element.is_enabled,
            "visible": element.is_visible,
            "checked": element.is_checked,
            "selected": element.is_selected,
            "package": element.package_name,
        }

        # Keep the prompt smaller.
        row = {
            k: v
            for k, v in row.items()
            if v is not None
        }

        rows.append(json.dumps(row, ensure_ascii=False))

    if not rows:
        return "No accessibility elements available."

    return "\n".join(rows)


def get_action_bbox(
    action: dict[str, Any],
    ui_elements: list[representation_utils.UIElement],
) -> list[int] | None:

    index = action.get("index")

    if index is None:
        return None

    try:
        index = int(index)
    except (TypeError, ValueError):
        return None

    if not 0 <= index < len(ui_elements):
        return None

    bbox = ui_elements[index].bbox_pixels

    if bbox is None:
        return None

    return [
        int(bbox.x_min),
        int(bbox.y_min),
        int(bbox.x_max),
        int(bbox.y_max),
    ]


def deterministic_grounding_score(
    action: dict[str, Any],
    ui_elements: list[representation_utils.UIElement],
) -> float:
    """
    Rule-based grounding score using AndroidWorld accessibility information.

    This is intentionally independent from the VLM.
    """

    action_type = action.get("action_type")

    # Actions that do not target a particular UI node.
    if action_type in {
        "navigate_back",
        "navigate_home",
        "open_app",
        "keyboard_enter",
        "wait",
        "status",
        "answer",
    }:
        return 1.0

    if action_type == "scroll" and action.get("index") is None:
        return 1.0

    index = action.get("index")

    if index is None:
        return 0.0

    try:
        index = int(index)
    except (TypeError, ValueError):
        return 0.0

    if not 0 <= index < len(ui_elements):
        return 0.0

    element = ui_elements[index]

    score = 0.0

    # Valid element exists.
    score += 0.20

    # Visible.
    if element.is_visible is not False:
        score += 0.20

    # Enabled.
    if element.is_enabled is not False:
        score += 0.15

    # Has a usable bounding box.
    if element.bbox_pixels is not None:
        score += 0.20

    # Action-specific compatibility.
    compatible = False

    if action_type == "click":
        compatible = bool(
            element.is_clickable
            or element.is_checkable
            or element.is_focusable
        )

    elif action_type == "long_press":
        compatible = bool(element.is_long_clickable)

    elif action_type == "input_text":
        compatible = bool(element.is_editable)

    elif action_type == "scroll":
        compatible = bool(element.is_scrollable)

    if compatible:
        score += 0.25

    return clamp01(score)


def action_signature(action: dict[str, Any]) -> str:
    """
    Used to measure particle disagreement.
    """

    return json.dumps(
        {
            "action_type": action.get("action_type"),
            "index": action.get("index"),
            "direction": action.get("direction"),
            "app_name": action.get("app_name"),
            "text": action.get("text"),
            "clear_text": action.get("clear_text"),
            "goal_status": action.get("goal_status"),
        },
        sort_keys=True,
        ensure_ascii=False,
    )


def to_android_world_action(
    action: dict[str, Any],
) -> json_action.JSONAction:
    return json_action.JSONAction(**action)


def screen_difference(
    before: np.ndarray,
    after: np.ndarray,
) -> float:
    """
    Cheap normalized screenshot difference.
    """

    if before.shape != after.shape:
        return 1.0

    a = before.astype(np.float32)
    b = after.astype(np.float32)

    return float(np.mean(np.abs(a - b)) / 255.0)
def action_effectiveness_prior(
    action: dict[str, Any],
) -> float:

    action_type = action.get("action_type")

    if action_type == "input_text":
        if action.get("clear_text") is True:
            return 1.0
        return 0.80

    if action_type == "navigate_back":
        return 0.85

    if action_type == "open_app":
        return 0.80

    if action_type == "scroll":
        return 0.65

    if action_type == "click":
        return 0.50

    if action_type == "long_press":
        return 0.45
    
    if action_type == "keyboard_enter":
        return 0.55

    if action_type == "wait":
        return 0.10

    if action_type == "status":
        return 0.50

    return 0.40
def repetition_penalty(
    action: dict[str, Any],
    history: list[HistoryStep],
    window: int = 4,
) -> float:

    if not history:
        return 0.0

    target = action_signature(action)

    recent = history[-window:]

    repeats = sum(
        action_signature(h.action) == target
        for h in recent
    )

    if repeats == 0:
        return 0.0

    if repeats == 1:
        return 0.25

    if repeats == 2:
        return 0.60

    return 1.0


def ui_signature(
    state: interface.State,
) -> tuple:
    values = []

    for e in state.ui_elements:
        values.append(
            (
                e.text,
                e.content_description,
                e.is_checked,
                e.is_selected,
                e.is_visible,
            )
        )

    return tuple(values)


def wait_for_human_transition(
    env: interface.AsyncEnv,
    before_state: interface.State,
    timeout_seconds: float,
    poll_interval_seconds: float,
    pixel_threshold: float = 0.01,
) -> bool:
    """
    Wait for the human to perform exactly one action.

    New behavior:
    1. Pause and wait for the human to press Enter in the terminal.
    2. After confirmation, fetch the new state.
    3. Return whether a meaningful UI/screen change was detected.
    """

    print()
    print("Press ENTER here after you finish the action in the emulator...")
    input()

    current = env.get_state(wait_to_stabilize=True)

    ui_changed = ui_signature(current) != ui_signature(before_state)
    pixel_changed = (
        screen_difference(before_state.pixels, current.pixels) >= pixel_threshold
    )

    return ui_changed or pixel_changed