"""Shared slide specification models and helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


DECK_CORE = "core"
DECK_FAILURE = "failure"
DECK_FUTURE = "future"


@dataclass(slots=True)
class SlideSpec:
    """One fully planned slide in the presentation system."""

    deck: str
    number: int
    section: str
    title: str
    subtitle: str
    accent: str
    layout: str
    bullets: list[str]
    visual_title: str
    visual_description: str
    animation_notes: list[str]
    interaction_notes: list[str]
    payload: dict[str, Any] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)


def slide(
    *,
    deck: str,
    number: int,
    section: str,
    title: str,
    subtitle: str,
    accent: str,
    layout: str,
    bullets: list[str],
    visual_title: str,
    visual_description: str,
    animation_notes: list[str],
    interaction_notes: list[str],
    payload: dict[str, Any] | None = None,
    tags: list[str] | None = None,
) -> SlideSpec:
    """Compact constructor used by the deck manifests."""

    return SlideSpec(
        deck=deck,
        number=number,
        section=section,
        title=title,
        subtitle=subtitle,
        accent=accent,
        layout=layout,
        bullets=bullets,
        visual_title=visual_title,
        visual_description=visual_description,
        animation_notes=animation_notes,
        interaction_notes=interaction_notes,
        payload=payload or {},
        tags=tags or [],
    )


ACCENTS = {
    "input": "#3A86FF",
    "model": "#7B61FF",
    "sampling": "#FF9F1C",
    "shap": "#2DC653",
    "failure": "#E63946",
    "neutral": "#A8B3CF",
    "future": "#00B4D8",
}


SECTION_LABELS = {
    DECK_CORE: "Core Workflow",
    DECK_FAILURE: "Failure Analysis",
    DECK_FUTURE: "InstaSHAP 2.0",
}
