"""Theme constants shared by the presentation generator."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Theme:
    canvas_width: int = 13_333
    canvas_height: int = 7_500
    title_font: str = "Aptos Display"
    body_font: str = "Segoe UI"
    mono_font: str = "Consolas"
    bg_top: tuple[int, int, int] = (8, 14, 32)
    bg_bottom: tuple[int, int, int] = (13, 22, 48)
    rail: tuple[int, int, int] = (18, 28, 58)
    input_blue: tuple[int, int, int] = (82, 167, 255)
    model_purple: tuple[int, int, int] = (152, 112, 255)
    sampling_orange: tuple[int, int, int] = (255, 165, 72)
    shap_green: tuple[int, int, int] = (74, 214, 131)
    error_red: tuple[int, int, int] = (255, 96, 109)
    pale_text: tuple[int, int, int] = (222, 232, 248)
    muted_text: tuple[int, int, int] = (147, 165, 200)
    card_bg: tuple[int, int, int] = (19, 31, 67)
    card_alt: tuple[int, int, int] = (24, 39, 82)
    line: tuple[int, int, int] = (62, 87, 144)
    white: tuple[int, int, int] = (255, 255, 255)
    black: tuple[int, int, int] = (0, 0, 0)


THEME = Theme()


SECTION_TO_COLOR = {
    "Motivation": THEME.input_blue,
    "What Is InstaSHAP": THEME.model_purple,
    "Full Workflow": THEME.sampling_orange,
    "Internal Mechanism": THEME.model_purple,
    "Why Fast": THEME.shap_green,
    "Limitations & Applicability": THEME.error_red,
    "Interactive Simulation": THEME.input_blue,
    "Failure Analysis": THEME.error_red,
    "InstaSHAP 2.0": THEME.shap_green,
}


def rgb_int(color: tuple[int, int, int]) -> int:
    """Convert an RGB tuple into the decimal color PowerPoint expects."""

    red, green, blue = color
    return red + (green << 8) + (blue << 16)
