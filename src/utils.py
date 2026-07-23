# src/utils.py

"""
+---------------------------------------------------------------------------+
| UTILITIES                                                                  |
+---------------------------------------------------------------------------+
Global helper functions for project.
"""

# Python Libraries
import time
import pandas as pd
import textwrap
import uuid

# Local Libraries
from src.constants import MSEC, SECS_IN_MIN, PEP8_LINE_LEN, METRIC_COLS


def get_run_id() -> str:
    """ Generates a unique ID for the current run. """
    return uuid.uuid4().hex[:6].upper()


def start_timer() -> float:
    """
    Start a timer
    """
    return time.time()


def get_time(start_time_float: float) -> str:
    """
    Get time in string format.
    :param start_time_float:
    :return:
    """
    diff = abs(time.time() - start_time_float)
    _, remainder = divmod(diff, SECS_IN_MIN*SECS_IN_MIN)
    minutes, seconds = divmod(remainder, SECS_IN_MIN)
    fractional_seconds = seconds - int(seconds)

    ms = fractional_seconds * MSEC
    return f"{int(minutes)}m {int(seconds)}s {int(ms)}ms"


def show_timer(start_time_int: float) -> None:
    print(f"⏰ Run Time: {get_time(start_time_int)}\n")


def _make_top_btm_line() -> str:
    open_close_len = 2 # open close of char `+` or `|`
    max_line_len = PEP8_LINE_LEN - open_close_len

    return '+' + ('-' * max_line_len) + '+'


def _create_title_banner(text: str, center_text: bool=True) -> None:
    """
    Creates the Title Banner
    :param text:
    :param center_text:
    :return:
    """
    open_close_len = 4 # open close of char `+` or `|`
    max_line_len = PEP8_LINE_LEN - open_close_len

    # Trim off any chars after limit plus two spaces for blanks
    text = text[0: max_line_len - open_close_len]
    text_len = len(text)
    padding_len = max_line_len - text_len

    if center_text:
        # If uneven padding add an extra length for the right side
        extra_len = 0 if padding_len % 2 == 0 else 1

        padding_len = padding_len // 2
        title_line = "| " + (' ' * padding_len) + text + (' ' * (padding_len + extra_len)) + " |"

    else:
        # Remove last two characters to account for open/close spacing
        title_line = "| " + text + (' ' * padding_len) + " |"

    top_btm_line = _make_top_btm_line()

    # Print title banner
    print("\n")
    print(top_btm_line)
    print(title_line)
    print(top_btm_line)


def _create_subtitle_banner(text: str | list, center_text: bool=False) -> None:
    """
    Creates the Subtitle Banner
    :param text:
    :param center_text:
    :return:
    """

    # Reconstructs the guard to safely catch wrong types OR empty values
    if not isinstance(text, (str, list)) or not text:
        return None

    open_close_len = 4 # open close of char `+` or `|` plus space
    max_line_len = PEP8_LINE_LEN - open_close_len
    wrapped_lines = _get_wrapped_lines(text, max_line_len)

    # Now that the data is a list format it for display.
    for line in wrapped_lines:
        # Clean up any rogue newline markers so they don't break string length math
        line = line.replace("\n", " ").strip()
        line_len = len(line)
        padding_len = max_line_len - line_len

        if center_text:
            extra_len = 0 if padding_len % 2 == 0 else 1
            padding_len = padding_len // 2
            padded_line = "| " + (' ' * padding_len) + line + (' ' * (padding_len + extra_len)) + " |"
        else:
            padded_line = "| " + line + (' ' * padding_len) + " |"

        print(padded_line)

    # Close the subtitle
    if len(wrapped_lines) > 0:
        print(_make_top_btm_line())

    return None


def _get_wrapped_lines(text: str | list, max_line_len: int) -> list:
    """
    Gets all wrapped lines and formats them accordingly.

    :param text:
    :param max_line_len:
    :return:
    """
    wrapped_lines = []

    if isinstance(text, list):
        # Explicitly wrap each individual item inside the list
        for item in text:
            if isinstance(item, str):
                wrapped_lines.extend(textwrap.wrap(item, width=max_line_len))
            else:
                wrapped_lines.append(str(item))

    elif isinstance(text, str):
        wrapped_lines = textwrap.wrap(text, width=max_line_len)

    return wrapped_lines


def show_banner(title: str, subtitle: str | list | None="", center_title_text: bool=True, center_subtitle_text: bool=False) -> None:
    """
    Shows Banner in formatted style.
    :param title:
    :param subtitle:
    :param center_title_text:
    :param center_subtitle_text:
    :return:
    """
    _create_title_banner(title, center_title_text)

    if subtitle:
        _create_subtitle_banner(subtitle, center_subtitle_text)


def format_performance(df: pd.DataFrame) -> list:
    """
    Formats model performances vertically for clean rendering inside text banners.
    Dynamically renders all available metrics (Accuracy, Precision, Recall, F1, AUC, etc.)
    """
    lines = []

    # 1. Transpose so models become the loop rows and metrics become columns
    flipped_matrix = df.T
    separator = "-" * (PEP8_LINE_LEN - 4)

    # 2. Iterate through each model row
    for model_title, row_data in flipped_matrix.iterrows():
        if str(model_title) in METRIC_COLS:
            # Single model layout (Metrics as Index)
            lines.append(f" {str(model_title).title():<10}: {row_data.iloc[0]:.4f}")

        else:
            # Multi-model comparison layout
            lines.append(f"Model: {model_title}")
            
            # Iterate through all available metrics in the row dynamically
            for metric, value in row_data.items():
                # Clean up naming: e.g. 'f1_score' -> 'F1 Score'
                display_name = str(metric).replace('_', ' ').title()
                lines.append(f" {display_name:<10}: {value:.4f}")
            
            lines.append(separator) 

    return lines
