"""
Module for utility functions used across the banking churn predictor project.

This module provides helper functions for timing, generating unique IDs,
formatting text banners for console output, and formatting model
performance metrics for display.
"""

# Standard Library Imports
import textwrap
import time
import uuid
from typing import List, Union

# Third-party Imports
import pandas as pd

# Local Imports
from src.constants import MSEC, METRIC_COLS, PEP8_LINE_LEN, SECS_IN_MIN


def get_run_id() -> str:
    """
    Generates a unique short ID for the current program run.

    Returns:
        str: A 6-character uppercase hexadecimal string representing the
             unique run ID.
    """
    return uuid.uuid4().hex[:6].upper()


def start_timer() -> float:
    """
    Starts a timer by returning the current time in seconds since the epoch.

    Returns:
        float: The current time as a floating-point number.
    """
    return time.time()


def get_time(start_time_float: float) -> str:
    """
    Calculates the elapsed time since `start_time_float` and returns it
    in a human-readable string format (minutes, seconds, milliseconds).

    Args:
        start_time_float (float): The starting time obtained from `time.time()`.

    Returns:
        str: A formatted string representing the elapsed time (e.g., "0m 1s 234ms").
    """
    diff: float = abs(time.time() - start_time_float)
    _, remainder = divmod(diff, SECS_IN_MIN * SECS_IN_MIN)
    minutes, seconds = divmod(remainder, SECS_IN_MIN)
    fractional_seconds: float = seconds - int(seconds)

    ms: int = int(fractional_seconds * MSEC)
    return f"{int(minutes)}m {int(seconds)}s {ms}ms"


def show_timer(start_time_int: float) -> None:
    """
    Prints the elapsed time since `start_time_int` in a formatted way.

    Args:
        start_time_int (float): The starting time obtained from `start_timer()`.
    """
    print(f"⏰ Run Time: {get_time(start_time_int)}\n")


def _make_top_btm_line() -> str:
    """
    Creates the top/bottom border line for the banner.

    Returns:
        str: A string representing the top or bottom border of the banner.
    """
    open_close_len: int = 2  # open close of char `+` or `|`
    max_line_len: int = PEP8_LINE_LEN - open_close_len

    return "+" + ("-" * max_line_len) + "+"


def _create_title_banner(text: str, center_text: bool = True) -> None:
    """
    Creates and prints the main title banner.

    Args:
        text (str): The title text to display.
        center_text (bool, optional): If True, the text will be centered.
                                      Defaults to True.
    """
    open_close_len: int = 4  # open close of char `+` or `|` plus space
    max_line_len: int = PEP8_LINE_LEN - open_close_len

    # Trim off any chars after limit plus two spaces for blanks
    text = text[0 : max_line_len - open_close_len]
    text_len: int = len(text)
    padding_len: int = max_line_len - text_len

    if center_text:
        # If uneven padding add an extra length for the right side
        extra_len: int = 0 if padding_len % 2 == 0 else 1

        padding_len = padding_len // 2
        title_line: str = (
            "| "
            + (" " * padding_len)
            + text
            + (" " * (padding_len + extra_len))
            + " |"
        )

    else:
        # Remove last two characters to account for open/close spacing
        title_line = "| " + text + (" " * padding_len) + " |"

    top_btm_line: str = _make_top_btm_line()

    # Print title banner
    print("\n")
    print(top_btm_line)
    print(title_line)
    print(top_btm_line)


def _create_subtitle_banner(
    text: Union[str, List[str]], center_text: bool = False
) -> None:
    """
    Creates and prints the subtitle banner.

    Args:
        text (Union[str, List[str]]): The subtitle text(s) to display.
                                      Can be a single string or a list of
                                      strings.
        center_text (bool, optional): If True, the text will be centered.
                                      Defaults to False.
    """
    # Reconstructs the guard to safely catch wrong types OR empty values
    if not isinstance(text, (str, list)) or not text:
        return None

    open_close_len: int = 4  # open close of char `+` or `|` plus space
    max_line_len: int = PEP8_LINE_LEN - open_close_len
    wrapped_lines: List[str] = _get_wrapped_lines(text, max_line_len)

    # Now that the data is a list format it for display.
    for line in wrapped_lines:
        # Clean up any rogue newline markers so they don't break string
        # length math
        line = line.replace("\n", " ").strip()
        line_len: int = len(line)
        padding_len: int = max_line_len - line_len

        if center_text:
            extra_len: int = 0 if padding_len % 2 == 0 else 1
            padding_len = padding_len // 2
            padded_line: str = (
                "| "
                + (" " * padding_len)
                + line
                + (" " * (padding_len + extra_len))
                + " |"
            )
        else:
            padded_line = "| " + line + (" " * padding_len) + " |"

        print(padded_line)

    # Close the subtitle
    if len(wrapped_lines) > 0:
        print(_make_top_btm_line())

    return None


def _get_wrapped_lines(text: Union[str, List[str]], max_line_len: int) -> List[str]:
    """
    Wraps text lines to fit within a specified maximum line length.

    Args:
        text (Union[str, List[str]]): The text to wrap. Can be a single
                                      string or a list of strings.
        max_line_len (int): The maximum length for each wrapped line.

    Returns:
        List[str]: A list of strings, where each string is a wrapped line.
    """
    wrapped_lines: List[str] = []

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


def show_banner(
    title: str,
    subtitle: Union[str, List[str], None] = "",
    center_title_text: bool = True,
    center_subtitle_text: bool = False,
) -> None:
    """
    Displays a formatted banner with a main title and an optional subtitle.

    Args:
        title (str): The main title for the banner.
        subtitle (Union[str, List[str], None], optional): The subtitle(s)
                                                          to display below
                                                          the title. Can be
                                                          a string, a list
                                                          of strings, or None.
                                                          Defaults to "".
        center_title_text (bool, optional): If True, the title text will be
                                            centered. Defaults to True.
        center_subtitle_text (bool, optional): If True, the subtitle text
                                               will be centered. Defaults
                                               to False.
    """
    _create_title_banner(title, center_title_text)

    if subtitle:
        _create_subtitle_banner(subtitle, center_subtitle_text)


def format_performance(df: pd.DataFrame) -> List[str]:
    """
    Formats model performance metrics from a DataFrame into a list of
    strings suitable for display in text banners.

    This function dynamically renders all available metrics
    (e.g., Accuracy, Precision, Recall, F1, AUC).

    Args:
        df (pd.DataFrame): A DataFrame containing model performance metrics.
                           It can be a single model's metrics (metrics as
                           index) or a comparison matrix (models as columns).

    Returns:
        List[str]: A list of formatted strings, each representing a line
                   of the performance report.
    """
    lines: List[str] = []

    # Transpose so models become the loop rows and metrics become columns
    flipped_matrix: pd.DataFrame = df.T
    separator: str = "-" * (PEP8_LINE_LEN - 4)

    # Iterate through each model row
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
                display_name: str = str(metric).replace("_", " ").title()
                lines.append(f" {display_name:<10}: {value:.4f}")

            lines.append(separator)

    return lines