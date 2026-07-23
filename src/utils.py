# src/utils.py

"""
+---------------------------------------------------------------------------+
| UTILITIES                                                                  |
+---------------------------------------------------------------------------+
Global helper functions for project.
"""

# Python Libraries
import time
import random
import numpy as np
import tensorflow as tf
import textwrap
import uuid

# Local Libraries
from src.constants import SECS_IN_MIN, PEP8_LINE_LEN


def get_run_id() -> str:
    """ Generates a unique ID for the current run. """
    return uuid.uuid4().hex[:5].upper()


def start_timer() -> float:
    """
    Start a timer
    """
    return time.time()


def get_time(start_time_float: float) -> str:
    diff = abs(time.time() - start_time_float)
    _, remainder = divmod(diff, SECS_IN_MIN*SECS_IN_MIN)
    minutes, seconds = divmod(remainder, SECS_IN_MIN)
    fractional_seconds = seconds - int(seconds)

    ms = fractional_seconds * 1000
    return f"{int(minutes)}m {int(seconds)}s {int(ms)}ms"


def show_timer(start_time_int: float) -> None:
    print(f"⏰Run Time: {get_time(start_time_int)}\n")


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
    _create_title_banner(title, center_title_text)

    if subtitle:
        _create_subtitle_banner(subtitle, center_subtitle_text)

# @TODO- defunct
def show_banner2(title: str, section: str='') -> None:
    padding = 2
    strlen = len(title) + padding

    # Top line
    print("\n")
    print('+ ', end='')
    print('-' * strlen)
    print('+', end='')

    # Show title
    print('  ' + title)

    print('+ ', end='')
    print('-' * strlen)
    print('+', end='')

    # Show section
    if section:
        print(' ' + section)
        print("\n")

# @TODO - defunct
def seed_script(seed_val: int):
    np.random.seed(seed_val)
    random.seed(seed_val)
    tf.random.set_seed(seed_val)
