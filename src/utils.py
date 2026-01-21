# utils.py

# =================================
#  HELPER FUNCTIONS
# ==================================

import time
import random
import numpy as np
import tensorflow as tf

from src.constants import SECS_IN_MIN

def start_timer() -> float:
    """
    Start a timer
    """
    return time.time()


def get_time(start_time_float: float) -> str:
    diff = abs(time.time() - start_time_float)
    hours, remainder = divmod(diff, SECS_IN_MIN*SECS_IN_MIN)
    minutes, seconds = divmod(remainder, SECS_IN_MIN)
    fractional_seconds = seconds - int(seconds)

    ms = fractional_seconds * 1000
    return f"{int(minutes)}m {int(seconds)}s {int(ms)}ms"


def show_timer(start_time_int: float) -> None:
    print(f"Run Time: {get_time(start_time_int)}")


def show_banner(title: str, section: str='') -> None:
    padding = 2
    strlen = len(title) + padding

    # Top line
    print("\n")
    print('# ', end='')
    print('=' * strlen)
    print('#', end='')

    # Show title
    print('  ' + title)

    print('# ', end='')
    print('=' * strlen)
    print('#', end='')

    # Show section
    if section:
        print(' ' + section)
        print("\n")

def seed_script(seed_val: int):
    np.random.seed(seed_val)
    random.seed(seed_val)
    tf.random.set_seed(seed_val)
