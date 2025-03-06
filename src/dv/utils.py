"""Utility functions"""

# %% [markdown]
# ## Imports

# %%
from collections.abc import Callable
from functools import wraps
from typing import ParamSpec, TypeVar

# %% [markdown]
# ## P = ParamSpec("P"):
# ParamSpec is used to capture ALL parameters of a function
# This includes positional args, keyword args, and their types
# Think of it as a "parameter specification"
# Without it, we couldn't properly type hint functions that preserve the original function
# signature
#
# ## R = TypeVar("R"):
# TypeVar represents a type variable
# In our case, it captures the return type of the decorated function
# It's generic, meaning it can be any type
# When used with ParamSpec, it completes the function's type signature

# %%
P = ParamSpec("P")  # capture all parameters of a function
R = TypeVar("R")  # captures the return type of the decorated function


# %%
def format_docstring(**kwargs: object) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Format function's docstring with provided variables."""

    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        if func.__doc__:
            func.__doc__ = func.__doc__.format(**kwargs)

        @wraps(func)
        def wrapper(*args: P.args, **kw: P.kwargs) -> R:
            return func(*args, **kw)

        return wrapper

    return decorator
