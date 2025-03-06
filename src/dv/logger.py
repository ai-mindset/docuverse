"""Set up and use a logging system with configurable levels and console output."""

# %%
import logging
from typing import Literal

# %% [markdown]
# ## Logging
# [logging levels](https://docs.python.org/3/library/logging.html#logging-levels)
# | Level | Value | Description |
# |---------|--------|-------------|
# | `logging.NOTSET` | 0 | When set on a logger, indicates that ancestor loggers are to be consulted to determine the effective level. If that still resolves to NOTSET, then all events are logged. When set on a handler, all events are handled. |
# | `logging.DEBUG` | 10 | Detailed information, typically only of interest to a developer trying to diagnose a problem. |
# | `logging.INFO` | 20 | Confirmation that things are working as expected. |
# | `logging.WARNING` | 30 | An indication that something unexpected happened, or that a problem might occur in the near future (e.g. 'disk space low'). The software is still working as expected. |
# | `logging.ERROR` | 40 | Due to a more serious problem, the software has not been able to perform some function. |
# | `logging.CRITICAL` | 50 | A serious error, indicating that the program itself may be unable to continue running. |


# %%
def setup_logging(
    name: str | None = None,
    level: str | Literal[0, 10, 20, 30, 40, 50] = "INFO",
    format_string: str | None = None,
) -> logging.Logger:
    """Set up a logger with specified level and console output.

    If a logger with the given name already exists, it will be returned without modification.
    If format_string is not provided, a default concise format will be used.

    Args:
        name: The name of the logger. If None, uses the module name.
        level: The logging level as a string or integer value. Default is 'INFO'.
        format_string: Custom format string for log messages. If None, uses default format.

    Returns:
        logging.Logger: A configured logger object with console handler and specified format.

    Examples:
        >>> setup_logging(level="DEBUG").info("This is a debug message")
    """
    # Use module name if name not provided
    if name is None:
        name = __name__

    # Get or create logger (getLogger will return existing logger if it exists)
    logger = logging.getLogger(name)

    # Only configure the logger if it doesn't have handlers yet
    if not logger.handlers:
        # Set the overall logging level of the logger
        if isinstance(level, str):
            level = getattr(logging, level.upper())
        logger.setLevel(level)

        # Create a console handler (outputs to terminal)
        ch = logging.StreamHandler()
        ch.setLevel(level)

        # Use provided format or default concise format
        if format_string is None:
            format_string = "%(asctime)s - %(module)s - %(levelname)s - %(message)s"

        # Create a formatter that specifies the format of log messages
        formatter = logging.Formatter(format_string)

        # Attach the formatter to the handler
        ch.setFormatter(formatter)

        # Add the handler to the logger
        logger.addHandler(ch)

    return logger


if __name__ == "__main__":
    # Set up the logger with INFO level
    logger = setup_logging(level=logging.INFO)

    # Log messages at different levels
    logger.debug("This is a debug message.")
    logger.info("This is an info message.")
    logger.warning("This is a warning message.")
    logger.error("This is an error message.")
    logger.critical("This is a critical message.")

    # Demonstrate that getting the same logger doesn't create duplicate handlers
    same_logger = setup_logging(name=__name__, level=logging.DEBUG)
    same_logger.debug("This debug message should appear if level was properly changed.")
    print(f"Number of handlers: {len(same_logger.handlers)}")  # Should be 1
