import sys
from datetime import (
    datetime,
    timedelta,
    timezone,
    tzinfo,
)
from typing import Union

from loguru import logger as log


DEFAULT_LOGURU_FORMAT = (
    "<g>{time:YYYY-MM-DD HH:mm:ss.SSS}</> | "
    "<lvl>{level: <8}</lvl> | "
    "<m>{extra[classname]: <4}</> | "
    "<c>{name}</>:<c>{function}</>:<c>{line}</> - <lvl>{message}</lvl>"
)
KST = timezone(timedelta(hours=9))
TIME_FORMAT = "%Y-%m-%d %H:%M:%S"


def now(tz: tzinfo = KST) -> datetime:
    """Return the current time in KST"""
    return datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S")


def set_logger(
    source: str,  # Specific to Rabbit
    level: Union[str, int] = "INFO",
    backtrace: bool = True,
    diagnose: bool = False,
    to_file: bool = False,
):
    """Call `set_logger` once to set `loguru` logging at the top of the main script.
    Usages:
        ```
        # the main script
        from loguru import logger as log  # It should be imported at every module
        from log import set_logger # Once
        set_logger(diagnose=True, to_file=True) # Once

        # Normal usage
        log.debug("This is debugging")

        # Class specific log
        class Test:
            def __init__(self):
                self.log = log.bind(classname=self.__class__.__name__)
                ...
            def test_method(self, msg):
                ...
                self.log.info("test_method is executed")
                ...
        ```

    Args:
        source (str): Add source name to slack logging.
        level (str | int, optional): Set log level of `stdout`. Defaults to "INFO".
        backtrace (bool, optional): Whether the exception trace formatted should be extended upward.
            Defaults to True.
        diagnose (bool, optional): Whether the exception trace should display the variables values
            to eases the debugging. Defaults to False.
        to_file (bool, optional): Whether to send log messages (DEBUG) to `rabbit_{now()}.log`.
            The logs in multiprocessing is sent to file. Defaults to False.
    """
    log.remove()
    log.configure(extra={"classname": "None", "source": source})
    log.add(
        sys.stdout,
        format=DEFAULT_LOGURU_FORMAT,
        level=level,
        backtrace=backtrace,
        diagnose=diagnose,
    )
    if to_file:
        filename = f"chameleon_{now()}.log"
        log.add(filename, format=DEFAULT_LOGURU_FORMAT, level="DEBUG")
