import copy
import logging.config
import logging.handlers
import pathlib
from queue import Queue, SimpleQueue
from typing import Any, Callable, Dict, Union

import yaml
from joblib import delayed

global_logger_config = {}


def setup_logging(
    config_path: str,
) -> None:
    config_file = pathlib.Path(config_path)
    with open(config_file) as f_in:
        config = yaml.safe_load(f_in)
    global_logger_config.update(config)
    logging.config.dictConfig(global_logger_config)


def delayed_with_logging(func: Callable[..., Any]) -> Callable[..., Any]:
    # filename = get_current_logfile()

    root_logger = logging.getLogger()
    root_logger_handlers = root_logger.handlers
    root_logger_level = root_logger.level
    logger_config = copy.deepcopy(global_logger_config)
    if (len(root_logger_handlers) > 0) and isinstance(
        root_logger_handlers[0], logging.handlers.QueueHandler
    ):
        queue = root_logger_handlers[0].queue

        def func_with_logging(*args: Any, **kwargs: Any) -> Callable[..., Any]:
            complete_setup_worker_logging(
                logger_config, queue, root_logger_level
            )
            return func(*args, **kwargs)

        return delayed(func_with_logging)

    return delayed(func)


def setup_worker_logging(
    queue: "Union[SimpleQueue[Any],Queue[Any]]", level: Union[int, str]
) -> None:
    worker_logger = logging.getLogger()
    if not worker_logger.hasHandlers():
        h = logging.handlers.QueueHandler(queue)
        worker_logger.addHandler(h)
        worker_logger.setLevel(level)


def set_levels_config(config: Dict[str, Any]) -> None:
    loggers = config["loggers"]
    for logger_name, logger_conf in loggers.items():
        level = logger_conf.get("level")
        if level is not None:
            logger = logging.getLogger(logger_name)
            logger.setLevel(level)


def complete_setup_worker_logging(
    logger_config: Dict[str, Any],
    queue: "Union[SimpleQueue[Any],Queue[Any]]",
    level: Union[int, str],
) -> None:
    setup_worker_logging(queue, level)
    set_levels_config(logger_config)
