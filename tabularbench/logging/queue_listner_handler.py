import atexit
from logging import Handler, LogRecord
from logging.config import ConvertingDict, ConvertingList, valid_ident
from logging.handlers import QueueHandler, QueueListener
from multiprocessing import Manager
from queue import Queue
from typing import Any, List, Optional, Union


def _resolve_handlers(handlers: List[Handler]) -> List[Handler]:
    if not isinstance(handlers, ConvertingList):
        return handlers

    # Indexing the list performs the evaluation.
    return [handlers[i] for i in range(len(handlers))]


def _resolve_queue(q: "Union[ConvertingDict, Queue[Any]]") -> "Queue[Any]":
    if not isinstance(q, ConvertingDict):
        return q
    if "__resolved_value__" in q:
        return q["__resolved_value__"]

    cname = q.pop("class")
    klass = q.configurator.resolve(cname)
    props = q.pop(".", None)
    kwargs = {k: q[k] for k in q if valid_ident(k)}
    result = klass(**kwargs)
    if props:
        for name, value in props.items():
            setattr(result, name, value)

    q["__resolved_value__"] = result
    return result


class QueueListenerHandler(QueueHandler):
    def __init__(
        self,
        handlers: List[Handler],
        respect_handler_level: bool = False,
        auto_run: bool = True,
        queue: "Optional[Queue[Any]]" = None,
    ) -> None:

        if queue is None:
            m = Manager()
            q = m.Queue(-1)
            # print(type(q))
        else:
            q = queue

        q = _resolve_queue(q)
        super().__init__(q)
        handlers = _resolve_handlers(handlers)
        self._listener = QueueListener(
            self.queue, *handlers, respect_handler_level=respect_handler_level
        )
        if auto_run:
            self.start()
            atexit.register(self.stop)

    def start(self) -> None:
        self._listener.start()

    def stop(self) -> None:
        self._listener.stop()

    def emit(self, record: LogRecord) -> None:
        return super().emit(record)
