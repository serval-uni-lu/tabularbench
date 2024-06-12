from pymoo.core.callback import Callback


class HistoryCallback(Callback):
    def __init__(self) -> None:
        super().__init__()
        self.data["F"] = []

    def notify(self, algorithm, **kwargs):
        self.data["F"].append(algorithm.pop.get("F"))
