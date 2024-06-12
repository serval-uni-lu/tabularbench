import json
import os
import time
import uuid
from pathlib import Path

DEFAULT_PROJECT_NAME = "project_name"
DEFAULT_PROJECT_WORKSPACE = "project_workspace"


class XP:
    def __init__(
        self,
        args,
        project_name=DEFAULT_PROJECT_NAME,
        workspace=DEFAULT_PROJECT_WORKSPACE,
    ):
        # Add a timestamp
        timestamp = time.time()
        args["timestamp"] = timestamp
        args["uuid"] = str(uuid.uuid4())[:8]

        self.experiment_name = "tabularbench_{}_{}_{}_{}_{}".format(
            args.get("model_name", ""),
            args.get("dataset_name", ""),
            args.get("attack_name", ""),
            timestamp,
            args.get("uuid", ""),
        )
        self.parameters = {}
        self.metrics = {}

        self.log_parameters(**args)

        self.workspace = workspace
        self.project_name = project_name
        self.path = os.path.join(
            os.getenv("XP_ROOT", "./data/xp"),
            self.workspace,
            self.project_name,
        )

    def log_parameters(self, *args, **kwargs):
        self.parameters.update(kwargs)

    def log_metrics(self, *args, **kwargs):
        self.metrics.update(kwargs)

    def get_name(self):
        return self.experiment_name + "_"

    def log_metric(self, name, value, **kwargs):
        self.log_metrics(**{name: value})

    def log_model(self, name, path):
        pass

    def log_asset(self, name, path):
        pass

    def save_json(self):
        Path(self.path).mkdir(parents=True, exist_ok=True)
        with open(
            os.path.join(self.path, self.experiment_name + ".json"), "w"
        ) as f:
            json.dump(
                {
                    "parameters": self.parameters,
                    "metrics": self.metrics,
                },
                f,
                indent=4,
            )

    def end(self):
        self.save_json()

    def flush(self):
        self.save_json()
