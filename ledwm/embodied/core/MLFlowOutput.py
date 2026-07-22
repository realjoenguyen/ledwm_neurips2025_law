import collections
import os


class MLFlowOutput:
    def __init__(self, run_name=None, resume_id=None, config=None, prefix=None):
        import mlflow

        self._mlflow = mlflow
        self._prefix = prefix
        self._setup(run_name, resume_id, config)

    def __call__(self, summaries):
        bystep = collections.defaultdict(dict)
        for step, name, value in summaries:
            if len(value.shape) == 0 and self._pattern.search(name):
                name = f"{self._prefix}/{name}" if self._prefix else name
                bystep[step][name] = float(value)
        for step, metrics in bystep.items():
            self._mlflow.log_metrics(metrics, step=step)

    def _setup(self, run_name, resume_id, config):
        tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "local")
        run_name = run_name or os.environ.get("MLFLOW_RUN_NAME")
        resume_id = resume_id or os.environ.get("MLFLOW_RESUME_ID")
        print("MLFlow Tracking URI:", tracking_uri)
        print("MLFlow Run Name:    ", run_name)
        print("MLFlow Resume ID:   ", resume_id)
        if resume_id:
            runs = self._mlflow.search_runs(None, f'tags.resume_id="{resume_id}"')
            assert len(runs), ("No runs to resume found.", resume_id)
            self._mlflow.start_run(run_name=run_name, run_id=runs["run_id"].iloc[0])
            for key, value in config.items():
                self._mlflow.log_param(key, value)
        else:
            tags = {"resume_id": resume_id or ""}
            self._mlflow.start_run(run_name=run_name, tags=tags)
