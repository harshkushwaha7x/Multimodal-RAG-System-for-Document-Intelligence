"""
Experiment Tracking Module.
MLflow integration for tracking experiments and models.
"""

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
import json

from ..utils import get_logger, get_config, LoggerMixin

logger = get_logger(__name__)
config = get_config()


@dataclass
class ExperimentRun:
    """Container for experiment run data."""
    
    run_id: str
    experiment_name: str
    params: Dict[str, Any]
    metrics: Dict[str, float]
    artifacts: List[str] = field(default_factory=list)
    tags: Dict[str, str] = field(default_factory=dict)
    start_time: str = ""
    end_time: str = ""
    status: str = "running"
    
    def to_dict(self) -> Dict:
        return {
            "run_id": self.run_id,
            "experiment_name": self.experiment_name,
            "params": self.params,
            "metrics": self.metrics,
            "artifacts": self.artifacts,
            "tags": self.tags,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "status": self.status
        }


class ExperimentTracker(LoggerMixin):
    """
    MLflow-based experiment tracker.
    
    Features:
    - Automatic experiment creation
    - Parameter and metric logging
    - Model artifact management
    - Run comparison
    """
    
    def __init__(
        self,
        tracking_uri: Optional[str] = None,
        experiment_name: Optional[str] = None,
        use_mlflow: bool = True
    ):
        """
        Initialize experiment tracker.
        
        Args:
            tracking_uri: MLflow tracking server URI
            experiment_name: Default experiment name
            use_mlflow: Whether to use MLflow (fallback to local JSON)
        """
        self.tracking_uri = tracking_uri or config.mlflow.tracking_uri
        self.experiment_name = experiment_name or config.mlflow.experiment_name
        self.use_mlflow = use_mlflow
        
        self._mlflow = None
        self._active_run = None
        self._local_runs: List[ExperimentRun] = []
    
    def _init_mlflow(self):
        """Initialize MLflow."""
        if self._mlflow is not None:
            return
        
        if not self.use_mlflow:
            return
        
        try:
            import mlflow
            self._mlflow = mlflow
            
            mlflow.set_tracking_uri(self.tracking_uri)
            mlflow.set_experiment(self.experiment_name)
            
            self.logger.info(f"MLflow initialized: {self.tracking_uri}")
            
        except ImportError:
            self.logger.warning("MLflow not installed, using local tracking")
            self.use_mlflow = False
    
    def start_run(
        self,
        run_name: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None
    ) -> str:
        """
        Start a new experiment run.
        
        Args:
            run_name: Optional name for the run
            tags: Optional tags for the run
            
        Returns:
            Run ID
        """
        self._init_mlflow()
        
        if self.use_mlflow and self._mlflow:
            run = self._mlflow.start_run(run_name=run_name, tags=tags)
            run_id = run.info.run_id
            self._active_run = run
        else:
            # Local tracking
            run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            self._active_run = ExperimentRun(
                run_id=run_id,
                experiment_name=self.experiment_name,
                params={},
                metrics={},
                tags=tags or {},
                start_time=datetime.now().isoformat()
            )
        
        self.logger.info(f"Started run: {run_id}")
        return run_id
    
    def log_params(self, params: Dict[str, Any]):
        """
        Log parameters.
        
        Args:
            params: Dict of parameter names to values
        """
        if self.use_mlflow and self._mlflow:
            self._mlflow.log_params(params)
        else:
            if isinstance(self._active_run, ExperimentRun):
                self._active_run.params.update(params)
        
        self.logger.debug(f"Logged {len(params)} parameters")
    
    def log_param(self, key: str, value: Any):
        """Log single parameter."""
        self.log_params({key: value})
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """
        Log metrics.
        
        Args:
            metrics: Dict of metric names to values
            step: Optional step number for tracking over time
        """
        if self.use_mlflow and self._mlflow:
            self._mlflow.log_metrics(metrics, step=step)
        else:
            if isinstance(self._active_run, ExperimentRun):
                self._active_run.metrics.update(metrics)
        
        self.logger.debug(f"Logged {len(metrics)} metrics")
    
    def log_metric(self, key: str, value: float, step: Optional[int] = None):
        """Log single metric."""
        self.log_metrics({key: value}, step=step)
    
    def log_artifact(
        self,
        local_path: str,
        artifact_path: Optional[str] = None
    ):
        """
        Log an artifact file.
        
        Args:
            local_path: Path to local file
            artifact_path: Optional subdirectory in artifact store
        """
        if self.use_mlflow and self._mlflow:
            self._mlflow.log_artifact(local_path, artifact_path)
        else:
            if isinstance(self._active_run, ExperimentRun):
                self._active_run.artifacts.append(local_path)
        
        self.logger.debug(f"Logged artifact: {local_path}")
    
    def log_model(
        self,
        model: Any,
        artifact_path: str = "model",
        model_type: str = "pytorch"
    ):
        """
        Log a model artifact.
        
        Args:
            model: Model object to log
            artifact_path: Path in artifact store
            model_type: Type of model ("pytorch", "sklearn", "custom")
        """
        if not self.use_mlflow or not self._mlflow:
            self.logger.warning("Model logging requires MLflow")
            return
        
        if model_type == "pytorch":
            import mlflow.pytorch
            mlflow.pytorch.log_model(model, artifact_path)
        elif model_type == "sklearn":
            import mlflow.sklearn
            mlflow.sklearn.log_model(model, artifact_path)
        elif model_type == "transformers":
            import mlflow.transformers
            mlflow.transformers.log_model(model, artifact_path)
        else:
            # Generic Python model
            import mlflow.pyfunc
            mlflow.pyfunc.log_model(artifact_path, python_model=model)
        
        self.logger.info(f"Logged {model_type} model: {artifact_path}")
    
    def set_tags(self, tags: Dict[str, str]):
        """
        Set run tags.
        
        Args:
            tags: Dict of tag names to values
        """
        if self.use_mlflow and self._mlflow:
            self._mlflow.set_tags(tags)
        else:
            if isinstance(self._active_run, ExperimentRun):
                self._active_run.tags.update(tags)
    
    def end_run(self, status: str = "FINISHED"):
        """
        End current run.
        
        Args:
            status: Run status ("FINISHED", "FAILED", "KILLED")
        """
        if self.use_mlflow and self._mlflow:
            self._mlflow.end_run(status)
        else:
            if isinstance(self._active_run, ExperimentRun):
                self._active_run.end_time = datetime.now().isoformat()
                self._active_run.status = status.lower()
                self._local_runs.append(self._active_run)
        
        self._active_run = None
        self.logger.info(f"Ended run with status: {status}")
    
    def get_run(self, run_id: str) -> Optional[ExperimentRun]:
        """
        Get run by ID.
        
        Args:
            run_id: Run ID
            
        Returns:
            ExperimentRun or None
        """
        if self.use_mlflow and self._mlflow:
            try:
                run = self._mlflow.get_run(run_id)
                return ExperimentRun(
                    run_id=run.info.run_id,
                    experiment_name=self.experiment_name,
                    params=run.data.params,
                    metrics=run.data.metrics,
                    tags=run.data.tags,
                    start_time=str(run.info.start_time),
                    end_time=str(run.info.end_time),
                    status=run.info.status
                )
            except Exception as e:
                self.logger.error(f"Failed to get run: {e}")
                return None
        else:
            for run in self._local_runs:
                if run.run_id == run_id:
                    return run
            return None
    
    def list_runs(
        self,
        max_results: int = 100
    ) -> List[ExperimentRun]:
        """
        List recent runs.
        
        Args:
            max_results: Maximum number of runs to return
            
        Returns:
            List of ExperimentRun objects
        """
        if self.use_mlflow and self._mlflow:
            try:
                runs = self._mlflow.search_runs(
                    experiment_names=[self.experiment_name],
                    max_results=max_results
                )
                
                result = []
                for _, row in runs.iterrows():
                    result.append(ExperimentRun(
                        run_id=row['run_id'],
                        experiment_name=self.experiment_name,
                        params={k.replace('params.', ''): v 
                               for k, v in row.items() if k.startswith('params.')},
                        metrics={k.replace('metrics.', ''): v 
                                for k, v in row.items() if k.startswith('metrics.')},
                        status=row.get('status', 'unknown')
                    ))
                return result
                
            except Exception as e:
                self.logger.error(f"Failed to list runs: {e}")
                return []
        else:
            return self._local_runs[-max_results:]
    
    def compare_runs(
        self,
        run_ids: List[str],
        metric_keys: Optional[List[str]] = None
    ) -> Dict[str, Dict[str, float]]:
        """
        Compare metrics across runs.
        
        Args:
            run_ids: List of run IDs to compare
            metric_keys: Metrics to compare (None = all)
            
        Returns:
            Dict mapping run_id to metrics
        """
        comparison = {}
        
        for run_id in run_ids:
            run = self.get_run(run_id)
            if run:
                if metric_keys:
                    comparison[run_id] = {
                        k: v for k, v in run.metrics.items()
                        if k in metric_keys
                    }
                else:
                    comparison[run_id] = run.metrics
        
        return comparison
    
    def save_local_runs(self, path: Path):
        """Save local run history to file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w') as f:
            json.dump([r.to_dict() for r in self._local_runs], f, indent=2)
        
        self.logger.info(f"Saved {len(self._local_runs)} runs to {path}")
    
    def load_local_runs(self, path: Path):
        """Load local run history from file."""
        path = Path(path)
        
        if path.exists():
            with open(path) as f:
                data = json.load(f)
            
            self._local_runs = [
                ExperimentRun(**run) for run in data
            ]
            
            self.logger.info(f"Loaded {len(self._local_runs)} runs from {path}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Experiment Tracker Test")
    parser.add_argument("--test", action="store_true", help="Run test mode")
    args = parser.parse_args()
    
    if args.test:
        print("Experiment Tracker Test\n" + "=" * 50)
        
        # Initialize tracker without MLflow
        tracker = ExperimentTracker(
            experiment_name="test_experiment",
            use_mlflow=False
        )
        
        # Start a run
        run_id = tracker.start_run(run_name="test_run")
        print(f"Started run: {run_id}")
        
        # Log parameters
        tracker.log_params({
            "model": "all-mpnet-base-v2",
            "embedding_dim": 768,
            "top_k": 10
        })
        
        # Log metrics
        tracker.log_metrics({
            "ndcg@5": 0.78,
            "mrr": 0.82,
            "latency_p50": 45.2
        })
        
        # End run
        tracker.end_run()
        
        # List runs
        runs = tracker.list_runs()
        print(f"\nTotal runs: {len(runs)}")
        
        for run in runs:
            print(f"\nRun: {run.run_id}")
            print(f"  Params: {run.params}")
            print(f"  Metrics: {run.metrics}")
