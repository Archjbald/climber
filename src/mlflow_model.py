"""MLflow pyfunc wrapper packaging the climbing-pose pipeline as a servable model."""

import mlflow.pyfunc
import pandas as pd
from src.pose import get_pose_tracker
from src.main import process_vid


class ClimbingPoseModel(mlflow.pyfunc.PythonModel):
    """MLflow pyfunc model that analyses a climbing video referenced by path."""

    def load_context(self, context) -> None:
        """Instantiate the shared pose tracker when the model is loaded."""
        self.pose_tracker = get_pose_tracker(
            mode="balanced", backend="onnxruntime", device="cpu"
        )

    def predict(self, context, model_input: pd.DataFrame) -> pd.DataFrame:
        """Run the pipeline on the input `video_path` and return the analysis as a DataFrame."""
        video_path = model_input["video_path"][0]
        result = process_vid(video_path, pose_tracker=self.pose_tracker)
        return pd.DataFrame([result])


mlflow.models.set_model(ClimbingPoseModel())
