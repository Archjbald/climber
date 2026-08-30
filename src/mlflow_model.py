import mlflow.pyfunc
import pandas as pd
from src.pose import get_pose_tracker
from main import process_vid


class ClimbingPoseModel(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        self.pose_tracker = get_pose_tracker(
            mode="balanced", backend="onnxruntime", device="cpu"
        )

    def predict(self, context, model_input: pd.DataFrame) -> pd.DataFrame:
        video_path = model_input["video_path"][0]
        result = process_vid(video_path, pose_tracker=self.pose_tracker)
        return pd.DataFrame([result])


mlflow.models.set_model(ClimbingPoseModel())
