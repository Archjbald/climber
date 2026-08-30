"""Package and register the climbing-pose pipeline as an MLflow pyfunc model."""

import mlflow
import pandas as pd

mlflow.set_experiment("climbing_model_packaging")

with mlflow.start_run(run_name="packaged_pipeline_v1"):
    input_example = pd.DataFrame({"video_path": ["data/test.mp4"]})
    mlflow.pyfunc.log_model(
        name="climbing_model",  # `artifact_path` is deprecated, per your log
        python_model="src/mlflow_model.py",
        input_example=input_example,
        registered_model_name="climbing_pose_analyzer",
    )
    print("Registered. Check the Models tab in mlflow ui for the version number.")