from pathlib import Path
import pandas as pd


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def write_submission(passenger_ids, predictions, output_path: Path):
    df = pd.DataFrame({"PassengerId": passenger_ids, "Survived": predictions})
    df.to_csv(output_path, index=False)
    return output_path
