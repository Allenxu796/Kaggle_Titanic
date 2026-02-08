from pathlib import Path
import pandas as pd

from src.data.load import load_raw
from src.features.build_features import add_basic_features
from src.models.train import build_tree_pipeline
from src.utils.io import ensure_dir, write_submission


def main():
    project_root = Path(__file__).resolve().parents[1]
    data_dir = project_root / "data" / "raw"
    submissions_dir = project_root / "submissions"
    ensure_dir(submissions_dir)

    train, test = load_raw(data_dir)

    train = add_basic_features(train)
    test = add_basic_features(test)

    features = [
        "Pclass",
        "Sex",
        "Age",
        "SibSp",
        "Parch",
        "Fare",
        "Embarked",
        "Title",
        "FamilySize",
        "IsAlone",
        "Deck",
        "TicketGroupSize",
        "FarePerPerson",
        "AgeBin",
        "FareBin",
    ]
    X = train[features]
    y = train["Survived"]
    X_test = test[features]

    num_features = [
        "Age",
        "SibSp",
        "Parch",
        "Fare",
        "FamilySize",
        "IsAlone",
        "TicketGroupSize",
        "FarePerPerson",
    ]
    cat_features = ["Pclass", "Sex", "Embarked", "Title", "Deck", "AgeBin", "FareBin"]

    model = build_tree_pipeline(num_features, cat_features, model_type="gradient_boosting")
    model.fit(X, y)
    pred = model.predict(X_test)

    output_path = submissions_dir / "submission_gb.csv"
    write_submission(test["PassengerId"], pred, output_path)
    print(output_path)


if __name__ == "__main__":
    main()
