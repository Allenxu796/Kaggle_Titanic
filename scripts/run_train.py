from pathlib import Path

from src.data.load import load_raw
from src.features.build_features import add_basic_features
from src.models.train import build_baseline_pipeline, build_tree_pipeline, cross_validate
from src.evaluation.metrics import summarize_scores


def main():
    project_root = Path(__file__).resolve().parents[1]
    data_dir = project_root / "data" / "raw"

    train, _ = load_raw(data_dir)
    train = add_basic_features(train)

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

    models = {
        "logistic_regression": build_baseline_pipeline(num_features, cat_features),
        "random_forest": build_tree_pipeline(num_features, cat_features, model_type="random_forest"),
        "gradient_boosting": build_tree_pipeline(num_features, cat_features, model_type="gradient_boosting"),
    }

    for name, model in models.items():
        scores = cross_validate(model, X, y)
        print(name, summarize_scores(scores))


if __name__ == "__main__":
    main()
