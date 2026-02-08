import pandas as pd


def add_basic_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add simple derived features; expand as needed."""
    out = df.copy()

    # Title from Name
    out["Title"] = out["Name"].str.extract(" ([A-Za-z]+)\\.", expand=False)
    out["Title"] = out["Title"].replace(
        [
            "Lady",
            "Countess",
            "Capt",
            "Col",
            "Don",
            "Dr",
            "Major",
            "Rev",
            "Sir",
            "Jonkheer",
            "Dona",
        ],
        "Rare",
    )
    out["Title"] = out["Title"].replace({"Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs"})

    # Family size features
    out["FamilySize"] = out["SibSp"].fillna(0) + out["Parch"].fillna(0) + 1
    out["IsAlone"] = (out["FamilySize"] == 1).astype(int)

    # Deck from Cabin
    out["Deck"] = out["Cabin"].str.slice(0, 1).fillna("U")

    # Ticket group size
    out["TicketGroupSize"] = out.groupby("Ticket")["Ticket"].transform("count")

    # Fare per person
    out["FarePerPerson"] = out["Fare"] / out["FamilySize"].replace(0, 1)

    # Binned features
    out["AgeBin"] = pd.cut(
        out["Age"],
        bins=[0, 5, 12, 18, 25, 35, 45, 55, 65, 80],
        labels=False,
        include_lowest=True,
    )
    out["FareBin"] = pd.qcut(out["Fare"], 4, labels=False, duplicates="drop")

    return out
