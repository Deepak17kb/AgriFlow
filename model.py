import numpy as np
from sklearn.ensemble import RandomForestClassifier

FEATURES = ["Yield", "Production", "Rainfall", "Temperature"]


def prepare_features(df):
    df = df.copy()

    # Target: does FSI rise by the next record? The latest record has no next
    # record yet, so its target stays NaN: it is predicted, never trained on.
    next_fsi = df["FSI"].shift(-1)
    df["target"] = np.where(next_fsi.isna(), np.nan, (next_fsi > df["FSI"]).astype(float))

    # Only the model's own columns decide which rows are usable; blanks in
    # unrelated text columns (certification, policy_support) must not drop rows.
    df = df.dropna(subset=FEATURES + ["FSI"])
    return df


def train_model(df):
    """Fit on rows with a known outcome. Returns (None, features) when the
    history holds fewer than two distinct outcomes, since a classifier cannot
    learn from a single class."""
    train = df.dropna(subset=["target"])
    y = train["target"].astype(int)

    if y.nunique() < 2:
        return None, FEATURES

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(train[FEATURES], y)
    return model, FEATURES


def predict(model, df, features):
    """Return (pred, p_up) for the latest record: pred is 1 when FSI is expected
    to improve, and p_up is the probability of improvement."""
    if model is None:
        # Fall back to the smoothed share of past improvements.
        y = df["target"].dropna()
        p_up = (y.sum() + 1) / (len(y) + 2)
    else:
        latest = df[features].iloc[-1:]
        p_up = model.predict_proba(latest)[0][list(model.classes_).index(1)]

    return int(p_up >= 0.5), float(p_up)
