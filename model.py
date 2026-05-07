from sklearn.ensemble import RandomForestClassifier

def prepare_features(df):
    df = df.copy()

    # Create prediction target
    df["target"] = (df["FSI"].shift(-1) > df["FSI"]).astype(int)

    df = df.dropna()
    return df


def train_model(df):
    features = ["Yield", "Production", "Rainfall", "Temperature"]

    X = df[features]
    y = df["target"]

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)

    return model, features


def predict(model, df, features):
    latest = df[features].iloc[-1:]

    pred = model.predict(latest)[0]
    prob = model.predict_proba(latest)[0][pred]

    return pred, prob