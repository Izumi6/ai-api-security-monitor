import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import joblib
from pathlib import Path

DATA_PATH = Path("data/requests_sample.csv")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)
MODEL_PATH = MODEL_DIR / "request_iforest.joblib"
SCALER_PATH = MODEL_DIR / "request_scaler.joblib"


def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp")
    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df_feat = df.copy()

    df_feat["endpoint"] = df_feat["endpoint"].astype("category").cat.codes
    df_feat["method"] = df_feat["method"].astype("category").cat.codes
    df_feat["ip_octet_1"] = df_feat["ip"].str.split(".").str[0].astype(int)

    features = df_feat[
        [
            "endpoint",
            "method",
            "bytes_in",
            "bytes_out",
            "status_code",
            "latency_ms",
            "ip_octet_1",
        ]
    ]
    return features


def main():
    print("Loading data...")
    df = load_data(DATA_PATH)

    print("Building features...")
    X = build_features(df).values

    print("Scaling features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print("Training IsolationForest...")
    model = IsolationForest(
        n_estimators=200,
        contamination=0.1,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_scaled)

    print(f"Saving model to {MODEL_PATH} and scaler to {SCALER_PATH}")
    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    print("Done.")


if __name__ == "__main__":
    main()
