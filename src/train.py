import os
import yaml
import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np

from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from mlflow.models.signature import infer_signature


def load_config():
    with open("config.yaml", "r") as file:
        return yaml.safe_load(file)


def load_data(path):
    df = pd.read_csv(
        path,
        sep=";",
        na_values="?",
        low_memory=False
    )

    df["datetime"] = pd.to_datetime(
        df["Date"] + " " + df["Time"],
        format="%d/%m/%Y %H:%M:%S"
    )

    df["hour"] = df["datetime"].dt.hour
    df["day"] = df["datetime"].dt.day
    df["month"] = df["datetime"].dt.month
    df["dayofweek"] = df["datetime"].dt.dayofweek

    numeric_cols = [
        "Global_active_power",
        "Global_reactive_power",
        "Voltage",
        "Global_intensity",
        "Sub_metering_1",
        "Sub_metering_2",
        "Sub_metering_3"
    ]

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["Global_active_power"])

    return df


def train():
    config = load_config()

    mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
    mlflow.set_experiment(config["mlflow"]["experiment_name"])

    df = load_data(config["data"]["path"])

    features = [
        "Global_reactive_power",
        "Voltage",
        "Global_intensity",
        "Sub_metering_1",
        "Sub_metering_2",
        "Sub_metering_3",
        "hour",
        "day",
        "month",
        "dayofweek"
    ]

    target = "Global_active_power"

    X = df[features]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config["data"]["test_size"],
        random_state=config["data"]["random_state"]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ]), features)
        ]
    )

    model = XGBRegressor(
        n_estimators=config["model"]["n_estimators"],
        max_depth=config["model"]["max_depth"],
        learning_rate=config["model"]["learning_rate"],
        random_state=config["data"]["random_state"]
    )

    pipeline = Pipeline([
        ("preprocessing", preprocessor),
        ("model", model)
    ])

    with mlflow.start_run():
        mlflow.log_params(config["model"])

        pipeline.fit(X_train, y_train)

        y_pred = pipeline.predict(X_test)

        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        mlflow.log_metric("mse", mse)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2", r2)

        signature = infer_signature(X_test, y_pred)
        input_example = X_test.head(5)

        mlflow.sklearn.log_model(
            sk_model=pipeline,
            artifact_path="energy_consumption_model",
            signature=signature,
            input_example=input_example
        )

        os.makedirs("artifacts", exist_ok=True)
        mlflow.sklearn.save_model(
            sk_model=pipeline,
            path="artifacts/model"
        )

        print("Modelo entrenado correctamente")
        print(f"MSE: {mse}")
        print(f"RMSE: {rmse}")
        print(f"MAE: {mae}")
        print(f"R2: {r2}")


if __name__ == "__main__":
    train()