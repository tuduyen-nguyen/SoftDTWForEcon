"""A simple API to make the prediction of time series."""

import logging

import s3fs
from datetime import datetime
import numpy as np
import torch
import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware

from data.data_preprocessing import DataConfig, get_normalization_metrics
from data.data_loader import DataLoaderS3
from model.mlp_baseline import MLP
from model.eval_model import eval_models, error

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),  # Affichage dans la console
    ],
)
logger = logging.getLogger(__name__)

load_dotenv()

app = FastAPI(
    title="Prédiction des valeurs suivants de la série",
    description='Prédiction du traffic de taxi pour les 5 prochaines heures <br>Une version par API pour faciliter la réutilisation du modèle 🚀 <br><br><img src="https://media.vogue.fr/photos/5faac06d39c5194ff9752ec9/1:1/w_2404,h_2404,c_limit/076_CHL_126884.jpg" width="200">',  # noqa: E501
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


device = "cpu"
model_names = ["Modèle Soft-DTW", "Modèle MSE"]


taxi_loader = DataLoaderS3(
    data_name="taxi",
    data_format="parquet",
    bucket_name="tnguyen",
    folder="diffusion/taxi_data",
)
df_taxi = taxi_loader.load_data()

weather_loader = DataLoaderS3(
    data_name="weather",
    data_format="csv",
    bucket_name="tnguyen",
    folder="diffusion/weather_data",
)
df_weather = weather_loader.load_data()


@app.get("/", tags=["Welcome"])
def show_welcome_page() -> dict:
    """Show welcome page with model name and version."""
    return {
        "Message": "API de prédiction heures de taxi ou de météo",
        "Model_name": "Taxi ML ou Weather ML",
        "Model_version": "0.2",
    }


@app.get("/predict_taxi", tags=["Predict_taxi"])
async def predict_taxi(
    date: str = Query(
        "2023-03-12 15:00:00", description="Date format: %Y-%m-%d %H:%M:%S"
    )
):
    """Predict taxi values."""

    try:
        input_date = datetime.strptime(date, "%Y-%m-%d %H:%M:%S")
    except ValueError:
        return {"error": "Date must be in format %Y-%m-%d %H:%M:%S"}

    data_config = DataConfig(
        split_train=0.6,
        split_val=0.2,
        input_size=20,
        output_size=5,
        stride=1,
        input_columns=["num_trips"],
        output_columns=["num_trips"],
    )

    models = []

    model_taxi_mse = MLP(
        input_size=20,
        hidden_size=300,
        output_size=5,
        num_features=1,
    )
    model_taxi_dtw = MLP(
        input_size=20,
        hidden_size=300,
        output_size=5,
        num_features=1,
    )

    s3_path_mse = "tnguyen/diffusion/model_weights/taxi_weights/model_taxi_MSE.pt"
    s3_path_dtw = (
        "tnguyen/diffusion/model_weights/taxi_weights/model_taxi_SDTW_gamma_1.pt"
    )
    fs = s3fs.S3FileSystem(
        client_kwargs={"endpoint_url": "https://minio.lab.sspcloud.fr"}
    )

    with fs.open(s3_path_mse, "rb") as f:
        model_taxi_mse.load_state_dict(torch.load(f, map_location=torch.device("cpu")))
    with fs.open(s3_path_dtw, "rb") as f:
        model_taxi_dtw.load_state_dict(torch.load(f, map_location=torch.device("cpu")))

    models.append(model_taxi_mse)
    models.append(model_taxi_dtw)

    input_array = df_taxi[
        pd.to_datetime(
            df_taxi["hour"],
            format="%Y-%m-%d %H:%M:%S",
        )
        < input_date
    ].tail(data_config.input_size)

    if input_array.shape[0] < data_config.input_size:
        return {
            "error": "Not enough data before the selected date to generate prediction."
        }

    input_array = input_array[data_config.input_columns].to_numpy().astype(np.float32)
    x_test = torch.tensor(input_array, dtype=torch.float32).unsqueeze(0)
    x_mean, x_std, y_mean, y_std = get_normalization_metrics(df_taxi, data_config)
    x_mean = torch.tensor(x_mean, dtype=torch.float32)
    x_std = torch.tensor(x_std, dtype=torch.float32)
    x_test = (x_test - x_mean) / x_std

    with torch.no_grad():
        y_pred_mse = model_taxi_mse(x_test).detach()
        prediction_mse = (y_pred_mse * y_std + y_mean).numpy().tolist()

        y_pred_dtw = model_taxi_dtw(x_test).detach()
        prediction_dtw = (y_pred_dtw * y_std + y_mean).numpy().tolist()

    results = eval_models(
        models,
        df_taxi,
        device,
        data_config,
    )

    mean_dtw, mean_mse, std_dtw, std_mse = error(
        results,
        df_taxi,
        data_config,
    )

    # Conversion en listes Python
    mean_dtw = np.round(mean_dtw, 3).tolist()
    std_dtw = np.round(std_dtw, 3).tolist()
    mean_mse = np.round(mean_mse, 3).tolist()
    std_mse = np.round(std_mse, 3).tolist()

    scores = {
        model_names[i]: {
            "MSE": {"mean": mean_mse[i], "std": std_mse[i]},
            "DTW": {"mean": mean_dtw[i], "std": std_dtw[i]},
        }
        for i in range(len(model_names))
    }
    return {
        "Date recue": {"date": input_date},
        "Prédiction taxi": [prediction_mse, prediction_dtw],
        "scores": scores,
    }


@app.get("/predict_weather", tags=["Predict_weather"])
async def predict_weather(
    date: str = Query(
        "12.03.2023 15:00:00", description="Date format: %d.%m.%Y %H:%M:%S"
    )
):
    """Prédiction météo à partir d'une date."""

    try:
        input_date = datetime.strptime(date, "%d.%m.%Y %H:%M:%S")
    except ValueError:
        return {"error": "Date must be in format %d.%m.%Y %H:%M:%S"}

    df_meteo = df_weather.drop(columns=["Date Time"])
    data_config = DataConfig(
        split_train=0.6,
        split_val=0.2,
        input_size=24,
        output_size=24,
        stride=1,
        input_columns=list(df_meteo.columns),
        output_columns=["T (degC)"],
    )
    x_mean, x_std, y_mean, y_std = get_normalization_metrics(df_meteo, data_config)
    x_mean = torch.tensor(x_mean, dtype=torch.float32)
    x_std = torch.tensor(x_std, dtype=torch.float32)
    input_array = df_weather[
        pd.to_datetime(
            df_weather["Date Time"],
            format="%d.%m.%Y %H:%M:%S",
        )
        < input_date
    ].tail(data_config.input_size)
    input_array = np.array(input_array[list(df_meteo.columns)].to_numpy())

    if input_array.shape[0] < data_config.input_size:
        return {
            "error": "Not enough data before the selected date to generate prediction."
        }

    models = []

    model_weather_mse = MLP(
        data_config.input_size,
        64,
        data_config.output_size,
        len(df_meteo.columns),
    )

    model_weather_dtw = MLP(
        data_config.input_size,
        64,
        data_config.output_size,
        len(df_meteo.columns),
    )

    s3_path_mse = "tnguyen/diffusion/model_weights/weather_weights/model_weather_MSE.pt"
    s3_path_dtw = (
        "tnguyen/diffusion/model_weights/weather_weights/model_weather_SDTW_gamma_1.pt"
    )
    fs = s3fs.S3FileSystem(
        client_kwargs={"endpoint_url": "https://minio.lab.sspcloud.fr"}
    )

    print(fs.ls("tnguyen/diffusion/model_weights/weather_weights"))
    with fs.open(s3_path_mse, "rb") as f:
        model_weather_mse.load_state_dict(
            torch.load(f, map_location=torch.device("cpu"))
        )
    with fs.open(s3_path_dtw, "rb") as f:
        model_weather_dtw.load_state_dict(
            torch.load(f, map_location=torch.device("cpu"))
        )
    model_weather_mse.eval()
    model_weather_dtw.eval()

    models.append(model_weather_mse)
    models.append(model_weather_dtw)

    x_test = torch.tensor(input_array, dtype=torch.float32).unsqueeze(0)
    x_norm = (x_test - x_mean) / (x_std)

    with torch.no_grad():
        y_pred_mse = model_weather_mse(x_norm)
        y_pred_denorm_mse = y_pred_mse * y_std + y_mean

        y_pred_dtw = model_weather_dtw(x_norm)
        y_pred_denorm_dtw = y_pred_dtw * y_std + y_mean
    prediction_mse = y_pred_denorm_mse.squeeze(0).tolist()
    prediction_dtw = y_pred_denorm_dtw.squeeze(0).tolist()

    results = eval_models(
        models,
        df_weather,
        device,
        data_config,
    )

    mean_dtw, mean_mse, std_dtw, std_mse = error(
        results,
        df_weather,
        data_config,
    )

    # Conversion en listes Python
    mean_dtw = np.round(mean_dtw, 3).tolist()
    std_dtw = np.round(std_dtw, 3).tolist()
    mean_mse = np.round(mean_mse, 3).tolist()
    std_mse = np.round(std_mse, 3).tolist()

    scores = {
        model_names[i]: {
            "MSE": {"mean": mean_mse[i], "std": std_mse[i]},
            "DTW": {"mean": mean_dtw[i], "std": std_dtw[i]},
        }
        for i in range(len(model_names))
    }
    return {
        "variables_reçues": {"annee": input_date},
        "prediction": [prediction_mse, prediction_dtw],
        "scores": scores,
    }
