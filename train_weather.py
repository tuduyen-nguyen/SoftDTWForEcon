"""Training script for weather data."""

import os


import logging

import torch
from dotenv import load_dotenv

from data.data_loader import DataLoaderS3
from data.data_preprocessing import DataConfig
from model.train_model import Trainer, TrainingConfig

load_dotenv()

os.makedirs("model_weights/weather_weights", exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),  # Affichage dans la console
    ],
)
logger = logging.getLogger(__name__)

weather_loader = DataLoaderS3(
    data_name="weather",
    data_format="csv",
    bucket_name="tnguyen",
    folder="diffusion/weather_data",
)
df_weather = weather_loader.load_data().drop(columns=["Date Time"])

data_config = DataConfig(
    split_train=0.6,
    split_val=0.2,
    input_size=24,
    output_size=24,
    stride=1,
    input_columns=list(df_weather.columns),
    output_columns=["T (degC)"],
)
training_config = TrainingConfig(
    hidden_size=64,
    epochs=50,
    batch_size=50,
    lr=1e-3,
    gammas=[1],
    max_norm=100.0,
    divergence=False,
)

DEV = "cuda:0" if torch.cuda.is_available() else "cpu"
device = torch.device(DEV)
logger.info("Using device %s", device)
trainer = Trainer(
    df_weather,
    device,
    data_config,
    training_config,
)

models = trainer.train_models()
for idx, model in enumerate(models):
    if idx != len(models) - 1:
        gamma = training_config.gammas[idx]
        torch.save(
            model.state_dict(),
            f"model_weights/weather_weights/model_weather_SDTW_gamma_{gamma}.pt",
        )
    else:
        torch.save(
            model.state_dict(), "model_weights/weather_weights/model_weather_MSE.pt"
        )
