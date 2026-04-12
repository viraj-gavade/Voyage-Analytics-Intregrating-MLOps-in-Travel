import numpy as np
from app.utils.logger import get_logger

logger = get_logger(__name__)


def run_prediction(model, feature_vector: np.ndarray) -> float:
    prediction = model.predict(feature_vector)
    predicted_price = float(round(prediction[0], 4))
    logger.info("Prediction result: %.4f", predicted_price)
    return predicted_price









import joblib





# Load gender model once
gender_model = joblib.load("models/gender_model.pkl")
gender_encoder = joblib.load("models/gender_encoder.pkl")









def predict_gender(feature_vector: np.ndarray) -> str:
    pred = gender_model.predict(feature_vector)[0]
    gender = gender_encoder.inverse_transform([pred])[0]

    logger.info("Predicted gender: %s", gender)

    return gender