# ============================================================
# DR. FARMER - FINAL BACKEND
# ML + WEATHER + HISTORICAL WEATHER + FARMER ADVISORY
# ============================================================

from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

import tensorflow as tf
import numpy as np
from PIL import Image, UnidentifiedImageError

import io
import json
import os
import requests

from datetime import datetime, date


# ============================================================
# FASTAPI SETUP
# ============================================================

app = FastAPI(title="Dr. Farmer API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# FILE PATHS
# ============================================================

# Always find the model/files next to backend.py.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "tomato_model.keras")
CLASS_NAMES_PATH = os.path.join(BASE_DIR, "class_names.json")
ACCURACY_PATH = os.path.join(BASE_DIR, "model_accuracy.json")


# ============================================================
# LOAD ML MODEL
# ============================================================

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Model file not found: {MODEL_PATH}\n"
        "Put tomato_model.keras in the same folder as backend.py."
    )

if not os.path.exists(CLASS_NAMES_PATH):
    raise FileNotFoundError(
        f"Class names file not found: {CLASS_NAMES_PATH}\n"
        "Put class_names.json in the same folder as backend.py."
    )

model = tf.keras.models.load_model(MODEL_PATH)

with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as f:
    CLASS_NAMES = json.load(f)


# Load actual validation accuracy from Colab.
MODEL_ACCURACY = None

if os.path.exists(ACCURACY_PATH):
    try:
        with open(ACCURACY_PATH, "r", encoding="utf-8") as f:
            accuracy_data = json.load(f)

        MODEL_ACCURACY = float(
            accuracy_data["validation_accuracy"]
        )

    except (KeyError, ValueError, TypeError, json.JSONDecodeError):
        MODEL_ACCURACY = None


CONFIDENCE_THRESHOLD = 0.60


# ============================================================
# DISEASE INFORMATION
# ============================================================

ADVISORY = {
    "Tomato_Healthy": {
        "disease": "Healthy",
        "message": "The tomato leaf appears healthy.",
        "action": (
            "Continue regular monitoring and good farming practices."
        ),
    },

    "Tomato_Early_Blight": {
        "disease": "Early Blight",
        "message": (
            "The leaf may show symptoms consistent with Early Blight."
        ),
        "action": (
            "Remove badly affected leaves, improve airflow, "
            "and avoid prolonged leaf wetness."
        ),
    },

    "Tomato_Late_Blight": {
        "disease": "Late Blight",
        "message": (
            "The leaf may show symptoms consistent with Late Blight."
        ),
        "action": (
            "Remove visibly infected plant material, "
            "avoid overhead watering, and monitor nearby plants."
        ),
    },
}


# ============================================================
# ML FUNCTIONS
# ============================================================

def preprocess_image(file_bytes):
    """Convert uploaded image to the format used by the model."""

    try:
        image = Image.open(
            io.BytesIO(file_bytes)
        ).convert("RGB")
    except (UnidentifiedImageError, OSError):
        raise ValueError("The uploaded file is not a valid image.")

    image = image.resize((224, 224))

    image_array = np.array(image, dtype=np.float32) / 255.0

    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    return image_array


def predict_disease(file_bytes):
    """Run the tomato disease classification model."""

    image = preprocess_image(file_bytes)

    predictions = model.predict(
        image,
        verbose=0
    )

    predicted_index = int(
        np.argmax(predictions[0])
    )

    confidence = float(
        predictions[0][predicted_index]
    )

    predicted_class = CLASS_NAMES[
        predicted_index
    ]

    low_confidence = (
        confidence < CONFIDENCE_THRESHOLD
    )

    advisory = ADVISORY.get(
        predicted_class,
        {
            "disease": predicted_class,
            "message": "Prediction completed.",
            "action": "Monitor the plant.",
        },
    )

    all_probabilities = {
        CLASS_NAMES[i]: round(
            float(predictions[0][i]) * 100,
            2,
        )
        for i in range(len(CLASS_NAMES))
    }

    return {
        "class": predicted_class,
        "confidence": round(
            confidence * 100,
            2,
        ),
        "low_confidence": low_confidence,
        "all_probabilities": all_probabilities,
        "advisory": advisory,
    }


# ============================================================
# WEATHER API
# ============================================================

WEATHER_API = (
    "https://api.open-meteo.com/v1/forecast"
)

HISTORICAL_WEATHER_API = (
    "https://archive-api.open-meteo.com/v1/archive"
)

REQUEST_TIMEOUT = 15


# ============================================================
# CURRENT + FUTURE WEATHER
# ============================================================

def get_weather_data(latitude, longitude):

    parameters = {
        "latitude": latitude,
        "longitude": longitude,

        "current": (
            "temperature_2m,"
            "relative_humidity_2m,"
            "precipitation"
        ),

        "hourly": (
            "temperature_2m,"
            "relative_humidity_2m,"
            "precipitation,"
            "precipitation_probability"
        ),

        "forecast_days": 7,
        "timezone": "auto",
    }

    try:
        response = requests.get(
            WEATHER_API,
            params=parameters,
            timeout=REQUEST_TIMEOUT,
        )

        response.raise_for_status()

        return response.json()

    except requests.RequestException:
        return None


# ============================================================
# HISTORICAL WEATHER
# ============================================================

def get_historical_weather(
    latitude,
    longitude,
    selected_date,
):

    parameters = {
        "latitude": latitude,
        "longitude": longitude,

        "start_date": selected_date,
        "end_date": selected_date,

        "hourly": (
            "temperature_2m,"
            "relative_humidity_2m,"
            "precipitation"
        ),

        "timezone": "auto",
    }

    try:
        response = requests.get(
            HISTORICAL_WEATHER_API,
            params=parameters,
            timeout=REQUEST_TIMEOUT,
        )

        response.raise_for_status()

        return response.json()

    except requests.RequestException:
        return None


# ============================================================
# WEATHER RISK CALCULATIONS
# ============================================================

def calculate_temperature_score(temperature):

    if 18 <= temperature <= 28:
        return 40

    elif 15 <= temperature < 18:
        return 25

    elif 28 < temperature <= 32:
        return 25

    elif 12 <= temperature < 15:
        return 10

    elif 32 < temperature <= 35:
        return 10

    return 0


def calculate_humidity_score(humidity):

    if humidity >= 85:
        return 40

    elif humidity >= 75:
        return 30

    elif humidity >= 65:
        return 20

    elif humidity >= 50:
        return 10

    return 0


def calculate_rainfall_score(rainfall):

    if rainfall >= 10:
        return 20

    elif rainfall >= 5:
        return 15

    elif rainfall >= 2:
        return 10

    elif rainfall > 0:
        return 5

    return 0


def get_risk_level(score):

    if score >= 80:
        return "HIGH"

    elif score >= 50:
        return "MEDIUM"

    elif score > 0:
        return "LOW"

    return "VERY LOW"


def calculate_risk(
    temperature,
    humidity,
    rainfall,
):

    temperature_score = calculate_temperature_score(
        temperature
    )

    humidity_score = calculate_humidity_score(
        humidity
    )

    rainfall_score = calculate_rainfall_score(
        rainfall
    )

    total_score = (
        temperature_score
        + humidity_score
        + rainfall_score
    )

    total_score = min(
        total_score,
        100,
    )

    return {
        "score": total_score,
        "level": get_risk_level(total_score),
    }


# ============================================================
# FUTURE FORECAST PARSER
# ============================================================

def parse_forecast(weather_data):

    if not weather_data:
        return []

    hourly = weather_data.get(
        "hourly",
        {},
    )

    times = hourly.get(
        "time",
        [],
    )

    temperatures = hourly.get(
        "temperature_2m",
        [],
    )

    humidities = hourly.get(
        "relative_humidity_2m",
        [],
    )

    rainfall = hourly.get(
        "precipitation",
        [],
    )

    rain_probabilities = hourly.get(
        "precipitation_probability",
        [],
    )

    daily_data = {}

    for i, time_string in enumerate(times):

        forecast_date = (
            time_string.split("T")[0]
        )

        if forecast_date not in daily_data:

            daily_data[forecast_date] = {
                "temperatures": [],
                "humidities": [],
                "rainfall": [],
                "rain_probabilities": [],
            }

        if i < len(temperatures):
            value = temperatures[i]
            if value is not None:
                daily_data[
                    forecast_date
                ]["temperatures"].append(value)

        if i < len(humidities):
            value = humidities[i]
            if value is not None:
                daily_data[
                    forecast_date
                ]["humidities"].append(value)

        if i < len(rainfall):
            value = rainfall[i]
            if value is not None:
                daily_data[
                    forecast_date
                ]["rainfall"].append(value)

        if i < len(rain_probabilities):
            value = rain_probabilities[i]
            if value is not None:
                daily_data[
                    forecast_date
                ]["rain_probabilities"].append(value)

    forecast = []

    for forecast_date, data in daily_data.items():

        if not data["temperatures"]:
            continue

        min_temp = min(
            data["temperatures"]
        )

        max_temp = max(
            data["temperatures"]
        )

        average_humidity = (
            sum(data["humidities"])
            / len(data["humidities"])
            if data["humidities"]
            else 0
        )

        total_rainfall = sum(
            data["rainfall"]
        )

        max_rain_probability = (
            max(data["rain_probabilities"])
            if data["rain_probabilities"]
            else 0
        )

        representative_temperature = (
            min_temp + max_temp
        ) / 2

        risk = calculate_risk(
            representative_temperature,
            average_humidity,
            total_rainfall,
        )

        # Both nested and flat values are returned.
        # This keeps the endpoint easy for different frontends to use.
        forecast.append({

            "date": forecast_date,

            "temperature": {
                "min": round(min_temp, 2),
                "max": round(max_temp, 2),
            },

            "humidity": round(
                average_humidity,
                2,
            ),

            "rainfall": round(
                total_rainfall,
                2,
            ),

            "rain_probability": round(
                max_rain_probability,
                2,
            ),

            "risk": risk,

            # Flat aliases for simpler frontend cards.
            "min_temp": round(min_temp, 2),
            "max_temp": round(max_temp, 2),
            "avg_humidity": round(
                average_humidity,
                2,
            ),
            "risk_score": risk["score"],
            "risk_level": risk["level"],
        })

    return forecast


# ============================================================
# HISTORICAL WEATHER PARSER
# ============================================================

def parse_historical_weather(
    weather_data,
    selected_date,
):

    if not weather_data:
        return None

    hourly = weather_data.get(
        "hourly",
        {},
    )

    temperatures = hourly.get(
        "temperature_2m",
        [],
    )

    humidities = hourly.get(
        "relative_humidity_2m",
        [],
    )

    rainfall = hourly.get(
        "precipitation",
        [],
    )

    temperatures = [
        value for value in temperatures
        if value is not None
    ]

    humidities = [
        value for value in humidities
        if value is not None
    ]

    rainfall = [
        value for value in rainfall
        if value is not None
    ]

    if not temperatures:
        return None

    min_temperature = min(
        temperatures
    )

    max_temperature = max(
        temperatures
    )

    average_temperature = (
        min_temperature
        + max_temperature
    ) / 2

    average_humidity = (
        sum(humidities)
        / len(humidities)
        if humidities
        else 0
    )

    total_rainfall = sum(
        rainfall
    )

    risk = calculate_risk(
        average_temperature,
        average_humidity,
        total_rainfall,
    )

    return {

        "date": selected_date,

        "temperature": {
            "min": round(
                min_temperature,
                2,
            ),

            "max": round(
                max_temperature,
                2,
            ),

            "average": round(
                average_temperature,
                2,
            ),
        },

        "humidity": round(
            average_humidity,
            2,
        ),

        "rainfall": round(
            total_rainfall,
            2,
        ),

        "risk": risk,
    }


# ============================================================
# WEATHER RISK REPORT
# ============================================================

def get_weather_risk_report(
    latitude,
    longitude,
    photo_date=None,
):

    # --------------------------------------------------------
    # CURRENT + FUTURE WEATHER
    # --------------------------------------------------------

    weather_data = get_weather_data(
        latitude,
        longitude,
    )

    if weather_data is None:

        return {
            "success": False,
            "error": "Unable to fetch weather data.",
        }

    current = weather_data.get(
        "current",
        {},
    )

    current_temperature = current.get(
        "temperature_2m",
        0,
    )

    current_humidity = current.get(
        "relative_humidity_2m",
        0,
    )

    current_rainfall = current.get(
        "precipitation",
        0,
    )

    current_risk = calculate_risk(
        current_temperature,
        current_humidity,
        current_rainfall,
    )

    forecast = parse_forecast(
        weather_data
    )

    # --------------------------------------------------------
    # WEATHER FOR PHOTO DATE
    # --------------------------------------------------------

    historical_weather = None

    if photo_date:

        selected_date = datetime.strptime(
            photo_date,
            "%Y-%m-%d",
        ).date()

        today = date.today()

        # Today's photo: use current weather.
        if selected_date == today:

            historical_weather = {
                "date": photo_date,

                "temperature": {
                    "average": round(
                        float(current_temperature),
                        2,
                    ),
                    "min": round(
                        float(current_temperature),
                        2,
                    ),
                    "max": round(
                        float(current_temperature),
                        2,
                    ),
                },

                "humidity": round(
                    float(current_humidity),
                    2,
                ),

                "rainfall": round(
                    float(current_rainfall),
                    2,
                ),

                "risk": current_risk,
            }

        # Past photo: use historical weather API.
        elif selected_date < today:

            historical_data = get_historical_weather(
                latitude,
                longitude,
                photo_date,
            )

            if historical_data:

                historical_weather = (
                    parse_historical_weather(
                        historical_data,
                        photo_date,
                    )
                )

    return {

        "success": True,

        "location": {
            "latitude": latitude,
            "longitude": longitude,
        },

        "current_weather": {
            "temperature": current_temperature,
            "humidity": current_humidity,
            "rainfall": current_rainfall,
        },

        "current_risk": current_risk,

        "photo_date_weather":
            historical_weather,

        "forecast": forecast,
    }


# ============================================================
# WEATHER TREND
# ============================================================

def analyze_weather_trend(forecast):

    if not forecast:

        return {
            "trend": "UNKNOWN",
            "message": (
                "Future weather trend is unavailable."
            ),
        }

    scores = [
        day["risk"]["score"]
        for day in forecast
        if "risk" in day and "score" in day["risk"]
    ]

    if len(scores) < 2:

        return {
            "trend": "UNKNOWN",
            "message": (
                "Not enough forecast data "
                "to determine a trend."
            ),
        }

    first = scores[0]
    last = scores[-1]

    high_days = sum(
        score >= 80
        for score in scores
    )

    if high_days >= 3:

        return {
            "trend": "PERSISTENT_HIGH",
            "message": (
                "Several upcoming days have "
                "high environmental risk."
            ),
        }

    if last > first + 10:

        return {
            "trend": "RISING",
            "message": (
                "Environmental risk is expected "
                "to increase over the coming days."
            ),
        }

    if last < first - 10:

        return {
            "trend": "FALLING",
            "message": (
                "Environmental risk is expected "
                "to decrease over the coming days."
            ),
        }

    return {
        "trend": "STABLE",
        "message": (
            "Environmental risk is expected "
            "to remain relatively stable."
        ),
    }


# ============================================================
# FARMER ADVISORY ENGINE
# ============================================================

def generate_farmer_advice(
    disease_result,
    weather_report,
    language="English",
):

    disease = disease_result["class"]
    confidence = disease_result["confidence"]
    low_confidence = disease_result["low_confidence"]

    forecast = weather_report.get(
        "forecast",
        [],
    )

    trend = analyze_weather_trend(
        forecast
    )

    current_risk = weather_report[
        "current_risk"
    ]

    historical = weather_report.get(
        "photo_date_weather"
    )

    # --------------------------------------------------------
    # LOW CONFIDENCE
    # --------------------------------------------------------

    if low_confidence:

        english = {
            "summary": (
                "The image could not be classified confidently."
            ),

            "why": (
                f"The model's confidence was only "
                f"{confidence}%. A clearer image of the "
                f"leaf would give a more reliable result."
            ),

            "next_few_days": trend["message"],

            "what_to_do": [
                "Take a clear photo of the affected leaf in good lighting.",
                "Check whether similar symptoms are appearing on other leaves.",
                "Continue monitoring the plant.",
            ],

            "urgency": "LOW",
        }

        return translate_advice(
            english,
            language,
        )

    # --------------------------------------------------------
    # HEALTHY
    # --------------------------------------------------------

    if disease == "Tomato_Healthy":

        if current_risk["level"] in [
            "HIGH",
            "MEDIUM",
        ]:

            summary = (
                "The leaf appears healthy, but the current "
                "weather conditions may favor disease development."
            )

        else:

            summary = (
                "The leaf appears healthy and current "
                "environmental risk is relatively low."
            )

        english = {
            "summary": summary,

            "why": (
                f"The model classified the leaf as healthy "
                f"with {confidence}% confidence."
            ),

            "next_few_days": trend["message"],

            "what_to_do": [
                "Continue checking new leaves regularly.",
                "Avoid keeping the leaves wet for long periods.",
                "Pay extra attention after rainy or very humid weather.",
            ],

            "urgency": (
                "WATCH"
                if current_risk["level"] == "HIGH"
                else "LOW"
            ),
        }

        return translate_advice(
            english,
            language,
        )

    # --------------------------------------------------------
    # DISEASE-SPECIFIC ACTIONS
    # --------------------------------------------------------

    if disease == "Tomato_Early_Blight":

        disease_name = "Early Blight"

        actions = [
            "Remove badly affected leaves if practical.",
            "Improve airflow around the plant.",
            "Avoid overhead watering and prolonged leaf wetness.",
            "Check nearby leaves for new spots.",
        ]

    elif disease == "Tomato_Late_Blight":

        disease_name = "Late Blight"

        actions = [
            "Remove visibly infected plant material.",
            "Avoid overhead watering.",
            "Inspect nearby tomato plants for similar symptoms.",
            "Monitor especially closely during wet or highly humid periods.",
        ]

    else:

        disease_name = disease

        actions = [
            "Continue monitoring the plant.",
            "Take another clear image if symptoms change.",
        ]

    # --------------------------------------------------------
    # WHY
    # --------------------------------------------------------

    why = (
        f"The model detected patterns consistent with "
        f"{disease_name} with {confidence}% confidence."
    )

    if historical:

        historical_risk = historical[
            "risk"
        ]

        why += (
            f" On the date the photo was taken, "
            f"the estimated environmental risk was "
            f"{historical_risk['level']} "
            f"({historical_risk['score']}/100)."
        )

    # --------------------------------------------------------
    # WEATHER CONTEXT
    # --------------------------------------------------------

    next_few_days = (
        f"Current environmental risk is "
        f"{current_risk['level']} "
        f"({current_risk['score']}/100). "
        f"{trend['message']}"
    )

    # --------------------------------------------------------
    # EXTRA WEATHER WARNING
    # --------------------------------------------------------

    high_days = [
        day["date"]
        for day in forecast
        if day.get("risk", {}).get("level") == "HIGH"
    ]

    if high_days:

        next_few_days += (
            f" There are {len(high_days)} "
            f"forecast day(s) currently classified "
            f"as HIGH environmental risk."
        )

    # --------------------------------------------------------
    # URGENCY
    # --------------------------------------------------------

    if disease == "Tomato_Late_Blight":

        urgency = "HIGH"

    elif current_risk["level"] == "HIGH":

        urgency = "HIGH"

    elif current_risk["level"] == "MEDIUM":

        urgency = "WATCH"

    else:

        urgency = "MEDIUM"

    english = {
        "summary": (
            f"The plant may be showing signs of "
            f"{disease_name}."
        ),

        "why": why,

        "next_few_days": next_few_days,

        "what_to_do": actions,

        "urgency": urgency,
    }

    return translate_advice(
        english,
        language,
    )


# ============================================================
# MULTILINGUAL FARMER-FRIENDLY OUTPUT
# ============================================================

def translate_advice(
    advice,
    language,
):

    selected = (
        str(language or "English")
        .strip()
        .lower()
    )

    # English: return original structured advice.
    if selected == "english":

        advice["requested_language"] = "English"
        advice["translation_status"] = "verified"

        return advice

    # --------------------------------------------------------
    # HINDI
    # --------------------------------------------------------

    if selected == "hindi":

        translated = {
            "summary": advice["summary"],
            "why": advice["why"],
            "next_few_days": advice["next_few_days"],
            "what_to_do": advice["what_to_do"],
            "urgency": advice["urgency"],
            "requested_language": "Hindi",
            "translation_status": (
                "Disease-specific Hindi templates are available."
            ),
        }

        if "confidently" in advice["summary"].lower():
            translated["summary"] = (
                "इस तस्वीर से बीमारी की पहचान "
                "विश्वास के साथ नहीं की जा सकी।"
            )

            translated["why"] = (
                f"मॉडल का भरोसा केवल {advice['why'].split('%')[0].split()[-1]}% था। "
                "पत्ते की साफ तस्वीर लेने से परिणाम अधिक विश्वसनीय हो सकता है।"
            )

            translated["next_few_days"] = (
                "अगले कुछ दिनों में मौसम का जोखिम: "
                + advice["next_few_days"]
            )

            translated["what_to_do"] = [
                "पत्ते की अच्छी रोशनी में साफ तस्वीर लें।",
                "जांचें कि दूसरे पत्तों पर भी ऐसे लक्षण दिखाई दे रहे हैं या नहीं।",
                "पौधे की नियमित निगरानी जारी रखें।",
            ]

            return translated

        if "healthy" in advice["summary"].lower():

            if "may favor" in advice["summary"].lower():

                translated["summary"] = (
                    "पत्ता स्वस्थ दिखाई देता है, लेकिन वर्तमान "
                    "मौसम बीमारी के विकास के लिए अनुकूल हो सकता है।"
                )

            else:

                translated["summary"] = (
                    "पत्ता स्वस्थ दिखाई देता है और वर्तमान "
                    "पर्यावरणीय जोखिम अपेक्षाकृत कम है।"
                )

            translated["why"] = (
                "मॉडल ने इस पत्ते को "
                f"{advice['why'].split('with ')[-1]} "
                "विश्वास के साथ स्वस्थ बताया है।"
            )

            translated["next_few_days"] = (
                "अगले कुछ दिनों का मौसम: "
                + advice["next_few_days"]
            )

            translated["what_to_do"] = [
                "नए पत्तों की नियमित जांच करते रहें।",
                "पत्तों को लंबे समय तक गीला न रहने दें।",
                "बारिश या बहुत अधिक नमी के बाद पौधे पर विशेष ध्यान दें।",
            ]

            return translated

        # Disease cases.
        if "Early Blight" in advice["summary"]:

            disease_hi = "अर्ली ब्लाइट"

        elif "Late Blight" in advice["summary"]:

            disease_hi = "लेट ब्लाइट"

        else:

            disease_hi = "बीमारी"

        translated["summary"] = (
            f"पौधे में {disease_hi} के लक्षण दिखाई दे सकते हैं।"
        )

        translated["why"] = (
            f"मॉडल ने {disease_hi} से मिलते-जुलते लक्षण "
            f"{advice['why'].split('with ')[-1]} विश्वास के साथ पहचाने हैं।"
        )

        translated["next_few_days"] = (
            "अगले कुछ दिनों में: "
            + advice["next_few_days"]
        )

        if "Early Blight" in advice["summary"]:

            translated["what_to_do"] = [
                "बहुत अधिक प्रभावित पत्तों को, यदि संभव हो, हटा दें।",
                "पौधे के आसपास हवा का प्रवाह बेहतर करें।",
                "पत्तों पर ऊपर से पानी देने और लंबे समय तक नमी रहने से बचें।",
                "आस-पास के पत्तों पर नए धब्बों की जांच करें।",
            ]

        elif "Late Blight" in advice["summary"]:

            translated["what_to_do"] = [
                "दिखाई देने वाले संक्रमित पौधे के हिस्सों को हटा दें।",
                "पत्तों पर ऊपर से पानी देने से बचें।",
                "आस-पास के टमाटर के पौधों की भी जांच करें।",
                "बारिश या बहुत अधिक नमी के समय विशेष निगरानी रखें।",
            ]

        return translated

    # --------------------------------------------------------
    # KANNADA
    # --------------------------------------------------------

    if selected == "kannada":

        translated = {
            "summary": advice["summary"],
            "why": advice["why"],
            "next_few_days": advice["next_few_days"],
            "what_to_do": advice["what_to_do"],
            "urgency": advice["urgency"],
            "requested_language": "Kannada",
            "translation_status": (
                "Disease-specific Kannada templates are available."
            ),
        }

        if "confidently" in advice["summary"].lower():

            translated["summary"] = (
                "ಈ ಚಿತ್ರದಿಂದ ರೋಗವನ್ನು ನಿಖರವಾಗಿ ಗುರುತಿಸಲು "
                "ಸಾಕಷ್ಟು ವಿಶ್ವಾಸವಿಲ್ಲ."
            )

            translated["why"] = (
                "ಎಲೆಯ ಸ್ಪಷ್ಟ ಚಿತ್ರವನ್ನು ಉತ್ತಮ ಬೆಳಕಿನಲ್ಲಿ "
                "ತೆಗೆದರೆ ಹೆಚ್ಚು ವಿಶ್ವಾಸಾರ್ಹ ಫಲಿತಾಂಶ ಸಿಗಬಹುದು."
            )

            translated["next_few_days"] = (
                "ಮುಂದಿನ ಕೆಲವು ದಿನಗಳ ಹವಾಮಾನ: "
                + advice["next_few_days"]
            )

            translated["what_to_do"] = [
                "ಬಾಧಿತ ಎಲೆಯ ಸ್ಪಷ್ಟ ಚಿತ್ರವನ್ನು ಉತ್ತಮ ಬೆಳಕಿನಲ್ಲಿ ತೆಗೆದುಕೊಳ್ಳಿ.",
                "ಇತರ ಎಲೆಗಳಲ್ಲಿಯೂ ಇದೇ ರೀತಿಯ ಲಕ್ಷಣಗಳಿವೆಯೇ ಎಂದು ಪರಿಶೀಲಿಸಿ.",
                "ಸಸ್ಯವನ್ನು ನಿಯಮಿತವಾಗಿ ಗಮನಿಸುತ್ತಿರಿ.",
            ]

            return translated

        if "healthy" in advice["summary"].lower():

            if "may favor" in advice["summary"].lower():

                translated["summary"] = (
                    "ಎಲೆ ಆರೋಗ್ಯಕರವಾಗಿ ಕಾಣುತ್ತದೆ, ಆದರೆ ಪ್ರಸ್ತುತ "
                    "ಹವಾಮಾನವು ರೋಗದ ಬೆಳವಣಿಗೆಗೆ ಅನುಕೂಲಕರವಾಗಿರಬಹುದು."
                )

            else:

                translated["summary"] = (
                    "ಎಲೆ ಆರೋಗ್ಯಕರವಾಗಿ ಕಾಣುತ್ತದೆ ಮತ್ತು ಪ್ರಸ್ತುತ "
                    "ಪರಿಸರದ ಅಪಾಯವು ಕಡಿಮೆಯಾಗಿದೆ."
                )

            translated["why"] = (
                "ಮಾದರಿಯು ಈ ಎಲೆಯನ್ನು ಆರೋಗ್ಯಕರವೆಂದು "
                "ಗುರುತಿಸಿದೆ."
            )

            translated["next_few_days"] = (
                "ಮುಂದಿನ ಕೆಲವು ದಿನಗಳ ಹವಾಮಾನ: "
                + advice["next_few_days"]
            )

            translated["what_to_do"] = [
                "ಹೊಸ ಎಲೆಗಳನ್ನು ನಿಯಮಿತವಾಗಿ ಪರಿಶೀಲಿಸಿ.",
                "ಎಲೆಗಳು ದೀರ್ಘಕಾಲ ತೇವವಾಗಿರುವುದನ್ನು ತಪ್ಪಿಸಿ.",
                "ಮಳೆ ಅಥವಾ ಹೆಚ್ಚಿನ ತೇವಾಂಶದ ನಂತರ ಹೆಚ್ಚು ಗಮನ ನೀಡಿ.",
            ]

            return translated

        if "Early Blight" in advice["summary"]:

            disease_kn = "ಅರ್ಲಿ ಬ್ಲೈಟ್"

        elif "Late Blight" in advice["summary"]:

            disease_kn = "ಲೇಟ್ ಬ್ಲೈಟ್"

        else:

            disease_kn = "ರೋಗ"

        translated["summary"] = (
            f"ಸಸ್ಯದಲ್ಲಿ {disease_kn} ಲಕ್ಷಣಗಳು ಕಾಣಿಸಬಹುದು."
        )

        translated["why"] = (
            f"ಮಾದರಿಯು {disease_kn} ಗೆ ಹೊಂದುವ ಲಕ್ಷಣಗಳನ್ನು "
            "ಗುರುತಿಸಿದೆ."
        )

        translated["next_few_days"] = (
            "ಮುಂದಿನ ಕೆಲವು ದಿನಗಳಲ್ಲಿ: "
            + advice["next_few_days"]
        )

        if "Early Blight" in advice["summary"]:

            translated["what_to_do"] = [
                "ತೀವ್ರವಾಗಿ ಬಾಧಿತ ಎಲೆಗಳನ್ನು ಸಾಧ್ಯವಾದರೆ ತೆಗೆದುಹಾಕಿ.",
                "ಸಸ್ಯದ ಸುತ್ತಲಿನ ಗಾಳಿಯ ಹರಿವನ್ನು ಉತ್ತಮಗೊಳಿಸಿ.",
                "ಎಲೆಗಳ ಮೇಲೆ ನೇರವಾಗಿ ನೀರು ಹಾಕುವುದನ್ನು ಮತ್ತು ದೀರ್ಘಕಾಲ ತೇವವಾಗುವುದನ್ನು ತಪ್ಪಿಸಿ.",
                "ಹತ್ತಿರದ ಎಲೆಗಳಲ್ಲಿ ಹೊಸ ಕಲೆಗಳಿವೆಯೇ ಎಂದು ಪರಿಶೀಲಿಸಿ.",
            ]

        elif "Late Blight" in advice["summary"]:

            translated["what_to_do"] = [
                "ಕಾಣುವ ಸೋಂಕಿತ ಸಸ್ಯದ ಭಾಗಗಳನ್ನು ತೆಗೆದುಹಾಕಿ.",
                "ಎಲೆಗಳ ಮೇಲೆ ನೇರವಾಗಿ ನೀರು ಹಾಕುವುದನ್ನು ತಪ್ಪಿಸಿ.",
                "ಹತ್ತಿರದ ಟೊಮ್ಯಾಟೊ ಸಸ್ಯಗಳನ್ನು ಪರಿಶೀಲಿಸಿ.",
                "ಮಳೆ ಅಥವಾ ಹೆಚ್ಚಿನ ತೇವಾಂಶದ ಸಮಯದಲ್ಲಿ ಹೆಚ್ಚು ಗಮನ ನೀಡಿ.",
            ]

        return translated

    # Unknown language: do not pretend to translate.
    advice["requested_language"] = language
    advice["translation_status"] = (
        "Language not yet supported. English returned."
    )

    return advice


# ============================================================
# FRONTEND WEATHER ENDPOINT
# ============================================================

@app.get("/api/health")
def weather_health(
    latitude: float,
    longitude: float,
):

    report = get_weather_risk_report(
        latitude,
        longitude,
    )

    if not report["success"]:
        return report

    return {
        "success": True,

        "location":
            report["location"],

        "current_weather":
            report["current_weather"],

        "current_risk":
            report["current_risk"],

        "forecast":
            report["forecast"],
    }


# ============================================================
# MAIN ANALYSIS ENDPOINT
# ============================================================

@app.post("/analyze")
async def analyze(

    image: UploadFile = File(...),

    latitude: float = Form(...),

    longitude: float = Form(...),

    photo_date: str = Form(...),

    language: str = Form("English"),
):

    # ========================================================
    # VALIDATE LOCATION
    # ========================================================

    if not -90 <= latitude <= 90:

        return JSONResponse(
            status_code=400,
            content={
                "success": False,
                "error": "Invalid latitude.",
            },
        )

    if not -180 <= longitude <= 180:

        return JSONResponse(
            status_code=400,
            content={
                "success": False,
                "error": "Invalid longitude.",
            },
        )

    # ========================================================
    # VALIDATE DATE
    # ========================================================

    try:

        selected_date = datetime.strptime(
            photo_date,
            "%Y-%m-%d",
        ).date()

    except ValueError:

        return JSONResponse(
            status_code=400,
            content={
                "success": False,
                "error": (
                    "photo_date must be in YYYY-MM-DD format."
                ),
            },
        )

    # A photo cannot have a future taken-date.
    if selected_date > date.today():

        return JSONResponse(
            status_code=400,
            content={
                "success": False,
                "error": (
                    "Photo date cannot be in the future."
                ),
            },
        )

    # ========================================================
    # READ IMAGE
    # ========================================================

    try:

        file_bytes = await image.read()

        if not file_bytes:

            return JSONResponse(
                status_code=400,
                content={
                    "success": False,
                    "error": "The uploaded image is empty.",
                },
            )

        # ====================================================
        # ML PREDICTION
        # ====================================================

        disease_result = predict_disease(
            file_bytes
        )

    except ValueError as error:

        return JSONResponse(
            status_code=400,
            content={
                "success": False,
                "error": str(error),
            },
        )

    except Exception as error:

        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": (
                    f"Image analysis failed: {str(error)}"
                ),
            },
        )

    # ========================================================
    # WEATHER
    # ========================================================

    weather_result = get_weather_risk_report(
        latitude,
        longitude,
        photo_date,
    )

    if not weather_result["success"]:

        return JSONResponse(
            status_code=503,
            content={
                "success": False,

                "error":
                    "Weather data could not be retrieved.",

                "disease_diagnosis":
                    disease_result,
            },
        )

    # ========================================================
    # FARMER ADVISORY
    # ========================================================

    farmer_advice = generate_farmer_advice(
        disease_result,
        weather_result,
        language,
    )

    # ========================================================
    # DISEASE PRESENT?
    # ========================================================

    disease_class = disease_result[
        "class"
    ]

    disease_present = (
        disease_class != "Tomato_Healthy"
        and not disease_result["low_confidence"]
    )

    # ========================================================
    # OVERALL RISK
    # ========================================================

    weather_risk_level = weather_result[
        "current_risk"
    ]["level"]

    if disease_present:

        if weather_risk_level == "VERY LOW":

            overall_risk = "LOW"

        elif weather_risk_level == "LOW":

            overall_risk = "MEDIUM"

        else:

            overall_risk = "HIGH"

    else:

        if weather_risk_level == "HIGH":

            overall_risk = "MEDIUM"

        elif weather_risk_level == "MEDIUM":

            overall_risk = "LOW"

        else:

            overall_risk = "LOW"

    # ========================================================
    # FINAL RESPONSE
    # ========================================================

    return {

        "success": True,

        "model_accuracy":
            MODEL_ACCURACY,

        "location":
            weather_result["location"],

        "photo_date":
            photo_date,

        "weather_on_photo_date":
            weather_result["photo_date_weather"],

        # Same data under the name used by the temporary frontend.
        "photo_date_weather":
            weather_result["photo_date_weather"],

        "current_weather":
            weather_result["current_weather"],

        "current_weather_risk":
            weather_result["current_risk"],

        "forecast":
            weather_result["forecast"],

        "disease_diagnosis":
            disease_result,

        "overall_risk_level":
            overall_risk,

        "farmer_advice":
            farmer_advice,
    }


# ============================================================
# BASIC HEALTH CHECK
# ============================================================

@app.get("/health")
def health_check():

    return {
        "status": "ok"
    }


# ============================================================
# RUN SERVER
# ============================================================

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
    )
