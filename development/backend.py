"""
AI Based Healthcare System — Backend
======================================
College mini project - INTEGRATED WITH TRAINED MODELS

This module contains all the "backend" logic integrated with trained ML models.
It connects to the model_integration module for real disease predictions.

Run a quick self-test with:
    python backend.py
"""

import sqlite3
import os
from datetime import datetime, date

# Import model integration
try:
    from model_integration import predict_disease, get_available_modules, get_module_features
    MODEL_INTEGRATION_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Model integration not available: {e}")
    MODEL_INTEGRATION_AVAILABLE = False
    predict_disease = None
    get_available_modules = None
    get_module_features = None


# -------------------------------------------------------------------
# DATABASE SETUP
# -------------------------------------------------------------------
DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "healthcare.db")


def get_connection():
    """Returns a new SQLite connection. Call close() on it when done,
    or use it as a context manager."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Creates all required tables if they don't already exist.
    Safe to call every time the app starts."""
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS chat_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            role TEXT NOT NULL,
            message TEXT NOT NULL,
            timestamp TEXT NOT NULL
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS mood_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            mood_score INTEGER NOT NULL,
            note TEXT,
            log_date TEXT NOT NULL
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS medications (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            dosage TEXT,
            reminder_time TEXT,
            taken_today INTEGER DEFAULT 0
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS sos_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            location TEXT NOT NULL,
            reason TEXT,
            timestamp TEXT NOT NULL
        )
    """)

    conn.commit()
    conn.close()


# -------------------------------------------------------------------
# CHAT LOGGING
# -------------------------------------------------------------------
def save_chat_message(role: str, message: str):
    conn = get_connection()
    conn.execute(
        "INSERT INTO chat_log (role, message, timestamp) VALUES (?, ?, ?)",
        (role, message, datetime.now().isoformat()),
    )
    conn.commit()
    conn.close()


def get_chat_history(limit: int = 50):
    conn = get_connection()
    rows = conn.execute(
        "SELECT role, message, timestamp FROM chat_log ORDER BY id DESC LIMIT ?",
        (limit,),
    ).fetchall()
    conn.close()
    return [dict(r) for r in reversed(rows)]


# -------------------------------------------------------------------
# CHATBOT / SYMPTOM NLP (rule-based placeholder for a trained model)
# -------------------------------------------------------------------
EMERGENCY_KEYWORDS = [
    "chest pain", "can't breathe", "cannot breathe", "suicidal",
    "unconscious", "heavy bleeding", "severe bleeding", "not breathing",
    "heart attack", "stroke", "overdose", "poisoning",
]

MENTAL_HEALTH_KEYWORDS = [
    "sad", "depressed", "anxious", "hopeless", "stressed", "panic",
    "lonely", "worthless", "overwhelmed", "can't sleep", "cannot sleep",
]

SYMPTOM_KEYWORDS = [
    "headache", "fever", "cough", "nausea", "fatigue", "dizziness",
    "sore throat", "vomiting", "pain", "ache", "breathless", "swelling",
    "itching", "rash", "bleeding", "weakness", "cold", "flu",
]

HEART_KEYWORDS     = ["heart", "cardiac", "cholesterol", "blood pressure", "hypertension", "angina"]
DIABETES_KEYWORDS  = ["diabetes", "blood sugar", "glucose", "insulin"]
KIDNEY_KEYWORDS    = ["kidney", "renal", "urine", "creatinine"]
LIVER_KEYWORDS     = ["liver", "jaundice", "hepatitis", "bilirubin"]
THYROID_KEYWORDS   = ["thyroid", "tsh", "hypothyroid", "hyperthyroid", "goitre", "goiter"]
LUNG_KEYWORDS      = ["lung", "breathing", "respiratory", "asthma", "copd", "smoking"]

CAPABILITY_KEYWORDS = [
    "what can you do", "help", "how do you work", "what are your features",
    "what do you offer", "capabilities", "what can i ask", "how to use",
    "features", "functions", "tools",
]

CAPABILITIES_REPLY = (
    "Here's what I can help you with:\n\n"
    "🔍 **Disease Prediction** — Use the sidebar to run ML-powered risk assessments for:\n"
    "  ❤️ Heart Disease  |  🫘 Kidney Disease  |  🩸 Diabetes  |  🫁 Lung Cancer\n"
    "  🦋 Thyroid  |  🫀 Liver Disease  |  📊 Glucose Survey Risk\n\n"
    "🟢 **Rule-Based Assessments** — Gallbladder risk & Mental Health screening\n\n"
    "📟 **Vitals Monitor** — Check heart rate, SpO2, and temperature for alerts\n\n"
    "💊 **Medication Reminders** — Log and track your daily medications\n\n"
    "🚑 **Emergency SOS** — Trigger an ambulance alert with your location\n\n"
    "💬 **Symptom Chat** — Describe your symptoms here and I'll guide you to the right tool.\n\n"
    "Just tell me what's bothering you, or pick a module from the sidebar!"
)


def get_bot_response(user_message: str) -> dict:
    """
    Rule-based intent router.

    TODO(model): Replace with a trained intent-classification model:
        import joblib
        clf = joblib.load("models/intent_classifier.pkl")
        intent = clf.predict([user_message])[0]

    Returns {"reply": str, "sos_triggered": bool}
    """
    text = user_message.lower()

    # --- Emergency ---
    if any(k in text for k in EMERGENCY_KEYWORDS):
        return {
            "reply": (
                "⚠️ This sounds like a medical emergency. "
                "I've flagged an SOS alert — please check the 🚑 SOS tab immediately, "
                "or call your local emergency number (112 / 911 / 999)."
            ),
            "sos_triggered": True,
        }

    # --- Capabilities / help ---
    if any(k in text for k in CAPABILITY_KEYWORDS):
        return {"reply": CAPABILITIES_REPLY, "sos_triggered": False}

    # --- Organ-specific routing ---
    if any(k in text for k in HEART_KEYWORDS):
        return {
            "reply": ("That sounds heart-related. Please use the ❤️ Heart Disease module in the sidebar "
                      "for a detailed ML-based risk assessment. It evaluates cholesterol, ECG, blood pressure, and more."),
            "sos_triggered": False,
        }
    if any(k in text for k in DIABETES_KEYWORDS):
        return {
            "reply": ("For diabetes or blood sugar concerns, head to the 🩸 Diabetes module. "
                      "You can also try the 📊 Survey module for a lifestyle-based glucose risk check."),
            "sos_triggered": False,
        }
    if any(k in text for k in KIDNEY_KEYWORDS):
        return {
            "reply": ("Kidney-related symptoms can be assessed in the 🫘 Kidney Disease module. "
                      "It evaluates creatinine, urea, blood pressure, and urinalysis results."),
            "sos_triggered": False,
        }
    if any(k in text for k in LIVER_KEYWORDS):
        return {
            "reply": ("For liver concerns, use the 🫀 Liver Disease module. "
                      "It analyses bilirubin, ALT, AST, albumin, and other liver function markers."),
            "sos_triggered": False,
        }
    if any(k in text for k in THYROID_KEYWORDS):
        return {
            "reply": ("Thyroid issues can be screened in the 🦋 Thyroid module. "
                      "Provide your TSH, T3, TT4 values and related history for a prediction."),
            "sos_triggered": False,
        }
    if any(k in text for k in LUNG_KEYWORDS):
        return {
            "reply": ("Respiratory symptoms can be checked in the 🫁 Lung module. "
                      "It uses a trained SVM model on smoking history, symptoms, and other factors."),
            "sos_triggered": False,
        }

    # --- Mental health ---
    if any(k in text for k in MENTAL_HEALTH_KEYWORDS):
        return {
            "reply": ("I'm sorry you're feeling this way. You're not alone. "
                      "Please visit the 🧠 Mental Health module for a structured assessment, "
                      "or consider reaching out to a mental health professional."),
            "sos_triggered": False,
        }

    # --- General symptoms ---
    if any(k in text for k in SYMPTOM_KEYWORDS):
        return {
            "reply": ("Thanks for sharing your symptoms. "
                      "For a structured assessment, please use one of the disease modules in the sidebar — "
                      "each one is tailored to specific conditions. "
                      "Type 'help' to see all available tools."),
            "sos_triggered": False,
        }

    # --- Greetings ---
    if any(k in text for k in ["hi", "hello", "hey", "good morning", "good evening"]):
        return {
            "reply": ("Hello! 👋 I'm your AI Health Assistant. "
                      "Describe your symptoms or type 'help' to see everything I can do."),
            "sos_triggered": False,
        }

    # --- Fallback ---
    return {
        "reply": ("I'm not sure I understood that. Try describing your symptoms, "
                  "or type 'help' to see all available features."),
        "sos_triggered": False,
    }


# -------------------------------------------------------------------
# SYMPTOM TRIAGE (rule-based placeholder for a trained classifier)
# -------------------------------------------------------------------
def check_symptoms(symptom_list: list, severity: int) -> dict:
    """
    TODO(model): Replace with your trained triage model
    (e.g. a Random Forest / classification model saved via joblib
    in the Model Saving stage):

        import joblib
        model = joblib.load("models/triage_model.pkl")
        risk = model.predict([[len(symptom_list), severity]])[0]
    """
    risk = "Low"
    if severity >= 8 or len(symptom_list) >= 4:
        risk = "High"
    elif severity >= 5:
        risk = "Medium"

    advice_map = {
        "Low": "Monitor your symptoms. Rest and stay hydrated.",
        "Medium": "Consider consulting a doctor within the next 24-48 hours.",
        "High": "Please seek medical attention soon. Consider using the SOS feature if this worsens.",
    }

    return {"risk_level": risk, "advice": advice_map[risk]}


# -------------------------------------------------------------------
# MENTAL HEALTH
# -------------------------------------------------------------------
def log_mood(mood_score: int, note: str = ""):
    conn = get_connection()
    conn.execute(
        "INSERT INTO mood_log (mood_score, note, log_date) VALUES (?, ?, ?)",
        (mood_score, note, date.today().isoformat()),
    )
    conn.commit()
    conn.close()

    return evaluate_mood(mood_score)


def evaluate_mood(mood_score: int) -> dict:
    """
    TODO(model): Replace with a trained sentiment/PHQ-9-style scoring
    model if you want something more rigorous than a threshold.
    """
    if mood_score <= 3:
        return {
            "level": "concerning",
            "message": ("It sounds like things are tough right now. Consider reaching out to a mental "
                        "health professional or a trusted person. You don't have to go through this alone."),
        }
    return {"level": "ok", "message": "Mood logged. Thanks for checking in with yourself today."}


def get_mood_history():
    conn = get_connection()
    rows = conn.execute("SELECT mood_score, note, log_date FROM mood_log ORDER BY id").fetchall()
    conn.close()
    return [dict(r) for r in rows]


# -------------------------------------------------------------------
# MEDICATION REMINDERS
# -------------------------------------------------------------------
def add_medication(name: str, dosage: str, reminder_time: str):
    """reminder_time should be a string like '09:00 AM' (frontend formats this)."""
    conn = get_connection()
    conn.execute(
        "INSERT INTO medications (name, dosage, reminder_time, taken_today) VALUES (?, ?, ?, 0)",
        (name, dosage, reminder_time),
    )
    conn.commit()
    conn.close()


def get_medications():
    conn = get_connection()
    rows = conn.execute(
        "SELECT id, name, dosage, reminder_time, taken_today FROM medications ORDER BY id"
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def mark_medication_taken(med_id: int, taken: bool):
    conn = get_connection()
    conn.execute(
        "UPDATE medications SET taken_today = ? WHERE id = ?",
        (1 if taken else 0, med_id),
    )
    conn.commit()
    conn.close()

    # TODO(scheduler): For real reminders (not just tracking), integrate
    # APScheduler or a push-notification service here, keyed off reminder_time.


# -------------------------------------------------------------------
# DEVICE / VITALS ALERTS
# -------------------------------------------------------------------
def check_vitals(heart_rate: float, spo2: float, temperature_c: float) -> list:
    """
    TODO(device): Replace number_input simulation in the frontend with
    a real device API integration (Fitbit / Apple Health / custom IoT
    sensor webhook) and feed those readings into this same function.
    """
    alerts = []
    if heart_rate > 120 or heart_rate < 45:
        alerts.append(f"Abnormal heart rate detected: {heart_rate} bpm")
    if spo2 < 92:
        alerts.append(f"Low blood oxygen detected: {spo2}%")
    if temperature_c > 38.5:
        alerts.append(f"High fever detected: {temperature_c}C")
    return alerts


# -------------------------------------------------------------------
# DISEASE PREDICTION - INTEGRATED WITH TRAINED MODELS
# -------------------------------------------------------------------
def _safe_predict_disease(module_name: str, patient_data: dict) -> dict:
    """Internal helper: calls the imported predict_disease only if it was
    actually bound (model_integration imported successfully). Checking
    `predict_disease is not None` directly (rather than the separate
    MODEL_INTEGRATION_AVAILABLE flag) lets type checkers correctly narrow
    away the Optional/None case, and guarantees this never calls None at
    runtime even if the two ever got out of sync."""
    if predict_disease is None:
        return {"error": "Model integration not available"}
    return predict_disease(module_name, patient_data)


def predict_heart_disease(patient_data):
    """Predict heart disease using trained model"""
    return _safe_predict_disease("heart", patient_data)


def predict_kidney_disease(patient_data):
    """Predict kidney disease using trained model"""
    return _safe_predict_disease("kidney", patient_data)


def predict_diabetes(patient_data):
    """Predict diabetes using trained model"""
    return _safe_predict_disease("diabetes", patient_data)


def predict_lung_disease(patient_data):
    """Predict lung disease using trained model"""
    return _safe_predict_disease("lung", patient_data)


def predict_thyroid_disease(patient_data):
    """Predict thyroid disease using trained model"""
    return _safe_predict_disease("thyroid", patient_data)


def predict_liver_disease(patient_data):
    """Predict liver disease using trained model"""
    return _safe_predict_disease("liver", patient_data)


def predict_survey_risk(patient_data):
    """Predict glucose risk from survey data using trained model"""
    return _safe_predict_disease("survey", patient_data)


def assess_gallbladder_risk(symptoms):
    """Assess gallbladder risk using rule-based engine"""
    return _safe_predict_disease("gallbladder", symptoms)


def assess_mental_health_risk(responses):
    """Assess mental health using rule-based engine"""
    return _safe_predict_disease("mental_health", responses)


def get_disease_modules():
    """Get list of available disease prediction modules"""
    if get_available_modules is not None:
        return get_available_modules()
    return []


def get_required_features(module_name):
    """Get required features for a disease module"""
    if get_module_features is not None:
        return get_module_features(module_name)
    return []


# -------------------------------------------------------------------
# EMERGENCY / SOS
# -------------------------------------------------------------------
def trigger_ambulance_call(location: str, reason: str = "User-triggered SOS") -> str:
    """
    TODO(integration): Replace this simulation with a real integration, e.g.:
      - Twilio Voice/SMS API to call/notify emergency contacts
      - Google Maps API to locate the nearest hospital
      - A webhook to a local emergency dispatch partner

    For a college project, simulating + logging this call is the
    appropriate (and safe) choice. Do not wire this to real dispatch
    services without proper authorization.
    """
    conn = get_connection()
    conn.execute(
        "INSERT INTO sos_log (location, reason, timestamp) VALUES (?, ?, ?)",
        (location, reason, datetime.now().isoformat()),
    )
    conn.commit()
    conn.close()

    return f"[SIMULATED] Ambulance dispatch requested for location: '{location}'. Reason: {reason}."


def get_sos_history():
    conn = get_connection()
    rows = conn.execute("SELECT location, reason, timestamp FROM sos_log ORDER BY id DESC").fetchall()
    conn.close()
    return [dict(r) for r in rows]


# -------------------------------------------------------------------
# SELF-TEST - run this file directly to sanity-check everything works
# -------------------------------------------------------------------
if __name__ == "__main__":
    init_db()
    print("Database initialized at:", DB_PATH)

    print("\n--- Chatbot test ---")
    print(get_bot_response("I have a bad headache and fever"))
    print(get_bot_response("I feel like I can't breathe"))

    print("\n--- Symptom checker test ---")
    print(check_symptoms(["Fever", "Cough", "Headache"], 7))

    print("\n--- Mood log test ---")
    print(log_mood(8, "Feeling good today"))
    print(get_mood_history())

    print("\n--- Medication test ---")
    add_medication("Paracetamol", "500mg", "09:00 AM")
    print(get_medications())

    print("\n--- Vitals test ---")
    print(check_vitals(130, 98, 37.0))

    print("\n--- SOS test ---")
    print(trigger_ambulance_call("123 MG Road, Bengaluru", reason="Self-test run"))
    print(get_sos_history())

    print("\nAll backend functions ran successfully.")