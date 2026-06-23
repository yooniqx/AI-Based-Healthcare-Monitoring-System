"""
Model Integration Module
========================
Connects trained ML models and rule-based engines to the frontend.
Handles model loading, prediction, and result formatting.
"""

import os
import sys
import pickle
import json
import numpy as np
import pandas as pd
from pathlib import Path

# Add parent directory to path to import rule-based engines
BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
sys.path.insert(0, str(MODELS_DIR))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Glucose / diabetes string-label map shared by both Diabetes and Survey models.
_GLUCOSE_LABEL_MAP = {
    "normal":      (0, "Normal",      False),
    "prediabetes": (1, "Prediabetes", True),
    "pre-diabetes":(1, "Prediabetes", True),
    "pre_diabetes":(1, "Prediabetes", True),
    "diabetes":    (2, "Diabetes",    True),
    "type1":       (2, "Type 1 Diabetes", True),
    "type2":       (2, "Type 2 Diabetes", True),
    "type_1":      (2, "Type 1 Diabetes", True),
    "type_2":      (2, "Type 2 Diabetes", True),
}

# Lung string-label map
_LUNG_LABEL_MAP = {
    "benign":     (0, "Benign",    False),
    "malignant":  (1, "Malignant", True),
    "normal":     (2, "Normal",    False),
    "no_cancer":  (2, "Normal",    False),
    "yes":        (1, "Malignant", True),
    "no":         (2, "Normal",    False),
}

# Liver string-label map  (1 = no disease, 2 = disease in original dataset)
_LIVER_LABEL_MAP = {
    "1":          (1, "Low",  False),
    "2":          (2, "High", True),
    "no_disease": (1, "Low",  False),
    "disease":    (2, "High", True),
    "normal":     (1, "Low",  False),
    "abnormal":   (2, "High", True),
}

# Heart / Kidney binary label map
_BINARY_LABEL_MAP = {
    "0": (0, False), "1": (1, True),
    "no": (0, False), "yes": (1, True),
    "negative": (0, False), "positive": (1, True),
    "normal": (0, False), "abnormal": (1, True),
    "ckd": (1, True), "notckd": (0, False), "not_ckd": (0, False),
}


def _safe_int(value) -> int:
    """Convert prediction to int if possible; return -1 otherwise."""
    try:
        return int(value)
    except (ValueError, TypeError):
        return -1


def _resolve_glucose(raw) -> tuple:
    """Return (int_label, display_label, is_abnormal)."""
    key = str(raw).strip().lower().replace(" ", "_").replace("-", "_")
    if key in _GLUCOSE_LABEL_MAP:
        return _GLUCOSE_LABEL_MAP[key]
    n = _safe_int(raw)
    if n >= 0:
        labels = ["Normal", "Prediabetes", "Diabetes"]
        return (n, labels[n] if n < len(labels) else str(raw), n >= 1)
    return (-1, str(raw), True)


def _resolve_lung(raw) -> tuple:
    """Return (int_label, display_label, is_abnormal)."""
    key = str(raw).strip().lower().replace(" ", "_")
    if key in _LUNG_LABEL_MAP:
        return _LUNG_LABEL_MAP[key]
    n = _safe_int(raw)
    if n >= 0:
        labels = ["Benign", "Malignant", "Normal"]
        return (n, labels[n] if n < len(labels) else str(raw), n == 1)
    return (-1, str(raw), True)


def _resolve_binary(raw) -> tuple:
    """Return (int_label, is_abnormal) for heart/kidney models."""
    key = str(raw).strip().lower().replace(" ", "_")
    if key in _BINARY_LABEL_MAP:
        return _BINARY_LABEL_MAP[key]
    n = _safe_int(raw)
    return (n, n != 0)


def _resolve_liver(raw) -> tuple:
    """Return (int_label, display_label, is_abnormal)."""
    key = str(raw).strip().lower().replace(" ", "_")
    if key in _LIVER_LABEL_MAP:
        return _LIVER_LABEL_MAP[key]
    n = _safe_int(raw)
    if n in (1, 2):
        return (n, "High" if n == 2 else "Low", n == 2)
    return (n, "High" if n != 1 else "Low", n != 1)


# Import rule-based engines
# NOTE: these modules live in MODELS_DIR, which is only added to sys.path
# at runtime (above). Static checkers like pyright/basedpyright can't see
# that, so they flag a false-positive "could not be resolved" here.
# See pyrightconfig.json (extraPaths) for the proper fix; the
# `type: ignore` comments below silence the residual warning either way.
try:
    from gallbladder_screening_engine import create_gallbladder_assessment  # type: ignore[reportMissingImports]
    from mental_health_screening_engine import create_mental_health_assessment  # type: ignore[reportMissingImports]
except ImportError as e:
    print(f"Warning: Could not import rule-based engines: {e}")
    create_gallbladder_assessment = None
    create_mental_health_assessment = None


class ModelRegistry:
    """Manages loading and accessing trained models"""
    
    def __init__(self):
        self.registry_path = MODELS_DIR / "model_registry.json"
        self.registry = self._load_registry()
        self.loaded_models = {}
        self.loaded_scalers = {}
    
    def _load_registry(self):
        """Load model registry JSON"""
        try:
            with open(self.registry_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading model registry: {e}")
            return {"models": {}}
    
    def get_module_info(self, module_name):
        """Get module information from registry"""
        return self.registry.get("models", {}).get(module_name)
    
    def load_model(self, module_name):
        """Load a trained model"""
        if module_name in self.loaded_models:
            return self.loaded_models[module_name]
        
        module_info = self.get_module_info(module_name)
        if not module_info:
            raise ValueError(f"Module {module_name} not found in registry")
        
        model_path = module_info.get("model_path")
        if not model_path:
            return None
        
        # Handle rule-based engines
        if module_info.get("model_type") == "RuleBasedScreeningEngine":
            return None  # Rule-based, no pickle file
        
        # Convert Windows path to cross-platform
        model_path = str(MODELS_DIR / Path(model_path).name)
        
        try:
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            self.loaded_models[module_name] = model
            return model
        except Exception as e:
            print(f"Error loading model for {module_name}: {e}")
            return None
    
    def load_scaler(self, module_name):
        """Load a scaler for a module"""
        if module_name in self.loaded_scalers:
            return self.loaded_scalers[module_name]
        
        module_info = self.get_module_info(module_name)
        if not module_info:
            return None
        
        scaler_path = module_info.get("scaler_path")
        if not scaler_path:
            return None
        
        # Convert Windows path to cross-platform
        scaler_path = str(MODELS_DIR / Path(scaler_path).name)
        
        try:
            with open(scaler_path, 'rb') as f:
                scaler = pickle.load(f)
            self.loaded_scalers[module_name] = scaler
            return scaler
        except Exception as e:
            print(f"Error loading scaler for {module_name}: {e}")
            return None
    
    def get_feature_names(self, module_name):
        """Get expected feature names for a module"""
        module_info = self.get_module_info(module_name)
        if module_info:
            return module_info.get("feature_names", [])
        return []
    
    def get_all_modules(self):
        """Get list of all available modules"""
        return list(self.registry.get("models", {}).keys())


class DiseasePredictor:
    """Handles disease prediction for all modules"""
    
    def __init__(self):
        self.registry = ModelRegistry()
    
    def predict_heart_disease(self, patient_data):
        """Predict heart disease risk"""
        try:
            model = self.registry.load_model("heart")
            scaler = self.registry.load_scaler("heart")
            features = self.registry.get_feature_names("heart")
            
            if not model or not scaler:
                return {"error": "Model not available"}
            
            # Prepare input
            X = np.array([[patient_data.get(f, 0) for f in features]])
            X_scaled = scaler.transform(X)
            
            # Predict
            prediction = model.predict(X_scaled)[0]
            probability = model.predict_proba(X_scaled)[0]

            pred_int, is_abnormal = _resolve_binary(prediction)
            confidence = max(probability) * 100

            return {
                "prediction": pred_int,
                "risk_level": "High" if is_abnormal else "Low",
                "confidence": f"{confidence:.1f}%",
                "probability_no_disease": f"{probability[0]*100:.1f}%",
                "probability_disease": f"{probability[1]*100:.1f}%",
                "recommendation": self._get_heart_recommendation(1 if is_abnormal else 0),
                "doctor_type": "Cardiologist" if is_abnormal else "General Physician",
            }
        except Exception as e:
            return {"error": str(e)}

    def predict_kidney_disease(self, patient_data):
        """Predict kidney disease risk"""
        try:
            model = self.registry.load_model("kidney")
            scaler = self.registry.load_scaler("kidney")
            features = self.registry.get_feature_names("kidney")
            
            if not model or not scaler:
                return {"error": "Model not available"}
            
            X = np.array([[patient_data.get(f, 0) for f in features]])
            X_scaled = scaler.transform(X)
            
            prediction = model.predict(X_scaled)[0]
            probability = model.predict_proba(X_scaled)[0]

            pred_int, is_abnormal = _resolve_binary(prediction)
            confidence = max(probability) * 100

            return {
                "prediction": pred_int,
                "risk_level": "High" if is_abnormal else "Low",
                "confidence": f"{confidence:.1f}%",
                "probability_no_disease": f"{probability[0]*100:.1f}%",
                "probability_disease": f"{probability[1]*100:.1f}%",
                "recommendation": self._get_kidney_recommendation(1 if is_abnormal else 0),
                "doctor_type": "Nephrologist" if is_abnormal else "General Physician",
            }
        except Exception as e:
            return {"error": str(e)}

    def predict_diabetes(self, patient_data):
        """Predict diabetes risk"""
        try:
            model = self.registry.load_model("diabetes")
            scaler = self.registry.load_scaler("diabetes")
            features = self.registry.get_feature_names("diabetes")
            
            if not model or not scaler:
                return {"error": "Model not available"}
            
            X = np.array([[patient_data.get(f, 0) for f in features]])
            X_scaled = scaler.transform(X)
            
            prediction = model.predict(X_scaled)[0]
            probability = model.predict_proba(X_scaled)[0]

            pred_int, risk_level, is_abnormal = _resolve_glucose(prediction)
            confidence = max(probability) * 100
            class_names = ["Normal", "Prediabetes", "Diabetes"]

            return {
                "prediction": pred_int,
                "risk_level": risk_level,
                "confidence": f"{confidence:.1f}%",
                "class_probabilities": {
                    class_names[i]: f"{probability[i]*100:.1f}%"
                    for i in range(min(len(class_names), len(probability)))
                },
                "recommendation": self._get_diabetes_recommendation(pred_int if pred_int >= 0 else 0),
                "doctor_type": "Endocrinologist" if is_abnormal else "General Physician",
            }
        except Exception as e:
            return {"error": str(e)}

    def predict_lung_disease(self, patient_data):
        """Predict lung disease risk"""
        try:
            model = self.registry.load_model("lung")
            scaler = self.registry.load_scaler("lung")
            features = self.registry.get_feature_names("lung")
            
            if not model or not scaler:
                return {"error": "Model not available"}
            
            X = np.array([[patient_data.get(f, 0) for f in features]])
            X_scaled = scaler.transform(X)
            
            prediction = model.predict(X_scaled)[0]

            pred_int, risk_level, is_abnormal = _resolve_lung(prediction)

            return {
                "prediction": pred_int,
                "risk_level": risk_level,
                "recommendation": self._get_lung_recommendation(pred_int if pred_int >= 0 else 2),
                "doctor_type": "Pulmonologist" if is_abnormal else "General Physician",
            }
        except Exception as e:
            return {"error": str(e)}
    
    # Thyroid class-label map: UCI dataset uses string labels like
    # 'subnormal', 'hypothyroid', 'negative', etc.  Models trained on it
    # predict strings, not integers.
    THYROID_LABEL_MAP = {
        "negative":               ("Normal",                         False),
        "subnormal":              ("Subnormal Thyroid",              True),
        "hypothyroid":            ("Hypothyroid",                    True),
        "primary_hypothyroid":    ("Primary Hypothyroid",            True),
        "compensated_hypothyroid":("Compensated Hypothyroid",        True),
        "t3_toxic":               ("Hyperthyroid (T3 Toxic)",        True),
        "hyperthyroid":           ("Hyperthyroid",                   True),
        "secondary_hypothyroid":  ("Secondary Hypothyroid",          True),
        "goitre":                 ("Goitre",                         True),
        "sick":                   ("Sick Euthyroid",                 True),
        "increased_binding_protein": ("Increased Binding Protein",   True),
        "decreased_binding_protein": ("Decreased Binding Protein",   True),
    }

    def predict_thyroid_disease(self, patient_data):
        """Predict thyroid disease risk.

        Handles models that return string class labels (e.g. 'subnormal',
        'hypothyroid') instead of integer indices — avoids ValueError from
        int(string_label).
        """
        try:
            model = self.registry.load_model("thyroid")
            scaler = self.registry.load_scaler("thyroid")
            features = self.registry.get_feature_names("thyroid")

            if not model or not scaler:
                return {"error": "Model not available"}

            X = np.array([[patient_data.get(f, 0) for f in features]])
            X_scaled = scaler.transform(X)

            raw_pred = model.predict(X_scaled)[0]
            probability = model.predict_proba(X_scaled)[0]
            confidence = max(probability) * 100

            # Normalise the prediction to a string key for map lookup
            pred_key = str(raw_pred).strip().lower().replace(" ", "_")

            matched = self.THYROID_LABEL_MAP.get(pred_key)
            if matched:
                risk_label, is_abnormal = matched
            else:
                # Numeric model: 0 = Normal, anything else = Abnormal
                try:
                    is_abnormal = int(raw_pred) != 0
                except (ValueError, TypeError):
                    is_abnormal = True   # unrecognised string label -> flag
                risk_label = "Abnormal" if is_abnormal else "Normal"

            # Safe serialisable value for JSON / display
            try:
                pred_out = int(raw_pred)
            except (ValueError, TypeError):
                pred_out = str(raw_pred)

            return {
                "prediction":    pred_out,
                "raw_label":     str(raw_pred),
                "risk_level":    risk_label,
                "status":        "Abnormal" if is_abnormal else "Normal",
                "confidence":    f"{confidence:.1f}%",
                "recommendation": self._get_thyroid_recommendation(1 if is_abnormal else 0),
                "doctor_type":   "Endocrinologist" if is_abnormal else "General Physician",
            }
        except Exception as e:
            return {"error": str(e)}
    
    def predict_liver_disease(self, patient_data):
        """Predict liver disease risk"""
        try:
            model = self.registry.load_model("liver")
            scaler = self.registry.load_scaler("liver")
            features = self.registry.get_feature_names("liver")
            
            if not model or not scaler:
                return {"error": "Model not available"}
            
            X = np.array([[patient_data.get(f, 0) for f in features]])
            X_scaled = scaler.transform(X)
            
            prediction = model.predict(X_scaled)[0]
            probability = model.predict_proba(X_scaled)[0]

            pred_int, risk_level, is_abnormal = _resolve_liver(prediction)
            confidence = max(probability) * 100

            return {
                "prediction": pred_int,
                "risk_level": risk_level,
                "confidence": f"{confidence:.1f}%",
                "recommendation": self._get_liver_recommendation(2 if is_abnormal else 1),
                "doctor_type": "Hepatologist" if is_abnormal else "General Physician",
            }
        except Exception as e:
            return {"error": str(e)}

    def predict_survey_risk(self, patient_data):
        """Predict glucose risk from survey data"""
        try:
            model = self.registry.load_model("survey")
            scaler = self.registry.load_scaler("survey")
            features = self.registry.get_feature_names("survey")
            
            if not model or not scaler:
                return {"error": "Model not available"}
            
            X = np.array([[patient_data.get(f, 0) for f in features]])
            X_scaled = scaler.transform(X)
            
            prediction = model.predict(X_scaled)[0]
            probability = model.predict_proba(X_scaled)[0]

            pred_int, risk_level, is_abnormal = _resolve_glucose(prediction)
            confidence = max(probability) * 100

            return {
                "prediction": pred_int,
                "risk_level": risk_level,
                "confidence": f"{confidence:.1f}%",
                "recommendation": self._get_diabetes_recommendation(pred_int if pred_int >= 0 else 0),
                "doctor_type": "Endocrinologist" if is_abnormal else "General Physician",
            }
        except Exception as e:
            return {"error": str(e)}

    def assess_gallbladder_risk(self, symptoms):
        """Assess gallbladder disease risk using rule-based engine"""
        try:
            if create_gallbladder_assessment is None:
                return {"error": "Gallbladder screening engine not available"}
            
            result = create_gallbladder_assessment(symptoms)
            return result
        except Exception as e:
            return {"error": str(e)}
    
    def assess_mental_health(self, responses):
        """Assess mental health using rule-based engine"""
        try:
            if create_mental_health_assessment is None:
                return {"error": "Mental health screening engine not available"}
            
            result = create_mental_health_assessment(responses)
            return result
        except Exception as e:
            return {"error": str(e)}
    
    # Recommendation helpers
    def _get_heart_recommendation(self, prediction):
        if prediction == 1:
            return "High risk detected. Consult a cardiologist immediately. Recommended tests: ECG, Echocardiogram, Stress Test."
        return "Low risk. Maintain healthy lifestyle: regular exercise, balanced diet, stress management."
    
    def _get_kidney_recommendation(self, prediction):
        if prediction == 1:
            return "Kidney disease indicators detected. Consult a nephrologist. Recommended tests: Creatinine, GFR, Urinalysis."
        return "Kidney function appears normal. Stay hydrated and maintain healthy blood pressure."
    
    def _get_diabetes_recommendation(self, prediction):
        if prediction == 2:
            return "Diabetes indicators detected. Consult an endocrinologist. Monitor blood glucose regularly."
        elif prediction == 1:
            return "Prediabetes detected. Lifestyle modifications recommended. Consult a doctor for prevention plan."
        return "Normal glucose levels. Maintain healthy diet and regular physical activity."
    
    def _get_lung_recommendation(self, prediction):
        if prediction == 1:
            return "Malignant indicators detected. Urgent consultation with pulmonologist required. Further imaging needed."
        elif prediction == 0:
            return "Benign findings. Follow-up with pulmonologist recommended for monitoring."
        return "Normal lung function. Continue healthy lifestyle and avoid smoking."
    
    def _get_thyroid_recommendation(self, prediction):
        if prediction != 0:
            return "Thyroid abnormality detected. Consult an endocrinologist. Recommended tests: TSH, T3, T4."
        return "Thyroid function appears normal. Regular monitoring recommended if symptoms develop."
    
    def _get_liver_recommendation(self, prediction):
        if prediction == 2:
            return "Liver disease indicators detected. Consult a hepatologist. Recommended tests: Liver function panel, ultrasound."
        return "Liver function appears normal. Limit alcohol consumption and maintain healthy weight."


# Global predictor instance
predictor = DiseasePredictor()


# Convenience functions
def predict_disease(module_name, patient_data):
    """
    Generic prediction function for any disease module
    
    Parameters:
    -----------
    module_name : str
        Name of the disease module (heart, kidney, diabetes, etc.)
    patient_data : dict
        Patient data with feature values
    
    Returns:
    --------
    dict: Prediction results
    """
    prediction_methods = {
        "heart": predictor.predict_heart_disease,
        "kidney": predictor.predict_kidney_disease,
        "diabetes": predictor.predict_diabetes,
        "lung": predictor.predict_lung_disease,
        "thyroid": predictor.predict_thyroid_disease,
        "liver": predictor.predict_liver_disease,
        "survey": predictor.predict_survey_risk,
        "gallbladder": predictor.assess_gallbladder_risk,
        "mental_health": predictor.assess_mental_health,
    }
    
    method = prediction_methods.get(module_name)
    if method:
        return method(patient_data)
    else:
        return {"error": f"Module {module_name} not supported"}


def get_available_modules():
    """Get list of all available disease modules"""
    return predictor.registry.get_all_modules()


def get_module_features(module_name):
    """Get required features for a module"""
    return predictor.registry.get_feature_names(module_name)


if __name__ == "__main__":
    # Test the integration
    print("Testing Model Integration...")
    print(f"Available modules: {get_available_modules()}")
    print(f"\nHeart features: {get_module_features('heart')[:5]}...")
    print("\nModel integration module ready!")

