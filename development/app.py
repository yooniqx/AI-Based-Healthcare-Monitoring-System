# AI Healthcare System - Integrated Frontend
import sys
import os

# Ensure 'development/' is on sys.path so that sibling modules (backend,
# model_integration) can be imported regardless of the working directory
# Render launches Streamlit from the repo root, so this is required.
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, date, time as dtime

try:
    from backend import (init_db, get_bot_response, check_vitals, trigger_ambulance_call,
        log_mood, get_mood_history, add_medication, get_medications,
        predict_heart_disease, predict_kidney_disease, predict_diabetes,
        predict_lung_disease, predict_thyroid_disease, predict_liver_disease,
        predict_survey_risk, assess_gallbladder_risk, assess_mental_health_risk)
    BACKEND_AVAILABLE = True
except ImportError:
    BACKEND_AVAILABLE = False
    init_db = None
    get_bot_response = None
    check_vitals = None
    trigger_ambulance_call = None
    log_mood = None
    get_mood_history = None
    add_medication = None
    get_medications = None
    predict_heart_disease = None
    predict_kidney_disease = None
    predict_diabetes = None
    predict_lung_disease = None
    predict_thyroid_disease = None
    predict_liver_disease = None
    predict_survey_risk = None
    assess_gallbladder_risk = None
    assess_mental_health_risk = None

st.set_page_config(page_title="AI Healthcare", page_icon="🩺", layout="wide")

if BACKEND_AVAILABLE and init_db is not None:
    init_db()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = [{'role': 'assistant', 'content': 'Hi, I am your AI Health Assistant. Tell me your symptoms or use the sidebar to navigate to a specific prediction tool.'}]
if "medications" not in st.session_state:
    st.session_state.medications = []
if "mood_log" not in st.session_state:
    st.session_state.mood_log = []
if "sos_triggered" not in st.session_state:
    st.session_state.sos_triggered = False

st.sidebar.title("🩺 AI Healthcare System")
page = st.sidebar.radio("Navigate", [
    "🏠 Home", "💬 Chatbot", "❤️ Heart", "🫘 Kidney", "🩸 Diabetes",
    "🫁 Lung", "🦋 Thyroid", "🫀 Liver", "📊 Survey", "🟢 Gallbladder",
    "🧠 Mental Health", "💊 Medications", "📟 Vitals", "🚑 SOS"])

if st.session_state.sos_triggered:
    st.sidebar.error("⚠️ SOS alert active")

# ── Home ─────────────────────────────────────────────────────────────────────
if page == "🏠 Home":
    st.title("🩺 AI Healthcare Monitoring System")
    st.info("Integrated with 9 trained ML models")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### ML Models")
        st.write("❤️ Heart (RF 86.7%)")
        st.write("🫘 Kidney (LR 98.8%)")
        st.write("🩸 Diabetes (LR 92.9%)")
        st.write("🫁 Lung (SVM)")
        st.write("🦋 Thyroid (RF 98.9%)")
        st.write("🫀 Liver (RF 72.5%)")
        st.write("📊 Survey (RF 100%)")
    with col2:
        st.markdown("### Rule-Based")
        st.write("🟢 Gallbladder")
        st.write("🧠 Mental Health")
    if BACKEND_AVAILABLE:
        st.success("✅ System Ready")
    else:
        st.error("❌ Backend not available")

# ── Chatbot ──────────────────────────────────────────────────────────────────
elif page == "💬 Chatbot":
    st.title("💬 Health Chatbot")
    for msg in st.session_state.chat_history:
        with st.chat_message(msg['role']):
            st.write(msg['content'])
    user_input = st.chat_input("Describe your symptoms or ask a health question...")
    if user_input:
        st.session_state.chat_history.append({'role': 'user', 'content': user_input})
        if get_bot_response is not None:
            bot_response = get_bot_response(user_input)
            reply = bot_response.get('reply', 'I am here to help.')
            if bot_response.get('sos_triggered'):
                st.session_state.sos_triggered = True
        else:
            reply = "Backend not available. Please check your installation."
        st.session_state.chat_history.append({'role': 'assistant', 'content': reply})
        st.rerun()

# ── Heart ─────────────────────────────────────────────────────────────────────
elif page == "❤️ Heart":
    st.title("❤️ Heart Disease Prediction")
    with st.form("heart_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            age = st.number_input("Age", 1, 120, 50)
            sex = st.selectbox("Sex", [1, 0], format_func=lambda x: "Male" if x==1 else "Female")
            cp = st.selectbox("Chest Pain Type", [0,1,2,3], help="0=Typical Angina, 1=Atypical Angina, 2=Non-anginal, 3=Asymptomatic")
            bp = st.number_input("Resting BP (mmHg)", 80, 200, 120)
            chol = st.number_input("Cholesterol (mg/dL)", 100, 600, 200)
        with c2:
            fbs = st.selectbox("Fasting Blood Sugar >120 mg/dL", [0,1], format_func=lambda x: "Yes" if x==1 else "No")
            ecg = st.selectbox("Resting ECG", [0,1,2], help="0=Normal, 1=ST-T Abnormality, 2=LV Hypertrophy")
            hr = st.number_input("Max Heart Rate", 60, 220, 150)
            angina = st.selectbox("Exercise Induced Angina", [0,1], format_func=lambda x: "Yes" if x==1 else "No")
        with c3:
            std = st.number_input("ST Depression", 0.0, 10.0, 0.0, 0.1)
            slope = st.selectbox("ST Slope", [0,1,2], help="0=Upsloping, 1=Flat, 2=Downsloping")
            vessels = st.selectbox("Major Vessels (0–3)", [0,1,2,3])
            thal = st.selectbox("Thalassemia", [0,1,2,3], help="0=Normal, 1=Fixed Defect, 2=Reversible Defect")
        if st.form_submit_button("🔍 Predict", type="primary"):
            if predict_heart_disease is None:
                st.error("Backend not available")
            else:
                data = {'age':age,'sex':sex,'chest_pain_type':cp,'resting_bp':bp,'cholesterol':chol,
                       'fasting_blood_sugar':fbs,'resting_ecg':ecg,'max_heart_rate':hr,
                       'exercise_induced_angina':angina,'st_depression':std,'st_slope':slope,
                       'num_major_vessels':vessels,'thalassemia':thal}
                result = predict_heart_disease(data)
                if 'error' in result:
                    st.error(result['error'])
                else:
                    c1,c2,c3 = st.columns(3)
                    c1.metric("Risk", result.get('risk_level'))
                    c2.metric("Confidence", result.get('confidence'))
                    c3.metric("Specialist", result.get('doctor_type'))
                    st.info(result.get('recommendation'))

# ── Kidney ────────────────────────────────────────────────────────────────────
elif page == "🫘 Kidney":
    st.title("🫘 Kidney Disease Prediction")
    with st.form("kidney_form"):
        c1,c2,c3 = st.columns(3)
        with c1:
            age = st.number_input("Age", 1, 120, 50)
            bp = st.number_input("Blood Pressure (mmHg)", 50, 200, 80)
            sg = st.number_input("Specific Gravity", 1.0, 1.03, 1.02, 0.005)
            al = st.selectbox("Albumin (0–5)", [0,1,2,3,4,5])
            su = st.selectbox("Sugar (0–5)", [0,1,2,3,4,5])
            rbc = st.selectbox("Red Blood Cells", [0,1], format_func=lambda x: "Abnormal" if x==1 else "Normal")
            pc = st.selectbox("Pus Cells", [0,1], format_func=lambda x: "Abnormal" if x==1 else "Normal")
            pcc = st.selectbox("Pus Cell Clumps", [0,1], format_func=lambda x: "Present" if x==1 else "Not Present")
        with c2:
            ba = st.selectbox("Bacteria", [0,1], format_func=lambda x: "Present" if x==1 else "Not Present")
            bgr = st.number_input("Blood Glucose Random (mg/dL)", 50, 500, 120)
            bu = st.number_input("Blood Urea (mg/dL)", 10, 200, 40)
            sc = st.number_input("Serum Creatinine (mg/dL)", 0.5, 15.0, 1.0, 0.1)
            sod = st.number_input("Sodium (mEq/L)", 100, 200, 140)
            pot = st.number_input("Potassium (mEq/L)", 2.0, 10.0, 4.5, 0.1)
            hemo = st.number_input("Hemoglobin (g/dL)", 5.0, 20.0, 13.0, 0.1)
            pcv = st.number_input("Packed Cell Volume (%)", 20, 60, 40)
        with c3:
            wc = st.number_input("WBC Count (cells/μL)", 2000, 30000, 8000)
            rc = st.number_input("RBC Count (millions/μL)", 2.0, 8.0, 5.0, 0.1)
            htn = st.selectbox("Hypertension", [0,1], format_func=lambda x: "Yes" if x==1 else "No")
            dm = st.selectbox("Diabetes Mellitus", [0,1], format_func=lambda x: "Yes" if x==1 else "No")
            cad = st.selectbox("Coronary Artery Disease", [0,1], format_func=lambda x: "Yes" if x==1 else "No")
            appet = st.selectbox("Appetite", [0,1], format_func=lambda x: "Poor" if x==1 else "Good")
            pe = st.selectbox("Pedal Edema", [0,1], format_func=lambda x: "Yes" if x==1 else "No")
            ane = st.selectbox("Anemia", [0,1], format_func=lambda x: "Yes" if x==1 else "No")
        if st.form_submit_button("🔍 Predict", type="primary"):
            if predict_kidney_disease is None:
                st.error("Backend not available")
            else:
                data = {'age':age,'blood_pressure':bp,'specific_gravity':sg,'albumin':al,'sugar':su,
                       'red_blood_cells':rbc,'pus_cells':pc,'pus_cell_clumps':pcc,'bacteria':ba,
                       'blood_glucose_random':bgr,'blood_urea':bu,'serum_creatinine':sc,'sodium':sod,
                       'potassium':pot,'hemoglobin':hemo,'packed_cell_volume':pcv,
                       'white_blood_cell_count':wc,'red_blood_cell_count':rc,'hypertension':htn,
                       'diabetes_mellitus':dm,'coronary_artery_disease':cad,'appetite':appet,
                       'pedal_edema':pe,'anemia':ane}
                result = predict_kidney_disease(data)
                if 'error' in result:
                    st.error(result['error'])
                else:
                    c1,c2,c3 = st.columns(3)
                    c1.metric("Risk", result.get('risk_level'))
                    c2.metric("Confidence", result.get('confidence'))
                    c3.metric("Specialist", result.get('doctor_type'))
                    st.info(result.get('recommendation'))

# ── Diabetes ──────────────────────────────────────────────────────────────────
elif page == "🩸 Diabetes":
    st.title("🩸 Diabetes Prediction")
    with st.form("diabetes_form"):
        c1,c2 = st.columns(2)
        with c1:
            pid = st.number_input("Patient ID", 1, 100000, 1001)
            ming = st.number_input("Min Blood Glucose (mg/dL)", 50, 300, 80)
            maxg = st.number_input("Max Blood Glucose (mg/dL)", 80, 500, 180)
            stdg = st.number_input("Std Dev Blood Glucose", 0.0, 100.0, 20.0, 0.1)
            medg = st.number_input("Median Blood Glucose (mg/dL)", 50, 400, 120)
            avri = st.number_input("Avg Regular Insulin (units)", 0.0, 100.0, 5.0, 0.1)
            avni = st.number_input("Avg NPH Insulin (units)", 0.0, 100.0, 10.0, 0.1)
            tdi = st.number_input("Total Daily Insulin (units)", 0.0, 200.0, 15.0, 0.1)
        with c2:
            ngm = st.number_input("# Glucose Measurements", 1, 1000, 50)
            nid = st.number_input("# Insulin Doses", 0, 500, 10)
            nhe = st.number_input("# Hypoglycemic Events", 0, 100, 2)
            dm = st.number_input("Days Monitored", 1, 365, 30)
            gv = st.number_input("Glucose Variability (%CV)", 0.0, 100.0, 15.0, 0.1)
            hyper = st.number_input("Hyperglycemia Time (%)", 0.0, 100.0, 20.0, 0.1)
            hypo = st.number_input("Hypoglycemia Time (%)", 0.0, 100.0, 5.0, 0.1)
            tr = st.number_input("Time in Target Range (%)", 0.0, 100.0, 75.0, 0.1)
        if st.form_submit_button("🔍 Predict", type="primary"):
            if predict_diabetes is None:
                st.error("Backend not available")
            else:
                data = {'Patient_ID':pid,'Min_Blood_Glucose':ming,'Max_Blood_Glucose':maxg,
                       'Std_Blood_Glucose':stdg,'Median_Blood_Glucose':medg,'Avg_Regular_Insulin':avri,
                       'Avg_NPH_Insulin':avni,'Total_Daily_Insulin':tdi,'Num_Glucose_Measurements':ngm,
                       'Num_Insulin_Doses':nid,'Num_Hypoglycemic_Events':nhe,'Days_Monitored':dm,
                       'Glucose_Variability':gv,'Hyperglycemia_Percentage':hyper,
                       'Hypoglycemia_Percentage':hypo,'Target_Range_Percentage':tr}
                result = predict_diabetes(data)
                if 'error' in result:
                    st.error(result['error'])
                else:
                    c1,c2,c3 = st.columns(3)
                    c1.metric("Classification", result.get('risk_level'))
                    c2.metric("Confidence", result.get('confidence'))
                    c3.metric("Specialist", result.get('doctor_type'))
                    st.info(result.get('recommendation'))

# ── Lung ──────────────────────────────────────────────────────────────────────
elif page == "🫁 Lung":
    st.title("🫁 Lung Disease Prediction")
    with st.form("lung_form"):
        c1,c2,c3 = st.columns(3)
        with c1:
            age = st.number_input("Age", 1, 120, 50)
            gender = st.selectbox("Gender", [1, 0], format_func=lambda x: "Male" if x==1 else "Female")
            smoking = st.selectbox("Smoking", [1, 2], format_func=lambda x: "Yes" if x==2 else "No")
            yellow_fingers = st.selectbox("Yellow Fingers", [1, 2], format_func=lambda x: "Yes" if x==2 else "No")
            anxiety = st.selectbox("Anxiety", [1, 2], format_func=lambda x: "Yes" if x==2 else "No")
        with c2:
            peer_pressure = st.selectbox("Peer Pressure", [1, 2], format_func=lambda x: "Yes" if x==2 else "No")
            chronic_disease = st.selectbox("Chronic Disease", [1, 2], format_func=lambda x: "Yes" if x==2 else "No")
            fatigue = st.selectbox("Fatigue", [1, 2], format_func=lambda x: "Yes" if x==2 else "No")
            allergy = st.selectbox("Allergy", [1, 2], format_func=lambda x: "Yes" if x==2 else "No")
            wheezing = st.selectbox("Wheezing", [1, 2], format_func=lambda x: "Yes" if x==2 else "No")
        with c3:
            alcohol = st.selectbox("Alcohol Consuming", [1, 2], format_func=lambda x: "Yes" if x==2 else "No")
            coughing = st.selectbox("Coughing", [1, 2], format_func=lambda x: "Yes" if x==2 else "No")
            shortness = st.selectbox("Shortness of Breath", [1, 2], format_func=lambda x: "Yes" if x==2 else "No")
            swallowing = st.selectbox("Swallowing Difficulty", [1, 2], format_func=lambda x: "Yes" if x==2 else "No")
            chest_pain = st.selectbox("Chest Pain", [1, 2], format_func=lambda x: "Yes" if x==2 else "No")
        if st.form_submit_button("🔍 Predict", type="primary"):
            if predict_lung_disease is None:
                st.error("Backend not available")
            else:
                data = {'GENDER':gender,'AGE':age,'SMOKING':smoking,'YELLOW_FINGERS':yellow_fingers,
                       'ANXIETY':anxiety,'PEER_PRESSURE':peer_pressure,'CHRONIC_DISEASE':chronic_disease,
                       'FATIGUE ':fatigue,'ALLERGY ':allergy,'WHEEZING':wheezing,
                       'ALCOHOL_CONSUMING':alcohol,'COUGHING':coughing,
                       'SHORTNESS_OF_BREATH':shortness,'SWALLOWING_DIFFICULTY':swallowing,
                       'CHEST_PAIN':chest_pain}
                result = predict_lung_disease(data)
                if 'error' in result:
                    st.error(result['error'])
                else:
                    c1,c2 = st.columns(2)
                    c1.metric("Classification", result.get('risk_level'))
                    c2.metric("Specialist", result.get('doctor_type'))
                    st.info(result.get('recommendation'))

# ── Thyroid ───────────────────────────────────────────────────────────────────
elif page == "🦋 Thyroid":
    st.title("🦋 Thyroid Disease Prediction")
    with st.form("thyroid_form"):
        c1,c2,c3 = st.columns(3)
        with c1:
            age = st.number_input("Age", 1, 120, 40)
            sex = st.selectbox("Sex", [1, 0], format_func=lambda x: "Male" if x==1 else "Female")
            on_thyroxine = st.selectbox("On Thyroxine", [0,1], format_func=lambda x: "Yes" if x==1 else "No")
            query_thyroxine = st.selectbox("Query on Thyroxine", [0,1], format_func=lambda x: "Yes" if x==1 else "No")
            on_antithyroid = st.selectbox("On Antithyroid Medication", [0,1], format_func=lambda x: "Yes" if x==1 else "No")
            sick = st.selectbox("Sick", [0,1], format_func=lambda x: "Yes" if x==1 else "No")
            pregnant = st.selectbox("Pregnant", [0,1], format_func=lambda x: "Yes" if x==1 else "No")
        with c2:
            thyroid_surgery = st.selectbox("Thyroid Surgery", [0,1], format_func=lambda x: "Yes" if x==1 else "No")
            i131_treatment = st.selectbox("I131 Treatment", [0,1], format_func=lambda x: "Yes" if x==1 else "No")
            query_hypothyroid = st.selectbox("Query Hypothyroid", [0,1], format_func=lambda x: "Yes" if x==1 else "No")
            query_hyperthyroid = st.selectbox("Query Hyperthyroid", [0,1], format_func=lambda x: "Yes" if x==1 else "No")
            lithium = st.selectbox("Lithium", [0,1], format_func=lambda x: "Yes" if x==1 else "No")
            goitre = st.selectbox("Goitre", [0,1], format_func=lambda x: "Yes" if x==1 else "No")
            tumor = st.selectbox("Tumor", [0,1], format_func=lambda x: "Yes" if x==1 else "No")
        with c3:
            hypopituitary = st.selectbox("Hypopituitary", [0,1], format_func=lambda x: "Yes" if x==1 else "No")
            psych = st.selectbox("Psych", [0,1], format_func=lambda x: "Yes" if x==1 else "No")
            tsh = st.number_input("TSH (mIU/L)", 0.0, 100.0, 2.5, 0.1)
            t3 = st.number_input("T3 (nmol/L)", 0.0, 15.0, 2.0, 0.1)
            tt4 = st.number_input("TT4 (nmol/L)", 0.0, 300.0, 100.0, 1.0)
            t4u = st.number_input("T4U (ratio)", 0.0, 3.0, 1.0, 0.01)
            fti = st.number_input("FTI (Free T4 Index)", 0.0, 300.0, 100.0, 1.0)
        if st.form_submit_button("🔍 Predict", type="primary"):
            if predict_thyroid_disease is None:
                st.error("Backend not available")
            else:
                data = {'age':age,'sex':sex,'on_thyroxine':on_thyroxine,'query_on_thyroxine':query_thyroxine,
                       'on_antithyroid_medication':on_antithyroid,'sick':sick,'pregnant':pregnant,
                       'thyroid_surgery':thyroid_surgery,'I131_treatment':i131_treatment,
                       'query_hypothyroid':query_hypothyroid,'query_hyperthyroid':query_hyperthyroid,
                       'lithium':lithium,'goitre':goitre,'tumor':tumor,
                       'hypopituitary':hypopituitary,'psych':psych,
                       'TSH':tsh,'T3':t3,'TT4':tt4,'T4U':t4u,'FTI':fti}
                result = predict_thyroid_disease(data)
                if 'error' in result:
                    st.error(result['error'])
                else:
                    c1,c2,c3 = st.columns(3)
                    c1.metric("Status", result.get('risk_level'))
                    c2.metric("Confidence", result.get('confidence'))
                    c3.metric("Specialist", result.get('doctor_type'))
                    st.info(result.get('recommendation'))

# ── Liver ─────────────────────────────────────────────────────────────────────
elif page == "🫀 Liver":
    st.title("🫀 Liver Disease Prediction")
    with st.form("liver_form"):
        c1,c2,c3 = st.columns(3)
        with c1:
            age = st.number_input("Age", 1, 120, 40)
            gender = st.selectbox("Gender", [1, 0], format_func=lambda x: "Male" if x==1 else "Female")
            tb = st.number_input("Total Bilirubin (mg/dL)", 0.1, 100.0, 1.0, 0.1)
            db = st.number_input("Direct Bilirubin (mg/dL)", 0.0, 50.0, 0.3, 0.1)
        with c2:
            alkphos = st.number_input("Alkaline Phosphotase (IU/L)", 50, 2000, 200)
            sgpt = st.number_input("SGPT / ALT (IU/L)", 5, 2000, 35)
            sgot = st.number_input("SGOT / AST (IU/L)", 5, 5000, 40)
        with c3:
            tp = st.number_input("Total Proteins (g/dL)", 1.0, 15.0, 7.0, 0.1)
            alb = st.number_input("Albumin (g/dL)", 0.5, 10.0, 3.5, 0.1)
            agr = st.number_input("Albumin/Globulin Ratio", 0.1, 5.0, 1.0, 0.01)
        if st.form_submit_button("🔍 Predict", type="primary"):
            if predict_liver_disease is None:
                st.error("Backend not available")
            else:
                data = {'Age':age,'Gender':gender,'Total_Bilirubin':tb,'Direct_Bilirubin':db,
                       'Alkaline_Phosphotase':alkphos,'Alamine_Aminotransferase':sgpt,
                       'Aspartate_Aminotransferase':sgot,'Total_Protiens':tp,
                       'Albumin':alb,'Albumin_and_Globulin_Ratio':agr}
                result = predict_liver_disease(data)
                if 'error' in result:
                    st.error(result['error'])
                else:
                    c1,c2,c3 = st.columns(3)
                    c1.metric("Risk", result.get('risk_level'))
                    c2.metric("Confidence", result.get('confidence'))
                    c3.metric("Specialist", result.get('doctor_type'))
                    st.info(result.get('recommendation'))

# ── Survey ────────────────────────────────────────────────────────────────────
elif page == "📊 Survey":
    st.title("📊 Glucose Risk Survey")
    st.caption("Survey-based glucose risk assessment using lifestyle and symptom data")
    with st.form("survey_form"):
        c1,c2 = st.columns(2)
        with c1:
            age = st.number_input("Age", 1, 120, 45)
            bmi = st.number_input("BMI", 10.0, 60.0, 25.0, 0.1)
            fasting_glucose = st.number_input("Fasting Glucose (mg/dL)", 50, 400, 100)
            hba1c = st.number_input("HbA1c (%)", 3.0, 15.0, 5.5, 0.1)
            family_history = st.selectbox("Family History of Diabetes", [0,1], format_func=lambda x: "Yes" if x==1 else "No")
            physical_activity = st.selectbox("Physical Activity Level", [0,1,2], format_func=lambda x: ["Low","Moderate","High"][x])
        with c2:
            diet_quality = st.selectbox("Diet Quality", [0,1,2], format_func=lambda x: ["Poor","Average","Good"][x])
            smoking = st.selectbox("Smoking", [0,1], format_func=lambda x: "Yes" if x==1 else "No")
            hypertension = st.selectbox("Hypertension", [0,1], format_func=lambda x: "Yes" if x==1 else "No")
            frequent_urination = st.selectbox("Frequent Urination", [0,1], format_func=lambda x: "Yes" if x==1 else "No")
            excessive_thirst = st.selectbox("Excessive Thirst", [0,1], format_func=lambda x: "Yes" if x==1 else "No")
            fatigue = st.selectbox("Fatigue", [0,1], format_func=lambda x: "Yes" if x==1 else "No")
        if st.form_submit_button("🔍 Assess Risk", type="primary"):
            if predict_survey_risk is None:
                st.error("Backend not available")
            else:
                data = {'age':age,'bmi':bmi,'fasting_glucose':fasting_glucose,'hba1c':hba1c,
                       'family_history':family_history,'physical_activity':physical_activity,
                       'diet_quality':diet_quality,'smoking':smoking,'hypertension':hypertension,
                       'frequent_urination':frequent_urination,'excessive_thirst':excessive_thirst,
                       'fatigue':fatigue}
                result = predict_survey_risk(data)
                if 'error' in result:
                    st.error(result['error'])
                else:
                    c1,c2,c3 = st.columns(3)
                    c1.metric("Classification", result.get('risk_level'))
                    c2.metric("Confidence", result.get('confidence'))
                    c3.metric("Specialist", result.get('doctor_type'))
                    st.info(result.get('recommendation'))

# ── Gallbladder ───────────────────────────────────────────────────────────────
elif page == "🟢 Gallbladder":
    st.title("🟢 Gallbladder Risk Assessment")
    with st.form("gallbladder_form"):
        c1,c2 = st.columns(2)
        with c1:
            age = st.number_input("Age", 1, 120, 45)
            gender = st.selectbox("Gender", ['male', 'female'])
            bmi = st.number_input("BMI", 10.0, 60.0, 25.0, 0.1)
            urp = st.checkbox("Upper Right Abdominal Pain")
            pafm = st.checkbox("Pain After Fatty Meals")
            nausea = st.checkbox("Nausea")
            vomit = st.checkbox("Vomiting")
        with c2:
            bloat = st.checkbox("Bloating")
            fever = st.checkbox("Fever")
            jaund = st.checkbox("Jaundice")
            fh = st.checkbox("Family History of Gallstones")
            pi = st.checkbox("Previous Gallbladder Issues")
        if st.form_submit_button("🔍 Assess Risk", type="primary"):
            if assess_gallbladder_risk is None:
                st.error("Backend not available")
            else:
                symptoms = {'age':age,'gender':gender,'bmi':bmi,'upper_right_pain':urp,
                           'pain_after_fatty_meals':pafm,'nausea':nausea,'vomiting':vomit,
                           'bloating':bloat,'fever':fever,'jaundice':jaund,
                           'family_history':fh,'previous_issues':pi}
                result = assess_gallbladder_risk(symptoms)
                if 'error' in result:
                    st.error(result['error'])
                else:
                    c1,c2 = st.columns(2)
                    c1.metric("Risk Score", result.get('risk_score'))
                    c2.metric("Risk Level", result.get('risk_level'))
                    st.info(result.get('recommendation'))

# ── Mental Health ─────────────────────────────────────────────────────────────
elif page == "🧠 Mental Health":
    st.title("🧠 Mental Health Assessment")
    st.caption("Rate each item: 0 = Never  |  1 = Sometimes  |  2 = Often  |  3 = Almost Always")
    questions = [
        'Feeling stressed or overwhelmed',
        'Difficulty relaxing even when free',
        'Trouble sleeping or poor sleep quality',
        'Excessive worrying about everyday things',
        'Difficulty concentrating or focusing',
        'Irritability or short temper',
        'Persistent fatigue or low energy',
        'Feeling anxious or on edge',
        'Loss of motivation or interest',
        'Feeling emotionally exhausted'
    ]
    with st.form("mental_health_form"):
        responses = {}
        for i, q in enumerate(questions, 1):
            responses[f'q{i}'] = st.slider(q, 0, 3, 0)
        if st.form_submit_button("🔍 Assess", type="primary"):
            if assess_mental_health_risk is None:
                st.error("Backend not available")
            else:
                result = assess_mental_health_risk(responses)
                if 'error' in result:
                    st.error(result['error'])
                else:
                    c1,c2,c3 = st.columns(3)
                    c1.metric("Score", f"{result.get('stress_score')}/30")
                    c2.metric("Level", result.get('stress_level'))
                    c3.metric("Severity", result.get('severity'))
                    st.info(result.get('recommendation'))

# ── Medications ───────────────────────────────────────────────────────────────
elif page == "💊 Medications":
    st.title("💊 Medication Reminders")
    with st.form("add_med", clear_on_submit=True):
        c1,c2,c3 = st.columns(3)
        name = c1.text_input("Medication Name")
        dose = c2.text_input("Dosage (e.g. 500mg)")
        time = c3.time_input("Reminder Time", value=dtime(9,0))
        if st.form_submit_button("➕ Add Medication"):
            if name:
                st.session_state.medications.append({'name':name,'dosage':dose,'time':time,'taken':False})
                st.success(f"Added {name}")
            else:
                st.warning("Please enter a medication name.")
    if st.session_state.medications:
        st.markdown("---")
        st.subheader("Today's Medications")
        for i, med in enumerate(st.session_state.medications):
            c1,c2,c3,c4 = st.columns([3,2,2,2])
            c1.write(f"**{med['name']}**")
            c2.write(med['dosage'] or "—")
            c3.write(med['time'].strftime('%I:%M %p'))
            taken = c4.checkbox("Taken ✓", value=med['taken'], key=f"med_{i}")
            st.session_state.medications[i]['taken'] = taken
    else:
        st.info("No medications added yet.")

# ── Vitals ────────────────────────────────────────────────────────────────────
elif page == "📟 Vitals":
    st.title("📟 Vitals Monitor")
    c1,c2,c3 = st.columns(3)
    hr = c1.number_input("Heart Rate (bpm)", 30, 220, 78)
    spo2 = c2.number_input("SpO2 (%)", 50, 100, 97)
    temp = c3.number_input("Temperature (°C)", 30.0, 43.0, 36.8, 0.1)
    if st.button("🔍 Check Vitals", type="primary"):
        if check_vitals is None:
            st.error("Backend not available")
        else:
            alerts = check_vitals(hr, spo2, temp)
            if alerts:
                for a in alerts:
                    st.error(a)
            else:
                st.success("✅ All vitals within normal range")

# ── SOS ───────────────────────────────────────────────────────────────────────
elif page == "🚑 SOS":
    st.title("🚑 Emergency SOS")
    if st.session_state.sos_triggered:
        st.error("⚠️ SOS condition detected via chatbot. Please confirm your location below.")
    st.warning("Use this only in a genuine medical emergency.")
    location = st.text_input("Your current location (address or landmark)")
    if st.button("🚨 CALL AMBULANCE", type="primary"):
        if not location:
            st.warning("Please enter your location first.")
        elif trigger_ambulance_call is None:
            st.error("Backend not available")
        else:
            result = trigger_ambulance_call(location)
            st.success(result)
            st.session_state.sos_triggered = False