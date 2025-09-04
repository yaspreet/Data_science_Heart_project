import streamlit as st
import pandas as pd 
import joblib 

model = joblib.load("knn_heart_model.pkl")
scaler = joblib.load("heart_scaler.pkl")
expected_columns = joblib.load("heart_columns.pkl")
if callable(expected_columns):
    expected_columns = expected_columns()

st.title('Heart Disease Prediction By Yash ❤')
st.markdown('Provide the following details')

age = st.slider('Age',18,100,40)
sex = st.selectbox('SEX',['M','F'])
chest_pain = st.selectbox("Chest Pain Type ",['ATA','NAP','TA','ASY'])
resting_bp = st.number_input('Resting Blood Pressure (mm Hg)',80,200,120)
cholesterol = st.number_input('Cholesterol (mg/dL)',100,600,200)
fasting_bs = st.selectbox('Fasting Blood Sugar >120 mg/dL',[0,1])
resting_ecg = st.selectbox('Resting ECG',['Normal','ST','LVH'])
max_hr = st.slider("Max Heart Rate",60,220,150)
exercise_angina = st.selectbox("Excercise-Induced Angina ",['Y','N'])
oldpeak = st.slider('Oldpekk (ST Depression )',0.0,6.0,1.0)
st_slope = st.selectbox('ST Slope',['Up','Flat','Down'])

if st.button('Predict'):
    raw_input = {
        'Age': age,
        'RestingBP': resting_bp,
        'Cholesterol': cholesterol,
        'FastingBS': fasting_bs,
        'MaxHR': max_hr,
        'Oldpeak': oldpeak,
        'Sex_' + sex: 1,
        'ChestPainType_' + chest_pain: 1,
        'RestingECG_' + resting_ecg: 1,
        'ExerciseAngina_' + exercise_angina: 1,
        'ST_Slope_' + st_slope: 1
    }
    input_df = pd.DataFrame([raw_input])

    for col in expected_columns:
        if col not in  input_df.columns:
            input_df[col]= 0
    input_df = input_df[expected_columns]
    scaled_input = scaler.transform(input_df)
    prediction = model.predict(scaled_input)[0]

    if prediction == 1:
        st.error("⚠️ High Risk of Heart Disease")
    else:
        st.success("✅ Low Risk of Heart Disease")

    