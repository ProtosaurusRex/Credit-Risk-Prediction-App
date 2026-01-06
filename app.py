# 1= good low risk 0 = bad high risk

import streamlit as st
import pandas as pd
import joblib as jb

model = jb.load('best_extra_trees_model.pkl')
encoders = {col: jb.load(f'{col}_encoder.pkl') for col in ['Sex', 'Housing', 'Saving accounts', 'Checking account', 'Purpose']}


st.title("Credit Risk Modeling App")
st.write("Enter the details below to predict credit risk.")

age = st.number_input('Age', min_value=18, max_value=100, value=30)
sex = st.selectbox('Sex', ['male', 'female'])
job = st.number_input('Job (0-3)', min_value=0, max_value=3, value=1)    
housing = st.selectbox('Housing', ['own', 'rent', 'free'])
saving_accounts = st.selectbox('Saving accounts', ['little', 'moderate', 'rich', 'quite rich', 'unknown'])
checking_account = st.selectbox('Checking account', ['little', 'moderate', 'rich', 'unknown'])
credit_amount = st.number_input('Credit Amount', min_value=250.0, max_value=20000.0, value=1000.0)
duration = st.number_input('Duration (months)', min_value=4, max_value=72, value=12)
purpose = st.selectbox('Purpose', ['car', 'furniture/equipment', 'radio/TV', 'domestic appliances', 'repairs', 'education', 'vacation/others', 'business', 'retraining'])

input_data = {
    'Age': [age],
    'Sex': [encoders['Sex'].transform([sex])[0]],
    'Job': [job],
    'Housing': [encoders['Housing'].transform([housing])[0]],
    'Saving accounts': [encoders['Saving accounts'].transform([saving_accounts])[0]],
    'Checking account': [encoders['Checking account'].transform([checking_account])[0]],
    'Credit amount': [credit_amount],
    'Duration': [duration],
    'Purpose': [encoders['Purpose'].transform([purpose])[0]]
}


# Build a single-row DataFrame so scikit-learn receives a 2D array with feature names
input_df = pd.DataFrame(input_data)

if st.button('Predict'):
    # Ensure feature names & order match the model's training data
    if hasattr(model, 'feature_names_in_'):
        expected = list(model.feature_names_in_)
        missing = [f for f in expected if f not in input_df.columns]
        if missing:
            st.error(f"Missing features required by the model: {missing}")
        else:
            input_df = input_df[expected]

    try:
        pred = model.predict(input_df)[0]
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        raise

    if pred == 1:
        st.success('The applicant is predicted to be LOW RISK (Good Credit).')
    else:
        st.error('The applicant is predicted to be HIGH RISK (Bad Credit).')
    