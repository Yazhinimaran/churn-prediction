import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image

# Load the model from disk
import joblib
model = joblib.load(r"./notebook/model.sav")

# Import preprocessing script
from preprocessing import preprocess

def main():
    # Setting Application title
    st.title('Telco Customer Churn Prediction App')

    # Setting Application description
    st.markdown("""
    :dart: This Streamlit app is designed to predict customer churn in a fictional telecommunication use case.
    The application supports both online prediction and batch data prediction.
    """)

    # Setting Application sidebar default
    image = Image.open('App.jpg')
    add_selectbox = st.sidebar.selectbox(
        "How would you like to predict?", ("Online", "Batch"))
    st.sidebar.info('This app is created to predict Customer Churn')
    st.sidebar.image(image)

    if add_selectbox == "Online":
        st.info("Input data below")

        # Demographic data
        st.subheader("Demographic Data")
        age = st.slider('Age', min_value=18, max_value=80, value=30)
        gender = st.selectbox('Gender', ('Male', 'Female'))

        # Contract and Payment Data
        st.subheader("Contract and Payment Data")
        contract = st.selectbox('Contract Type', ('Month-to-month', 'One year', 'Two year'))
        tenure = st.slider('Tenure (Months)', min_value=0, max_value=72, value=12)
        monthlycharges = st.number_input('Monthly Charges ($)', min_value=0, max_value=150, value=50)
        totalcharges = st.number_input('Total Charges ($)', min_value=0, max_value=10000, value=600)
        paperlessbilling = st.selectbox('Paperless Billing', ('Yes', 'No'))
        paymentmethod = st.selectbox('Payment Method', 
                                    ('Electronic check', 'Mailed check', 
                                     'Bank transfer (automatic)', 'Credit card (automatic)'))

        # Services signed up for
        st.subheader("Services Signed Up For")
        internetservice = st.selectbox('Internet Service', ('DSL', 'Fiber optic', 'No'))
        techsupport = st.selectbox('Tech Support', ('Yes', 'No'))

        # Compile input data into a dictionary
        data = {
            'Age': age,
            'Gender': gender,
            'ContractType': contract,
            'Tenure': tenure,
            'MonthlyCharges': monthlycharges,
            'TotalCharges': totalcharges,
            'PaperlessBilling': paperlessbilling,
            'PaymentMethod': paymentmethod,
            'InternetService': internetservice,
            'TechSupport': techsupport
        }

        # Convert input data to DataFrame
        features_df = pd.DataFrame.from_dict([data])
        st.markdown("<h3></h3>", unsafe_allow_html=True)
        st.write('Overview of input is shown below')
        st.markdown("<h3></h3>", unsafe_allow_html=True)
        st.dataframe(features_df)

        # Preprocess inputs
        preprocess_df = preprocess(features_df, 'Online')

        # Make prediction
        prediction = model.predict(preprocess_df)

        if st.button('Predict'):
            if prediction == 1:
                st.warning('Yes, the customer is likely to churn.')
            else:
                st.success('No, the customer is not likely to churn.')

    else:
        st.subheader("Dataset Upload")
        uploaded_file = st.file_uploader("Choose a file")
        if uploaded_file is not None:
            data = pd.read_csv(uploaded_file)
            st.write(data.head())  # Display the first few rows of the dataset

            # Preprocess inputs
            preprocess_df = preprocess(data, "Batch")
            if st.button('Predict'):
                prediction = model.predict(preprocess_df)
                prediction_df = pd.DataFrame(prediction, columns=["Predictions"])
                prediction_df = prediction_df.replace({1: 'Yes, the customer is likely to churn.', 
                                                       0: 'No, the customer is not likely to churn.'})

                st.markdown("<h3></h3>", unsafe_allow_html=True)
                st.subheader('Prediction Results')
                st.write(prediction_df)
            
if __name__ == '__main__':
    main()
