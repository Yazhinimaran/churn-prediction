import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def preprocess(df, option):
    """
    This function covers all the preprocessing steps on the churn dataframe. It involves selecting important features, 
    encoding categorical data, handling missing values, feature scaling, and preparing the data for modeling.
    """
    # Define the binary map function for encoding
    def binary_map(feature):
        return feature.map({'Yes': 1, 'No': 0})

    # Encode binary categorical features
    binary_list = ['PaperlessBilling', 'TechSupport']
    df[binary_list] = df[binary_list].apply(binary_map)

    # Encode Gender feature
    df['Gender'] = df['Gender'].map({'Male': 0, 'Female': 1})

    # Encode ContractType feature
    df['ContractType'] = df['ContractType'].map({'Month-to-month': 0, 'One year': 1, 'Two year': 2})

    # Encode InternetService feature
    df['InternetService'] = df['InternetService'].map({'No': 0, 'DSL': 1, 'Fiber optic': 2})

    # Drop CustomerID as it is not useful for prediction
    df = df.drop('CustomerID', axis=1)

    # Drop values based on operational options
    if option == "Online":
        # Select relevant features
        columns = ['Age', 'Gender', 'ContractType', 'Tenure', 'MonthlyCharges', 'TotalCharges', 'PaperlessBilling',
                   'PaymentMethod', 'InternetService', 'TechSupport']
        df = pd.get_dummies(df, columns=['PaymentMethod']).reindex(columns=columns, fill_value=0)
    elif option == "Batch":
        df = df[['Age', 'Gender', 'ContractType', 'Tenure', 'MonthlyCharges', 'TotalCharges', 'PaperlessBilling',
                 'PaymentMethod', 'InternetService', 'TechSupport']]
        df = pd.get_dummies(df, columns=['PaymentMethod'])
    else:
        print("Incorrect operational option")

    # Feature scaling
    scaler = MinMaxScaler()
    df[['Tenure', 'MonthlyCharges', 'TotalCharges']] = scaler.fit_transform(df[['Tenure', 'MonthlyCharges', 'TotalCharges']])

    return df
