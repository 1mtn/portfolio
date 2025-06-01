import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# Set page config
st.set_page_config(page_title="Heart Disease Prediction", layout="wide")

# Load and preprocess data
@st.cache_data
def load_data():
    # Read CSV file
    df = pd.read_csv('heart_disease.csv')
    # Drop Unnamed: 0 
    if 'Unnamed: 0' in df.columns:
        df = df.drop('Unnamed: 0', axis=1)
    # Handle missing values - replacing ? with mode
    df = df.replace('?', np.nan)
    df = df.fillna(df.mode().iloc[0])
    # Ensure categorical columns are strings
    df[['cp', 'restecg', 'slope', 'thal']] = df[['cp', 'restecg', 'slope', 'thal']].astype(str)
    # Encode categorical variables
    df_encoded = pd.get_dummies(df, columns=['cp', 'restecg', 'slope', 'thal'], drop_first=True)
    return df, df_encoded

df, df_encoded = load_data()
# Ensure no unexpected columns
X = df_encoded.drop('present', axis=1)
y = df_encoded['present']

# Train model
@st.cache_resource
def train_model():
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    model = LogisticRegression(solver='liblinear', max_iter=1000)
    model.fit(X_train_scaled, y_train)
    return model, scaler, X_train, X_test, y_train, y_test

model, scaler, X_train, X_test, y_train, y_test = train_model()

# Debugging: Display expected feature names
st.sidebar.write("**Expected Features**")
st.sidebar.write(X.columns.tolist())

# Tabs for navigation
tabs = st.tabs(["Introduction", "Data Exploration", "Model Details", "Prediction Tool", "Code & Methodology"])

# Tab 1: Introduction
with tabs[0]:
    st.title("Heart Disease Prediction with Logistic Regression")
    st.write("""
        This app demonstrates logistic regression using the Cleveland Heart Disease dataset (303 patients, 13 predictors).
        Explore the data, understand the model, and predict heart disease risk for new patients.
        Logistic regression predicts the probability of heart disease based on factors like age, sex, and cholesterol.
    """)
    st.markdown("[Heart Disease Dataset](https://archive.ics.uci.edu/dataset/45/heart+disease)")

# Tab 2: Data Exploration
with tabs[1]:
    st.header("Data Exploration")
    st.write("The dataset includes 303 patients with 13 predictors and a binary outcome (0 = no heart disease, 1 = heart disease).")
    
    # Show dataset
    st.subheader("Raw Data Sample")
    st.dataframe(df.head())
    
    # Missing values
    st.subheader("Missing Values")
    missing = df.isin(['?']).sum()
    st.write(f"Missing values ('?') in 'ca' and 'thal' were imputed with mode: {missing[missing > 0].to_dict()}")
    
    # Feature distributions
    st.subheader("Feature Distributions")
    col = st.selectbox("Select feature to visualize", df.columns.drop('present'))
    fig, ax = plt.subplots()
    sns.histplot(df[col], ax=ax)
    st.pyplot(fig)

    # Feature description table
    st.subheader("Feature Descriptions")
    feature_descriptions = pd.DataFrame({
    'Feature': ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'present'],
    'Full Name': [
        'Age (years)', 
        'Sex (0=Female, 1=Male)', 
        'Chest Pain Type (1=Typical Angina, 2=Atypical Angina, 3=Non-Anginal, 4=Asymptomatic)', 
        'Resting Blood Pressure (mmHg)', 
        'Cholesterol (mg/dl)', 
        'Fasting Blood Sugar > 120 mg/dl (0=No, 1=Yes)', 
        'Resting ECG (0=Normal, 1=ST-T Abnormality, 2=LV Hypertrophy)', 
        'Maximum Heart Rate (beats per minute)', 
        'Exercise-Induced Angina (0=No, 1=Yes)', 
        'ST Depression (mm)', 
        'Slope of ST Segment (1=Upsloping, 2=Flat, 3=Downsloping)', 
        'Number of Major Vessels (0-3)', 
        'Thalassemia (3=Normal, 6=Fixed Defect, 7=Reversible Defect)', 
        'Heart Disease (0=No, 1=Yes)']    
        })  
    st.dataframe(feature_descriptions)



# Tab 3: Model Details
with tabs[2]:
    st.header("Model Details")
    
    # Coefficients
    st.subheader("Logistic Regression Coefficients")
    coef_df = pd.DataFrame({
        'Feature': X.columns,
        'Log-Odds': model.coef_[0],
        'Odds Ratio': np.exp(model.coef_[0])
    })
    st.dataframe(coef_df)
    
    # Plot coefficients
    fig, ax = plt.subplots()
    sns.barplot(x='Log-Odds', y='Feature', data=coef_df, ax=ax)
    ax.set_title("Feature Coefficients (Log-Odds)")
    st.pyplot(fig)
    
    # Performance metrics
    y_test_pred = model.predict(X_test)
    tn, fp, fn, tp = confusion_matrix(y_test, y_test_pred).ravel()
    test_accuracy = accuracy_score(y_test, y_test_pred)
    test_sensitivity = tp / (tp + fn)
    test_specificity = tn / (tn + fp)
    
    st.subheader("Test Set Performance")
    st.write(f"""
        - Accuracy: {test_accuracy:.4f} (Correct predictions)
        - Sensitivity: {test_sensitivity:.4f} (True positive rate)
        - Specificity: {test_specificity:.4f} (True negative rate)
        - Confusion Matrix: TP={tp}, TN={tn}, FP={fp}, FN={fn}
    """)
    
    # Cross-validation
    cv_scores = cross_val_score(model, scaler.transform(X), y, cv=5)
    st.subheader("Cross-Validation")
    st.write(f"Mean Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    #Explanation 
    st.subheader("How to Read Graph")
    st.write("The \"Feature Coefficients (Log-Odds)\" graph shows how each feature in the logistic regression model affects the likelihood of heart disease. Each horizontal bar represents a feature’s coefficient, where the length indicates the strength of its impact, and the direction (right or left) shows whether it increases or decreases the risk. A positive bar (extending right) means a higher feature value raises the chance of heart disease, while a negative bar (extending left) lowers it. For example, a long positive bar for `sex` (coefficient: 0.8390) indicates that males (`sex=1`) have a significantly higher risk of heart disease compared to females. " \
    "The graph uses standardized features, so each coefficient reflects the effect of a one-standard-deviation change in the feature. For instance, a negative bar for `thalach` (coefficient: -0.3812) suggests that a higher maximum heart rate (about 19 beats per minute) reduces the risk of heart disease, as it’s linked to better heart fitness. This visualization helps you understand which factors, like blocked vessels (`ca`) or chest pain type (`cp_4`), are most important in predicting heart disease and whether they increase or decrease the risk.")
    
    #Feature Descriptions
    st.subheader("Feature Descriptions")
    feature_descriptions = pd.DataFrame({
    'Feature': ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'present'],
    'Full Name': [
        'Age (years)', 
        'Sex (0=Female, 1=Male)', 
        'Chest Pain Type (1=Typical Angina, 2=Atypical Angina, 3=Non-Anginal, 4=Asymptomatic)', 
        'Resting Blood Pressure (mmHg)', 
        'Cholesterol (mg/dl)', 
        'Fasting Blood Sugar > 120 mg/dl (0=No, 1=Yes)', 
        'Resting ECG (0=Normal, 1=ST-T Abnormality, 2=LV Hypertrophy)', 
        'Maximum Heart Rate (beats per minute)', 
        'Exercise-Induced Angina (0=No, 1=Yes)', 
        'ST Depression (mm)', 
        'Slope of ST Segment (1=Upsloping, 2=Flat, 3=Downsloping)', 
        'Number of Major Vessels (0-3)', 
        'Thalassemia (3=Normal, 6=Fixed Defect, 7=Reversible Defect)', 
        'Heart Disease (0=No, 1=Yes)']    
        })  
    st.dataframe(feature_descriptions)


# Tab 4: Prediction Tool
with tabs[3]:
    st.header("Predict Heart Disease Risk")
    st.write("Enter patient data to predict the risk of heart disease.")
    
    # Input widgets
    col1, col2 = st.columns(2)
    with col1:
        age = st.slider("Age", 20, 80, 50)
        sex = st.selectbox("Sex", [0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
        trestbps = st.number_input("Resting Blood Pressure (mmHg)", 80, 200, 120)
        chol = st.number_input("Cholesterol (mg/dl)", 100, 600, 200)
        fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
        thalach = st.number_input("Max Heart Rate", 60, 220, 150)
        exang = st.selectbox("Exercise-Induced Angina", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
    with col2:
        oldpeak = st.number_input("ST Depression", 0.0, 6.0, 1.0, step=0.1)
        ca = st.number_input("Number of Major Vessels (0-3)", 0, 3, 0)
        cp = st.selectbox("Chest Pain Type", ["1", "2", "3", "4"], format_func=lambda x: {1: "Typical Angina", 2: "Atypical Angina", 3: "Non-Anginal", 4: "Asymptomatic"}[int(x)])
        restecg = st.selectbox("Resting ECG", ["0", "1", "2"], format_func=lambda x: {0: "Normal", 1: "ST-T Abnormality", 2: "LV Hypertrophy"}[int(x)])
        slope = st.selectbox("Slope of ST Segment", ["1", "2", "3"], format_func=lambda x: {1: "Upsloping", 2: "Flat", 3: "Downsloping"}[int(x)])
        thal = st.selectbox("Thalassemia", ["3", "6", "7"], format_func=lambda x: {3: "Normal", 6: "Fixed Defect", 7: "Reversible Defect"}[int(x)])
    
    # Prepare input for prediction
    input_data = pd.DataFrame(0, index=[0], columns=X.columns)  # Initialize with zeros
    input_data['age'] = age
    input_data['sex'] = sex
    input_data['trestbps'] = trestbps
    input_data['chol'] = chol
    input_data['fbs'] = fbs
    input_data['thalach'] = thalach
    input_data['exang'] = exang
    input_data['oldpeak'] = oldpeak
    input_data['ca'] = ca
    input_data['cp_2'] = 1 if cp == '2' else 0
    input_data['cp_3'] = 1 if cp == '3' else 0
    input_data['cp_4'] = 1 if cp == '4' else 0
    input_data['restecg_1'] = 1 if restecg == '1' else 0
    input_data['restecg_2'] = 1 if restecg == '2' else 0
    input_data['slope_2'] = 1 if slope == '2' else 0
    input_data['slope_3'] = 1 if slope == '3' else 0
    input_data['thal_6.0'] = 1 if thal == '6' else 0
    input_data['thal_7.0'] = 1 if thal == '7' else 0
    
    # Debug: Check input_data columns
    st.write("**Input Data Columns**", input_data.columns.tolist())
    
    # Scale input and predict
    input_scaled = scaler.transform(input_data)
    prob = model.predict_proba(input_scaled)[0][1]
    pred = model.predict(input_scaled)[0]
    
    # Display prediction
    st.subheader("Prediction Result")
    risk = "High" if prob >= 0.7 else "Medium" if prob >= 0.3 else "Low"
    st.write(f"Probability of Heart Disease: {prob:.2%}")
    st.write(f"Prediction: {'Heart Disease' if pred == 1 else 'No Heart Disease'}")
    st.write(f"Risk Level: {risk}")
    st.info("Note: This is a demonstration tool. Consult a doctor for medical advice.")

# Tab 5: Code & Methodology
with tabs[4]:
    st.header("Code & Methodology")
    st.write("""
        **Preprocessing**:
        - Dropped 'Unnamed: 0' if present.
        - Replaced '?' with mode.
        - Encoded categorical variables (`cp`, `restecg`, `slope`, `thal`) using `pd.get_dummies(drop_first=True)`.
        - Scaled numerical features with `StandardScaler`.
        
        **Model**:
        - Logistic regression with `liblinear` solver and L1 regularization.
        - Trained on 80% of data, tested on 20%.
        
        **Validation**:
        - 5-fold cross-validation (mean accuracy: 0.8281).
        - Evaluated with accuracy, sensitivity, specificity.
    """)
    st.code("""
# Preprocessing and model training
df = pd.read_csv('heart_disease.csv')
if 'Unnamed: 0' in df.columns:
    df = df.drop('Unnamed: 0', axis=1)
df = df.replace('?', np.nan).fillna(df.mode().iloc[0])
df[['cp', 'restecg', 'slope', 'thal']] = df[['cp', 'restecg', 'slope', 'thal']].astype(str)
df_encoded = pd.get_dummies(df, columns=['cp', 'restecg', 'slope', 'thal'], drop_first=True)
X = df_encoded.drop('present', axis=1)
y = df_encoded['present']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
model = LogisticRegression(solver='liblinear', max_iter=1000)
model.fit(X_train_scaled, y_train)
    """)
