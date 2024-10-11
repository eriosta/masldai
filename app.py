import streamlit as st
import pandas as pd
import pickle
import os
import numpy as np
from sklearn.metrics import roc_auc_score, confusion_matrix, accuracy_score
import base64

# Set page config
st.set_page_config(
    page_title="MASLD AI",
    page_icon="🩺",  
)

# def load_model(model_path):
#     with open(model_path, 'rb') as file:
#         model = pickle.load(file)
#     return model

# def calculate_fast_score(data, ast_col, cap_col, lsm_col):
#     exponent = -1.65 + 1.07 * np.log(data[lsm_col]) + \
#                2.66*(10**-8) * data[cap_col]**3 - \
#                63.3 * data[ast_col]**-1
#     return np.exp(exponent) / (1 + np.exp(exponent))

# def predict_and_save(model, data, output_path):
#     y_pred = model.predict(data)
#     y_pred_proba = model.predict_proba(data)[:, 1]

#     results_df = pd.DataFrame()
#     results_df['Predicted Label'] = y_pred
#     results_df['Probability'] = y_pred_proba

#     results_df.to_csv(output_path, index=False)
#     return output_path

# def calculate_metrics(y_true, y_pred):
#     tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
#     sensitivity = tp / (tp + fn)
#     specificity = tn / (tn + fp)
#     accuracy = accuracy_score(y_true, y_pred)
#     ppv = tp / (tp + fp)
#     npv = tn / (tn + fn)
#     auroc = roc_auc_score(y_true, y_pred)
#     return sensitivity, specificity, accuracy, ppv, npv, auroc

# st.title('XGBoost MASLD AI')

# st.markdown("""
# Welcome to the XGBoost MASLD AI application. This tool allows you to predict and analyze medical data using a machine learning model.
# Please follow the steps below to upload your data and receive predictions.
# """)

# st.header("Step 1: Upload Your Data")
# uploaded_file = st.file_uploader("Choose a CSV or Excel file", type=["csv", "xlsx"])

# if uploaded_file is not None:
#     file_name = uploaded_file.name
#     if file_name.endswith('.csv'):
#         data = pd.read_csv(uploaded_file)
#     elif file_name.endswith('.xlsx'):
#         data = pd.read_excel(uploaded_file)
#     else:
#         st.error("Please upload a valid CSV or Excel file.")
#         st.stop()

#     model_features = ['ALANINE AMINOTRANSFERASE (ALT) (U/L)', 'GAMMA GLUTAMYL TRANSFERASE (GGT) (IU/L)', 'PLATELET COUNT (1000 CELLS/UL)', 'AGE IN YEARS AT SCREENING', 'BODY MASS INDEX (KG/M**2)']
    
#     st.header("Step 2: Configure Your Data")
#     st.subheader("Select columns for FAST score calculation")

#     st.latex(r'FAST = -1.65 + 1.07 * log(LSM) + 2.66*(10**-8) * (CAP)^3 - 63.3 * (AST)^-1')
    
#     col1, col2, col3 = st.columns(3)
#     with col1:
#         ast_col = st.selectbox('Column for AST:', data.columns)
#     with col2:
#         cap_col = st.selectbox('Column for CAP:', data.columns)
#     with col3:
#         lsm_col = st.selectbox('Column for LSM:', data.columns)

#     st.subheader("Select columns to run the model")
#     feature_mapping = {}
#     for feature in model_features:
#         options = list(data.columns)
#         selected_feature = st.selectbox(f'Column for {feature}:', options)
#         feature_mapping[selected_feature] = feature

#     if st.button('Run Model'):
#         with st.spinner('Running the model...'):
#             data = data.rename(columns=feature_mapping)
#             data['FASTscore'] = calculate_fast_score(data, ast_col, cap_col, lsm_col)
#             data['isAtRiskMASH35'] = np.where(data['FASTscore'] >= 0.35, 1, 0)
#             model = load_model('xgboost_mashai_35.pkl')
#             output_file = predict_and_save(model, data[model_features], 'predictions.csv')
#             y_pred = pd.read_csv(output_file)['Predicted Label']
#             sensitivity, specificity, accuracy, ppv, npv, auroc = calculate_metrics(data['isAtRiskMASH35'], y_pred)
#             st.success('Model ran successfully!')

#         st.header("Step 3: Results")

#         # Create a DataFrame for the results
#         results = pd.DataFrame({
#             'Metric': ['Sensitivity', 'Specificity', 'Accuracy', 'PPV', 'NPV', 'AUROC'],
#             'Value': [sensitivity, specificity, accuracy, ppv, npv, auroc]
#         })

#         # Display the results in a table
#         html = results.to_html(index=False, justify='center')
#         st.markdown(html, unsafe_allow_html=True)

#         # Create a CSV download link
#         csv = results.to_csv(index=False)
#         b64 = base64.b64encode(csv.encode()).decode()
#         href = f'<a href="data:file/csv;base64,{b64}" download="model_results.csv">Download CSV File</a>'
#         st.markdown(href, unsafe_allow_html=True)

# else:
#     if uploaded_file is not None:
#         st.error("Please upload a valid CSV file.")

import shap

# Set the page layout to wide
st.set_page_config(layout="wide")

st.sidebar.markdown(
    'Read our paper here: [Njei et al. (2024). Scientific Reports.](https://www.nature.com/articles/s41598-024-59183-4)'
)


def load_model(model_path):
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
    return model

def predict(model, data):
    y_pred = model.predict(data)
    y_pred_proba = model.predict_proba(data)[:, 1]
    return y_pred, y_pred_proba


st.sidebar.header("Enter Data")

model_features = ['ALANINE AMINOTRANSFERASE (ALT) (U/L)', 
                  'GAMMA GLUTAMYL TRANSFERASE (GGT) (IU/L)', 
                  'PLATELET COUNT (1000 CELLS/UL)', 
                  'AGE IN YEARS AT SCREENING', 
                  'BODY MASS INDEX (KG/M**2)']

display_labels = ['ALT (U/L)', 'GGT (U/L)', 'Platelets (1000 cells/µL)', 'Age (years)', 'BMI (kg/m²)']

data = {}
for feature, label in zip(model_features, display_labels):
    data[feature] = st.sidebar.number_input(f'{label}', min_value=0.0)

data_df = pd.DataFrame([data])

if st.sidebar.button('Run Model'):
    with st.spinner('Running the model...'):
        model = load_model('xgboost_mashai_35.pkl')
        y_pred, y_pred_proba = predict(model, data_df)
        
        # Calculate SHAP values
        explainer = shap.Explainer(model)
        shap_values = explainer(data_df)
        
        # Rename features in SHAP values to display shorter feature names
        shap_values.feature_names = display_labels
        
        st.toast('Model ran successfully! 🎉')

    predicted_label = "Likely High-Risk MASH" if y_pred[0] == 1 else "Unlikely High-Risk MASH"

    # st.subheader("Model Predictions")

    col1, col2 = st.columns(2)
    with col1:
        st.metric(label="Prediction", value=predicted_label)
    with col2:
        st.metric(label="Probability", value=f"{y_pred_proba[0]:.2f}")

    # Display SHAP waterfall plot with high quality image
    fig = shap.plots.waterfall(shap_values[0], show=False)
    fig.savefig("shap_waterfall.png", dpi=500, bbox_inches='tight')
    st.image("shap_waterfall.png", caption="SHAP Explanations", use_column_width=True)