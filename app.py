import streamlit as st
import pandas as pd
import atrisk_ai
import fibrox_ai
import shap
import matplotlib.pyplot as plt

# Streamlit app
st.set_page_config(
    page_title="MASLD AI",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)
st.sidebar.title("MASLD AI")

st.sidebar.markdown(
    'Read our paper here: [Njei et al. (2024). Scientific Reports.](https://www.nature.com/articles/s41598-024-59183-4)'
)

selected_model = st.sidebar.radio(
    "Choose the model:",
    ["At-Risk AI", "FibroX AI"],
    help="""
    **At-Risk AI**: Identifies adults with MASLD who may develop at-risk MASH. 
    Uses ALT, GGT, Platelets, Age and BMI.
    
    **FibroX AI**: Identifies adults with MASLD who may develop advanced fibrosis and are at risk of 30-year cause-specific mortality. Uses Age, HbA1c, ALT, AST, Platelets, BMI and GFR.
    """
)

# Define display labels for user input fields
display_labels_map = {
    "At-Risk AI": {
        'ALANINE AMINOTRANSFERASE (ALT) (U/L)': 'ALT (U/L)',
        'GAMMA GLUTAMYL TRANSFERASE (GGT) (IU/L)': 'GGT (U/L)',
        'PLATELET COUNT (1000 CELLS/UL)': 'Platelets (1000 cells/µL)',
        'AGE IN YEARS AT SCREENING': 'Age (years)',
        'BODY MASS INDEX (KG/M**2)': 'BMI (kg/m²)'
    },
    "FibroX AI": {
        'Age_in_years_at_screening': 'Age (years)',
        'Glycohemoglobin (%)': 'HbA1c (%)',
        'Alanine Aminotransferase (ALT) (U/L)': 'ALT (U/L)',
        'Aspartate Aminotransferase (AST) (U/L)': 'AST (U/L)',
        'Platelet count (1000 cells/uL)': 'Platelets (1000 cells/µL)',
        'Body Mass Index (kg/m**2)': 'BMI (kg/m²)',
        'GFR': 'MDRD GFR'
    }
}

# Load model and features
if selected_model == "At-Risk AI":
    model_features = atrisk_ai.MODEL_FEATURES
    display_labels = display_labels_map["At-Risk AI"]
    model = atrisk_ai.load_model()
    prediction_function = atrisk_ai.predict
else:
    model_features = fibrox_ai.MODEL_FEATURES
    display_labels = display_labels_map["FibroX AI"]
    model = fibrox_ai.load_model()
    prediction_function = fibrox_ai.predict

# Collect user inputs dynamically based on model features
data = {}
all_inputs_filled = True  # Initialize check for all required inputs
if selected_model == "FibroX AI":
    serum_cr = None
    is_female = None

    for feature in model_features:
        if feature == "GFR":
            serum_cr = st.sidebar.number_input('Creatinine (mg/dL)', min_value=0.0, value=None)
            gender = st.sidebar.radio('Gender', ['Male', 'Female'])
            is_female = gender == 'Female'

            # Use the age already entered for the model
            age_feature = 'Age_in_years_at_screening'
            if age_feature in data:
                age = data[age_feature]
            else:
                st.sidebar.warning("Please enter a valid age to calculate GFR.")
                all_inputs_filled = False

            # Calculate GFR if all required inputs are present
            if serum_cr is not None and 'Age_in_years_at_screening' in data:
                age = data['Age_in_years_at_screening']
                data['GFR'] = fibrox_ai.calculate_gfr(serum_cr, age, is_female)
                st.sidebar.info(f"eGFR: {data['GFR']:.2f} mL/min/1.73 m²")
            else:
                data['GFR'] = None
                all_inputs_filled = False
        else:
            label = display_labels[feature]
            input_value = st.sidebar.number_input(f'{label}', min_value=0.0, value=None)
            data[feature] = input_value
            if input_value is None:
                all_inputs_filled = False
else:
    for feature in model_features:
        label = display_labels[feature]
        input_value = st.sidebar.number_input(f'{label}', min_value=0.0, value=None)
        data[feature] = input_value
        if input_value is None:
            all_inputs_filled = False

# Convert collected data into DataFrame
data_df = pd.DataFrame([data])

# Conditionally display the Run button
if all_inputs_filled:
    if st.sidebar.button(f'Run {selected_model} Model'):
        with st.spinner('Running the model...'):
            y_pred, y_pred_proba = prediction_function(model, data_df)

            # Calculate SHAP values
            explainer = shap.Explainer(model)
            shap_values = explainer(data_df)

            # Update SHAP feature names to use display labels
            shap_values.feature_names = [display_labels[feature] for feature in model_features]

            st.toast(f'{selected_model} ran successfully! 🎉')

            # Custom output for FibroX AI
            st.subheader("Results")
            if selected_model == "FibroX AI" and y_pred[0] == 1:
                st.warning(
                    """
                    **Result**: High-Risk Mortality  
                    - **1.3 times** at a higher adjusted* risk of **30-year all-cause mortality**  
                    - **1.9 times** at a higher adjusted* risk of **30-year cardiovascular-related mortality**  
                    
                    *age- and sex-adjusted
                    """
                )
            elif selected_model == "FibroX AI" and y_pred[0] == 0:
                st.info("""
                    **Result**: Low-Risk Mortality                    
                    - No significant association with all-cause or cardiovascular-related mortality when adjusting for age and sex 
                    
                """)
            else:
                # For At-Risk AI
                st.warning("**Result**: High-Risk MASH" if y_pred[0] == 1 else "**Result**: No High-Risk MASH")

            # Display SHAP waterfall plot
            st.subheader("SHapley Additive exPlanations (SHAP) Explanations")
            fig = shap.plots.waterfall(shap_values[0], show=False)
            plt.savefig("shap_waterfall.png", dpi=500, bbox_inches='tight')
            st.image("shap_waterfall.png", caption="SHAP Explanations", use_column_width=True)

            # Add disclaimer about AI role
            st.info("""
                **Important**: This AI tool is designed to supplement, not replace, clinical decision making. 
                The results should be interpreted alongside clinical expertise, patient history, and other relevant medical information. 
                Healthcare providers should use their professional judgment when incorporating these predictions into patient care decisions.
            """)
else:
    st.sidebar.warning("Please fill in all required fields.")
