import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder

# Load the trained model
model = joblib.load("best_model.pkl")

st.set_page_config(page_title="Employee Salary Classification", page_icon="💼", layout="centered")

# Add custom CSS
st.markdown("""
<style>
/* Apply Cambria font and larger size globally */
html, body, [class*="css"], .stApp, .markdown-text-container, .stTextInput > label, .stSlider > label, .stSelectbox > label,
.stNumberInput > label, .stFileUploader > label, .stDataFrame, .stMarkdown, .stButton > button, .stCheckbox > label,
label, p, span, div {
    font-family: Cambria, Georgia, serif !important;
    font-size: 18px !important;
}

/* Header styles */
h1 {
    color: #1F4E79;
    font-size: 2.5em !important; /* Increased font size */
    font-weight: bold;
}
h2 {
    font-size: 30px !important;
}
h3 {
    font-size: 24px !important;
}

/* Background */
body {
    background-color: #F0FFFF;
}

/* Gradient text */
.gradient-text {
    background: linear-gradient(#1F4E79);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: bold;
    font-size: 2.5em; /* Ensure the span doesn't override h1 font size */
}

/* Sidebar slider track */
.stSlider > div[data-baseweb="slider"] > div {
    background-color: #F8F8FF !important;
    height: 6px !important;
    border-radius: 4px;
}

/* Selected (filled) range in slider */
.stSlider > div[data-baseweb="slider"] > div > div:first-child {
    background-color: #FFFFFF !important;
    height: 6px !important;
    border-radius: 4px;
}

/* Slider thumb */
.stSlider > div[data-baseweb="slider"] span[role="slider"] {
    background-color: #1E90FF !important;
    border: none !important;
    height: 20px !important;
    width: 20px !important;
    border-radius: 50% !important;
    margin-top: -7px;
    box-shadow: 0 0 4px rgba(0,0,0,0.2);
}

/* Slider label */
.stSlider > label {
    color: #1E90FF !important;
    font-weight: bold;
    font-size: 18px !important;
}

</style>
""", unsafe_allow_html=True)
st.markdown('<h1>💼 <span class="gradient-text">Employee Salary Classification App</span></h1>', unsafe_allow_html=True)
st.markdown("Predict whether an employee earns >50K or ≤50K based on input features.")

# Function to get tentative amount in Rs
def get_tentative_amount_in_rs(income_class):
  """
  Maps the income class to a tentative annual salary range in INR.

  Args:
    income_class: The predicted income class ('<=50K' or '>50K').

  Returns:
    A string representing the tentative annual salary range in Indian Rupees.
  """
  # Approximate conversion factor (USD to INR) - this can fluctuate
  # Using a rough estimate for demonstration purposes
  usd_to_inr = 83

  if income_class == '>50K':
    # Assuming >50K USD is roughly equivalent to a range like 50K - 100K USD annually
    lower_bound_usd = 50000
    upper_bound_usd = 100000 # This is an arbitrary upper bound for demonstration
    lower_bound_inr = lower_bound_usd * usd_to_inr
    upper_bound_inr = upper_bound_usd * usd_to_inr
    return f"Tentative Annual Amount: Rs. {lower_bound_inr:,} - Rs. {upper_bound_inr:,}"
  elif income_class == '<=50K':
    # Assuming <=50K USD is roughly equivalent to a range like 20K - 50K USD annually
    lower_bound_usd = 20000 # This is an arbitrary lower bound for demonstration
    upper_bound_usd = 50000
    lower_bound_inr = lower_bound_usd * usd_to_inr
    upper_bound_inr = upper_bound_usd * usd_to_inr
    return f"Tentative Annual Amount: Rs. {lower_bound_inr:,} - Rs. {upper_bound_inr:,}"
  else:
    return "Invalid income class"


# Sidebar inputs (these must match your training feature columns)
st.sidebar.header("Input Employee Details")

# Use the actual columns from your training data
# The order of these columns matters for the model prediction
age = st.sidebar.slider("Age", 17, 75, 30)
workclass = st.sidebar.selectbox("Workclass", ['Private', 'Self-emp-not-inc', 'Local-gov', 'Others', 'State-gov', 'Self-emp-inc', 'Federal-gov'])
fnlwgt = st.sidebar.number_input("Fnlwgt", value=200000)
educational_num = st.sidebar.slider("Educational Number", 5, 16, 9)
marital_status = st.sidebar.selectbox("Marital Status", ['Never-married', 'Married-civ-spouse', 'Divorced', 'Separated', 'Widowed', 'Married-spouse-absent', 'Married-AF-spouse'])
occupation = st.sidebar.selectbox("Job Role", [
    'Prof-specialty', 'Craft-repair', 'Exec-managerial', 'Adm-clerical', 'Sales',
    'Other-service', 'Machine-op-inspct', 'Others', 'Transport-moving', 'Handlers-cleaners',
    'Farming-fishing', 'Tech-support', 'Protective-serv', 'Priv-house-serv', 'Armed-Forces'
])
relationship = st.sidebar.selectbox("Relationship", ['Husband', 'Not-in-family', 'Own-child', 'Unmarried', 'Wife', 'Other-relative'])
race = st.sidebar.selectbox("Race", ['Black', 'White', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other'])
gender = st.sidebar.selectbox("Gender", ['Male', 'Female'])
capital_gain = st.sidebar.number_input("Capital Gain", value=0)
capital_loss = st.sidebar.number_input("Capital Loss", value=0)
hours_per_week = st.sidebar.slider("Hours per week", 1, 99, 40)
native_country = st.sidebar.selectbox("Native Country", ['United-States', 'Mexico', 'Philippines', 'Germany', 'Canada', 'Puerto-Rico', 'El-Salvador', 'India', 'Cuba', 'England', 'Jamaica', 'South', 'China', 'Italy', 'Dominican-Republic', 'Vietnam', 'Guatemala', 'Japan', 'Poland', 'Columbia', 'Taiwan', 'Haiti', 'Iran', 'Portugal', 'Nicaragua', 'Peru', 'Greece', 'France', 'Ecuador', 'Ireland', 'Hong', 'Trinadad&Tobago', 'Thailand', 'Cambodia', 'Laos', 'Yugoslavia', 'Outlying-US(Guam-USVI-etc)', 'Honduras', 'Hungary', 'Scotland', 'Holand-Netherlands'])


# Build input DataFrame (⚠️ must match preprocessing of your training data)
# Need to apply the same label encoding as done during training
encoder = LabelEncoder()

input_data = {
    'age': [age],
    'workclass': [workclass],
    'fnlwgt': [fnlwgt],
    'educational-num': [educational_num],
    'marital-status': [marital_status],
    'occupation': [occupation],
    'relationship': [relationship],
    'race': [race],
    'gender': [gender],
    'capital-gain': [capital_gain],
    'capital-loss': [capital_loss],
    'hours-per-week': [hours_per_week],
    'native-country': [native_country]
}

input_df = pd.DataFrame(input_data)

# Apply label encoding to categorical features in the input DataFrame
# It's important to fit the encoder on the full data or a representative sample
# to ensure all categories are seen. For simplicity here, I will fit on a dummy
# list of all possible values for each categorical column.
# In a real application, load the encoders or fit on the training data.
workclass_categories = ['Private', 'Self-emp-not-inc', 'Local-gov', 'Others', 'State-gov', 'Self-emp-inc', 'Federal-gov']
marital_status_categories = ['Never-married', 'Married-civ-spouse', 'Divorced', 'Separated', 'Widowed', 'Married-spouse-absent', 'Married-AF-spouse']
occupation_categories = ['Prof-specialty', 'Craft-repair', 'Exec-managerial', 'Adm-clerical', 'Sales', 'Other-service', 'Machine-op-inspct', 'Others', 'Transport-moving', 'Handlers-cleaners', 'Farming-fishing', 'Tech-support', 'Protective-serv', 'Priv-house-serv', 'Armed-Forces']
relationship_categories = ['Husband', 'Not-in-family', 'Own-child', 'Unmarried', 'Wife', 'Other-relative']
race_categories = ['Black', 'White', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other']
gender_categories = ['Male', 'Female']
native_country_categories = ['United-States', 'Mexico', 'Philippines', 'Germany', 'Canada', 'Puerto-Rico', 'El-Salvador', 'India', 'Cuba', 'England', 'Jamaica', 'South', 'China', 'Italy', 'Dominican-Republic', 'Vietnam', 'Guatemala', 'Japan', 'Poland', 'Columbia', 'Taiwan', 'Haiti', 'Iran', 'Portugal', 'Nicaragua', 'Peru', 'Greece', 'France', 'Ecuador', 'Ireland', 'Hong', 'Trinadad&Tobago', 'Thailand', 'Cambodia', 'Laos', 'Yugoslavia', 'Outlying-US(Guam-USVI-etc)', 'Honduras', 'Hungary', 'Scotland', 'Holand-Netherlands']


input_df['workclass'] = encoder.fit(workclass_categories).transform(input_df['workclass'])
input_df['marital-status'] = encoder.fit(marital_status_categories).transform(input_df['marital-status'])
input_df['occupation'] = encoder.fit(occupation_categories).transform(input_df['occupation'])
input_df['relationship'] = encoder.fit(relationship_categories).transform(input_df['relationship'])
input_df['race'] = encoder.fit(race_categories).transform(input_df['race'])
input_df['gender'] = encoder.fit(gender_categories).transform(input_df['gender'])
input_df['native-country'] = encoder.fit(native_country_categories).transform(input_df['native-country'])


st.write("### 🔎 Input Data (Processed)")
st.write(input_df)

# Predict button
if st.button("Predict Salary Class"):
    prediction = model.predict(input_df)
    predicted_class = prediction[0]
    st.success(f"✅ Prediction: {predicted_class}")

    # Get and display tentative amount in Rs
    tentative_amount = get_tentative_amount_in_rs(predicted_class)
    st.info(tentative_amount)


# Batch prediction
st.markdown("---")
st.markdown("#### 📂 Batch Prediction")
uploaded_file = st.file_uploader("Upload a CSV file for batch prediction", type="csv")

if uploaded_file is not None:
    batch_data = pd.read_csv(uploaded_file)
    st.write("Uploaded data preview:", batch_data.head())

    # Apply the same preprocessing (label encoding) to batch data
    batch_data['workclass'] = encoder.fit(workclass_categories).transform(batch_data['workclass'])
    batch_data['marital-status'] = encoder.fit(marital_status_categories).transform(batch_data['marital-status'])
    batch_data['occupation'] = encoder.fit(occupation_categories).transform(batch_data['occupation'])
    batch_data['relationship'] = encoder.fit(relationship_categories).transform(batch_data['relationship'])
    batch_data['race'] = encoder.fit(race_categories).transform(batch_data['race'])
    batch_data['gender'] = encoder.fit(gender_categories).transform(batch_data['gender'])
    batch_data['native-country'] = encoder.fit(native_country_categories).transform(batch_data['native-country'])


    batch_preds = model.predict(batch_data)
    batch_data['PredictedClass'] = batch_preds
    st.write("✅ Predictions:")
    st.write(batch_data.head())

    # Add tentative amount for batch predictions (optional, but good for consistency)
    batch_data['TentativeINR'] = batch_data['PredictedClass'].apply(lambda x: get_tentative_amount_in_rs(x))
    st.write("✅ Predictions with Tentative INR:")
    st.write(batch_data.head())


    csv = batch_data.to_csv(index=False).encode('utf-8')
    st.download_button("Download Predictions CSV", csv, file_name='predicted_classes.csv', mime='text/csv')
