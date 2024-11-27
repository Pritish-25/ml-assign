import streamlit as st
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Load the dataset
df = pd.read_csv(r"./diabetes.csv")

# Header with dark theme
st.markdown("""
    <style>
        body {
            background-color: #1a1a1a;
            color: #ffffff;
        }
        .header {
            font-size: 48px;
            font-family: 'Montserrat', sans-serif;
            font-weight: bold;
            text-align: center;
            color: #00ff88;
            margin-bottom: 20px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.5);
        }
        .sub-header {
            font-size: 22px;
            font-family: 'Montserrat', sans-serif;
            color: #b8b8b8;
            text-align: center;
            margin-bottom: 15px;
        }
        .highlight {
            font-size: 20px;
            color: #00ff88;
            font-weight: bold;
        }
        hr {
            border: none;
            border-top: 2px solid #333333;
        }
        .stButton>button {
            background: linear-gradient(45deg, #00ff88, #00b8ff);
            color: black;
            font-size: 20px;
            font-weight: bold;
            padding: 15px 30px;
            border: none;
            border-radius: 25px;
            cursor: pointer;
            transition: all 0.3s ease;
            box-shadow: 0 4px 15px rgba(0,255,136,0.2);
        }
        .stButton>button:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(0,255,136,0.3);
        }
        .sidebar .sidebar-content {
            background-color: #2d2d2d;
        }
        .stSidebar {
            background-color: #2d2d2d;
            color: #ffffff;
        }
        .stNumberInput > div > div > input {
            color: #ffffff;
            background-color: #333333;
            border: 1px solid #444444;
        }
        .stSlider > div > div > div {
            background-color: #00ff88;
        }
    </style>
    <div class="header">🔬 Health Analytics Pro</div>
    <div class="sub-header">Advanced Diabetes Risk Assessment System</div>
    <hr>
""", unsafe_allow_html=True)

# Updated sidebar
st.sidebar.title("🎯 Patient Profile")
st.sidebar.markdown("### 📊 Enter Health Metrics")

def get_user_input():
    pregnancies = st.sidebar.number_input('👶 Pregnancies', min_value=0, max_value=17, value=3, step=1)
    bp = st.sidebar.number_input('💓 Blood Pressure (mm Hg)', min_value=0, max_value=122, value=70, step=1)
    bmi = st.sidebar.number_input('⚖️ BMI', min_value=0.0, max_value=67.0, value=20.0, step=0.1)
    glucose = st.sidebar.number_input('🔬 Glucose (mg/dL)', min_value=0, max_value=200, value=120, step=1)
    skinthickness = st.sidebar.number_input('📏 Skin Thickness (mm)', min_value=0, max_value=100, value=20, step=1)
    dpf = st.sidebar.number_input('🧬 Diabetes Pedigree Factor', min_value=0.0, max_value=2.4, value=0.47, step=0.01)
    insulin = st.sidebar.number_input('💉 Insulin (IU/mL)', min_value=0, max_value=846, value=79, step=1)
    age = st.sidebar.slider('🎂 Age (years)', min_value=21, max_value=88, value=33)

    return pd.DataFrame({
        'Pregnancies': [pregnancies],
        'Glucose': [glucose],
        'BloodPressure': [bp],
        'SkinThickness': [skinthickness],
        'Insulin': [insulin],
        'BMI': [bmi],
        'DiabetesPedigreeFunction': [dpf],
        'Age': [age]
    })

user_data = get_user_input()

st.markdown("<h2 style='color: #00ff88;'>📊 Health Metrics Overview</h2>", unsafe_allow_html=True)
st.table(user_data)

# Model preparation
x = df.drop(['Outcome'], axis=1)
y = df['Outcome']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
rf = RandomForestClassifier()
rf.fit(x_train, y_train)

if st.button('🔍 Analyze Risk Profile'):
    st.markdown("<h3 style='text-align: center; color: #00ff88;'>🔄 Processing Health Data...</h3>", unsafe_allow_html=True)
    
    progress = st.progress(0)
    for percent in range(100):
        progress.progress(percent + 1)
    
    prediction = rf.predict(user_data)
    
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("<h2 style='color: #00ff88;'>🎯 Analysis Result</h2>", unsafe_allow_html=True)
    
    result = '✅ Low Risk - No Diabetes Indicators' if prediction[0] == 0 else '⚠️ High Risk - Diabetes Indicators Present'
    st.markdown(f"<p class='highlight'>{result}</p>", unsafe_allow_html=True)
    
    accuracy = accuracy_score(y_test, rf.predict(x_test)) * 100
    st.markdown(f"<p style='color: #b8b8b8; font-size: 18px;'>🎯 Model Accuracy: {accuracy:.2f}%</p>", unsafe_allow_html=True)

else:
    st.markdown("<h3 style='text-align: center; color: #b8b8b8;'>👈 Complete your profile and click 'Analyze Risk Profile'</h3>", unsafe_allow_html=True)
