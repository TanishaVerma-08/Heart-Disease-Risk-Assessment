import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
st.write("""
# Heart Disease Risk Assessment
""")

st.sidebar.header('User Input Features')
heart_dataset = pd.read_csv("D:\Practice ML\CodeClause\Heart Disease Risk Management\heart.csv")

x = heart_dataset.drop(['target'], axis = 1)
y = heart_dataset.iloc[:, -1]
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.2, random_state = 0)

def user_input_features():
    age = st.sidebar.number_input('Enter your age: ')

    sex  = st.sidebar.selectbox('Sex',(0,1))
    cp = st.sidebar.selectbox('Chest pain type',(0,1,2,3))
    tres = st.sidebar.number_input('Resting blood pressure: ')
    chol = st.sidebar.number_input('Serum cholestoral in mg/dl: ')
    fbs = st.sidebar.selectbox('Fasting blood sugar',(0,1))
    res = st.sidebar.number_input('Resting electrocardiographic results: ')
    tha = st.sidebar.number_input('Maximum heart rate achieved: ')
    exa = st.sidebar.selectbox('Exercise induced angina: ',(0,1))
    old = st.sidebar.number_input('oldpeak ')
    slope = st.sidebar.number_input('he slope of the peak exercise ST segmen: ')
    ca = st.sidebar.selectbox('number of major vessels',(0,1,2,3))
    thal = st.sidebar.selectbox('thal',(0,1,2))

    data = {'age': age,
            'sex': sex, 
            'cp': cp,
            'trestbps':tres,
            'chol': chol,
            'fbs': fbs,
            'restecg': res,
            'thalach':tha,
            'exang':exa,
            'oldpeak':old,
            'slope':slope,
            'ca':ca,
            'thal':thal
                }
    features = pd.DataFrame(data, index=[0])
    return features
# PATIENT DATA
user_data = user_input_features()
st.subheader('Patient Data')
st.write(user_data)

df = pd.concat([user_data,heart_dataset],axis=0)

df = pd.get_dummies(df, columns = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal'])

df = df[:1] 

rf  = RandomForestClassifier()
rf.fit(x_train, y_train)
user_result = rf.predict(user_data)

st.subheader('Your Report: ')
output=''
if user_result[0]==0:
  output = 'No Risk of Heart Disease'
else:
  output = 'High Risk of Heart Disease'
st.title(output)