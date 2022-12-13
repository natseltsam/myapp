import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

st.markdown("# LinkedIn Predictor")

income = st.selectbox("Income",
             options = ["< $10,000",
                        "$10,000 to < $20,000",
                        "$20,000 to < $30,000",
                        "$30,000 to < $40,000",
                        "$40,000 to < $50,000",
                        "$50,000 to < $75,000",
                        "$75,000 to < $100,000",
                        "$100,000 to < $150,000",
                        "> $150,000"])

# st.write(f"Income: {income}")

if income == "< $10,000":
    income = 1
elif income == "$10,000 to < $20,000":
    income = 2
elif income == "$20,000 to < $30,000":
    income = 3
elif income == "$30,000 to < $40,000":
    income = 4
elif income == "$40,000 to < $50,000":
    income = 5
elif income == "$50,000 to < $75,000":
    income = 6
elif income == "$75,000 to < $100,000":
    income = 7
elif income == "$100,000 to < $150,000":
    income = 8
else: 
    income = 9    

education = st.selectbox("Education",
             options = ["Less than High School",
                        "High School Incomplete",
                        "High School Graduate",
                        "Some College, No Degree",
                        "Two-year Associates Degree",
                        "Four-year Bachelors Degree",
                        "Some Postgraduate Education",
                        "Postgraduate or Professional Degree"])

# st.write(f"Education: {education}")

if education == "Less than High School":
    education = 1
elif education == "High School Incomplete":
    education = 2
elif education == "High School Graduate":
    education = 3
elif education == "Some College, No Degree":
    education = 4
elif education == "Two-year Associates Degree":
    education = 5
elif education == "Four-year Bachelors Degree":
    education = 6
elif education == "Some Postgraduate Education":
    education = 7
else: 
    education = 8

parent = st.selectbox("Parent",
             options = ["Yes",
                        "No"])

# st.write(f"Parent: {parent}")

if parent == "Yes":
    parent = 1
else: 
    parent = 0

married = st.selectbox("Married",
             options = ["Married",
                        "Living with a Partner",
                        "Divorced",
                        "Seperated",
                        "Widowed",
                        "Never Been Married"])

# st.write(f"Married: {married}")

if married == "Married":
    married = 1
else: 
    married = 0

gender = st.selectbox("Gender",
             options = ["Male",
                        "Female",
                        "Other"])

# st.write(f"Gender: {gender}")

if gender == "Female":
    gender = 1
else: 
    gender = 0

age = st.slider(label="Enter an Age",
                      min_value=1,
                      max_value=98,
                      value=1)

# st.write(f"Age: {age}")

s = pd.read_csv("social_media_usage.csv")

def clean_sm(x):
    x = np.where(x == 1, 1, 0)
    return x

ss = pd.DataFrame({
    "income":np.where(s["income"] <= 9, s["income"], np.nan),
    "education":np.where(s["educ2"] <= 8, s["educ2"], np.nan),
    "parent":np.where(s["par"] == 2, 1, 0),
    "married":np.where(s["marital"] == 1, 1, 0),
    "gender":np.where(s["gender"] == 2, 1, 0),
    "age":np.where(s["age"] <= 98, s["age"], np.nan),
    "sm_li":clean_sm(s["web1h"])
})

ss = ss.dropna()

y = ss["sm_li"]
X = ss[["income", "education", "parent", "married", "gender", "age"]]

X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    stratify=y,
                                                    test_size=0.2,
                                                    random_state=42)

lr = LogisticRegression(class_weight="balanced")

lr.fit(X_train, y_train)

person = pd.DataFrame({
    "income": [income],
    "education": [education],
    "parent": [parent],
    "married": [married],
    "gender": [gender],
    "age": [age]
})

predicted_class = lr.predict(person)
probs = lr.predict_proba(person)

st.markdown(f"Predicted User (If 1, You are Predicted to be a LinkedIn User): {predicted_class[0]}")
st.markdown(f"Probability that you are a LinkedIn User: {probs[0][1]}")