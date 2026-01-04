import streamlit as st
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(page_title="Spam Detector", page_icon="📩")

st.title("📩 Spam Message Detector")
st.write("Check whether a message is **Spam** or **Ham** using Machine Learning")

# -------------------------------
# Load Dataset (Hidden)
# -------------------------------
@st.cache_data
def load_data():
    return pd.read_csv("spam.csv")

df = load_data()

# -------------------------------
# Vectorization
# -------------------------------
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df['Message'])
y = df['Category']

# -------------------------------
# Train-Test Split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.5, random_state=42
)

# -------------------------------
# Train Model
# -------------------------------
model = MultinomialNB()
model.fit(X_train, y_train)

# -------------------------------
# User Input
# -------------------------------
st.subheader("✉️ Enter Your Message")

user_message = st.text_area(
    "",
    height=120,
    placeholder="Type your message here..."
)

# -------------------------------
# Prediction
# -------------------------------
if st.button("🔍 Check Message"):
    if user_message.strip() == "":
        st.warning("⚠️ Please enter a message")
    else:
        msg_vectorized = vectorizer.transform([user_message])
        prediction = model.predict(msg_vectorized)[0]

        if prediction.lower() == "spam":
            st.error("🚨 This message is **SPAM**")
        else:
            st.success("✅ This message is **HAM (Not Spam)**")
