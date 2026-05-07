import streamlit as st
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier

# --- APP CONFIGURATION ---
st.set_page_config(page_title="AI Phishing Detector", page_icon="🛡️")

# --- ML LOGIC (From previous steps) ---
@st.cache_resource # This keeps the model in memory so it doesn't re-train on every click
def train_model():
    # Load and clean data
    df = pd.read_csv('email_data.csv').dropna(subset=['text', 'label'])
    
    nltk.download('punkt')
    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))

    def clean_text(text):
        text = str(text).lower()
        text = re.sub(r'[^a-z\s]', '', text)
        tokens = word_tokenize(text)
        return ' '.join([w for w in tokens if w not in stop_words])

    df['cleaned_text'] = df['text'].apply(clean_text)
    
    # Vectorize
    vectorizer = TfidfVectorizer(max_features=5000)
    X = vectorizer.fit_transform(df['cleaned_text'])
    y = df['label']
    
    # Train Random Forest
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    return model, vectorizer, clean_text

# Initialize model and vectorizer
model, vectorizer, clean_func = train_model()

# --- FRONTEND UI ---
st.title("🛡️ AI Email Phishing Detector")
st.write("Paste the content of a suspicious email below to check if it's a phishing attempt.")

# Input area
email_input = st.text_area("Email Content:", placeholder="Enter email text here...", height=200)

if st.button("Analyze Email"):
    if email_input.strip() == "":
        st.warning("Please paste some text first!")
    else:
        # Process input
        cleaned_input = clean_func(email_input)
        features = vectorizer.transform([cleaned_input])
        
        # Predict
        prediction = model.predict(features)[0]
        probability = model.predict_proba(features)[0][1] # Probability of being phishing
        
        # Display Results
        st.divider()
        if prediction == 1:
            st.error(f"⚠️ **Verdict: PHISHING DETECTED**")
            st.write(f"Confidence Score: {probability*100:.2f}%")
            st.info("💡 **Red Flags:** This email contains language patterns common in cyber attacks, such as urgency or suspicious links.")
        else:
            st.success(f"✅ **Verdict: LIKELY SAFE**")
            st.write(f"Risk Score: {probability*100:.2f}%")
            st.write("This email appears to be legitimate correspondence.")

st.sidebar.markdown("### How it works")
st.sidebar.info("This AI uses a Random Forest Classifier and NLP (Natural Language Processing) to detect social engineering patterns in text.")