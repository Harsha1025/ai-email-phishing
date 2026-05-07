import pandas as pd

# Load the dataset
print("Loading dataset...")
try:
    df = pd.read_csv('email_data.csv')
    print(f"Dataset loaded successfully with {len(df)} rows.")
except FileNotFoundError:
    print("Error: email_data.csv not found. Please ensure the file is in the same directory.")
    # Exiting or creating dummy data would go here in production

# Handle missing values by dropping any rows with NaN in 'text' or 'label' columns
df = df.dropna(subset=['text', 'label'])
print(f"Dataset size after dropping missing values: {len(df)} rows.")




import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download required NLTK data (run once)
nltk.download('punkt')
nltk.download('stopwords')

stop_words = set(stopwords.words('english'))

def clean_text(text):
    """
    Cleans the input email text by lowercasing, removing special characters,
    and removing stopwords.
    """
    # Convert to lowercase
    text = str(text).lower()
    
    # Remove special characters and numbers (keep only letters)
    text = re.sub(r'[^a-z\s]', '', text)
    
    # Tokenize the text (split into individual words)
    tokens = word_tokenize(text)
    
    # Remove stop words
    filtered_tokens = [word for word in tokens if word not in stop_words]
    
    # Join tokens back into a single string for the vectorizer
    cleaned_text = ' '.join(filtered_tokens)
    
    return cleaned_text

print("Cleaning text data... this may take a moment depending on dataset size.")
df['cleaned_text'] = df['text'].apply(clean_text)
print("Text preprocessing complete.")



from sklearn.feature_extraction.text import TfidfVectorizer

print("Extracting features using TF-IDF...")

# Initialize the TF-IDF Vectorizer
# max_features limits the vocabulary size to the top 5000 words, preventing memory issues
vectorizer = TfidfVectorizer(max_features=5000)

# Fit the vectorizer on the text and transform it into numerical features (X)
X = vectorizer.fit_transform(df['cleaned_text'])

# Isolate our target labels (y)
y = df['label']

print(f"Feature matrix shape: {X.shape}")



from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier

# Split the data 80% for training, 20% for testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Training Multinomial Naive Bayes model...")
nb_model = MultinomialNB()
nb_model.fit(X_train, y_train)

print("Training Random Forest Classifier... (This might take a bit longer)")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

print("Model training complete.")






from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate_model(model, X_test, y_test, model_name):
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Print metrics
    print(f"\n{'='*40}")
    print(f"EVALUATION FOR: {model_name}")
    print(f"{'='*40}")
    print(f"Accuracy Score: {accuracy_score(y_test, y_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Legitimate (0)', 'Phishing (1)']))
    
    # Plot Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Legitimate', 'Phishing'], 
                yticklabels=['Legitimate', 'Phishing'])
    plt.title(f'Confusion Matrix: {model_name}')
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    plt.show()

# Evaluate both models
evaluate_model(nb_model, X_test, y_test, "Multinomial Naive Bayes")
evaluate_model(rf_model, X_test, y_test, "Random Forest Classifier")




def predict_email(email_text, model, vectorizer):
    """
    Takes a raw email string, preprocesses it, extracts features, 
    and returns a prediction using the trained model.
    """
    # 1. Clean the incoming text
    cleaned = clean_text(email_text)
    
    # 2. Convert to TF-IDF features (use transform, NOT fit_transform)
    features = vectorizer.transform([cleaned])
    
    # 3. Predict using the provided model
    prediction = model.predict(features)[0]
    
    # 4. Map the numeric prediction back to a human-readable label
    result = "Phishing" if prediction == 1 else "Safe (Legitimate)"
    
    return result

# --- Let's test it out! ---

suspicious_email = """
URGENT: Your account has been compromised. 
Please click the link below immediately to verify your password and secure your funds.
http://secure-update-account.com/login
Failure to do so will result in account suspension within 24 hours.
"""

safe_email = """
Hey team, 
Just a reminder that we have our weekly sync at 10 AM tomorrow in Conference Room B. 
I've attached the agenda for review. Let me know if you have questions.
Best,
Sarah
"""

print("\n--- INFERENCE TEST ---")
print(f"Suspicious Email Verdict (RF): {predict_email(suspicious_email, rf_model, vectorizer)}")
print(f"Safe Email Verdict (RF): {predict_email(safe_email, rf_model, vectorizer)}")