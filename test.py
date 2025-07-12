import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import string
from wordcloud import WordCloud

# NLTK downloads - run these once if you haven't already
# These lines check if the NLTK data is already downloaded.
# If not, they will download it. This ensures the script is self-contained.
import nltk

# Correct way to check and download NLTK resources
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try: 
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')


from nltk.corpus import stopwords
from nltk.stem import PorterStemmer # Using PorterStemmer for simplicity

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay

# --- Phase 2: Data Loading and Initial Exploration (EDA) ---

print("--- Phase 2: Data Loading and Initial Exploration (EDA) ---")

# Load the dataset
# The 'encoding' is often 'latin-1' or 'ISO-8859-1' for this type of dataset
# Added a try-except block to handle cases where the file might not be found.
try:
    df = pd.read_csv('spam.csv', encoding='latin-1')
    print("Dataset loaded successfully.")
except FileNotFoundError:
    print("Error: 'spam.csv' not found. Please ensure the file is in the correct directory.")
    exit() # Exit the script if the file is not found

# Inspect the data
print("\nDataFrame Head:")
print(df.head())

print("\nDataFrame Info:")
df.info()

print("\nDataFrame Shape:", df.shape)

# Rename columns for clarity
# The dataset typically has 'v1' for label and 'v2' for message.
# It also often has empty 'Unnamed' columns which we will drop.
df = df[['v1', 'v2']] # Select only relevant columns
df.rename(columns={'v1': 'label', 'v2': 'message'}, inplace=True)
print("\nColumns renamed:")
print(df.head())

# Check for missing values
print("\nMissing values per column:")
print(df.isnull().sum())

# Check Class Distribution
print("\nClass Distribution (Spam vs. Ham):")
print(df['label'].value_counts())

# Visualize Class Distribution
plt.figure(figsize=(6, 4))
sns.countplot(x='label', data=df)
plt.title('Distribution of Spam vs. Ham Messages')
plt.xlabel('Category')
plt.ylabel('Count')
plt.show()

# Visualize Text (Word Clouds)
print("\nGenerating Word Clouds for Spam and Ham messages...")
# Combine all spam messages into a single string for the word cloud
spam_words = ' '.join(df[df['label'] == 'spam']['message'])
# Combine all ham messages into a single string for the word cloud
ham_words = ' '.join(df[df['label'] == 'ham']['message'])

plt.figure(figsize=(15, 7))

# Create and display Word Cloud for Spam messages
plt.subplot(1, 2, 1) # 1 row, 2 columns, first plot
spam_wordcloud = WordCloud(width=800, height=400, background_color='white').generate(spam_words)
plt.imshow(spam_wordcloud, interpolation='bilinear')
plt.title('Spam Word Cloud')
plt.axis('off') # Hide axes for cleaner look

# Create and display Word Cloud for Ham messages
plt.subplot(1, 2, 2) # 1 row, 2 columns, second plot
ham_wordcloud = WordCloud(width=800, height=400, background_color='white').generate(ham_words)
plt.imshow(ham_wordcloud, interpolation='bilinear')
plt.title('Ham Word Cloud')
plt.axis('off') # Hide axes for cleaner look
plt.show()
print("Word clouds displayed.")

# --- Phase 3: Text Preprocessing ---

print("\n--- Phase 3: Text Preprocessing ---")

# Initialize Porter Stemmer and English stopwords
ps = PorterStemmer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    """
    Performs text preprocessing steps:
    1. Lowercasing: Converts all characters to lowercase.
    2. Removing punctuation and numbers: Keeps only alphabetical characters and spaces.
    3. Tokenization: Splits the text into individual words.
    4. Removing stopwords: Filters out common words that don't add much meaning.
    5. Stemming: Reduces words to their root form (e.g., 'running' -> 'run').
    """
    text = text.lower() # 1. Lowercasing
    text = re.sub(r'[^a-zA-Z\s]', '', text) # 2. Remove punctuation and numbers, keep only letters and spaces
    tokens = text.split() # 3. Simple tokenization by splitting on space

    processed_tokens = []
    for word in tokens:
        if word not in stop_words: # 4. Removing stopwords
            processed_tokens.append(ps.stem(word)) # 5. Stemming
    return ' '.join(processed_tokens)

# Apply the preprocessing function to the 'message' column to create a new 'cleaned_message' column
print("Applying text preprocessing to messages...")
df['cleaned_message'] = df['message'].apply(preprocess_text)
print("\nCleaned Messages Head (original vs. cleaned):")
print(df[['message', 'cleaned_message']].head())

# --- Phase 4: Feature Extraction (Text Vectorization) ---

print("\n--- Phase 4: Feature Extraction (Text Vectorization) ---")

# Convert categorical labels ('ham', 'spam') to numerical (0, 1)
df['label_encoded'] = df['label'].map({'ham': 0, 'spam': 1})
print("\nLabels encoded (original vs. numerical):")
print(df[['label', 'label_encoded']].head())

# Initialize TF-IDF Vectorizer
# TF-IDF (Term Frequency-Inverse Document Frequency) is a numerical statistic
# that is intended to reflect how important a word is to a document in a corpus.
# max_features limits the number of features (words) to consider, picking the most frequent/important ones.
print("Vectorizing text data using TF-IDF...")
tfidf_vectorizer = TfidfVectorizer(max_features=5000) # Limit to top 5000 features

# Fit the vectorizer to the cleaned messages and transform them into a numerical matrix (X)
X = tfidf_vectorizer.fit_transform(df['cleaned_message']).toarray()
# The target variable (labels)
y = df['label_encoded']

print("Shape of feature matrix (X):", X.shape)
print("Shape of label vector (y):", y.shape)

# --- Phase 5: Model Training and Evaluation ---

print("\n--- Phase 5: Model Training and Evaluation ---")

# Split the dataset into training and testing sets
# test_size=0.2 means 20% of the data will be used for testing, 80% for training.
# random_state ensures reproducibility (you get the same split every time).
# stratify=y is crucial for imbalanced datasets: it ensures that the proportion
# of 'spam' and 'ham' messages is maintained in both the training and testing sets.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"Training set shapes: X_train={X_train.shape}, y_train={y_train.shape}")
print(f"Testing set shapes: X_test={X_test.shape}, y_test={y_test.shape}")

# Initialize and train the Multinomial Naive Bayes model
# Multinomial Naive Bayes is a probabilistic classifier that is often used for text classification.
print("Training Multinomial Naive Bayes model...")
model = MultinomialNB()
model.fit(X_train, y_train)
print("Model training complete.")

# Make predictions on the unseen test set
y_pred = model.predict(X_test)

# Evaluate the model's performance
print("\n--- Model Evaluation ---")

# Calculate Accuracy: The proportion of correctly classified instances.
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")

print("\nConfusion Matrix:")
# A confusion matrix shows the number of correct and incorrect predictions,
# broken down by each class.
# Rows: Actual classes, Columns: Predicted classes
# [[True Negatives (Ham correctly predicted Ham), False Positives (Ham predicted Spam)]
#  [False Negatives (Spam predicted Ham), True Positives (Spam correctly predicted Spam)]]
cm = confusion_matrix(y_test, y_pred)
print(cm)

# Display the Confusion Matrix visually
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Ham', 'Spam'])
disp.plot(cmap=plt.cm.Blues) # Use a blue color map
plt.title('Confusion Matrix')
plt.show()

print("\nClassification Report:")
# The classification report provides precision, recall, and F1-score for each class.
# Precision: Of all messages predicted as spam, how many were actually spam? (TP / (TP + FP))
# Recall: Of all actual spam messages, how many were correctly identified? (TP / (TP + FN))
# F1-Score: Harmonic mean of precision and recall, a balanced metric.
# 'target_names' makes the report more readable.
print(classification_report(y_test, y_pred, target_names=['Ham', 'Spam']))

# --- Phase 6: Simple Prediction Function ---

print("\n--- Phase 6: Simple Prediction Function ---")

def predict_message_type(message, vectorizer, model, preprocessor_func):
    """
    Predicts whether a given message is 'Spam' or 'Ham'.

    Args:
        message (str): The input text message to classify.
        vectorizer: The trained TF-IDF or CountVectorizer used for feature extraction.
        model: The trained machine learning model (e.g., MultinomialNB).
        preprocessor_func (function): The text preprocessing function used during training.

    Returns:
        str: 'Spam' if the message is predicted as spam, 'Ham' otherwise.
    """
    # 1. Preprocess the input message using the same function used for training data
    cleaned_message = preprocessor_func(message)
    # 2. Vectorize the cleaned message.
    #    Note: .transform() expects an iterable (like a list) of documents,
    #    even if it's just one message.
    vectorized_message = vectorizer.transform([cleaned_message])
    # 3. Make prediction using the trained model
    prediction = model.predict(vectorized_message)
    # 4. Return 'Spam' or 'Ham' based on the numerical prediction (0 or 1)
    return "Spam" if prediction[0] == 1 else "Ham"

# Test the prediction function with some example messages
example_spam_1 = "WINNER!! As a valued network customer you have been selected to receive a £900 prize reward! Call 09061701461 now."
example_ham_1 = "Hey, just wanted to check if you're free for coffee tomorrow afternoon?"
example_spam_2 = "URGENT! Your account has been suspended. Click here to reactivate: http://malicious-link.com"
example_ham_2 = "Thanks for the update. I'll get back to you by end of day."
example_spam_3 = "FreeMsg: Txt: CALL to reply for a free call to a UK number. Svc is £1.50/per min."
example_ham_3 = "Can we meet at 7 PM at the usual spot?"

print(f"'{example_spam_1}' is classified as: {predict_message_type(example_spam_1, tfidf_vectorizer, model, preprocess_text)}")
print(f"'{example_ham_1}' is classified as: {predict_message_type(example_ham_1, tfidf_vectorizer, model, preprocess_text)}")
print(f"'{example_spam_2}' is classified as: {predict_message_type(example_spam_2, tfidf_vectorizer, model, preprocess_text)}")
print(f"'{example_ham_2}' is classified as: {predict_message_type(example_ham_2, tfidf_vectorizer, model, preprocess_text)}")
print(f"'{example_spam_3}' is classified as: {predict_message_type(example_spam_3, tfidf_vectorizer, model, preprocess_text)}")
print(f"'{example_ham_3}' is classified as: {predict_message_type(example_ham_3, tfidf_vectorizer, model, preprocess_text)}")
