import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix

# Load the datasets
fake_df = pd.read_csv('C:/Users/mucah/OneDrive/Desktop/python/Fake News Detection/Fake.csv')
true_df = pd.read_csv('C:/Users/mucah/OneDrive/Desktop/python/Fake News Detection/True.csv')

# Label the datasets
fake_df['label'] = 0  # 0 for fake news
true_df['label'] = 1  # 1 for real news

# Combine datasets
df = pd.concat([fake_df, true_df], ignore_index=True)
df = df.sample(frac=1).reset_index(drop=True)  # Shuffle the dataset

# Text Preprocessing
df['text'] = df['text'].str.lower().str.replace('[^a-zA-Z]', ' ', regex=True)

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.2, random_state=42)

# Convert text to TF-IDF features
tfidf = TfidfVectorizer(stop_words='english', max_df=0.7)
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# Train the model
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

# Predict and evaluate
y_pred = model.predict(X_test_tfidf)
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print(f'Accuracy: {accuracy:.4f}')
print('Confusion Matrix:')
print(conf_matrix)

# Function to test new news text
def test_news(news_text):
    news_tfidf = tfidf.transform([news_text.lower()])
    prediction = model.predict(news_tfidf)
    return 'Real' if prediction[0] == 1 else 'Fake'

# Test the model with new news data
new_article = "The UK government has announced it will cut spending on defence contracts by 50% in 2023"
print(f"The new article is classified as: {test_news(new_article)}")
