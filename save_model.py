import os
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

ARTIFACT_DIR = "artifacts"
os.makedirs(ARTIFACT_DIR, exist_ok=True)

csv_path = "spam.csv"
df = pd.read_csv(csv_path, encoding='latin-1')

df = df[['v1', 'v2']]
df = df.rename(columns={'v1': 'label', 'v2': 'text'})
df['label_num'] = df['label'].map({'ham': 0, 'spam': 1})

X = df['text']
y = df['label_num']

X_train, X_test, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)

feature_extraction = TfidfVectorizer(min_df=1, stop_words='english', lowercase=True)
X_train_features = feature_extraction.fit_transform(X_train)

model = LogisticRegression()
model.fit(X_train_features, y_train)

joblib.dump(feature_extraction, "artifacts/tfidf.joblib")
joblib.dump(model, "artifacts/spam_model.joblib")

print("Artifacts saved successfully inside /artifacts folder")
