# train.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import joblib

# 1. Load data
df = pd.read_csv("cuisine_data_5000.csv")

# 2. Vectorize
vectorizer = TfidfVectorizer(ngram_range=(1, 2))
X = vectorizer.fit_transform(df['ingredients'])
y = df['label']

# 3. Train
model = MultinomialNB(alpha=0.1)
model.fit(X, y)

# 4. Save
joblib.dump(model, 'cuisine_model.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')
print(f"🔥 Đã huấn luyện xong mô hình cục bộ với {len(df)} dòng dữ liệu!")