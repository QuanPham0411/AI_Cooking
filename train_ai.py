import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix
import joblib

# 1. Load Data
df = pd.read_csv("cuisine_data_5000.csv")

# 2. Vectorization với Stop Words (Loại bỏ từ vô nghĩa)
vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words=['và', 'với', 'của', 'là'])
X = vectorizer.fit_transform(df['ingredients'])
y = df['label']

# 3. Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Huấn luyện với Alpha tối ưu (thường là 0.1 hoặc 1.0)
model = MultinomialNB(alpha=0.1)
model.fit(X_train, y_train)

# 5. Đánh giá chuyên sâu
print("--- BÁO CÁO CHI TIẾT ---")
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# Kiểm tra chéo (Cross-validation)
cv_scores = cross_val_score(model, X, y, cv=5)
print(f"🎯 Độ chính xác ổn định: {cv_scores.mean()*100:.2f}%")

# 6. Lưu Model
joblib.dump(model, 'cuisine_model.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')