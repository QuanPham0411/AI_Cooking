import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix
import joblib

# 1. Load Data
df = pd.read_csv("cuisine_data_5000.csv")

X = df["ingredients"]
y = df["label"]

label_names = sorted(y.unique())

# 2. Train/Test Split trên text thô để tránh rò rỉ dữ liệu từ tập test
X_train, X_test, y_train, y_test = train_test_split(
	X,
	y,
	test_size=0.2,
	random_state=42,
	stratify=y,
)

# 3. Pipeline vectorization + classifier
model = Pipeline(
	[
		(
			"tfidf",
			TfidfVectorizer(
				ngram_range=(1, 2),
				stop_words=['và', 'với', 'của', 'là'],
			),
		),
		("nb", MultinomialNB(alpha=0.3)),
	]
)

# 4. Huấn luyện
model.fit(X_train, y_train)

# 5. Đánh giá
print("--- BÁO CÁO CHI TIẾT ---")
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred, labels=label_names))
print("\n--- CONFUSION MATRIX ---")
print(confusion_matrix(y_test, y_pred, labels=label_names))

# Kiểm tra chéo trên text thô
cv_scores = cross_val_score(model, X, y, cv=5, scoring="accuracy")
print(f"🎯 Độ chính xác ổn định: {cv_scores.mean()*100:.2f}%")

# 6. Lưu Model
joblib.dump(model, 'cuisine_model.pkl')