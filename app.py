import streamlit as st
import json
import joblib
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns

# --- 1. LOAD MODELS & DATABASE ---
try:
    model = joblib.load('cuisine_model.pkl') # [source: 2, 5]
    vectorizer = joblib.load('vectorizer.pkl') # [source: 6]
    with open('recipes.json', 'r', encoding='utf-8') as f:
        recipes_db = json.load(f) # [source: 8]
    model_ready = True
except:
    model_ready = False

st.set_page_config(page_title="AI Chef Local Pro", page_icon="🍳", layout="wide")

# --- 2. GIAO DIỆN CHÍNH ---
st.title("👨‍🍳 AI Chef Agent: Local Expert System")
st.write("Dự án STU 2026 - Công nghệ: Naive Bayes + TF-IDF Vectorization (100% No-API)")

if not model_ready:
    st.error("Lỗi: Thiếu file mô hình hoặc database! Hãy chạy seed_data.py và train.py.")
    st.stop()

tab1, tab2 = st.tabs(["🔍 Tìm kiếm món ăn", "📊 Phân tích mô hình (For Teacher)"])

with tab1:
    input_text = st.text_area("Nhập nguyên liệu (ngăn cách bằng dấu phẩy):", placeholder="Ví dụ: bún, mắm tôm, đậu phụ...")
    
    if st.button("🔥 PHÂN TÍCH & GỢI Ý"):
        if input_text:
            # Bước A: Phân loại vùng miền (AI Classification) [source: 5]
            user_vec = vectorizer.transform([input_text])
            region = model.predict(user_vec)[0]
            probs = model.predict_proba(user_vec)[0]
            max_prob = max(probs) * 100

            # Bước B: Tìm món ăn (Local Search)
            candidates = [r for r in recipes_db if r['region'] == region]
            
            best_recipe = None
            max_score = 0
            
            # Dùng thuật toán so khớp dựa trên từ khóa
            user_words = set([w.strip().lower() for w in input_text.split(',')])
            for dish in candidates:
                dish_words = set([w.lower() for w in dish['ingredients']])
                # Điểm số = Số nguyên liệu khớp / Tổng nguyên liệu món đó cần
                score = len(user_words & dish_words) / len(dish_words)
                if score > max_score:
                    max_score = score
                    best_recipe = dish

            # Hiển thị
            st.info(f"📍 Mô hình nhận diện: **{region}** (Độ tin cậy: {max_prob:.1f}%)")
            
            if best_recipe and max_score > 0:
                st.success(f"Món ăn phù hợp nhất: **{best_recipe['name']}**")
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Nguyên liệu cần có:**")
                    st.write(", ".join(best_recipe['ingredients']))
                with col2:
                    st.write(f"🔥 Năng lượng: **{best_recipe['calories']} kcal**")
                
                st.write("**Cách chế biến:**")
                for i, step in enumerate(best_recipe['steps']):
                    st.write(f"{i+1}. {step}")
            else:
                st.warning("Xin lỗi, nguyên liệu này chưa có trong kho tri thức địa phương.")

with tab2:
    st.header("📈 Báo cáo hiệu năng mô hình")
    # Đọc dữ liệu từ file csv để vẽ biểu đồ
    df = pd.read_csv("cuisine_data_5000.csv") # [source: 3]
    
    col_a, col_b = st.columns(2)
    with col_a:
        st.write("📊 Phân bố tập dữ liệu huấn luyện (5.000 dòng)")
        fig, ax = plt.subplots()
        df['label'].value_counts().plot(kind='pie', autopct='%1.1f%%', ax=ax)
        st.pyplot(fig)
    
    with col_b:
        st.write("🎯 Thuật toán: Multinomial Naive Bayes")
        st.code("""
        P(Vùng | Nguyên liệu) = 
        [P(Nguyên liệu | Vùng) * P(Vùng)] / P(Nguyên liệu)
        """)
        st.info("Mô hình được tối ưu với alpha=0.1 và N-gram(1,2)")