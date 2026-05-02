import os
import streamlit as st
import joblib
from google import genai
from PIL import Image
from dotenv import load_dotenv

# --- 1. CẤU HÌNH & TẢI MÔ HÌNH ---
load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
MODEL_NAME = "gemini-3-flash-preview"

# Tải mô hình Naive Bayes và Vectorizer đã huấn luyện
try:
    local_model = joblib.load('cuisine_model.pkl')
    vectorizer = joblib.load('vectorizer.pkl')
    model_ready = True
except FileNotFoundError:
    model_ready = False

st.set_page_config(page_title="AI Chef Agent - Hybrid Edition", page_icon="🍳", layout="wide")

# Khởi tạo lịch sử trong Session State
if "history" not in st.session_state:
    st.session_state.history = []

# --- 2. GIAO DIỆN SIDEBAR (LỊCH SỬ) ---
with st.sidebar:
    st.title("📜 Lịch sử gợi ý")
    if not st.session_state.history:
        st.write("Chưa có món nào được lưu.")
    else:
        for i, item in enumerate(reversed(st.session_state.history)):
            with st.expander(f"Lần {len(st.session_state.history) - i}: {item['name']}"):
                st.caption(f"Phong cách: {item['region']}")
                st.write(item['content'])
    
    if st.button("Xóa lịch sử"):
        st.session_state.history = []
        st.rerun()

# --- 3. GIAO DIỆN CHÍNH ---
st.title("👨‍🍳 AI Chef Agent: Hybrid ML System")
st.write("Sự kết hợp giữa **Naive Bayes (Local)** để phân loại vùng miền và **Gemini 3 Flash** để sáng tạo công thức.")

if not model_ready:
    st.error("⚠️ Không tìm thấy file `cuisine_model.pkl` hoặc `vectorizer.pkl`. Vui lòng chạy file `train.py` trước khi khởi động App!")
    st.stop()

# Chia Tab nhập liệu
tab1, tab2 = st.tabs(["📝 Nhập văn bản", "📸 Quét hình ảnh"])

with tab1:
    ingredients_text = st.text_area("Nhập nguyên liệu bạn có:", placeholder="Ví dụ: bún bò, mắm ruốc, hạt nén...")

with tab2:
    st.write("Chụp ảnh hoặc tải ảnh nguyên liệu:")
    img_file = st.camera_input("Chụp ảnh")
    upload_file = st.file_uploader("Tải ảnh lên", type=["jpg", "png", "jpeg"])
    input_img = img_file if img_file else upload_file
    if input_img:
        st.image(input_img, caption="Ảnh đầu vào", width=300)

# Cấu hình khẩu phần
st.divider()
col_a, col_b = st.columns(2)
with col_a:
    diet = st.selectbox("Chế độ ăn:", ["Bình thường", "Eat Clean", "Thuần chay", "Keto", "Tăng cơ (Gym)"])
with col_b:
    servings = st.slider("Khẩu phần (người):", 1, 6, 2)

# --- 4. HÀM XỬ LÝ HYBRID LOGIC ---
def process_cooking_request(text_input, image_input):
    final_ingredients = ""
    
    # Bước 1: Xử lý đầu vào ảnh (nếu có) bằng Gemini để lấy danh sách nguyên liệu
    if image_input:
        raw_img = Image.open(image_input)
        vision_prompt = "Hãy liệt kê tên tất cả nguyên liệu thực phẩm bạn thấy trong ảnh, ngăn cách bằng dấu phẩy."
        vision_response = client.models.generate_content(model=MODEL_NAME, contents=[vision_prompt, raw_img])
        final_ingredients = vision_response.text
    else:
        final_ingredients = text_input

    # Bước 2: Sử dụng mô hình Local (Naive Bayes) để phân loại vùng miền
    input_vector = vectorizer.transform([final_ingredients])
    predicted_region = local_model.predict(input_vector)[0]
    
    # Bước 3: Gửi kết quả phân loại sang Gemini để viết công thức
    final_prompt = f"""
    Bạn là một đầu bếp chuyên nghiệp. 
    Nguyên liệu: {final_ingredients}.
    Vùng miền xác định bởi mô hình Local: {predicted_region}.
    Chế độ ăn: {diet}. Khẩu phần: {servings} người.
    
    Yêu cầu:
    1. Gợi ý 1 món ăn đặc sắc nhất chuẩn vị {predicted_region}.
    2. Định dạng: Tên món (Bold), Calo, các bước thực hiện chi tiết.
    3. Nếu có vật không ăn được trong nguyên liệu, hãy từ chối lịch sự.
    """
    
    recipe_response = client.models.generate_content(model=MODEL_NAME, contents=final_prompt)
    return predicted_region, recipe_response.text

# --- 5. NÚT THỰC THI ---
if st.button("🔥 PHÂN TÍCH & GỢI Ý", use_container_width=True):
    if ingredients_text.strip() or input_img:
        with st.spinner("Đang chạy Pipeline: Phân loại Local -> Sáng tạo Gemini..."):
            try:
                region, result = process_cooking_request(ingredients_text, input_img)
                
                st.success("Xong rồi!")
                st.info(f"📍 **Kết quả phân loại từ Naive Bayes:** Đây là phong cách ẩm thực **{region}**")
                st.markdown(result)
                
                # Lưu vào lịch sử
                desc = ingredients_text[:20] + "..." if ingredients_text else "Gợi ý từ ảnh"
                st.session_state.history.append({"name": desc, "region": region, "content": result})
            except Exception as e:
                st.error(f"Lỗi hệ thống: {e}")
    else:
        st.warning("Vui lòng cung cấp nguyên liệu!")

st.divider()
st.caption("Dự án AI Agent Hybrid - IT Student STU 2026")