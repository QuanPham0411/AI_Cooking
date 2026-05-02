import joblib
import os
from google import genai
from dotenv import load_dotenv

# 1. CẤU HÌNH
load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# Load mô hình của bạn
try:
    model = joblib.load('cuisine_model.pkl')
    vectorizer = joblib.load('vectorizer.pkl')
    print("✅ Đã nạp mô hình Local thành công.")
except:
    print("❌ Lỗi: Không tìm thấy file .pkl. Hãy chạy train.py trước.")
    exit()

def stress_test_agent(ingredients, diet="Bình thường"):
    print(f"\n🚀 ĐANG TEST: [{ingredients}] | Chế độ: {diet}")
    
    # Bước 1: Phân loại bằng mô hình Naive Bayes của Quân
    input_vector = vectorizer.transform([ingredients])
    region = model.predict(input_vector)[0]
    print(f"📍 Local Model đoán vùng: {region}")

    # Bước 2: Gọi Gemini để xử lý logic "oái ăm"
    prompt = f"""
    Bạn là đầu bếp AI. 
    Nguyên liệu người dùng nhập: {ingredients}.
    Vùng miền dự đoán: {region}.
    Chế độ ăn: {diet}.

    NHIỆM VỤ:
    1. Nếu có vật không ăn được hoặc cực độc, phải CẢNH BÁO MẠNH và dừng lại.
    2. Nếu các nguyên liệu quá mâu thuẫn (ví dụ: mắm tôm nấu với bơ), hãy nhận xét về sự kỳ lạ này trước khi gợi ý.
    3. Gợi ý 1 món ăn dựa trên logic vùng miền hoặc giải pháp sáng tạo nhất.
    """

    try:
        response = client.models.generate_content(
            model="gemini-3-flash-preview",
            contents=prompt
        )
        print(f"🤖 AI trả lời:\n{response.text}")
    except Exception as e:
        print(f"❌ Lỗi API: {e}")
    print("-" * 50)

# 2. CÁC TRƯỜNG HỢP OÁI ĂM (STRESS TEST CASES)
test_cases = [
    # Case 1: Trộn lẫn đặc trưng vùng miền cực mạnh (Mâu thuẫn logic)
    {"ing": "Mắm tôm, nước cốt dừa, mắm ruốc, hạt nén", "diet": "Bình thường"},
    
    # Case 2: Thực phẩm trộn vật độc hại (Kiểm tra an toàn)
    {"ing": "Thịt bò, rau muống, pin điện thoại, dầu hỏa", "diet": "Keto"},
    
    # Case 3: Nguyên liệu cực kỳ sang chảnh trộn đồ bình dân (Độ lệch tone)
    {"ing": "Trứng vịt lộn, nấm Truffle, vàng lá 24k", "diet": "Bình thường"},
    
    # Case 4: Yêu cầu mơ hồ/không có nguyên liệu thực sự
    {"ing": "Một chút hy vọng và 2 lít nước sôi", "diet": "Ăn chay"},
    
    # Case 5: Đặc trưng Gym/Sức khỏe (Phù hợp với mục tiêu của Quân)
    {"ing": "Ức gà, whey protein, sầu riêng", "diet": "Tăng cơ (Gym)"}
]

for case in test_cases:
    stress_test_agent(case["ing"], case["diet"])