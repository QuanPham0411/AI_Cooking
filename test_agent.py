import google.generativeai as genai
import os
from dotenv import load_dotenv

# 1. Cấu hình
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel('gemini-2.5-flash')

def test_chef_agent(ingredients):
    print(f"\n--- Thử nghiệm với: {ingredients} ---")
    
    prompt = f"""
    Bạn là một đầu bếp AI. Người dùng cung cấp danh sách nguyên liệu: {ingredients}.
    1. Nếu có vật không ăn được, hãy từ chối và giải thích lý do.
    2. Nếu ăn được, gợi ý 1 món ăn nhanh gọn kèm 3 bước thực hiện.
    """
    
    response = model.generate_content(prompt)
    print(response.text)

# 2. Các kịch bản thử nghiệm
test_cases = [
    "Trứng, cà chua, hành lá",          # Kịch bản 1: Bình thường
    "Gạo, muối",                       # Kịch bản 2: Quá ít nguyên liệu
    "Xà phòng, Kim cương, nước sôi"    # Kịch bản 3: Đồ không ăn được
]

for case in test_cases:
    test_chef_agent(case)