import pandas as pd
import random

# Mở rộng kho dữ liệu cực lớn để tạo sự đa dạng
data_pool = {
    "Miền Bắc": {
        "ingredients": [
            "bún chả", "phở bò", "nem rán", "mắm tôm", "tía tô", "lá lốt", "hành lá", "sườn xào", 
            "giấm bỗng", "bánh đa", "su hào", "quả sấu", "tương bần", "mắc khén", "hạt dổi", 
            "rau muống", "thịt đông", "cốm", "bánh chưng", "dưa hành", "cá kho làng Vũ Đại", 
            "măng khô", "mộc nhĩ", "miến dong", "rau cần", "su su", "thịt chim", "bún thang"
        ],
        "keywords": ["Hà Nội", "Tây Bắc", "Đông Bắc", "chuẩn vị Bắc", "thanh đạm"]
    },
    "Miền Trung": {
        "ingredients": [
            "mắm ruốc", "ớt bột", "ớt xanh", "bún bò huế", "tôm chua", "hạt nén", "vả", "bánh tráng", 
            "mì quảng", "hến", "chả bò", "củ nén", "ớt xiêm", "mắm nêm", "tương ớt Hội An", 
            "cá nục", "vịt lộn", "bánh lọc", "bánh bèo", "cao lầu", "ram ít", "kẹo mè xửng",
            "mắm cá cơm", "ớt sừng", "măng chua", "thịt heo luộc", "mắm cái"
        ],
        "keywords": ["Huế", "Đà Nẵng", "Quảng Nam", "cay nồng", "đậm đà"]
    },
    "Miền Nam": {
        "ingredients": [
            "nước cốt dừa", "mắm sặc", "mắm linh", "đường thốt nốt", "rau đắng", "bông súng", 
            "cá kho tộ", "canh chua", "hủ tiếu", "khô sặc", "me chua", "chuối", "lá giang", 
            "chuột đồng", "cá lóc", "tôm càng xanh", "bánh xèo", "bánh khọt", "lẩu mắm",
            "kho quẹt", "rau choại", "bông điên điển", "kèo nèo", "thịt vắt", "mắm bò hóc"
        ],
        "keywords": ["Miền Tây", "Sài Gòn", "ngọt thanh", "phóng khoáng", "dân dã"]
    },
    "Món Âu": {
        "ingredients": [
            "cheese", "butter", "olive oil", "pasta", "rosemary", "thyme", "beefsteak", "cream", 
            "mayonnaise", "bacon", "salmon", "parsley", "basil", "oregano", "truffle oil", 
            "balsamic vinegar", "parmesan", "mozzarella", "asparagus", "lamb chop", "wine",
            "pepperoni", "spaghetti", "lasagna", "tuna", "avocado", "shrimp cocktail"
        ],
        "keywords": ["Italy", "Pháp", "tinh tế", "béo ngậy", "sang trọng"]
    }
}

rows = []
for _ in range(5000):
    label = random.choice(list(data_pool.keys()))
    
    # Chọn ngẫu nhiên số lượng nguyên liệu từ 4 đến 8 để dữ liệu phong phú hơn
    num_items = random.randint(4, 8)
    selected_items = random.sample(data_pool[label]["ingredients"], num_items)
    
    # Thêm từ khóa đặc trưng hoặc tên vùng miền (tỉ lệ 40%)
    if random.random() > 0.6:
        selected_items.append(random.choice(data_pool[label]["keywords"]))
        
    # Xáo trộn thứ tự các nguyên liệu
    random.shuffle(selected_items)
    
    ingredients_str = ", ".join(selected_items)
    rows.append({"ingredients": ingredients_str, "label": label})

df = pd.DataFrame(rows)
# Xóa các dòng trùng lặp nếu có
df = df.drop_duplicates()
df.to_csv("cuisine_data_5000.csv", index=False, encoding="utf-8-sig")

print(f"Hoàn thành! Đã tạo {len(df)} dòng dữ liệu độc nhất.")