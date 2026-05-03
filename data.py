import pandas as pd
import random

SEED = 42
TARGET_ROWS = 5000
random.seed(SEED)
rng = random.Random(SEED)

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

shared_pool = [
    "muối", "đường", "tiêu", "hành", "tỏi", "gừng", "ớt", "nước mắm", "rau thơm",
    "ngò rí", "chanh", "dầu ăn", "gia vị", "hạt nêm", "nước lọc"
]


def build_sample(label):
    label_data = data_pool[label]
    selected_items = []

    # Giữ phần lõi đặc trưng, nhưng không cho nó chiếm toàn bộ tín hiệu
    core_count = rng.randint(2, 5)
    selected_items.extend(rng.sample(label_data["ingredients"], core_count))

    # Thêm nguyên liệu chung giữa nhiều vùng để bài toán khó hơn
    shared_count = rng.randint(1, 3)
    selected_items.extend(rng.sample(shared_pool, shared_count))

    # Thỉnh thoảng trộn một nguyên liệu gây nhiễu từ vùng khác
    if rng.random() < 0.32:
        other_labels = [name for name in data_pool.keys() if name != label]
        other_label = rng.choice(other_labels)
        selected_items.append(rng.choice(data_pool[other_label]["ingredients"]))

    # Từ khóa vùng miền chỉ xuất hiện không quá thường xuyên
    if rng.random() < 0.28:
        selected_items.append(rng.choice(label_data["keywords"]))

    # Thêm một chút nhiễu từ ngữ chung để text giống thực tế hơn
    if rng.random() < 0.22:
        selected_items.append(rng.choice(["tươi", "sạch", "thơm", "đậm vị", "nhẹ"]))

    # Loại trùng và xáo trộn thứ tự
    selected_items = list(dict.fromkeys(selected_items))
    rng.shuffle(selected_items)

    return ", ".join(selected_items)

rows = []

labels = list(data_pool.keys())
base_count = TARGET_ROWS // len(labels)
remainder = TARGET_ROWS % len(labels)

for index, label in enumerate(labels):
    target_count = base_count + (1 if index < remainder else 0)

    for _ in range(target_count):
        rows.append({"ingredients": build_sample(label), "label": label})

df = pd.DataFrame(rows)
# Xóa các dòng trùng lặp nếu có
df = df.drop_duplicates()
df.to_csv("cuisine_data_5000.csv", index=False, encoding="utf-8-sig")

print(f"Hoàn thành! Đã tạo {len(df)} dòng dữ liệu độc nhất.")