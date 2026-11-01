# -*- coding: utf-8 -*-
"""TechBot Store - Giao diện Web & Chatbot (Streamlit + Google Gemini)."""

import json
import os
import re
import unicodedata
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

MODEL_NAME = "gemini-3.5-flash-lite"
CATALOG_PATH = Path("techbot_products_catalog_300_filled.csv")
CONFIG_PATH = Path("config.json")

st.set_page_config(page_title="TechBot Store", page_icon="💻", layout="wide")


# ----------------------------------------------------------------------
# 1. Nạp cấu hình & dữ liệu
# ----------------------------------------------------------------------
DEFAULT_CONFIG = {
    "store_name": "TechBot Store",
    "store_address": "Địa chỉ cửa hàng của bạn",
    "hotline": "1900-0000",
    "functions": [
        "giới thiệu cửa hàng",
        "giới thiệu sản phẩm công nghệ",
        "giải đáp thông số và so sánh sản phẩm",
    ],
    "out_of_scope_message": (
        "Tôi đang không hỗ trợ chức năng này. Xin vui lòng liên hệ nhân viên "
        "cửa hàng qua hotline 1900-0000 để được trợ giúp."
    ),
}

SAMPLE_PRODUCTS = [
    {
        "product_id": "0251", "product_name": "PlayStation 5", "brand": "Sony",
        "category": "Thiết bị chơi game", "release_year": "2020",
        "launch_price": "499", "currency": "USD",
        "specifications": "CPU AMD Zen 2 8 nhân 3.5GHz; GPU RDNA 2 10.28 TFLOPS; RAM 16GB GDDR6; SSD 825GB; hỗ trợ 4K 120Hz, ray tracing",
        "advantages": "SSD siêu nhanh; DualSense có haptic và trigger thích ứng; thư viện game độc quyền mạnh",
        "disadvantages": "Dung lượng SSD hạn chế",
        "successor_model": "PlayStation 5 Pro",
        "official_source_url": "https://www.sony.com",
        "data_status": "Điền từ kiến thức nền, cần đối chiếu nguồn hãng",
    },
    {
        "product_id": "0132", "product_name": "WH-1000XM5", "brand": "Sony",
        "category": "Tai nghe và âm thanh", "release_year": "2022",
        "launch_price": "399.99", "currency": "USD",
        "specifications": "Tai nghe over-ear không dây; ANC với 8 mic; Bluetooth 5.2; pin khoảng 30 giờ",
        "advantages": "ANC rất tốt; thoải mái; âm thanh chi tiết",
        "disadvantages": "Không gập phẳng; giá cao",
        "successor_model": "WH-1000XM6",
        "official_source_url": "https://www.sony.com",
        "data_status": "Điền từ kiến thức nền, cần đối chiếu nguồn hãng",
    },
    {
        "product_id": "0039", "product_name": "MacBook Air M3 (2024)", "brand": "Apple",
        "category": "Laptop", "release_year": "2024",
        "launch_price": "1099", "currency": "USD",
        "specifications": "Chip M3 (CPU 8 nhân); RAM 8/16/24GB; SSD 256GB-2TB; màn hình 13.6\" Liquid Retina; Wi-Fi 6E",
        "advantages": "Giá khởi điểm 1099 USD; hỗ trợ hai màn hình ngoài; chip M3 mạnh",
        "disadvantages": "RAM khởi điểm 8GB; chỉ 2 cổng",
        "successor_model": "MacBook Air M4 (2025)",
        "official_source_url": "https://www.apple.com",
        "data_status": "Điền từ kiến thức nền, cần đối chiếu nguồn hãng",
    },
]


@st.cache_data
def load_config() -> dict:
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            return {**DEFAULT_CONFIG, **json.load(f)}
    return DEFAULT_CONFIG


@st.cache_data
def load_catalog() -> tuple[pd.DataFrame, bool]:
    """Trả về (catalog, dùng_dữ_liệu_mẫu)."""
    if CATALOG_PATH.exists():
        df = pd.read_csv(CATALOG_PATH, dtype=str, encoding="utf-8-sig").fillna("")
        return df, False
    return pd.DataFrame(SAMPLE_PRODUCTS).fillna(""), True


config = load_config()
catalog_df, using_sample = load_catalog()


# ----------------------------------------------------------------------
# 2. Tìm kiếm sản phẩm
# ----------------------------------------------------------------------
def normalize(text) -> str:
    text = unicodedata.normalize("NFD", str(text).lower())
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    text = text.replace("đ", "d")
    return re.sub(r"[^a-z0-9]+", " ", text).strip()


CATEGORY_KEYWORDS = {
    "Điện thoại": ["dien thoai", "smartphone", "phone"],
    "Laptop": ["laptop", "may tinh xach tay", "notebook"],
    "Chuột máy tính": ["chuot", "mouse"],
    "Bàn phím": ["ban phim", "keyboard"],
    "Tai nghe và âm thanh": ["tai nghe", "loa", "am thanh", "headphone", "earbuds"],
    "Màn hình": ["man hinh", "monitor"],
    "Đồng hồ và vòng đeo": ["dong ho", "vong deo", "smartwatch", "watch", "band"],
    "Máy tính bảng và e-reader": ["may tinh bang", "tablet", "e reader", "ereader", "ebook"],
    "Thiết bị chơi game": ["choi game", "may choi game", "console", "game"],
    "Mạng và lưu trữ": ["mang", "luu tru", "router", "wifi", "ssd", "nas", "usb", "o cung"],
}

STOPWORDS = set(
    "co la cua cho va nhung cac mot nao gi nhu the bao nhieu gia may thong so "
    "so sanh khac nhau ra mat nam khi tu den nay do hay voi ve thi nen o mau loai hang cua hang nhe nhat tot nhat".split()
)


def search_products(df: pd.DataFrame, question: str) -> list:
    q_norm = normalize(question)
    q_tokens = {t for t in q_norm.split() if t not in STOPWORDS}
    scored = []
    for idx, row in df.iterrows():
        name_tokens = normalize(row["product_name"]).split()
        core = [t for t in name_tokens if not re.fullmatch(r"(19|20)\d{2}", t)]
        brand_norm = normalize(row["brand"])
        matched = [t for t in core if t in q_tokens]
        score = 3.0 * len(matched)
        if core and len(matched) == len(core):
            score += 10
        brand_hit = brand_norm in q_tokens or any(t in q_tokens for t in brand_norm.split())
        if brand_hit:
            score += 2
        cat_hit = any(
            (kw in f" {q_norm} ") if " " in kw else (kw in q_tokens)
            for kw in CATEGORY_KEYWORDS.get(row["category"], [])
        )
        if cat_hit:
            score += 1.5
        if brand_hit and cat_hit:
            score += 3
        if str(row["release_year"]) in q_tokens:
            score += 1
        if score > 0:
            scored.append((score, idx))
    scored.sort(key=lambda x: (-x[0], x[1]))
    return [idx for _, idx in scored]


def reliability_text(status: str) -> str:
    status = str(status)
    note = ""
    if "Ghi chú:" in status:
        note = " Ghi chú: " + status.split("Ghi chú:", 1)[1].strip()
    if status.startswith("Độ tin cậy thấp"):
        return "thấp, cần xác minh từ nguồn hãng." + note
    return "trung bình, cần đối chiếu nguồn hãng." + note


def format_product(row) -> str:
    year = row["release_year"] or "chưa rõ"
    price = f"{row['launch_price']} {row['currency']}".strip() if row["launch_price"] else "chưa có dữ liệu giá"
    successor = row["successor_model"] or "chưa rõ"
    return (
        f"- {row['product_name']} ({row['brand']} | {row['category']} | ra mắt: {year}). "
        f"Giá khởi điểm: {price}. Thông số: {row['specifications']}. "
        f"Ưu điểm: {row['advantages']}. Nhược điểm: {row['disadvantages']}. "
        f"Đời kế nhiệm: {successor}. "
        f"Độ tin cậy dữ liệu: {reliability_text(row['data_status'])}"
    )


def catalog_overview(df: pd.DataFrame) -> str:
    parts = [f"{cat}: {n} sản phẩm" for cat, n in df["category"].value_counts().items()]
    return f"Catalog có {len(df)} sản phẩm. Các nhóm: " + "; ".join(parts) + "."


def build_catalog_context(df: pd.DataFrame, question: str, top_k: int = 8, extra_names: int = 25) -> str:
    matches = search_products(df, question)
    lines = [catalog_overview(df)]
    if not matches:
        lines.append("Không tìm thấy sản phẩm nào trong catalog khớp với câu hỏi này.")
        return "\n".join(lines)
    lines.append("Các sản phẩm liên quan nhất:")
    for idx in matches[:top_k]:
        lines.append(format_product(df.loc[idx]))
    rest = matches[top_k:top_k + extra_names]
    if rest:
        short = ", ".join(f"{df.loc[i, 'product_name']} ({df.loc[i, 'release_year'] or '?'})" for i in rest)
        lines.append(f"Các sản phẩm liên quan khác (chỉ có tên và năm): {short}.")
    return "\n".join(lines)


# ----------------------------------------------------------------------
# 3. Gemini
# ----------------------------------------------------------------------
def get_api_key() -> str | None:
    key = st.session_state.get("manual_api_key")
    if key:
        return key
    try:
        key = st.secrets.get("GEMINI_API_KEY") or st.secrets.get("GOOGLE_API_KEY")
    except Exception:
        key = None
    return key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")


@st.cache_resource
def get_client(api_key: str):
    from google import genai
    return genai.Client(api_key=api_key)


def build_system_instruction(cfg: dict) -> str:
    functions = ", ".join(cfg.get("functions", []))
    return f"""
Bạn tên là TechBot, một trợ lý AI thông minh, nhiệt tình và lịch sự của cửa hàng {cfg['store_name']}.
Địa chỉ cửa hàng: {cfg['store_address']}. Hotline: {cfg['hotline']}.
Các nhiệm vụ được hỗ trợ: {functions}.

Quy tắc trả lời BẮT BUỘC:
1. Luôn trả lời bằng tiếng Việt, giọng điệu thân thiện, rõ ràng và chuyên nghiệp.
2. Chỉ tư vấn và trả lời dựa trên thông tin sản phẩm trong dữ liệu được cung cấp. Tuyệt đối KHÔNG tự bịa ra sản phẩm, thông số hay giá không có trong dữ liệu.
3. Nếu dữ liệu ghi "chưa có dữ liệu giá" hoặc thiếu thông tin, hãy nói rõ là chưa có dữ liệu thay vì đoán.
4. Khi giới thiệu sản phẩm, nêu thông số chính, ưu điểm, nhược điểm, năm ra mắt và đời kế nhiệm (nếu có). Nếu "Độ tin cậy dữ liệu" là thấp hoặc trung bình, nhắc khách nên đối chiếu với nguồn chính hãng trước khi mua. Giá là giá khởi điểm khi ra mắt, không phải giá bán hiện tại.
5. Khi khách muốn so sánh, hãy đối chiếu từng tiêu chí dựa trên dữ liệu cung cấp.
6. Nếu khách hỏi sản phẩm không có trong dữ liệu nhưng vẫn thuộc lĩnh vực công nghệ, hãy nói sản phẩm đó hiện chưa có trong catalog.
7. Nếu khách hàng hỏi những câu nằm NGOÀI phạm vi cửa hàng/sản phẩm công nghệ (ví dụ: món ăn, thời tiết, giải toán, chính trị...), hãy trả lời ĐÚNG câu sau:
   "{cfg['out_of_scope_message']}"
8. Giữ câu trả lời súc tích, mạch lạc, dễ đọc.
""".strip()


SYSTEM_INSTRUCTION = build_system_instruction(config)

MOCK_REPLY = (
    "[Mock Response]: Chào bạn! Đây là câu trả lời mẫu vì chưa có GEMINI_API_KEY. "
    "Hãy nhập API key ở thanh bên trái để TechBot trả lời thật nhé!"
)


def generate_text(prompt: str, api_key: str | None) -> str:
    if not api_key:
        return MOCK_REPLY
    try:
        from google.genai import types

        client = get_client(api_key)
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_INSTRUCTION,
                thinking_config=types.ThinkingConfig(thinking_level="minimal"),
            ),
        )
        return response.text or "Xin lỗi, tôi chưa tạo được câu trả lời. Bạn thử hỏi lại nhé."
    except Exception as error:
        return f"⚠️ Không gọi được Gemini API: {error}"


def ask_bot(question: str, history: list[dict], api_key: str | None) -> str:
    # Nếu câu hỏi ngắn/nối tiếp và không khớp sản phẩm nào, thêm câu hỏi trước đó để tìm kiếm
    search_text = question
    if not search_products(catalog_df, question):
        prev_users = [m["content"] for m in history if m["role"] == "user"]
        if prev_users:
            search_text = prev_users[-1] + " " + question

    catalog_context = build_catalog_context(catalog_df, search_text)

    recent = history[-6:]
    history_text = "\n".join(
        f"{'Khách' if m['role'] == 'user' else 'TechBot'}: {m['content']}" for m in recent
    )

    prompt = f"""
Dưới đây là thông tin sản phẩm liên quan trong catalog của cửa hàng:
{catalog_context}

Lịch sử hội thoại gần đây:
{history_text or '(chưa có)'}

Khách hàng hỏi:
"{question}"

Hãy trả lời khách hàng theo đúng vai trò TechBot và các quy tắc đã được hướng dẫn.
"""
    return generate_text(prompt, api_key)


# ----------------------------------------------------------------------
# 4. Giao diện
# ----------------------------------------------------------------------
with st.sidebar:
    st.header(f"🏪 {config['store_name']}")
    st.write(f"📍 {config['store_address']}")
    st.write(f"☎️ Hotline: {config['hotline']}")
    st.caption(f"📊 {len(catalog_df)} sản phẩm trong catalog")
    if using_sample:
        st.warning("Không thấy file CSV catalog — đang dùng dữ liệu mẫu.")

    st.divider()
    if not (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")):
        st.text_input(
            "GEMINI_API_KEY",
            type="password",
            key="manual_api_key",
            help="Hoặc đặt trong st.secrets / biến môi trường / file .env",
        )
    if get_api_key():
        st.success(f"Đang dùng model: {MODEL_NAME}")
    else:
        st.info("Chưa có API key → chạy Mock Mode.")

    if st.button("🗑️ Xoá cuộc trò chuyện", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

st.title("💻 TechBot - Trợ lý công nghệ")
st.caption("Hỏi về sản phẩm, thông số, so sánh... TechBot sẽ trả lời dựa trên catalog của cửa hàng.")

if "messages" not in st.session_state:
    st.session_state.messages = []

if not st.session_state.messages:
    with st.chat_message("assistant"):
        st.write(f"Xin chào! Mình là TechBot của {config['store_name']}. Bạn cần tư vấn sản phẩm công nghệ nào ạ? 😊")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if question := st.chat_input("Nhập câu hỏi của bạn..."):
    history = list(st.session_state.messages)
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("TechBot đang soạn câu trả lời..."):
            answer = ask_bot(question, history, get_api_key())
        st.markdown(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})
