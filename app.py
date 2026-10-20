"""Streamlit semantic FAQ assistant."""

import csv
from pathlib import Path

import numpy as np
import streamlit as st
from sentence_transformers import SentenceTransformer


MODEL_NAME = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
DATA_PATH = Path(__file__).with_name("QA.csv")


@st.cache_resource(show_spinner="正在載入語意模型，第一次可能需要較久時間…")
def load_knowledge_base():
    """Load the FAQ rows and create normalized Transformer embeddings once."""
    with DATA_PATH.open("r", encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))

    if not rows or set(rows[0]) != {"question", "answer"}:
        raise ValueError("QA.csv 必須包含 question 與 answer 欄位，且不可為空。")

    model = SentenceTransformer(MODEL_NAME)
    texts = [f"問題：{row['question']} 答案：{row['answer']}" for row in rows]
    embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return model, rows, np.asarray(embeddings, dtype=np.float32)


def search_faq(question: str):
    question = question.strip()
    model, rows, embeddings = load_knowledge_base()
    query_embedding = model.encode([question], normalize_embeddings=True)[0]
    scores = embeddings @ np.asarray(query_embedding, dtype=np.float32)
    best_index = int(np.argmax(scores))
    return rows[best_index], float(scores[best_index])


def render_answer(question: str):
    if not question.strip():
        return "請先輸入您的問題。"

    match, score = search_faq(question)
    if score < 0.25:
        return (
            "抱歉，知識庫中沒有足夠相近的問題。\n\n"
            "請改用更具體的關鍵字，例如：訂單、付款、配送或退貨。"
        )

    return (
        f"{match['answer']}\n\n"
        f"**參考問題：** {match['question']}\n\n"
        f"相似度：`{score:.1%}`"
    )


st.set_page_config(page_title="購物網站線上客服", page_icon="🛍️", layout="centered")
st.title("🛍️ 購物網站線上客服")
st.caption("使用多語言 Transformer 語意搜尋最相關的客服答案。")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

question = st.chat_input("例如：忘記會員密碼該怎麼辦？")
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("正在查找最相關的客服答案…"):
            response = render_answer(question)
        st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})

with st.expander("常見問題範例"):
    st.markdown("- 忘記會員密碼該怎麼辦？")
    st.markdown("- 如何查詢訂單配送進度？")
    st.markdown("- 收到瑕疵商品要怎麼退貨？")
