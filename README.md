# 購物網站線上客服

這是一個 Streamlit 語意搜尋客服應用，使用
`paraphrase-multilingual-mpnet-base-v2` 將使用者問題與 `QA.csv` 轉成向量，
再以 cosine similarity 找出最相關的客服答案。

## 執行

```bash
streamlit run app.py
```

部署到 Streamlit Community Cloud 時，將此專案連接到 GitHub，主檔案選擇
`app.py`，依賴會由 `requirements.txt` 自動安裝。
