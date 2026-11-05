# Bursa Pro Analyzer

    python3.11 -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
    pip install -r requirements.txt
    streamlit run app.py

Default ticker: 1155.KL (Maybank). Use the sidebar custom box for any Yahoo code (e.g. 5148.KL).

Foreign flow CSV format (optional): columns `date,net_flow` (RM mil).
Educational tool, not financial advice.
