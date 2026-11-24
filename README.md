# Phishing Link Checker – كاشف روابط التصيّد الاحتيالي

Bachelor thesis prototype, "Cybersecurity in Digital Marketing", Business Informatics, German International University (GIU), 2026.

Rates the phishing risk of links for digital marketing teams (Arab region focus), from the URL text alone.

- **Check one link:** risk level (Low / Medium / High / Critical) from a stacked engine (LightGBM on 47 URL features + a character n-gram model) with SHAP reasons.
- **Live check (outside the model):** Google Safe Browsing and domain age (RDAP), shown next to the score; never changes it.
- **Check a list:** CSV/Excel upload or pasted links; summary per channel, alert list, downloadable report.
- **Feedback button:** answers saved to a private Hugging Face dataset.

Time-based test (newer phishing, unseen websites): F1 0.903, ROC-AUC 0.969.

Run locally: `pip install -r requirements.txt` then `python app.py` (Python 3.12).
Optional environment variables: `GSB_API_KEY`, `HF_TOKEN`, `FEEDBACK_REPO`. Google Safe Browsing is used for non-commercial research only.
