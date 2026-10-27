# email-poc: Email Classification and Routing (AI/ML POC)

A proof of concept that reads an email (subject and body) and returns a category,
priority, department, confidence score and a short explanation. When the system
isn't sure, it hands the email to a human instead of guessing.

**How it works**
- Classifies emails into 8 categories: Technical Support, Payment Issue, Account & Login,
  Sales Inquiry, Complaint, Refund Request, Job Application, General Inquiry
- Ensemble of two logistic regression models (TF-IDF and MiniLM sentence embeddings)
  with calibrated confidence
- Confidence-based routing: auto-route, route with review flag, or manual review
- Unknown fallback for emails unlike the training data, plus a rule-based filter for
  automated notices
- Hybrid priority (ML model + keyword rules) and department lookup from a config file
- Explanations: top contributing terms and the most similar training emails
- Streamlit dashboard with single-email, batch CSV, review queue and history views

**Results (see REPORT.md)**
- Held-out test: 95.5% accuracy on routed emails (98.3% when auto-routed)
- Hand-written stress test: 71.4% on routed emails, which shows confidence is
  overestimated on unfamiliar emails

An honest evaluation and a full list of limitations are in [REPORT.md](REPORT.md).

## Run it
    python -m venv .venv
    .venv\Scripts\activate
    pip install -r requirements.txt
    python demo.py                 # terminal demo
    streamlit run app/app.py       # dashboard

## Data
The Kaggle dataset "Multilingual Customer Support Tickets" is not redistributed here;
download it from Kaggle. The 24 synthetic email files are in data/synthetic/.
The hand-written evaluation emails are not published (some come from a personal inbox).

## Layout
src/ (pipeline code), app/ (dashboard), config/ (categories, routing, thresholds),
models/ (saved models), reports/ (figures and prediction files without email text).