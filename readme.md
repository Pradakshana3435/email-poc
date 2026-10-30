# Email Classification & Routing (proof of concept)

A system that reads an email (subject and body) and returns a category, a priority, a department, a confidence score and a short explanation. When it isn't sure, it hands the email to a human. It also blocks spam, rejects unusable input, and sets aside automated notices.

**Status: a triage assistant with a human in the loop. It is not ready to route real mail on its own.** The full write-up, with every number and limitation, is in [REPORT.md](REPORT.md).

## What it does

Each email passes through these steps, in order. The first one that applies decides the outcome.

| Step | Purpose | Outcome |
|---|---|---|
| Junk filter | Rejects gibberish, very short or greeting-only text | Manual review ("unusable input") |
| Notice filter | Sets aside newsletters, auto-replies and other automated mail | Filtered, no action needed |
| Spam gate | A separate spam model (score 0 to 1) | ≥ 0.90 blocked and quarantined; 0.50 to 0.90 manual review; below 0.50 continues |
| Category model | Average of two logistic regressions (TF-IDF and MiniLM embeddings), calibrated | Auto-route (≥ 85%), route with a review flag (60 to 84%), or manual review (< 60% or unlike the training data) |
| Priority | ML model plus keyword rules | High, Medium or Low |

Categories: Technical Support, Payment Issue, Account & Login, Sales Inquiry, Complaint, Refund Request, Job Application, General Inquiry. The category-to-department routing table is in `config/pipeline.yaml`, so it can be changed without retraining.

## Headline results

| Test | Result |
|---|---|
| Held-out test (152 emails, same generator as training): accuracy of routed emails | 127 / 133 = 95.5% (auto-routed only: 98.3%) |
| Hand-written test (82 messy, short, novel or borderline emails): accuracy of routed emails | 45 / 63 = 71.4% (auto-routed only: 79.6%) |
| Spam gate v2, real Enron test (50 spam / 50 ham) at the chosen thresholds | 37 / 50 spam blocked, 47 / 50 blocked or reviewed; 1 / 50 ham blocked |

On data that looks like the training data the system is accurate. On unfamiliar emails its confidence is too high, so the review flag is a better warning than the auto-route threshold. About 95% of the training data is synthetic (written by an LLM). The spam gate, junk filter and session limit were added after the frozen evaluation and are evaluated separately (REPORT.md, Section 13).

## Quick start

Python 3.11.

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux
pip install -r requirements.txt
streamlit run app/app.py
```

The first run downloads the MiniLM model (`all-MiniLM-L6-v2`) from the Hugging Face Hub. The app opens at http://localhost:8501.

The dashboard has four tabs:

- **Single email**: paste a subject and body, press Classify, and see the category, department, priority, confidence and an explanation (the words that pushed the decision and the most similar training emails).
- **Batch (CSV)**: upload a CSV with `subject` and `body` columns and download the results. A sample is in `data/batch_demo.csv`.
- **Review queue**: emails that need a human (manual review, review flag, blocked spam). A reviewer picks the correct category and marks the email as reviewed.
- **History**: everything classified so far, with charts.

Each browser session is limited to 10 emails (single and batch combined). The counter resets when the page is refreshed. This is a demo guard, not a security feature.

Streamlit may print long `torchvision` tracebacks in the terminal. They are harmless; `.streamlit/config.toml` turns off the file watcher that causes them. After editing code, restart the app.

## Configuration

Everything below lives in `config/pipeline.yaml`:

- the category list and the department each category routes to;
- the confidence thresholds (auto-route, flag, manual review) and the similarity threshold for Unknown;
- the `spam:` block: `enabled`, `block` (0.90), `review` (0.50), `department`, and `model_path` (the spam model file to use);
- `limits.max_emails_per_session` (10).

## Data

Included in `data/`:

- the merged corpus (`emails_final.csv`) and its train, validation and test splits (1,008 emails: 960 synthetic and 48 derived from the Kaggle dataset "Multilingual Customer Support Tickets", credited to its authors);
- the 24 synthetic batches in `data/synthetic/`;
- synthetic spam and hard-ham files in `data/spam/` (genuine-looking emails that sound spammy, used so the spam model doesn't just learn "has a link");
- `batch_demo.csv`, a small sample for the batch tab.

Not included:

- `data/test_handwritten.csv` and the hand-written development set: some emails come from my own inbox;
- `data/spam/real/`: real Enron emails from the Hugging Face dataset `SetFit/enron_spam`, used to train and test the spam model. They are real people's text and are not redistributed. To rebuild them, download the dataset and run `build_real.py`;
- `data/results.db`: the local history of classified emails, which can hold real inputs.

## Training and reproducing

- The saved models are in `models/`: `tfidf_lr`, `minilm_lr`, `priority_model`, `meta` (the calibration temperature), `E_train.npy`, and the spam models `spam_tfidf_lr` (v1) and `spam_v2_tfidf_lr` (v2, in use).
- The held-out and hand-written evaluation numbers come from the saved prediction files in `reports/`, with the email text removed. `make_figures.py` rebuilds the confusion matrices from them without re-running any model.
- Spam model scripts: `check_spam_data.py` and `merge_spam.py` (data checks and merging), `train_spam.py` (v1), `train_spam_v2.py` (v2), `inspect_spam.py` (top spam words and a link-shortcut test), `check_test_spam.py` (runs the full pipeline over `data/test.csv` and lists spam-related decisions).

`python train_spam_v2.py` selects the model on validation data only. **Do not run `python train_spam_v2.py --final` again**, and do not re-run the final evaluation cell in the notebook: both give a one-time look at test data, and the numbers in the report come from those runs.

## Project layout

```
email-poc/
├── app/                  Streamlit dashboard (app.py, pages/)
├── config/pipeline.yaml  categories, routing, thresholds, spam gate, session limit
├── data/                 corpus, splits, synthetic batches, spam data (see above)
├── models/               trained models
├── notebooks/            exploration and evaluation notebook
├── reports/              saved predictions, confusion matrices, screenshots
├── src/                  preprocess, decision, priority, notices, junk, spam, quota, predict, store
├── .streamlit/           dashboard settings
└── *.py                  spam data and model scripts
```

## Known limitations

The main ones (all of them, with detail, are in REPORT.md, Section 10):

- most training data is synthetic, so held-out scores are optimistic;
- confidence does not transfer well to unfamiliar emails;
- the test sets are small and were written by the builder;
- priority is the weakest part of the system;
- the spam model has not been tested on a real support inbox, and one real legitimate email in 50 would be blocked at the chosen threshold in the Enron test;
- the junk filter has no proper evaluation set;
- English only, text only.

## Credits

Kaggle dataset: "Multilingual Customer Support Tickets" (credit to its authors). Spam model real data: `SetFit/enron_spam` on Hugging Face. Sentence embeddings: `all-MiniLM-L6-v2` (sentence-transformers). Synthetic emails were generated with Claude (Anthropic). All hand-written emails, labels, rubrics and evaluation choices are my own.
