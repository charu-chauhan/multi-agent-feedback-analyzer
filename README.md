# Intelligent User Feedback Analysis and Action System

A multi-agent AI system built with CrewAI that processes user feedback from CSV files, classifies it, extracts insights, creates structured tickets, and provides a Streamlit monitoring UI.

## Setup

```bash
pip install -r requirements.txt
export GEMINI_API_KEY="your-gemini-api-key"
```

Requires a Google Gemini API key (free tier works). Get one at https://aistudio.google.com/apikey

## Run

**Pipeline only (CLI):**
```bash
python pipeline.py
```

**Streamlit UI:**
```bash
streamlit run app.py
```

## Architecture

Uses 6 specialized CrewAI agents in a sequential pipeline:

```
CSV Reader Agent → Feedback Classifier Agent → Bug Analysis Agent
                                              → Feature Extractor Agent
                                              → Ticket Creator Agent → Quality Critic Agent → Output CSVs
```

| Agent | Role |
|---|---|
| CSV Reader | Validates and cleans input data |
| Feedback Classifier | Categorizes into Bug / Feature Request / Praise / Complaint / Spam |
| Bug Analyst | Extracts technical details from bug reports |
| Feature Extractor | Analyzes feature requests and estimates impact |
| Ticket Creator | Generates structured tickets |
| Quality Critic | Reviews tickets for completeness and accuracy |

## Tech Stack
- CrewAI (agent orchestration)
- Google Gemini 2.5 Flash (LLM)
- Streamlit (UI dashboard)
- Pandas (data handling)

## Input Files (in `data/`)
- `app_store_reviews.csv` — 25 app store reviews (Google Play + App Store)
- `support_emails.csv` — 10 customer support emails
- `expected_classifications.csv` — 35 ground truth entries for accuracy measurement

## Output Files (in `output/`)
- `generated_tickets.csv` — structured tickets with quality scores
- `processing_log.csv` — processing history and timestamps
- `metrics.csv` — category accuracy, priority accuracy, processing time

## Streamlit UI Tabs
1. **Dashboard** — Overview of input data and generated tickets
2. **Process Feedback** — Run the pipeline with API key input and progress tracking
3. **Manual Override** — Edit/approve generated tickets inline
4. **Analytics** — Bar charts by category/priority, accuracy metrics, processing log
