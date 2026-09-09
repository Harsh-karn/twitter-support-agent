# Twitter Support Agent (Hiver Take-Home)

This repository contains an end-to-end AI support agent for AppleSupport on Twitter, built for the Hiver SDE Intern Take-Home Assignment.

## Features
- **Intent Classification**: Classifies customer tweets into 6 predefined intents using an LLM.
- **RAG Reply Generator**: Retrieves historically resolved similar issues using `sentence-transformers` and FAISS, then drafts a contextual, grounded reply.
- **Escalation Logic**: A deterministic rules engine to flag severe complaints or complex messages for human intervention.
- **Evaluation Harness**: Scripts to compute baseline metrics and run an LLM-as-judge rubric.

## Setup & Reproducibility (Under 15 minutes)

1. **Environment Setup**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

2. **Configure API Keys**
Copy `.env.example` to `.env` and insert your OpenAI API key.
```bash
cp .env.example .env
# Edit .env and set OPENAI_API_KEY=sk-...
```

3. **Data Download**
If you have the Kaggle CLI configured (`~/.kaggle/kaggle.json`), run:
```bash
mkdir -p data/raw
kaggle datasets download thoughtvector/customer-support-on-twitter -p data/raw
unzip data/raw/customer-support-on-twitter.zip -d data/raw
```
*Alternatively, download `twcs.csv` manually and place it in `data/raw/`.*

4. **Run Data Pipeline**
Extracts conversation pairs for AppleSupport and subsamples 2,000 rows.
```bash
python src/data_pipeline.py
```

5. **Generate Golden Set**
Randomly samples 200 pairs and runs the LLM classifiers to bootstrap the labels.
```bash
python src/generate_golden_set.py
```

6. **Run Evaluation Harness**
Runs trivial/simple baselines and the LLM-as-judge on generated replies.
```bash
python src/eval_harness.py
```

7. **Test the Pipeline Live**
Run the interactive inference script.
```bash
python src/main.py
```

## Reports & Decisions
- [REPORT.md](REPORT.md) - Problem framing, failure analysis, and baseline results.
- [DECISION_LOG.md](DECISION_LOG.md) - List of non-obvious architecture and design decisions.

## Citations
- Dataset: [Customer Support on Twitter (Kaggle)](https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter)
- Retrievals: [SentenceTransformers](https://sbert.net/) and [FAISS](https://github.com/facebookresearch/faiss)
