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
Copy `.env.example` to `.env` and insert your Groq API key.
```bash
cp .env.example .env
# Edit .env and set GROQ_API_KEY=gsk-...
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

5. **Generate Golden Set (Optional - Pre-computed in Repo)**
*Note: Due to API rate limits, this step can take >1 hour. We have committed the `data/golden_set.csv` directly in the repository so you can skip this step and still reproduce the results in under 15 minutes.*
```bash
python src/generate_golden_set.py
```

6. **Run Evaluation Harness (Headline Results)**
Runs the trivial/keyword baselines and the LLM-as-judge rubric on generated replies. This script uses the pre-computed golden set and runs in under 1 minute.
```bash
python src/eval_harness.py
```

7. **Evaluate Optional Banking77 Dataset**
A completely offline baseline evaluation of 77 intents without needing LLM APIs.
```bash
python src/banking77_intents.py
```

8. **Test the Pipeline Live**
Run the interactive inference script.
```bash
python src/main.py
```

## Reports & Decisions
- [REPORT.md](REPORT.md) - Problem framing, failure analysis, and baseline results.
- [DECISION_LOG.md](DECISION_LOG.md) - List of non-obvious architecture and design decisions.

## Citations
- Dataset: [Customer Support on Twitter (Kaggle)](https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter)
- Secondary Dataset: [Banking77 (Hugging Face)](https://huggingface.co/datasets/PolyAI/banking77)
- Retrievals: [SentenceTransformers](https://sbert.net/) and [FAISS](https://github.com/facebookresearch/faiss)
