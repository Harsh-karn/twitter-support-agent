# Decision Log

- **Chose AppleSupport over SpotifyCares/Uber_Support**: High volume of tweets, distinct intent clusters (battery, updates, hardware) making classification rules clearer.
- **Used LLM prompts for Classification over fine-tuned BERT**: Faster to implement and iterate on intent taxonomy without needing to build and train on a labelled dataset first.
- **Bootstrapped Golden Set using LLM**: Labeling 200 tweets manually is time-consuming. I used the Groq LLM to bootstrap the labels and manually reviewed a subset, saving hours of manual annotation.
- **Adopted `sentence-transformers` for RAG instead of BM25/TF-IDF**: Semantic similarity is superior for matching customer issues where vocabulary differs (e.g., "phone dying" vs "battery drain").
- **Used FAISS Index**: Extremely fast local nearest-neighbor search, highly scalable even if we expand from 2,000 to the full 100,000+ brand replies.
- **Defined a narrow, 6-class intent taxonomy**: Keeping the classes small avoids overlapping boundaries and simplifies the LLM prompt, increasing accuracy.
- **Implemented a Rule-Based Escalator over a learned classifier**: Easier to interpret and instantly adjust. If legal threats are missed, a single keyword can be added instantly, whereas an ML model requires retraining.
- **Subsampled dataset to 2,000 pairs**: Allows the entire pipeline (embedding, evaluation, execution) to run in under 15 minutes locally on a CPU, adhering to assignment constraints.
- **Removed empty texts and URLs during preprocessing**: URLs are mostly noise or standard Apple support links, and they skew the TF-IDF baselines.
- **Chose a JSON schema for LLM-as-judge output**: Guarantees parseable output for the automated evaluation metrics, avoiding regex extraction bugs.
- **Pre-computed and committed the Golden Set**: To ensure evaluators can run the repo in under 15 minutes without hitting strict free-tier daily token limits on the Groq API, the LLM-labeled golden set is pre-computed and stored in the repo.
- **Chose `openai/gpt-oss-20b` via Groq over local HuggingFace models**: Enabled lightning-fast inference and bypassed the need for the evaluator to download 10GB+ weights or possess a powerful local GPU.
- **Used `temperature=0.0` for LLM-as-judge**: Ensures deterministic, reproducible evaluation scores for the exact same text inputs every time the harness runs.
- **Separated the `Banking77` evaluation into a standalone script**: Keeps the core AppleSupport pipeline uncluttered and strictly focused on the primary project requirements, while still robustly proving the offline intent baseline concept.
- **Added exponential backoff and retry loops to API calls**: Groq's free tier has strict rate limits. Implementing a robust 3-try loop with `time.sleep` ensures the evaluation pipeline won't unpredictably crash mid-execution for the evaluator.
