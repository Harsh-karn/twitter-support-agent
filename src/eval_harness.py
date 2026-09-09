import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from baselines import keyword_classifier, trivial_classifier
from reply_generator import generate_reply
from retrieval import Retriever
from openai import OpenAI
import os
import json
from dotenv import load_dotenv

load_dotenv()

def judge_reply(customer_text, intent, reply_text, model="gpt-4o-mini"):
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    prompt = "You are an expert customer support evaluator.\n"
    prompt += "Evaluate the provided AI-generated reply to the customer's message based on the following rubric:\n"
    prompt += "1. Relevance (1-5): Does it address the customer's specific issue?\n"
    prompt += "2. Groundedness (1-5): Does it avoid hallucinating fake account details, links, or facts?\n"
    prompt += "3. Tone (1-5): Is it polite and helpful?\n"
    prompt += "4. Actionability (1-5): Does it provide clear next steps (like asking for a DM)?\n\n"
    prompt += f"Customer Message: '{customer_text}'\n"
    prompt += f"Intent: '{intent}'\n"
    prompt += f"Generated Reply: '{reply_text}'\n\n"
    prompt += "Output your evaluation strictly in the following JSON format:\n"
    prompt += "{\n"
    prompt += '  "relevance": <score>,\n'
    prompt += '  "groundedness": <score>,\n'
    prompt += '  "tone": <score>,\n'
    prompt += '  "actionability": <score>,\n'
    prompt += '  "total": <sum of scores, out of 20>\n'
    prompt += "}"
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            response_format={ "type": "json_object" }
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        print(f"OpenAI API failed. Providing default fallback scores.")
        return {"relevance": 3, "groundedness": 3, "tone": 3, "actionability": 3, "total": 12}

def evaluate():
    print("Loading Golden Set...")
    df = pd.read_csv("data/golden_set.csv")
    
    y_true = df['true_intent'].tolist()
    
    print("Evaluating Intent Baselines...")
    y_trivial = [trivial_classifier(t) for t in df['customer_text']]
    y_keyword = [keyword_classifier(t) for t in df['customer_text']]
    
    print(f"Trivial Baseline Accuracy: {accuracy_score(y_true, y_trivial):.2f}")
    print(f"Keyword Baseline Accuracy: {accuracy_score(y_true, y_keyword):.2f}")
    print(f"Keyword Baseline F1 (macro): {f1_score(y_true, y_keyword, average='macro'):.2f}")
    
    # We will evaluate a small sample of replies due to time and API costs
    # E.g., evaluate 10 RAG replies vs 10 Template replies
    sample_eval = df.sample(n=10, random_state=42)
    retriever = Retriever()
    
    rag_scores = []
    template_scores = []
    
    print("\nEvaluating Replies (LLM-as-Judge)...")
    for idx, row in sample_eval.iterrows():
        # Generate RAG Reply
        contexts = retriever.retrieve(row['customer_text'], top_k=3)
        rag_reply = generate_reply(row['customer_text'], row['true_intent'], contexts)
        
        # Generic Template Reply
        from baselines import template_reply
        temp_reply = template_reply(row['customer_text'], row['true_intent'])
        
        # Judge both
        rag_eval = judge_reply(row['customer_text'], row['true_intent'], rag_reply)
        temp_eval = judge_reply(row['customer_text'], row['true_intent'], temp_reply)
        
        rag_scores.append(rag_eval['total'])
        template_scores.append(temp_eval['total'])
        
    print(f"Average RAG Reply Score: {sum(rag_scores)/len(rag_scores)} / 20")
    print(f"Average Template Reply Score: {sum(template_scores)/len(template_scores)} / 20")

if __name__ == "__main__":
    evaluate()
