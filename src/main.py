import os
from intent_classifier import classify_intent
from retrieval import Retriever
from reply_generator import generate_reply
from escalator import decide_escalation
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

def run_agent():
    print("Initializing Twitter Support Agent Pipeline...\n")
    
    # 1. Load Data
    data_path = "data/processed/apple_support_pairs.csv"
    if not os.path.exists(data_path):
        print("Data not found. Please run src/data_pipeline.py first.")
        return
        
    df = pd.read_csv(data_path)
    
    # 2. Initialize Retriever
    retriever = Retriever(data_path)
    
    # 3. Process a test sample
    test_sample = df.sample(n=3, random_state=42)
    
    print("--- Running Pipeline on Samples ---\n")
    for idx, row in test_sample.iterrows():
        customer_msg = row['customer_text']
        print(f"Customer Message: {customer_msg}")
        
        # Step A: Classify Intent
        intent = classify_intent(customer_msg)
        print(f"Detected Intent: {intent}")
        
        # Step B: Escalate or Auto-handle
        escalate, reason = decide_escalation(customer_msg, intent)
        print(f"Action: {'ESCALATE' if escalate else 'AUTO-HANDLE'} ({reason})")
        
        # Step C: Generate Grounded Reply
        if not escalate:
            contexts = retriever.retrieve(customer_msg, top_k=3)
            reply = generate_reply(customer_msg, intent, contexts)
            print(f"Drafted Reply: {reply}\n")
        else:
            print("Drafted Reply: [Escalated to human agent]\n")
        print("-" * 50 + "\n")

if __name__ == "__main__":
    run_agent()
