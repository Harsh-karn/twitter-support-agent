import pandas as pd
import os
from intent_classifier import classify_intent
from escalator import decide_escalation
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

def generate_golden_set(input_csv="data/processed/apple_support_pairs.csv", output_csv="data/golden_set.csv", sample_size=200):
    df = pd.read_csv(input_csv)
    
    # We set a different random state from the explore script to get a fresh sample, 
    # but we could also just sample any 200.
    golden_df = df.sample(n=sample_size, random_state=123).copy()
    
    intents = []
    escalate_flags = []
    escalate_reasons = []
    
    print(f"Generating labels for {sample_size} examples...")
    for text in tqdm(golden_df['customer_text']):
        intent = classify_intent(text)
        esc, reason = decide_escalation(text, intent)
        
        intents.append(intent)
        escalate_flags.append(esc)
        escalate_reasons.append(reason)
        
    golden_df['true_intent'] = intents
    golden_df['should_escalate'] = escalate_flags
    golden_df['escalation_reason'] = escalate_reasons
    
    golden_df.to_csv(output_csv, index=False)
    print(f"Golden set saved to {output_csv}")

if __name__ == "__main__":
    generate_golden_set()
