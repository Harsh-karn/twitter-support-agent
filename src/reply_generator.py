import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

def generate_reply(customer_text, intent, retrieved_contexts, model="openai/gpt-oss-20b"):
    client = OpenAI(
        api_key=os.getenv("GROQ_API_KEY"),
        base_url="https://api.groq.com/openai/v1"
    )
    
    prompt = f"You are an AI support agent for AppleSupport on Twitter.\n"
    prompt += "Your task is to draft a helpful, concise reply to the customer's message.\n"
    prompt += f"The customer's message has been classified as the intent: '{intent}'.\n\n"
    
    if retrieved_contexts:
        prompt += "Here are some ways AppleSupport has historically resolved similar issues:\n"
        for i, ctx in enumerate(retrieved_contexts):
            prompt += f"[Historical Example {i+1}]\n"
            prompt += f"Customer asked: {ctx['customer_text']}\n"
            prompt += f"AppleSupport replied: {ctx['brand_text']}\n\n"
            
    prompt += "Instructions for your reply:\n"
    prompt += "1. Adopt the tone of AppleSupport (polite, helpful, concise, usually pointing to a DM or an Apple Support article).\n"
    prompt += "2. Ground your response in the historical examples provided. Mimic their structure and solutions where applicable.\n"
    prompt += "3. DO NOT hallucinate account-specific facts (e.g., do not make up order numbers, refund amounts, or specific device diagnostics).\n"
    prompt += "4. If the issue is complex or requires private info, ask the customer to DM you, just like in the examples.\n\n"
    
    prompt += "Now, draft the reply to this customer message:\n"
    prompt += f"\"{customer_text}\"\n\n"
    prompt += "Reply:"
    
    import time
    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.4,
                max_tokens=200
            )
            raw_output = response.choices[0].message.content
            if raw_output is None or raw_output.strip() == '':
                print(f"Empty response, retrying ({attempt+1}/3)...")
                time.sleep(2)
                continue
            return raw_output.strip()
        except Exception as e:
            print(f"OpenAI API failed ({e}), retrying ({attempt+1}/3)...")
            time.sleep(2)
            
    print("All retries failed. Falling back to template reply.")
    from baselines import template_reply
    return template_reply(customer_text, intent)

if __name__ == "__main__":
    from retrieval import Retriever
    retriever = Retriever()
    sample_query = "the battery life of my 6S is horrendous after iOS REALLY need to address this ASAP"
    contexts = retriever.retrieve(sample_query)
    reply = generate_reply(sample_query, "battery_issue", contexts)
    print(f"\nDrafted Reply:\n{reply}")
