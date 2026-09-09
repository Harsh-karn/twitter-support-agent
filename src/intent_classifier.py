import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# Intent Guide
INTENT_GUIDE = {
    "battery_issue": "Complaints about battery draining fast, device not charging, or poor battery health.",
    "update_glitch": "Software bugs explicitly tied to an iOS or macOS update (e.g., autocorrect bugs, app freezing, slow performance).",
    "hardware_failure": "Physical device issues such as broken touchscreen, boot loops, unresponsive buttons, or hardware defects.",
    "account_and_services": "Issues with Apple services like iCloud, Apple Music, iTunes, App Store, or compromised accounts.",
    "customer_service_complaint": "Explicit frustration with a support representative, threats of legal action, or demands for a refund.",
    "other": "Anything that doesn't fit the above or is too ambiguous."
}

def classify_intent(text, model="openai/gpt-oss-20b"):
    client = OpenAI(
        api_key=os.getenv("GROQ_API_KEY"),
        base_url="https://api.groq.com/openai/v1"
    )
    
    prompt = f"You are an expert customer support classifier for AppleSupport.\n"
    prompt += f"Classify the following customer tweet into exactly one of these intents:\n"
    prompt += f"{list(INTENT_GUIDE.keys())}\n\n"
    prompt += f"Tweet: '{text}'\n"
    prompt += f"Intent:"
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=20
        )
        intent = response.choices[0].message.content.strip().lower()
        if intent not in INTENT_GUIDE:
            from baselines import keyword_classifier
            return keyword_classifier(text)
        return intent
    except Exception as e:
        print(f"OpenAI API failed (likely quota limit). Falling back to keyword classifier.")
        from baselines import keyword_classifier
        return keyword_classifier(text)

if __name__ == "__main__":
    sample_text = "the battery life of my 6S is horrendous after iOS REALLY need to address this ASAP"
    print(f"Text: {sample_text}")
    print(f"Intent: {classify_intent(sample_text)}")
