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

def classify_intent(text, model="gpt-4o-mini"):
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    prompt = "You are a customer support intent classifier for AppleSupport on Twitter.\n\n"
    prompt += "Classify the following customer message into exactly ONE of these intents:\n"
    for intent, desc in INTENT_GUIDE.items():
        prompt += f"- {intent}: {desc}\n"
    
    prompt += "\nCustomer message:\n"
    prompt += f"\"{text}\"\n\n"
    prompt += "Output ONLY the intent name from the list above, nothing else."
    
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
