def trivial_classifier(text):
    return "other"

def keyword_classifier(text):
    text_lower = text.lower()
    
    if any(k in text_lower for k in ["battery", "charge", "power", "drain"]):
        return "battery_issue"
    elif any(k in text_lower for k in ["update", "ios", "install", "sierra", "glitch", "freeze"]):
        return "update_glitch"
    elif any(k in text_lower for k in ["screen", "broken", "button", "reboot", "stuck"]):
        return "hardware_failure"
    elif any(k in text_lower for k in ["icloud", "password", "music", "id", "itunes", "store"]):
        return "account_and_services"
    elif any(k in text_lower for k in ["sue", "manager", "terrible", "worst", "disrespectful", "refund"]):
        return "customer_service_complaint"
    else:
        return "other"

def template_reply(text, intent):
    return "We're here to help. Please DM us your device details and iOS version so we can assist you further."

if __name__ == "__main__":
    sample = "my battery is draining so fast since the update"
    print(f"Text: {sample}")
    print(f"Trivial: {trivial_classifier(sample)}")
    print(f"Keyword: {keyword_classifier(sample)}")
    print(f"Template Reply: {template_reply(sample, 'battery_issue')}")
