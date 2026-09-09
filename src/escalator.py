import re

def decide_escalation(customer_text, intent):
    """
    Decides whether a message should be escalated to a human agent.
    Returns: (should_escalate (bool), reason (str))
    """
    
    text_lower = customer_text.lower()
    
    # 1. Intent-based rules
    if intent == "customer_service_complaint":
        return True, "Intent indicates a severe complaint or anger."
    
    # 2. Keyword-based safety rules
    escalation_keywords = ['lawyer', 'sue', 'sued', 'stolen', 'lost', 'manager', 'police', 'urgent', 'scam', 'fraud']
    for keyword in escalation_keywords:
        if re.search(r'\b' + keyword + r'\b', text_lower):
            return True, f"Contains escalation keyword: '{keyword}'."
            
    # 3. Complexity rules (e.g. very long messages often contain complex multi-part issues)
    if len(customer_text.split()) > 50:
        return True, "Message is very long (>50 words), likely complex."
        
    # Auto-handle by default
    return False, "Standard issue, safe to auto-handle."

if __name__ == "__main__":
    test_cases = [
        ("My phone is broken and I want to sue you", "customer_service_complaint"),
        ("How do I restart my iPhone?", "other"),
        ("I lost my macbook on the train", "hardware_failure")
    ]
    for text, intent in test_cases:
        esc, reason = decide_escalation(text, intent)
        print(f"Text: {text}\nEscalate: {esc} | Reason: {reason}\n")
