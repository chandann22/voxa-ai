BLOCKED_WORDS = [
    "bomb", "hack", "exploit", "malware", "illegal weapons",
    "how to kill", "drug synthesis"
]

MAX_INPUT_LENGTH = 2000

def check_input(text: str) -> tuple[bool, str]:
    if len(text) > MAX_INPUT_LENGTH:
        return False, "Message too long. Please keep it under 2000 characters."
    text_lower = text.lower()
    for word in BLOCKED_WORDS:
        if word in text_lower:
            return False, "I can't help with that topic."
    return True, ""

def clean_output(text: str) -> str:
    return text.strip()