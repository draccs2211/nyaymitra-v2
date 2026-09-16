def detect_language(text: str) -> str:
    """
    Detect language of input text.
    Returns: 'hi' for Hindi, 'en' for English, 'hinglish' for mixed
    """
    if not text:
        return "en"

    hindi_chars = sum(1 for c in text if '\u0900' <= c <= '\u097F')
    total_chars = len(text.replace(" ", ""))

    if total_chars == 0:
        return "en"

    hindi_ratio = hindi_chars / total_chars

    if hindi_ratio > 0.5:
        return "hi"

    hinglish_keywords = [
        "kya", "hai", "mera", "mere", "meri", "aap", "hum", "tum",
        "kaise", "kyun", "kahan", "kab", "kaun", "kya", "nahi", "haan",
        "chahiye", "karna", "dena", "lena", "bolta", "kehna", "batao",
        "problem", "issue", "help", "please", "landlord", "police",
        "court", "lawyer", "vakil", "paisa", "rupee", "case"
    ]

    text_lower = text.lower()
    hinglish_count = sum(1 for kw in hinglish_keywords if kw in text_lower)

    if hinglish_count >= 2:
        return "hinglish"

    return "en"


def get_response_language(detected: str) -> str:
    """Map detected language to response language."""
    if detected in ["hi", "hinglish"]:
        return "hi"
    return "en"
