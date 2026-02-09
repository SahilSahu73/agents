import html
import re
from typing import Any, Dict, List


# Input Sanitization
def sanitize_string(value: str) -> str:
    """
    Sanitize a string to prevent xss and other injection attacks.
    """
    if not isinstance(value, str):
        value = str(value)

    # 1. HTML Escape: Converts <script> to &lt;script&gt;
    value = html.escape(value)

    # 2. Agressive Scrubbing: Remove script tags entirely if they slipped through
    value = re.sub(r"&lt;script.*?&gt;.*?&lt;/script&gt;", "", value, flags=re.DOTALL)

    # 3. Null byte removal: Prevents low-level binary exploitation attempts
    value = value.replace("\0", "")

    return value


def sanitize_email(email: str) -> str:
    """
    Sanitize and validate an email address format.
    """
    # Basic cleaning
    email = sanitize_string(email)

    # regex validation for standard email format
    if not re.match(r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$", email):
        raise ValueError("Invalid email format")
    
    return email.lower()