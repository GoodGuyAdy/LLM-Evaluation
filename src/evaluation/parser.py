import re
import json

def extract_star(text: str):
    match = re.search(r"[1-5]", text)
    return int(match.group()) if match else 3

def parse_json_response(raw: str) -> dict:
    """Attempt to parse a JSON response, returning None on failure."""
    try:
        # Strip markdown fences if present
        cleaned = re.sub(r"```json|```", "", raw).strip()
        return json.loads(cleaned)
    except json.JSONDecodeError:
        # Try extracting first JSON object
        match = re.search(r"\{.*?\}", raw, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except:
                pass
        return None