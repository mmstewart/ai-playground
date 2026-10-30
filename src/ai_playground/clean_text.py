import re


def clean_text(raw: str) -> str:
    stripped_text = raw.strip()
    
    cleaned_text = re.sub(r'\n{2,}', '\n\n', stripped_text)
    
    return cleaned_text