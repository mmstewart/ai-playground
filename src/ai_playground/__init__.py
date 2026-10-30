import sys

from .clean_text import clean_text


def main() -> None:
    print('Paste text, then press Ctrl+D (Mac/Linux) or Ctrl+Z then Enter (Windows):')
    
    raw = sys.stdin.read()
    # raw = 'hello\n\n\n\nworld'
    
    cleaned_text = repr(clean_text(raw))
    
    print(cleaned_text)
