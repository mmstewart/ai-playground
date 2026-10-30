from ai_playground.clean_text import clean_text


def test_trim_whitespace():
    assert clean_text("  hello\n\n\n\nworld  ") == "hello\n\nworld"

def test_reduce_multiple_newlines():
    assert clean_text("hello\n\n\n\nworld") == "hello\n\nworld"
    
def test_single_new_space():
    assert clean_text("hello\nworld") == "hello\nworld"