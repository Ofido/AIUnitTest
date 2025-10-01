"""String utility functions for the fake project."""


def reverse_string(text: str) -> str:
    """Reverse a string."""
    return text[::-1]


def capitalize_words(text: str) -> str:
    """Capitalize each word in a string."""
    return " ".join(word.capitalize() for word in text.split())


def count_words(text: str) -> int:
    """Count the number of words in a string."""
    if not text.strip():
        return 0
    return len(text.split())


def remove_duplicates(text: str) -> str:
    """Remove duplicate characters from a string."""
    seen = set()
    result = []
    for char in text:
        if char not in seen:
            seen.add(char)
            result.append(char)
    return "".join(result)


def is_palindrome(text: str) -> bool:
    """Check if a string is a palindrome."""
    clean_text = "".join(char.lower() for char in text if char.isalnum())
    return clean_text == clean_text[::-1]


def find_longest_word(text: str) -> str | None:
    """Find the longest word in a string."""
    if not text.strip():
        return None
    words = text.split()
    return max(words, key=len)


def truncate_string(text: str, max_length: int, suffix: str = "...") -> str:
    """Truncate a string to a maximum length."""
    if len(text) <= max_length:
        return text
    return text[: max_length - len(suffix)] + suffix
