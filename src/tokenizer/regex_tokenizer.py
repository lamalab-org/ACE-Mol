import re
from typing import List

class RegexTokenizer:
    """Run regex tokenization"""

    def __init__(self, regex_pattern: str) -> None:
        """Constructs a RegexTokenizer.

        Args:
            regex_pattern: regex pattern used for tokenization
        """
        self.regex_pattern = regex_pattern
        self.regex = re.compile(self.regex_pattern)

    def tokenize(self, text: str) -> List[str]:
        """Regex tokenization.

        Args:
            text: text to tokenize.

        Returns:
            extracted tokens.
        """
        tokens = [token for token in self.regex.findall(text)]
        return tokens
