# mood_analyzer.py
"""
Rule based mood analyzer for short text snippets.

This class starts with very simple logic:
  - Preprocess the text
  - Look for positive and negative words
  - Compute a numeric score
  - Convert that score into a mood label
"""

from typing import List, Dict, Tuple, Optional
import string

from dataset import POSITIVE_WORDS, NEGATIVE_WORDS


class MoodAnalyzer:
    """
    A very simple, rule based mood classifier.
    """

    def __init__(
        self,
        positive_words: Optional[List[str]] = None,
        negative_words: Optional[List[str]] = None,
    ) -> None:
        # Use the default lists from dataset.py if none are provided.
        positive_words = positive_words if positive_words is not None else POSITIVE_WORDS
        negative_words = negative_words if negative_words is not None else NEGATIVE_WORDS

        # Store as sets for faster lookup.
        self.positive_words = set(w.lower() for w in positive_words)
        self.negative_words = set(w.lower() for w in negative_words)

    # ---------------------------------------------------------------------
    # Preprocessing
    # ---------------------------------------------------------------------

    def preprocess(self, text: str) -> List[str]:
        """
        Convert raw text into a list of normalized tokens.

        Enhancements:
          - Strip whitespace and lowercase.
          - Replace common emoji strings with sentiment tokens.
          - Remove punctuation but keep emoji/emoticon meaning.
          - Normalize repeated characters ("soooo" -> "soo").
          - Split into tokens and map slang to canonical forms.
        """
        import re

        emoji_map = {
            ":)": "smile",
            ":-)": "smile",
            ":(": "sad",
            ":-(": "sad",
            "🥲": "sad",
            "😂": "laugh",
            "💀": "dead",
            "😍": "love",
            "😡": "angry",
        }

        slang_map = {
            "lowkey": "lowkey",
            "highkey": "highkey",
            "no cap": "truth",
            "lol": "laugh",
            "lmao": "laugh",
            "omg": "surprise",
            "fml": "sad",
        }

        cleaned = text.strip().lower()

        # Replace emoji / emoticons before removing punctuation, to preserve meaning.
        for symbol, replacement in emoji_map.items():
            cleaned = cleaned.replace(symbol, f" {replacement} ")

        # Remove punctuation; keep words separated.
        cleaned = cleaned.translate(str.maketrans('', '', string.punctuation))

        raw_tokens = cleaned.split()
        norm_tokens: List[str] = []

        for token in raw_tokens:
            # Normalize repeated characters, but keep at most 2 repeats.
            token = re.sub(r"(.)\1{2,}", r"\1\1", token)

            # Map slang if known.
            if token in slang_map:
                norm_tokens.append(slang_map[token])
            else:
                norm_tokens.append(token)

        return norm_tokens

    # ---------------------------------------------------------------------
    # Scoring logic
    # ---------------------------------------------------------------------

    def score_text(self, text: str) -> int:
        """
        Compute a numeric "mood score" for the given text.

        Positive words increase the score by +1.
        Negative words decrease the score by -1.

        Additional behavior:
          - Handle simple negation words ("not", "never", "no", "can't").
          - Treat repeated occurrences as cumulative.
          - Use lexicon from positive/negative sets and emoji/slang tokens.
        """
        tokens = self.preprocess(text)

        score = 0
        negation_tokens = {"not", "never", "no", "nobody", "none", "nothing", "nowhere", "hardly", "barely", "cant", "isnt", "wasnt", "dont", "doesnt", "didnt"}
        invert_next = False

        for token in tokens:
            if token in negation_tokens:
                invert_next = True
                continue

            token_score = 0
            if token in self.positive_words:
                token_score = 1
            elif token in self.negative_words:
                token_score = -1

            if invert_next and token_score != 0:
                token_score *= -1
                invert_next = False

            score += token_score

        return score

    # ---------------------------------------------------------------------
    # Label prediction
    # ---------------------------------------------------------------------

    def predict_label(self, text: str) -> str:
        """
        Turn the numeric score for a piece of text into a mood label.

        Mapping:
          - score > 0  -> "positive"
          - score < 0  -> "negative"
          - score == 0 -> "neutral"

        We keep the starter label set consistent with TRUE_LABELS.
        """
        score = self.score_text(text)

        if score > 0:
            return "positive"
        if score < 0:
            return "negative"

        return "neutral"

    # ---------------------------------------------------------------------
    # Explanations (optional but recommended)
    # ---------------------------------------------------------------------

    def explain(self, text: str) -> str:
        """
        Return a short string explaining WHY the model chose its label.

        TODO:
          - Look at the tokens and identify which ones counted as positive
            and which ones counted as negative.
          - Show the final score.
          - Return a short human readable explanation.

        Example explanation (your exact wording can be different):
          'Score = 2 (positive words: ["love", "great"]; negative words: [])'

        The current implementation is a placeholder so the code runs even
        before you implement it.
        """
        tokens = self.preprocess(text)

        positive_hits: List[str] = []
        negative_hits: List[str] = []
        score = 0
        negation_tokens = {"not", "never", "no", "nobody", "none", "nothing", "nowhere", "hardly", "barely", "cant", "isnt", "wasnt", "dont", "doesnt", "didnt"}
        invert_next = False

        for token in tokens:
            if token in negation_tokens:
                invert_next = True
                continue

            token_score = 0
            if token in self.positive_words:
                token_score = 1
            elif token in self.negative_words:
                token_score = -1

            if invert_next and token_score != 0:
                token_score *= -1
                invert_next = False

            if token_score > 0:
                positive_hits.append(token)
            elif token_score < 0:
                negative_hits.append(token)

            score += token_score

        return (
            f"Score = {score} "
            f"(positive: {positive_hits or '[]'}, "
            f"negative: {negative_hits or '[]'})"
        )
