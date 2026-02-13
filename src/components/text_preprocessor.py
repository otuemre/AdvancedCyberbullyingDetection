import re
import html
import unicodedata
import contractions

from src.logger import get_logger
from src.exception import CustomException

logger = get_logger(__name__)

def preprocess_text(text: str) -> str:
    try:
        logger.debug("Starting text preprocessing.")

        # Ensure string
        if not isinstance(text, str):
            text = "" if text is None else str(text)

        if not text or text.isspace():
            logger.debug("Empty or whitespace-only text received.")
            return ""

        # Decode HTML entities
        text = html.unescape(text)

        # Normalize unicode
        text = unicodedata.normalize("NFKC", text)

        # Expand contractions
        text = contractions.fix(text)

        # Replace URLs
        text = re.sub(r"https?://\S+|www\.\S+", "[URL]", text)

        # Replace emails
        text = re.sub(r"\S+@\S+", "[EMAIL]", text)

        # Replace phone numbers
        text = re.sub(r"\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}", "[PHONE]", text)

        # Replace mentions
        text = re.sub(r"@\w+", "[USER]", text)

        # Remove formatting noise like ==== ::: ```
        text = re.sub(r"(?:^|\s)([=:`]{3,})(?:\s|$)", " ", text)

        # Normalize whitespace
        text = re.sub(r"\s+", " ", text).strip()

        logger.debug("Text preprocessing completed successfully.")

        return text

    except Exception as e:
        logger.error("Error occurred during text preprocessing.")
        raise CustomException("Error in preprocess_text()", cause=e)
