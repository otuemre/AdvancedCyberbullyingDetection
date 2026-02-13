from transformers import AutoTokenizer, AutoModelForSequenceClassification

from src.config import Config
from src.exception import CustomException
from src.logger import get_logger

logger = get_logger(__name__)

class ModelLoader:
    def __init__(self):
        self.model = None
        self.tokenizer = None

    def load(self):
        try:
            logger.info("Loading model and tokenizer...")

            self.tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_DIR)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                Config.MODEL_DIR
            )

            self.model.to(Config.DEVICE)
            self.model.eval()

            logger.info(f"Model loaded successfully on device: {Config.DEVICE}")

            return self.model, self.tokenizer

        except Exception as e:
            logger.error("Failed to load model.")
            raise CustomException("Error in ModelLoader.load()", cause=e)
