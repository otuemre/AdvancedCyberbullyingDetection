import torch

from src.components.model_loader import ModelLoader
from src.components.text_preprocessor import preprocess_text
from src.config import Config
from src.exception import CustomException
from src.logger import get_logger

logger = get_logger(__name__)

class PredictionPipeline:
    def __init__(self, threshold: float = 0.5, max_len: int = 256):
        self.threshold = float(threshold)
        self.max_len = int(max_len)

        loader = ModelLoader()
        self.model, self.tokenizer = loader.load()

    @torch.inference_mode()
    def predict_one(self, text: str) -> dict:
        """
        Predict for a single input text.
        Returns: dict with label, probability, and cleaned text.
        """
        try:
            cleaned = preprocess_text(text)

            if not cleaned:
                return {
                    "text_cleaned": "",
                    "prob_cyberbullying": 0.0,
                    "pred_label": 0,
                    "pred_name": "not_cyberbullying",
                    "threshold": self.threshold,
                }

            inputs = self.tokenizer(
                cleaned,
                truncation=True,
                padding="max_length",
                max_length=self.max_len,
                return_tensors="pt",
            )

            inputs = {k: v.to(Config.DEVICE) for k, v in inputs.items()}

            outputs = self.model(**inputs)
            logits = outputs.logits  # shape: [1, 2]

            probs = torch.softmax(logits, dim=-1).detach().cpu().numpy()[0]
            prob_cyberbullying = float(probs[1])

            pred_label = 1 if prob_cyberbullying >= self.threshold else 0
            pred_name = "cyberbullying" if pred_label == 1 else "not_cyberbullying"

            return {
                "preprocessed_text": cleaned,
                "prob_cyberbullying": prob_cyberbullying,
                "pred_label": pred_label,
                "pred_name": pred_name,
            }

        except Exception as e:
            logger.error("Prediction failed.")
            raise CustomException("Error in PredictionPipeline.predict_one()", cause=e)

    @torch.inference_mode()
    def predict_batch(self, texts: list[str]) -> list[dict]:
        """
        Predict for a batch of texts. Returns list of dict outputs.
        """
        try:
            cleaned_texts = [preprocess_text(t) for t in texts]

            # If all empty after cleaning:
            if all(not t for t in cleaned_texts):
                return [
                    {
                        "text_cleaned": "",
                        "prob_cyberbullying": 0.0,
                        "pred_label": 0,
                        "pred_name": "not_cyberbullying",
                        "threshold": self.threshold,
                    }
                    for _ in cleaned_texts
                ]

            inputs = self.tokenizer(
                cleaned_texts,
                truncation=True,
                padding="max_length",
                max_length=self.max_len,
                return_tensors="pt",
            )

            inputs = {k: v.to(Config.DEVICE) for k, v in inputs.items()}

            outputs = self.model(**inputs)
            logits = outputs.logits  # [batch, 2]
            probs = torch.softmax(logits, dim=-1).detach().cpu().numpy()
            cyber_probs = probs[:, 1]

            results = []
            for cleaned, p in zip(cleaned_texts, cyber_probs):
                p = float(p)
                pred_label = 1 if p >= self.threshold else 0
                pred_name = "cyberbullying" if pred_label == 1 else "not_cyberbullying"

                results.append(
                    {
                        "text_cleaned": cleaned,
                        "prob_cyberbullying": p,
                        "pred_label": pred_label,
                        "pred_name": pred_name,
                        "threshold": self.threshold,
                    }
                )

            return results

        except Exception as e:
            logger.error("Batch prediction failed.")
            raise CustomException("Error in PredictionPipeline.predict_batch()", cause=e)
