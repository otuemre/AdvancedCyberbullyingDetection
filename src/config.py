from pathlib import Path
import torch

class Config:
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    MODEL_DIR = PROJECT_ROOT / "models" / "bert_cyberbullying"
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
