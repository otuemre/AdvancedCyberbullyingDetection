# Advanced Cyberbullying Detection System
This project presents an AI-driven system designed to detect cyberbullying and toxic content in real-time within chat environments. The system integrates a trained machine learning model with a Discord bot (`dc_bot.py`) to automatically monitor messages, classify them, and take moderation actions when necessary.

---

## Features
- Real-time message monitoring via Discord bot
- AI-based toxicity and cyberbullying detection
- Automatic message deletion for harmful content
- Admin notification system with confidence scores
- Simple and scalable API integration

---

## System Architecture
The system consists of two main components:

1. Discord Bot (`dc_bot.py`)
   - Listens to messages in real-time
   - Sends message content to the API
   - Deletes toxic messages and notifies admin

2. Backend API
   - Receives text input
   - Returns prediction label and probability
   - Deployed on AWS using container-based deployment

---

## Tech Stack
- Python
- PyTorch
- Hugging Face Transformers
- Discord.py
- FastAPI (Uvicorn)
- AWS CodeRunner

---

## AWS Deployment
The backend API is deployed using AWS with a container-based approach. The model and API are packaged into a container and deployed to allow scalable and reliable inference.

---

## Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/otuemre/AdvancedCyberbullyingDetection.git
cd AdvancedCyberbullyingDetection
```

### 2. Create Environment Variables
Create a `.env` file:
```
DISCORD_TOKEN=your_discord_token
API_URL=your_api_endpoint
ADMIN_ID=your_discord_user_id
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Bot
```bash
python dc_bot.py
```

---

## Model Details
- Model: BERT (bert-base-uncased)
- Task: Binary classification (toxic vs non-toxic)
- Output:
  - `pred_label`: 0 (non-toxic), 1 (toxic)
  - `prob_cyberbullying`: confidence score

---

## Datasets
No real user data was collected or used in this project. All experiments were conducted using publicly available datasets.

The model was trained using the following datasets:

1. Jason Wang, Kaiqun Fu, Chang-Tien Lu, "Fine-Grained Balanced Cyberbullying Dataset", IEEE Dataport, November 13, 2020  
   DOI: 10.21227/kn1c-zx22  
   Link: https://ieee-dataport.org/open-access/fine-grained-balanced-cyberbullying-dataset

2. Ejaz, Naveed; Choudhury, Salimur; Razi, Fakhra (2024), "A Comprehensive Dataset for Automated Cyberbullying Detection", Mendeley Data, V2  
   DOI: 10.17632/wmx9jj2htd.2  
   Link: https://data.mendeley.com/datasets/wmx9jj2htd/2

---

## Future Improvements
- Improve sarcasm detection by collecting and training on more context-aware datasets
- Test the system in large-scale environments (e.g., handling 100+ concurrent users/messages)
- Extend the system to support more advanced moderation strategies
