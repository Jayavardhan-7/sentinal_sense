# 🛡️ Sentinel_Sense – Sentiment-Aware Movie Recommendation System

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Django](https://img.shields.io/badge/Django-Framework-green)
![PyTorch](https://img.shields.io/badge/PyTorch-Model-red)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

**Sentinel_Sense** is an AI-powered movie recommendation web app that understands how you feel and recommends movies accordingly. By leveraging the power of **Sentiment Analysis** using **Transformer-based models**, it captures the mood behind your input and returns tailored movie suggestions.

---

## 🎯 Features

- 🔍 **Sentiment Analysis**: Understands user mood from text using pre-trained Transformer models (e.g., BERT).
- 🎥 **Smart Movie Recommendations**: Suggests movies based on the detected sentiment.
- 🌐 **Web App Interface**: Built with Django for a clean and user-friendly experience.
- ⚙️ **Modular NLP Pipeline**: Easy to plug in different models or improve logic.
- 🧠 **Machine Learning Powered**: Backend models trained/fine-tuned using PyTorch & Hugging Face Transformers.

---

## 🧪 Tech Stack

| Layer         | Tools & Frameworks                     |
| ------------- | -------------------------------------- |
| Frontend      | HTML, CSS, JavaScript (Optional: Bootstrap/Tailwind) |
| Backend       | Django (Python)                        |
| ML/NLP        | PyTorch, Hugging Face Transformers     |
| Sentiment Model | BERT / RoBERTa / DistilBERT (fine-tuned) |
| Deployment (Optional) | Heroku / Render / Vercel               |
| Data Sources  | IMDb, TMDB, or custom movie dataset    |

---

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/your-username/Sentinel_Sense.git
cd Sentinel_Sense
```
### 2. Set up the virtual environment


Create virtual environment
```bashpython -m venv venv```

 Activate virtual environment
 For Windows:
```bash venv\Scripts\activate  ```

For macOS/Linux:
```bash source venv/bin/activat e```
### 3. Install the dependencies

```bash pip install -r requirements.txt```
### 4. Apply migrations

python manage.py migrate
### 5. Run the development server

```bash python manage.py runserver```
Open your browser and go to:
http://127.0.0.1:8000

### 📦 Project Structure

Sentinel_Sense/
│
├── app/                   # Main Django app
├── sentiment_model/       # Transformer-based sentiment analysis logic
├── movie_recommender/     # Movie filtering and suggestion logic
├── templates/             # HTML templates for frontend
├── static/                # Static files like CSS, JS, images
├── manage.py              # Django project manager
├── requirements.txt       # Python dependencies
└── README.md              # This file

🧠 Future Enhancements
🎙️ Voice-based sentiment input

🌐 Support for multiple languages

🤝 Collaborative filtering with sentiment fusion

📈 Real-time feedback and rating integration

🤝 Contributing
Contributions are welcome!

Fork the repo

Create your branch (git checkout -b feature/your-feature)

Commit your changes (git commit -m 'Add feature')

Push and open a pull request

🛡️ License
This project is licensed under the MIT License.

📬 Contact
Made with ❤️ by P. Sai Jayavardhan
