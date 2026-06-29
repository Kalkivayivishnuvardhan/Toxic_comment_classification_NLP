---
title: Toxic Comment Classification Nlp
emoji: 😻
colorFrom: indigo
colorTo: indigo
sdk: docker
pinned: false
---

# Toxic Comment Classification using NLP

## Overview

This project is a web application that detects and classifies toxic comments using Natural Language Processing (NLP) and Machine Learning/Deep Learning models. Users can enter a comment and receive predictions for multiple toxicity categories.

## Features

- Multi-label toxic comment classification
- Interactive Flask web interface
- Supports multiple trained models:
  - Logistic Regression
  - XGBoost
  - BiLSTM
- TF-IDF vectorization for traditional ML models
- Keras Tokenizer for deep learning model
- Ready for deployment on Hugging Face Spaces

## Tech Stack

- Python
- Flask
- Scikit-learn
- TensorFlow / Keras
- XGBoost
- Pandas
- NumPy
- HTML/CSS

## Project Structure

```text
Toxic_comment_classification_NLP/
│── app.py
│── Dockerfile
│── requirements.txt
│── templates/
│── models/
│── README.md
```

## Installation

```bash
git clone https://github.com/Kalkivayivishnuvardhan/Toxic_comment_classification_NLP.git

cd Toxic_comment_classification_NLP

pip install -r requirements.txt

python app.py
```

Open:

```
http://127.0.0.1:5000
```

## Models Used

- Logistic Regression
- XGBoost
- BiLSTM

## Future Improvements

- BERT
- RoBERTa
- REST API
- Dashboard

## Author

**Kalkivayi Vishnu Vardhan**

GitHub:
https://github.com/Kalkivayivishnuvardhan