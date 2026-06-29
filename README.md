# Toxic Comment Classification using NLP

## Overview

This project is a web application that detects and classifies toxic comments using Natural Language Processing (NLP) and Machine Learning/Deep Learning models. Users can enter a comment and receive predictions for multiple toxicity categories.

## Features

* Multi-label toxic comment classification
* Interactive Flask web interface
* Supports multiple trained models:

  * Logistic Regression
  * XGBoost
  * BiLSTM
* TF-IDF vectorization for traditional ML models
* Keras Tokenizer for deep learning model
* Ready for deployment on Render

## Tech Stack

* Python
* Flask
* Scikit-learn
* TensorFlow / Keras
* XGBoost
* Pandas
* NumPy
* HTML/CSS

## Project Structure

```
Toxic_comment_classification_NLP/
│── app.py
│── requirements.txt
│── render.yaml
│── templates/
│   └── index.html
│── models/
│   ├── bilstm_model.h5
│   ├── final_lr_models.pkl
│   ├── tfidf_vectorizer.pkl
│   ├── keras_tokenizer.json
│   └── xgb_models.pkl
│── .gitignore
└── README.md
```

## Installation

Clone the repository:

```bash
git clone https://github.com/Kalkivayivishnuvardhan/Toxic_comment_classification_NLP.git
```

Move into the project folder:

```bash
cd Toxic_comment_classification_NLP
```

Install the required packages:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
python app.py
```

Open your browser and visit:

```
http://127.0.0.1:5000
```

## Models Used

* Logistic Regression
* XGBoost
* BiLSTM Neural Network

## Future Improvements

* Add BERT and RoBERTa models
* REST API support
* Docker containerization
* User authentication
* Model performance dashboard

## Author

**Kalkivayi Vishnu Vardhan**

GitHub:
https://github.com/Kalkivayivishnuvardhan
