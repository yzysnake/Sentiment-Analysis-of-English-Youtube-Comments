# YouTube Comment Sentiment Analysis

This project focuses on analyzing the sentiment of YouTube video comments. The goal is to predict the sentiment (positive, neutral, or negative) of comments on a YouTube video using a deep learning model built with TensorFlow and LSTM layers. The application can fetch YouTube comments via the YouTube API, preprocess the text, and classify comments into sentiment categories.

## Objectives

- **Fetch YouTube Comments**: Automatically collect comments from a specified YouTube video using the YouTube Data API.
- **Preprocess Text**: Clean and preprocess the text data (tokenization, removing stopwords, etc.).
- **Train Sentiment Analysis Model**: Build and train a Bi-directional LSTM model to classify sentiments.
- **Predict Sentiments**: Use the trained model to predict the sentiment of YouTube comments.

## Dataset

- **Source**: The model is trained using a custom dataset, `Combined_Filtered_Sentiment.csv`, containing labeled YouTube comments with their respective sentiment (positive, neutral, or negative).
- **Features**:
  - **comment**: The actual text of the comment.
  - **label**: The sentiment label (0 for negative, 1 for neutral, 2 for positive).

## Project Structure

```bash
├── data/                    # Folder for storing the CSV dataset
├── sentiment_analysis_model.h5   # Saved Keras model after training
├── training_history.pkl      # Training history for performance analysis
├── src/                     # Python source files
    ├── sentiment_analysis.py   # Main script for model training and sentiment analysis
├── README.md                # Project documentation
└── requirements.txt         # Dependencies for the project
