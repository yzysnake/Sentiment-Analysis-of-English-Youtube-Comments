import os
import re
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from googleapiclient.discovery import build
import googleapiclient.discovery
import googleapiclient.errors
import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
from tkinter.scrolledtext import ScrolledText
from collections import Counter
from langdetect import detect
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# YouTube API key
API_KEY = 'AIzaSyC8Y7glTka8H9WqymYDKTN190EoJUFlqWU'  # Enter your API key here

# Train data path
csv_file_path = '/Users/ziyuanye/Documents/PSU/2023 Spring/DS 340W/Final Project/Combined_Filtered_Sentiment.csv'


def collect_comments(video_id, max_comments=None, api_key=API_KEY):
    comments = []

    youtube = googleapiclient.discovery.build("youtube", "v3", developerKey=api_key)

    request = youtube.commentThreads().list(
        part="snippet,replies",
        videoId=video_id,
        maxResults=100,
        textFormat="plainText"
    )

    while request and (max_comments is None or len(comments) < max_comments):
        response = request.execute()
        for item in response["items"]:
            comments.append(item["snippet"]["topLevelComment"]["snippet"]["textDisplay"])
            if "replies" in item and item["replies"]["comments"]:
                for reply in item["replies"]["comments"]:
                    if is_english(reply["snippet"]["textDisplay"]):
                        comments.append(reply["snippet"]["textDisplay"])

        # Check if there is a nextPageToken and update the request accordingly
        if "nextPageToken" in response:
            request = youtube.commentThreads().list(
                part="snippet,replies",
                videoId=video_id,
                maxResults=100,
                textFormat="plainText",
                pageToken=response["nextPageToken"]
            )
        else:
            request = None

    # If max_comments is specified, return only the requested number of comments
    if max_comments is not None:
        comments = comments[:max_comments]

    return comments


# Function to remove non-BMP characters
def remove_nonbmp_chars(text):
    return ''.join(c for c in text if c <= '\uFFFF')


def is_english(text):
    try:
        return detect(text) == 'en'
    except:
        return False


def calculate_max_length(sentiment_data, percentile=0.95):
    # Calculate the length of each comment
    comment_lengths = sentiment_data['comment'].apply(lambda x: len(x.split()))

    # Find the length at the desired percentile
    max_length = int(np.percentile(comment_lengths, percentile * 100))

    return max_length


def preprocess_text(text):
    # Create a stopword set and a lemmatizer object
    stopwords_set = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()

    # Remove URLs
    text = re.sub(r'https?://\S+', '', text)

    # Remove special characters and convert text to lowercase
    text = re.sub(r'\W+', ' ', text).lower()

    # Tokenize the text
    tokens = word_tokenize(text)

    # Remove stopwords and lemmatize the tokens
    tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stopwords_set]

    # Reconstruct the preprocessed text
    preprocessed_text = ' '.join(tokens)

    return preprocessed_text


def create_model(vocab_size, embedding_dim, max_length):
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True)),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(3, activation='softmax')
    ])
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


# Function to handle the Analyze button click event
def analyze_comments():
    # Check if we need to train the model
    if train_model_var.get():
        train_and_save_model()

    # Load the model
    try:
        model = tf.keras.models.load_model('sentiment_analysis_model.h5')
    except Exception as e:
        messagebox.showerror("Error", "Failed to load the model. Please train the model first.")
        return

    video_url = video_url_entry.get()
    if not video_url:
        messagebox.showerror("Error", "Please enter a valid video URL.")
        return

    video_id = extract_video_id(video_url)
    if not video_id:
        messagebox.showerror("Error", "Please enter a valid video URL.")
        return

    try:
        max_comments = max_comments_entry.get()
        max_comments = int(max_comments) if max_comments.isdigit() else None

        comments = collect_comments(video_id, max_comments=max_comments)
        preprocessed_comments = [preprocess_text(comment) for comment in comments]
        comment_sequences = tokenizer.texts_to_sequences(preprocessed_comments)
        padded_comment_sequences = pad_sequences(comment_sequences, maxlen=max_length)
        predictions = np.argmax(model.predict(padded_comment_sequences), axis=-1)
        sentiments = ['negative' if pred == 0 else ('neutral' if pred == 1 else 'positive') for pred in predictions]

        sentiment_count = Counter(sentiments)
        summary_text.delete('1.0', tk.END)

        # Clear existing Treeview items
        for item in result_treeview.get_children():
            result_treeview.delete(item)

        # Insert new results
        for comment, sentiment in zip(comments, sentiments):
            sanitized_comment = remove_nonbmp_chars(comment)
            result_treeview.insert("", "end", values=(sanitized_comment, sentiment))

        summary_text.insert(tk.END, f"Negative comments: {sentiment_count['negative']}\n")
        summary_text.insert(tk.END, f"Neutral comments: {sentiment_count['neutral']}\n")
        summary_text.insert(tk.END, f"Positive comments: {sentiment_count['positive']}")

    except Exception as e:
        messagebox.showerror("Error", str(e))


def extract_video_id(video_url):
    video_id_regex = r'(?:youtube(?:-nocookie)?\.com/(?:[^/]+/.+/|(?:v|e(?:mbed)?)/|.*[?&]v=)|youtu\.be/)([^"&?/\s]{11})'
    match = re.search(video_id_regex, video_url)
    if match:
        return match.group(1)
    else:
        return None


def prepare_data(csv_file_path, percentile=0.95):
    # Load sentiment dataset from CSV and preprocess
    sentiment_data = pd.read_csv(csv_file_path)
    sentiment_data['comment'] = sentiment_data['comment'].apply(preprocess_text)

    # Calculate the max_length based on the desired percentile
    max_length = calculate_max_length(sentiment_data, percentile)

    # Tokenize and pad sequences
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(sentiment_data['comment'])
    sequences = tokenizer.texts_to_sequences(sentiment_data['comment'])
    padded_sequences = pad_sequences(sequences, maxlen=max_length)

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(padded_sequences, sentiment_data['label'], test_size=0.2,
                                                        random_state=42)

    return X_train, X_test, y_train, y_test, tokenizer, max_length


def train_and_save_model():
    # Create and train the model

    X_train, X_test, y_train, y_test, tokenizer, max_length = prepare_data(csv_file_path)
    vocab_size = len(tokenizer.word_index) + 1
    embedding_dim = 100
    model = create_model(vocab_size, embedding_dim, max_length)
    history = model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))
    for epoch in range(len(history.history['accuracy'])):
        train_acc = history.history['accuracy'][epoch]
        val_acc = history.history['val_accuracy'][epoch]
        print(f"Epoch {epoch + 1}: Train accuracy = {train_acc:.4f}, Validation accuracy = {val_acc:.4f}")

    # Save the trained model
    model.save('sentiment_analysis_model.h5')


def main():
    # Create the main application window
    # global critical variables to use other functions
    global train_model_var, video_url_entry, max_comments_entry, summary_text, result_treeview, tokenizer, max_length

    X_train, X_test, y_train, y_test, tokenizer, max_length = prepare_data(csv_file_path)

    app = tk.Tk()
    app.title("YouTube Video Comment Sentiment Analysis")

    # Add input fields, labels, and buttons
    video_url_label = ttk.Label(app, text="Video URL:")
    video_url_label.grid(column=0, row=0, padx=10, pady=10, sticky=tk.W)

    video_url_entry = ttk.Entry(app)
    video_url_entry.grid(column=1, row=0, padx=10, pady=10, sticky=tk.W)

    max_comments_label = ttk.Label(app, text="Max Comments:")
    max_comments_label.grid(column=0, row=1, padx=10, pady=10, sticky=tk.W)

    max_comments_entry = ttk.Entry(app)
    max_comments_entry.grid(column=1, row=1, padx=10, pady=10, sticky=tk.W)

    train_model_var = tk.IntVar()
    train_model_checkbox = ttk.Checkbutton(app, text="Train model", variable=train_model_var)
    train_model_checkbox.grid(column=1, row=2, padx=10, pady=10, sticky=tk.E)

    analyze_button = ttk.Button(app, text="Analyze", command=analyze_comments)
    analyze_button.grid(column=1, row=2, padx=10, pady=10, sticky=tk.W)

    result_label = ttk.Label(app, text="Sentiment Analysis Results:")
    result_label.grid(column=0, row=3, padx=10, pady=10, sticky=tk.W)

    # Create Treeview widget
    result_treeview = ttk.Treeview(app, columns=("comment", "sentiment"), show="headings")
    result_treeview.heading("comment", text="Comment")
    result_treeview.heading("sentiment", text="Sentiment")
    result_treeview.column("comment", width=500, anchor=tk.W)
    result_treeview.column("sentiment", width=100, anchor=tk.CENTER)
    result_treeview.grid(column=0, row=4, padx=10, pady=10, columnspan=2)

    # Add a scrollbar
    treeview_scrollbar = ttk.Scrollbar(app, orient="vertical", command=result_treeview.yview)
    treeview_scrollbar.grid(column=2, row=4, padx=0, pady=10, sticky=tk.N + tk.S)
    result_treeview.configure(yscrollcommand=treeview_scrollbar.set)

    summary_label = ttk.Label(app, text="Sentiment Summary:")
    summary_label.grid(column=0, row=5, padx=10, pady=10, sticky=tk.W)

    summary_text = ScrolledText(app, wrap=tk.WORD, width=50, height=6)
    summary_text.grid(column=0, row=6, padx=10, pady=10, columnspan=2)

    # Run the application
    app.mainloop()


if __name__ == "__main__":
    main()
