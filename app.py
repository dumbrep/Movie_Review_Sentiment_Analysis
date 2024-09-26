import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import nltk

lemmatizer = WordNetLemmatizer()

word_index = imdb.get_word_index()

max_length = 200

model = load_model('LSTM_imdb.h5')

def pre_process(text):
    for i in range(len(text)):
        words = nltk.word_tokenize(text)
        words = [lemmatizer.lemmatize(word) for word in words if word not in set(stopwords.words('english')) ]
        text = ' '.join(words) #converting all words into sentences
    words = text.lower().split()
    encoded_review = [ word_index.get(i,2) + 3 for i in words]
    padded_review = sequence.pad_sequences([encoded_review],maxlen = max_length)
    return padded_review

def predict_sentiment(review):
    prepocessed_input = pre_process(review)

    predicted = model.predict(prepocessed_input)

    sentiment = 'Positive' if predicted[0][0]>0.5 else 'Negative'

    return sentiment,predicted[0][0]




st.title("Movie review sentimation")

review = st.text_area('Please enter your review about movie')

if st.button('Classify'):
    sentiment,score = predict_sentiment(review)
    if(score > 0.5):
        st.write('Positive review \nProbability : ' ,score)
    else :
        st.write('Negative review \nProbability : ' , score)




