import streamlit as st
import pickle
import nltk
import string
import sklearn
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps=PorterStemmer()
from PIL import Image

image = Image.open("D:\\4202011_email_gmail_mail_logo_social_icon (2).png")

st.image(image, caption='EMAIL/SMS')

#1. Preprocssing
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

tf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('vmodel.pkl', 'rb'))

st.title("Email/SMS Spam Detection")

input_msg = st.text_area("Enter your message here")

option = st.selectbox("You got message from :", ["Via email", "Via SMS", "Via Whatsapp", "Other"])

if st.button('ClickMe'):

   transformed_sms = transform_text(input_msg)

   #2. Vectorization
   vector_input = tf.transform([transformed_sms])

   #3. Prediction
   result = model.predict(vector_input)[0]

   #4. Display
   if result == 1:
      st.header("ALERT!! SPAM MESSAGE")
   else:
      st.header("GO ON!! NOT SPAM")