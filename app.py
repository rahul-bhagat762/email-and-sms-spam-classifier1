import pickle
import string
import streamlit as st
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer


ps = PorterStemmer()

def trasform_text(text):
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

#tfidf = pickle.load(('vectorizer.pkl','rb'))
with open("vectorizer.pkl","rb") as f:
    tfidf = pickle.load(f)
#model = pickle.load(open('model.pkl','rb'))
with open("model2.pkl","rb") as f:
    model = pickle.load(f)

st.title("Email/sms spam Classifier")
input_sms = st.text_area("Enter The message/email")
if st.button('predict'):


    #1. preprocess
    trasformed_sms = trasform_text(input_sms)
    #2. Vectorize
    vector_input = tfidf.transform([trasformed_sms])
    #3. predict
    result = model.predict(vector_input)[0]
    #4. Display
    if result == 1:
        st.header('spam')
    else:
        st.header('Not spam')
