import streamlit as st
import streamlit as st
import tensorflow as tf
import keras 
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.utils import pad_sequences

mod1 = keras.models.load_model("models/sumn.keras")
mod2 = keras.models.load_model("models/whoops.keras")


def getpercent(sent):
    max_words = 10000
    max_len = 300
    tok = Tokenizer(num_words=max_words)
    tok.fit_on_texts([sent])
    sequences = tok.texts_to_sequences([sent])
    sequences_matrix = pad_sequences(sequences,maxlen = max_len)
    if mod1.predict(sequences_matrix)[0][0] > mod2.predict(sequences_matrix)[0][0]:
        a = mod1.predict(sequences_matrix)[0][0]
        if a>0.30 and a<0.49:
            return "Ehh, don't think so"
        elif a>=0.491:
            return "Yup, seems like it"
        else:
            return "Did you even type anything funny?"
    else:
        a = mod2.predict(sequences_matrix)[0][0]
        if a>0.30 and a<0.49:
            return "Ehh, don't think so"
        elif a>=0.491:
            return "Yup, seems like it"
        else:
            return "Did you even type anything funny?"

with open("styles.css") as css:
    st.markdown( f'<style>{css.read()}</style>' , unsafe_allow_html= True)
st.header("Sarcasm Detection using Tensorflow :fire:")

inputcol,outputcol = st.columns(2,gap="large")

with inputcol:
    st.markdown("### Enter your sentence")
    sent = st.text_area("You aint gon see this",label_visibility="hidden")
    a = st.button("Calculate")

with outputcol:
    st.markdown("### Is it sarcastic?")
    if a:
        st.write(getpercent(sent))
    else:
        pass
