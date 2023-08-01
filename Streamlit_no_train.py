import streamlit as st

import os
import numpy as np
import wget
import tarfile

from nltk.tokenize import WordPunctTokenizer
import gensim.downloader as api

import torch

@st.cache_resource
def load_data():

    tokenizer = WordPunctTokenizer()

    with st.spinner('Loading embedding model'):
        gensim_embedding_model = api.load('glove-twitter-200')

    with st.spinner('Loading models'):
        
        model_label = 'model_label.pth'
        url = 'https://drive.google.com/uc?export=view&id=1OmxJYLqsfa6fU8IuHzdN_Z53BgolA6NW&confirm=t'
        wget.download(url, model_label)
        model_label = torch.load(model_label)

        model_rating = 'model_rating.pth'
        url = 'https://drive.google.com/uc?export=view&id=1PHGT0NY6iDxbkEW4gIRawdN-XfHR1lhs&confirm=t'
        wget.download(url, model_rating)
        model_rating = torch.load(model_rating)

    st.success('Success!')
        
    return tokenizer, gensim_embedding_model, model_label, model_rating

def tokenize(tokenizer, texts):

    preprocess = lambda text: ' '.join(tokenizer.tokenize(text.lower()))
    texts = [preprocess(text) for text in texts]

    return texts

def text_to_average_embedding(text, tokenizer, gensim_embedding_model):

    embedding_for_text = np.zeros([gensim_embedding_model.vector_size], dtype='float32')
    phrase_tokenized = tokenizer.tokenize(text.lower())
    phrase_vectors = [gensim_embedding_model[x] for x in phrase_tokenized if
    gensim_embedding_model.has_index_for(x)]

    if len(phrase_vectors) != 0:
        embedding_for_text = np.mean(phrase_vectors, axis=0)

    return embedding_for_text

def get_prediction(rewiew, tokenizer, gensim_embedding_model, model_label, model_rating):

    rewiew = [rewiew]
    data = tokenize(tokenizer, rewiew)
    data_emb = [text_to_average_embedding(text, tokenizer, gensim_embedding_model) for text in data]
    data_emb_torch = torch.Tensor(np.array(data_emb))
    
    res = model_label(data_emb_torch).detach().cpu().numpy().argmax(axis=1).item()
    if res == 1:
        label = 'pos'
    if res == 0:
        label = 'neg'

    rating = model_rating(data_emb_torch).detach().cpu().numpy().argmax(axis=1).item() + 1

    if label == 'pos' and rating <= 4 or label == 'neg' and rating >= 7:
        st.markdown("It's complicated to analyse this rewiew")

    st.markdown('Rewiew label is  ' + label)
    st.markdown('Rewiew rating is ' + str(rating))

def main():

    st.header('Loading data')

    tokenizer, gensim_embedding_model, model_label, model_rating = load_data()
        
    st.header('Enter rewiew to movie')
    with st.form(key='my_form'):
        rewiew = st.text_input(label='')
        submit_button = st.form_submit_button(label='Submit')
    
        if submit_button:
            get_prediction(rewiew, tokenizer, gensim_embedding_model, model_label, model_rating)
    

if __name__ == "__main__":
    main()
