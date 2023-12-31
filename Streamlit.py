import streamlit as st

import warnings
warnings.filterwarnings("ignore")

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rcParams

import wget
import tarfile

from nltk.tokenize import WordPunctTokenizer
import gensim.downloader as api

from sklearn.utils import class_weight

import torch
from torch import nn

from sklearn.metrics import balanced_accuracy_score, confusion_matrix, ConfusionMatrixDisplay

from IPython import display
import matplotlib.pyplot as plt
#%matplotlib inline

def extract_tar():

    zip_file = 'aclImdb_v1.tar.gz'
    data_status = os.path.exists('./' + zip_file)
        
    if data_status == False:
        url = 'https://drive.google.com/uc?export=view&id=1Azkk7zzqxPSBOfGR99JHuldNy1-ZD865&confirm=t'
        wget.download(url, zip_file)
    
    folder_name = 'aclImdb'
    data_status = os.path.exists('./' + folder_name)
        
    if data_status == False:
        tar = tarfile.open(zip_file)
        tar.extractall()
        tar.close()

def load_data():

    folder_name = 'aclImdb'

    train_data = []
    test_data = []

    train_label = []
    test_label = []

    train_rating = []
    test_rating = []

    for folder in ['train', 'test']:
        path = './' + folder_name + '/' + folder + '/'
        for label in ['neg', 'pos']:
            current_path = path + label
            if not os.path.exists(current_path):
                continue
            for file in os.listdir(current_path):
                with open(current_path + '/' + file) as f:
                    try:
                        data = f.read()
                        if label == 'neg':
                            label = 0
                        if label == 'pos':
                            label = 1

                        rating = int(file.split('_')[1].split('.')[0]) - 1
                        for pref in ['data', 'label', 'rating']:
                            array_name = folder + '_' + pref
                            locals()[array_name].append(locals()[pref])
                    except:
                        pass

    train_label = np.array(train_label)
    test_label = np.array(test_label)

    return train_data, train_label, train_rating, test_data, test_label, test_rating

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

def calculate_weights(train, target_size):
    class_weights = class_weight.compute_class_weight(class_weight ='balanced', classes=np.unique(train), y=train)
    if target_size > 2:
        class_weights = list(class_weights[0:(target_size//2-1)]) + [0, 0] + list(class_weights[(target_size//2-1):])
    class_weights = torch.tensor(class_weights,dtype=torch.float)
    return class_weights

def create_model(input_len, target_size):
    
    model = nn.Sequential(
        nn.Linear(input_len, 1000),
        nn.ReLU(),
        nn.Linear(1000, target_size)
    )

    return model

def train_model(
    model,
    opt,
    loss_function,
    X_train,
    y_train,
    X_val,
    y_val,
    n_iterations=1000,
    batch_size=128,
    show_plots=True,
    eval_every=100
):
    train_loss_history = []
    train_acc_history = []
    val_loss_history = []
    val_acc_history = []

    local_train_loss_history = []
    local_train_acc_history = []

    X_train_torch = torch.Tensor(np.array(X_train))
    X_val_torch = torch.Tensor(np.array(X_val))

    y_train_torch = torch.Tensor(y_train).type(torch.LongTensor)
    y_val_torch = torch.Tensor(y_val).type(torch.LongTensor)
    
    for i in range(n_iterations):

        # sample batch_size random observations
        ix = np.random.randint(0, len(X_train_torch), batch_size)
        x_batch = X_train_torch[ix]
        y_batch = y_train_torch[ix]

        # predict log-probabilities or logits
        y_predicted = model(x_batch)
        
        # compute loss, just like before
        loss = loss_function(y_predicted, y_batch)

        # compute gradients
        loss.backward()

        # Adam step
        opt.step()

        # clear gradients
        opt.zero_grad()

        local_train_loss_history.append(loss.item())
        local_train_acc_history.append(
            balanced_accuracy_score(
                y_batch.to('cpu').detach().numpy(),
                y_predicted.to('cpu').detach().numpy().argmax(axis=1)
            )
        )

        if i % eval_every == 0:
            train_loss_history.append(np.mean(local_train_loss_history))
            train_acc_history.append(np.mean(local_train_acc_history))
            local_train_loss_history, local_train_acc_history = [], []

            predictions_val = model(X_val_torch)
            val_loss_history.append(loss_function(predictions_val, y_val_torch).to('cpu').detach().item())

            acc_score_val = balanced_accuracy_score(y_val_torch.cpu().numpy(), predictions_val.to('cpu').detach().numpy().argmax(axis=1))
            val_acc_history.append(acc_score_val)

    if show_plots:
        fig1 = plot_train_process(train_loss_history, val_loss_history, train_acc_history, val_acc_history)
        st.pyplot(fig1)

    return model

def plot_train_process(train_loss, val_loss, train_accuracy, val_accuracy, title_suffix=''):
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    axes[0].set_title(' '.join(['Loss', title_suffix]))
    axes[0].plot(train_loss, label='train')
    axes[0].plot(val_loss, label='validation')
    axes[0].legend()
    axes[0].grid()

    axes[1].set_title(' '.join(['Validation accuracy', title_suffix]))
    axes[1].plot(train_accuracy, label='train')
    axes[1].plot(val_accuracy, label='validation')
    axes[1].legend()
    axes[1].grid()

    return fig

def visualize_results(model, X_train, X_test, y_train, y_test, target_size):

    X_train = torch.Tensor(np.array(X_train))
    X_test = torch.Tensor(np.array(X_test))

    y_train = torch.Tensor(y_train).type(torch.LongTensor)
    y_test = torch.Tensor(y_test).type(torch.LongTensor)

    proba = model(X_test).detach().cpu().numpy().argmax(axis=1)

    fig_size = (target_size - 5) ** 2 / 3

    fig, ax = plt.subplots(figsize=(fig_size, fig_size))
    
    cm = confusion_matrix(y_test, proba, labels=np.linspace(0, target_size-1, num=target_size, dtype=int), normalize='true')
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.linspace(0, target_size-1, num=target_size, dtype=int))
    disp.plot(ax=ax, cmap='Blues')
    if target_size == 2:
        plt.xticks([0, 1], ['neg', 'pos'])
        plt.yticks([0, 1], ['neg', 'pos'])
    else:
        plt.xticks(np.linspace(0, target_size-1, num=target_size, dtype=int), np.linspace(1, target_size, num=target_size, dtype=int))
        plt.yticks(np.linspace(0, target_size-1, num=target_size, dtype=int), np.linspace(1, target_size, num=target_size, dtype=int))
    
    plt.title('Confusion matrix')

    return fig

def get_prediction(review, tokenizer, gensim_embedding_model, model_label, model_rating):

    review = [review]
    data = tokenize(tokenizer, review)
    data_emb = [text_to_average_embedding(text, tokenizer, gensim_embedding_model) for text in data]
    data_emb_torch = torch.Tensor(np.array(data_emb))
    
    res = model_label(data_emb_torch).detach().cpu().numpy().argmax(axis=1).item()
    if res == 1:
        label = 'pos'
    if res == 0:
        label = 'neg'

    rating = model_rating(data_emb_torch).detach().cpu().numpy().argmax(axis=1).item() + 1

    if label == 'pos' and rating <= 4 or label == 'neg' and rating >= 7:
        st.markdown("It's complicated to analyse this review")

    st.markdown('Review label is  ' + label)
    st.markdown('Review rating is ' + str(rating))


def main():

    st.title('Review classification')

    @st.cache_resource()
    def model_training():

        tokenizer = WordPunctTokenizer()

        st.header('Data loading')
    
        with st.spinner('Extracting data archive'):
            extract_tar()
            
        with st.spinner('Loading embedding model'):  
            gensim_embedding_model = api.load('glove-twitter-200')

        st.success('Success!')

        st.header('Data processing')
    
        with st.spinner('Data loading'):
            train_data, train_label, train_rating, test_data, test_label, test_rating = load_data()
    
        with st.spinner('Tokenizing'):
            texts_train = tokenize(tokenizer, train_data)
            texts_test = tokenize(tokenizer, test_data)
    
        with st.spinner('Creating embeddings'):
            X_train_emb = [text_to_average_embedding(text, tokenizer, gensim_embedding_model) for text in texts_train]
            X_test_emb = [text_to_average_embedding(text, tokenizer, gensim_embedding_model) for text in texts_test]
    
        st.success('Success!')
    
        st.header('Model for classifying reviews into positive and negative')
    
        with st.spinner('Creating and training model'):
            
            target_size = max(train_label) + 1
            class_weights = calculate_weights(train_label, target_size)
            model_label = create_model(len(X_train_emb[0]), target_size)
            
            loss_function = nn.CrossEntropyLoss(weight=class_weights)
            opt = torch.optim.Adam(model_label.parameters(), lr=1e-3)
            
            model_label = train_model(model_label, opt, loss_function, X_train_emb, train_label, X_test_emb, test_label, n_iterations=5000)
    
            torch.save(model_label, 'model_label.pth')
    
        st.success('Success!')
    
        fig2 = visualize_results(model_label, X_train_emb, X_test_emb, train_label, test_label, target_size=2)
        st.pyplot(fig2)
    
        st.header('Model for classifying reviews by rating')
    
        with st.spinner('Creating and training model'):
            
            target_size = max(train_rating) + 1
            class_weights = calculate_weights(train_rating, target_size)
            model_rating = create_model(len(X_train_emb[0]), target_size)
            
            loss_function = nn.CrossEntropyLoss(weight=class_weights)
            opt = torch.optim.Adam(model_rating.parameters(), lr=1e-3)
            
            model_rating = train_model(model_rating, opt, loss_function, X_train_emb, train_rating, X_test_emb, test_rating, n_iterations=5000)
    
            torch.save(model_rating, 'model_rating.pth')
    
        st.success('Success!')
    
        fig3 = visualize_results(model_rating, X_train_emb, X_test_emb, train_rating, test_rating, target_size=10)
        st.pyplot(fig3)

        return tokenizer, gensim_embedding_model, model_label, model_rating

    tokenizer, gensim_embedding_model, model_label, model_rating = model_training()

    st.header('Enter review to movie')
    with st.form(key='my_form'):
        review = st.text_input(label='')
        submit_button = st.form_submit_button(label='Submit')
    
        if submit_button:
            get_prediction(review, tokenizer, gensim_embedding_model, model_label, model_rating)
    

if __name__ == "__main__":
    main()
