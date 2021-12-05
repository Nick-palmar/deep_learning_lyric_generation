from fastai.text.all import *
from flask import Flask
from model import preprocess_data
import pandas as pd
# import spacy
app = Flask(__name__)


def generate_lyrics(artist,starting_text):
    """
    generates lyrics for a particular artist
    """

    #create model
    dls = preprocess_data(artist)

    # load the model
    learn = language_model_learner(dls, AWD_LSTM, drop_mult=0.3, metrics=accuracy)

    if artist == "drake":
        learn = learn.load("Drake_model")

    elif artist == "beach boys":
        predicter = load_learner("models/beach_boys_model.pkl", cpu=True, pickle_module=pickle)

    elif artist == "red hot chili peppers":
        predicter = load_learner("models/red_hot_chili_peppers_model.pkl", cpu=True, pickle_module=pickle)

    #generate lyrics in a string 
    words = 100
    sentences = 5
    preds = [learn.predict(starting_text, words, temperature=0.75)
            for sentence in range(sentences)]

    
    


    
    #join the list into a string
    preds = "\n".join(preds)

    return preds   


if __name__ == '__main__':
    # app.run()
    print(generate_lyrics("Drake","jumped out of bed"))