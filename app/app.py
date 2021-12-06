from fastai.text.all import *
from flask import Flask, request, jsonify
from model import preprocess_data, infer
import pandas as pd

# import spacy
app = Flask(__name__)

@app.route('/api/gen-lyrics', methods=['GET'])
def generate_lyrics():
    """
    Generates lyrics for a particular artist
    """
    artist = request.args.get('artist_name')
    starting_lyrics = request.args.get('starting_lyrics')
    words = request.args.get('words')
    sentences = request.args.get('sentences')
    temperature = request.args.get('temperature')

    if artist == "" or artist == None or starting_lyrics == None or starting_lyrics == "":
        return jsonify({"Bad Request: Either artist or starting_lyrics were none"}), 404
    # call the function to infer the artist and starting text
    try:
        pred = infer(artist, starting_lyrics, words=words, sentences=sentences, temperature=temperature)
        return jsonify({"Success": pred}), 200
    except:
        return jsonify({"Not Found: Artist was not found, please try again"}), 404


if __name__ == '__main__':
    # app.run()
    artist = input("Enter an artist to generate lyrics for: ")
    starting_lyrics = input("Enter starting lyrics: ")
    print(infer(artist, starting_lyrics))