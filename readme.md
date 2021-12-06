# Lyric Gen

Lyric gen uses deep learning to make new lyrics in the sytle of your favourite artist after being given some starting text.  

***

## Motivation

In order to apply knowledge in NLP as well as transfer learning, I attempted to make a program to generate lyrics. The idea is that this can be turned into a karaoke style game later where the generated lyrcis are used to have fun with friends.

***

## Tech/framework
**Built with**
- Fastai (Python, high level deep learning library built on top of pytorch)
- Pytorch (Python, deep learning)
- Flask (Python)
- Pandas (Python)


## Project Checkpoints
1. Perform data analysis on the song lyric dataset (10_song_data_analysis.ipynb)
2. Train a model in a jupyter notebook (example_notebooks folder)
3. Modularize the code to create models and generate lyrics based on these models (app folder; model.py)

***


## Screenshots
This is an example of a lyric generated in Drake style:
![alt text](https://github.com/Nick-palmar/deep_learning_lyric_generation/blob/main/images/drake_example.png "Drake Example Text")

## Next Steps
This project requires a large variety of models to be successful (different people would be interested in different artists). The next steps should be to train a number of models with data from different artists so that lyrics can be generated in the styles of multiple artists. Different model architectures could also be attempted since this version of the project is using an LSTM based architecture. Now, transformer based architectures are being favoured in the NLP field instead as they perform better than LSTM models on langage tasks. 

