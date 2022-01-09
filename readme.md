# Lyric Gen

Lyric gen uses deep learning to make new lyrics in the sytle of your favourite artist after being given some starting text.  

**Please scroll down to the *screenshots* section to see examples of the models in action.**

***

## Motivation

In order to apply knowledge in NLP as well as transfer learning, I attempted to make a program to generate lyrics. The idea is that this can be turned into a karaoke style game later where the generated lyrics are used to have fun with friends.

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
4. Train varioius models (**25+ models**) on CUDA compatible GPUs and test the results

***


## Screenshots
1. The Beatles Style
2. Drake Style
3. Eminem Style
4. Madonna Style
5. Michael Jackson Style
6. Pitbull Style
7. Taylor Swift Style

### The Beatles Style
![The Beatles Example Text](https://github.com/Nick-palmar/deep_learning_lyric_generation/blob/main/images/beatles_example.png "The Beatles Example Text")

### Drake Style
![Drake Example Text](https://github.com/Nick-palmar/deep_learning_lyric_generation/blob/main/images/drake_example.png "Drake Example Text")

### Eminem Style
![Enimem Example Text](https://github.com/Nick-palmar/deep_learning_lyric_generation/blob/main/images/eminem_example.png "Enimem Example Text")

### Madonna Style
![Madonna Example Text](https://github.com/Nick-palmar/deep_learning_lyric_generation/blob/main/images/madonna_example.png "Madonna Example Text")

### Michael Jackson Style
![Michael Jackson Example Text](https://github.com/Nick-palmar/deep_learning_lyric_generation/blob/main/images/mj_example.png "Michael Jackson Example Text")

### Pitbull Style
![Pitbull Example Text](https://github.com/Nick-palmar/deep_learning_lyric_generation/blob/main/images/pitbull_example.png "Pitbull Example Text")

### Taylor Swift Style
![Taylor Swift Example Text](https://github.com/Nick-palmar/deep_learning_lyric_generation/blob/main/images/taylor_swift_example.png "Taylor Swift Example Text")



## Next Steps
This project requires a large variety of models to be successful (different people would be interested in different artists). The next steps should be to train more models with data from different artists so that lyrics can be generated in the styles of multiple artists. Different model architectures could also be attempted since this version of the project is using an LSTM based architecture. Now, transformer based architectures are being favoured in the NLP field instead as they perform better than LSTM models on langage tasks. Finally, a different dataset with other song lyrics from the same artists could be used to enhance the transfer learning and improve results. 

