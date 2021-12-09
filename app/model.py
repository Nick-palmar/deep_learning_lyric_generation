
from fastai.text.all import *
import pandas as pd
import torch

mode = 'infer'
seed = 8
# boolean if you want to wait for training on a cpu
want_to_wait = False
gpu = torch.cuda.is_available()

def preprocess_data(artist_name):
    # read and combine the data
    lyrics_data = pd.read_csv('../data/lyrics-data.csv')
    lyrics_data.rename(columns={"ALink": "Link"}, inplace=True)
    artist_data = pd.read_csv('../data/artists-data.csv')
    merged_dfs = lyrics_data.merge(artist_data, how='inner', on='Link')
    eng_artists = merged_dfs.loc[merged_dfs['Idiom'] == 'ENGLISH', ['Artist', 'SName', 'Lyric', 'Genre']].drop_duplicates(subset=['SName'])
    eng_artists.reset_index(inplace=True, drop=True)
    
    # choose the specific artist
    artist_df = eng_artists.loc[eng_artists['Artist'] == artist_name].reset_index(drop=True)
    lyric_block = DataBlock(
        blocks=TextBlock.from_df('artist_df', seq_len=72, is_lm=True),
        get_items=ColReader('Lyric'),
        splitter=RandomSplitter(seed=seed)
    )

    # create data loader
    dls_lm = lyric_block.dataloaders(artist_df, bs=128, seq_len=80)

    return dls_lm


def save_model(learner, new_fname, path='../models'):
    checkpoint = {
        "state_dict": learner.model.state_dict()
    }
    try:
        torch.save(checkpoint, path + '/' + new_fname + '.pth')
        return 'success'
    except:
        return 'error saving'


def train_model(artist_name, model=AWD_LSTM):
    dls = preprocess_data(artist_name)
    # load the model
    learn = language_model_learner(dls, model, drop_mult=0.3, metrics=accuracy)
    
    # transfer learning on the model
    if gpu or want_to_wait:
        if gpu:
            learn.model = learn.model.cuda()
        learn.fit_one_cycle(3, 1e-3)
        learn.freeze_to(-2)
        learn.fit_one_cycle(2, 1e-3)
        learn.freeze_to(-3)
        learn.fit_one_cycle(2, 1e-3)
        learn.freeze_to(-4)
        learn.fit_one_cycle(2, 1e-3)
        learn.freeze_to(-5)
        learn.fit_one_cycle(2, 1e-3)
        learn.unfreeze()
        print("Unfreezing model")
        learn.fit_one_cycle(6, lr_max=slice(1e-5, 1e-3))

    else:
        learn.fit_one_cycle(1, 1e-2)
    
    # save the trained model
    save_model(learn, artist_name + '_model')

    # return the learner
    return learn


def get_model(artist_name, model=AWD_LSTM, folder='../models'):
    fname = artist_name + '_model'
    dls = preprocess_data(artist_name)

    # load the model
    learn = language_model_learner(dls, model, drop_mult=0.3, metrics=accuracy)
    checkpoint = torch.load(folder + '/' + fname + '.pth', map_location=torch.device('cpu'))
    learn.model.load_state_dict(checkpoint['state_dict'])
    if gpu:
        learn.model = learn.model.cuda()
    else:
        learn.model = learn.model.cpu()

    return learn

def get_most_complex(start_text, preds):
  max_len = 0
  max_i = -1
  for i, pred in enumerate(preds):
    pred_cardinality = len(set(pred.split()))
    if pred_cardinality > max_len:
      max_len = pred_cardinality
      max_i = i
  
  return_str = preds[max_i]

  val = -1
  occurrence = len(start_text.split())
  for i in range(0, occurrence):
    val = return_str.find(' ', val + 1)

  return start_text + return_str[val:return_str.rfind('.')+1]

def infer(artist_name, start_text, words=50, sentences=3, temperature=0.75):
    if words == None:
        words = 50
    if sentences == None:
        sentences = 3
    if temperature == None:
        temperature = 0.75
    learn = get_model(artist_name)
    preds = [learn.predict(start_text, words, temperature=temperature)
            for sentence in range(sentences)]

    pred = get_most_complex(start_text, preds)
    return pred


if __name__ == '__main__':
    artists = []
   
    if mode == 'train':
        for artist in artists:
            print(artist)
            learn = train_model(artist)
    
    elif mode == 'infer':
        pred = infer("Eminem", "Hello world", )
        print(pred)
    