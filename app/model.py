from fastai.text.all import *
import pandas as pd

def preprocess_data(artist_name):
    # read and combine the data
    lyrics_data = pd.read_csv('data/lyrics-data.csv')
    lyrics_data.rename(columns={"ALink": "Link"}, inplace=True)
    artist_data = pd.read_csv('data/artists-data.csv')
    merged_dfs = lyrics_data.merge(artist_data, how='inner', on='Link')
    eng_artists = merged_dfs.loc[merged_dfs['Idiom'] == 'ENGLISH', ['Artist', 'SName', 'Lyric', 'Genre']].drop_duplicates(subset=['SName'])
    eng_artists.reset_index(inplace=True, drop=True)
    
    # choose the specific artist
    artist_df = eng_artists.loc[eng_artists['Artist'] == artist_name].reset_index(drop=True)
    lyric_block = DataBlock(
        blocks=TextBlock.from_df('artist_df', seq_len=72, is_lm=True),
        get_items=ColReader('Lyric')
    )

    # create data loader
    dls_lm = lyric_block.dataloaders(artist_df, bs=128, seq_len=80)

    return dls_lm



def train_model(artist_name, model=AWD_LSTM):

    dls = preprocess_data(artist_name)

    # load the model
    learn = language_model_learner(dls, model, drop_mult=0.3, metrics=accuracy)
    
    # transfer learning on the model
    print(learn.fit_one_cycle(4, 0.005))
    learn.unfreeze()
    print("Unfreezing model")
    print(learn.fit_one_cycle(20, lr_max=slice(1e-5, 1e-3)))
    
    # save the trained model
    learn.save(artist_name + '_model')

    # return the learner
    return learn


def get_model(artist_name, model=AWD_LSTM):
    dls = preprocess_data(artist_name)

    # load the model
    learn = language_model_learner(dls, model, drop_mult=0.3, metrics=accuracy)
    learn = learn.load(artist_name + "_model")

    return learn


# def predict()


if __name__ == '__main__':
    # learn = train_model('Drake')
    learn = get_model("Drake")

    start_text = "When a good thing goes bad"
    words = 100
    sentences = 3
    preds = [learn.predict(start_text, words, temperature=0.75)
            for sentence in range(sentences)]
    print("Preds", preds)