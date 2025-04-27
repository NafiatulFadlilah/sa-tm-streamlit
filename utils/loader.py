import pandas as pd
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

def load_alay_dictionary(kamusalay_filepath, additional_filepath=None):
    # Load kamus utama
    alay_df = pd.read_csv(kamusalay_filepath, header=None, names=['alay', 'formal'], encoding='ISO-8859-1')
    alay_dict = dict(zip(alay_df['alay'], alay_df['formal']))
    
    # Kalau ada file tambahan, load juga
    if additional_filepath is not None:
        additional_df = pd.read_csv(additional_filepath, header=None, names= ['alay', 'formal'], encoding='ISO-8859-1')
        additional_dict = dict(zip(additional_df['alay'], additional_df['formal']))
        
        # Gabungkan dictionary utama + tambahan
        alay_dict.update(additional_dict)
    
    return alay_dict

def load_stopwords(stopwords_filepath, additional_stopwords=[], excluded_stopwords=[]):
    factory = StopWordRemoverFactory()
    default_stopwords = factory.get_stop_words()
    
    custom_stopwords = pd.read_csv(stopwords_filepath, header=None)[0].tolist()
    
    # Gabungkan semua stopwords
    stopwords = set(default_stopwords + custom_stopwords + additional_stopwords)
    
    # Keluarkan kata yang dikecualikan
    stopwords = stopwords.difference(set(excluded_stopwords))
    
    return stopwords
