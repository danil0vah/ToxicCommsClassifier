import nltk
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer

nltk.download('punkt')
nltk.download('stopwords')

snowball = SnowballStemmer('russian')
stop_words = set(stopwords.words('russian'))

def tokenize(text: str, remove_stopwords: bool =True) -> list:
    tokens = word_tokenize(text, language='russian')
    tokens = [t for t in tokens if t not in string.punctuation]
    if remove_stopwords:
        tokens = [t for t in tokens if t not in stop_words]
    
    return [snowball.stem(t) for t in tokens]