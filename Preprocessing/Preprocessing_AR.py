import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import re

nltk.download("punkt")
nltk.download("punkt_tab")
nltk.download("stopwords")



class Preprocessing_AR:
  @staticmethod
  def process(sentance):
    sentance = Preprocessing_AR.remove_punctuation(sentance)
    sentance = Preprocessing_AR.normalizer(sentance)
    sentance = Preprocessing_AR.tokenizer(sentance)
    sentance = Preprocessing_AR.remove_stopwords(sentance)
    sentance = Preprocessing_AR.remove_deplicate(sentance)
    sentance = Preprocessing_AR.stemmer(sentance)
    
    return sentance

  @staticmethod
  def tokenizer(sentance):
    words = word_tokenize(sentance)
    return words

  @staticmethod
  def normalizer(sentance):
    sentance = re.sub(r"[إأٱآا]","ا",sentance)
    sentance = re.sub(r"ي","ى",sentance)
    sentance = re.sub(r"ء","ؤ",sentance)
    sentance = re.sub(r"ء","ئ",sentance)
    sentance = re.sub(r"[^ا-ي]"," ",sentance)
    sentance = re.sub(r"ة","ه",sentance)
    return sentance


  @staticmethod
  def remove_stopwords(sentance):
    
    stop_words = set(stopwords.words('arabic'))

    sentance = [word for word in sentance if word not in stop_words]
    return sentance
  
  @staticmethod
  def stemmer(sentance):
    stemmer = PorterStemmer()
    sentance = [stemmer.stem(word) for word in sentance]
    return sentance

  @staticmethod
  def remove_punctuation(sentance):
    return re.sub(r'[^ا-ي 0-9\s]',' ',sentance)

  @staticmethod
  def remove_deplicate(sentance):
    return list(set(sentance))