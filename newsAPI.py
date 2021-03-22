!pip install spacy
!pip install newsapi-python
!python -m spacy download en_core_web_lg

import spacy
import en_core_web_lg
from newsapi import NewsApiClient
import pickle
import pandas as pd
import string
import numpy as np
from collections import Counter 
import nltk
from nltk.tokenize import RegexpTokenizer
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from wordcloud import WordCloud 
from textblob import TextBlob
nltk.download('stopwords')
nltk.download('brown')
nltk.download('punkt')

nlp_eng = en_core_web_lg.load()
newsapi = NewsApiClient(api_key='953f24f39c0b4b4bb665181e6ff83d05') 
temp = newsapi.get_everything(q='coronavirus', language='en', from_param='2021-02-22', to='2021-03-21', sort_by='relevancy')

filename = 'articlesCOVID.pckl'
pickle.dump(temp, open(filename, 'wb'))
filename = 'articlesCOVID.pckl'
loaded_model = pickle.load(open(filename, 'rb'))
filepath = '/content/articlesCOVID.pckl'
pickle.dump(loaded_model, open(filepath, 'wb'))
df = pd.DataFrame(temp['articles'])
articles = temp['articles']
dados = list() 
for i, article in enumerate(articles):
  title = article['title']
  description = article['description']
  content = article['content']
  date = article['publishedAt']
  dados.append({'title':title, 'date':date, 'desc':description, 'content':content})
df = pd.DataFrame(dados)
df = df.dropna()
df.head()

pos_tag = {'VERB', 'NOUN', 'PROPN'}

def get_keywords_eng(text):
  result = []
  for token in nlp_eng(text):

    if (token.text in nlp_eng.Defaults.stop_words):
      continue
    if (token.pos_ in pos_tag):
      result.append(token.text)

  return result

for content in df.content.values:
  results = list()
    
  results.append([x[0] for x in Counter(get_keywords_eng(content)).most_common(5)])

text = str(results)
wordcloud = WordCloud(max_font_size=50, max_words=100, background_color="white").generate(text)
plt.figure()
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()
