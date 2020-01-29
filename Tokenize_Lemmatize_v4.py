# =============================================================================
# This is for the tweets!
# =============================================================================

import nltk
nltk.download('all')


import nltk
nltk.download('popular')



### Import excel ###
import numpy as np
import pandas as pd
import re
import nltk
import string
from nltk import word_tokenize

#df = pd.read_excel ('C:/Users/Friendly_Snek_Box/Desktop/Anul 2 - Semestrul 1/Analiza Text/Proiect Pornhub/all_comments.xlsx') 
# =============================================================================
df_csv = pd.read_csv("C:/Users/Friendly_Snek_Box/Desktop/Anul 2 - Semestrul 1/Analiza Text/Proiect Pornhub/twt_training.csv") 
#df_csv = df[['tweet']]



### Make everything lowercase NNN -> nnn
df_csv['comment']  = df_csv['tweet'].str.lower()
df_csv['comment'] = df_csv['comment'].astype(str)



#=============================================================================
#  :)) Emojis & Emoticons <3  -  not working yet
#pip install demoji
# =============================================================================
import re
import sys
import demoji
demoji.download_codes()

def replaceEmoticons(comment):
    dict_emotion = demoji.findall(comment)
    for key in dict_emotion:
        comment = comment.replace(key," "+dict_emotion[key]+" ")
    return comment;

df_csv['comments_emoticon'] = df_csv['comment'].apply(lambda x: replaceEmoticons(x))
df_csv['comments_emoticon'] = df_csv['comments_emoticon'].astype(str)




# =============================================================================
# Remove html
# =============================================================================
def remove_html_from_comment(comment):
    comment = comment.replace('{html}', "")
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, '', comment)
    rem_url = re.sub(r'http\S+', '', cleantext)
    rem_num = re.sub('[0-9]+', '', rem_url)
    return rem_num

df_csv['comments_html'] = df_csv['comments_emoticon'].apply(lambda x: remove_html_from_comment(x))



# =============================================================================
# =============================================================================
# #Good one! Majestic lambda! Trying to remove non-normal characters & Non-English 
#Tokenize
# =============================================================================
from nltk.tokenize import RegexpTokenizer

def clean_non_english_characters(df_csv, tokenizer =  RegexpTokenizer(r'[a-zA-Z]{2,}')):
    #df_csv['comments_lemma'] = df_csv['comments_lemma'].astype(str)
    df_csv['comments_noneng'] = df_csv['comments_html'].apply(lambda x: tokenizer.tokenize(x))
    return df_csv
 
clean_non_english_characters(df_csv)



# =============================================================================
# Detokenize
# =============================================================================
from nltk.tokenize.treebank import TreebankWordDetokenizer
#TreebankWordDetokenizer().detokenize(['the', 'quick', 'brown'])
df_csv['comments_noneng'] = df_csv['comments_noneng'].apply(lambda x: TreebankWordDetokenizer().detokenize(x))




# =============================================================================
# Stemming - e cam eronat, nu prea scoate pluralul corect
#stopwords - scoate cam multe
# =============================================================================


# =============================================================================
# Marius Tag POS
# =============================================================================

from nltk.corpus import wordnet as wn
from nltk.stem.wordnet import WordNetLemmatizer
from nltk import word_tokenize, pos_tag
from collections import defaultdict

tag_map = defaultdict(lambda : wn.NOUN)
tag_map['J'] = wn.ADJ
tag_map['V'] = wn.VERB
tag_map['R'] = wn.ADV

def lemma_pos(text):
    tokens = word_tokenize(text)
    lmtzr = WordNetLemmatizer()
    lemma = []    
    for token, tag in pos_tag(tokens):
        lemma.append( lmtzr.lemmatize(token, tag_map[tag[0]]))    
    return lemma

df_csv['comments_lemma'] = df_csv['comments_noneng'].apply(lambda x: lemma_pos(x))

# =============================================================================
# df_csv['comments_noneng'] = df_csv['comments_noneng'].astype(str)
# df_csv['comments_lemma'] = df_csv['comments_noneng'].apply(lambda x: lemma_pos(x))
# =============================================================================








from nltk.corpus import words
"Christmas" in words.words()

# =============================================================================
# remove non-english
# =============================================================================
from nltk.corpus import words
words = set(nltk.corpus.words.words())

def remove_noneng(text):  
    text = [w for w in text if w.lower() in words or not w.isalpha()]
    return text

df_csv['comments_clean'] = df_csv['comments_lemma'].apply(lambda x: remove_noneng(x))


# =============================================================================
# Remove stopwords
# =============================================================================
import nltk
from nltk.corpus import stopwords
result = set(stopwords.words('english'))
print("List of stopwords in English:")
print(result)


import nltk
from nltk.corpus import stopwords
stopword = set(stopwords.words('english')) - set(['they', 'she', 'above', 'more', 'me', 'now', 'my', 'any', 'down', 'few', 'over', 'with', 'him', 'ours', 'that', 'other', 'does', 'through', 'off', 'just', 'such', 'was', 'have', 'for', 'about', 'his', 'only', 'too', 'each', 'because', 'under', 'those', 'isn', 'no', 'when', 'we', 'all', 'why', 'same', 'against', 'what', 'her', 'if', 'do', 'into', 'most', 'has', 'your', 'you', 'not', 'this', 'once', 'below', 'our', 'after', 'where', 'he', 'some', 'again', 'yours', 'between', 'there', 'from', 'by', 'then', 'them', 'further', 'i', 'himself', 'had', 'were', 'very', 'during', 'be', 'but', 'before', 'myself', 'here', 'how', 'than', 'hers', 'out', 'until', 'which', 'up', 'will'])

 
def remove_stopwords(text):
    text = [word for word in text if word not in stopword]
    return text
     
df_csv['comments_clean2'] = df_csv['comments_clean'].apply(lambda x: remove_stopwords(x))
df_csv.head(10)





# =============================================================================
# Detokenize
# =============================================================================
from nltk.tokenize.treebank import TreebankWordDetokenizer
#TreebankWordDetokenizer().detokenize(['the', 'quick', 'brown'])
df_csv['comments_clean'] = df_csv['comments_clean'].apply(lambda x: TreebankWordDetokenizer().detokenize(x))
df_csv['comments_clean2'] = df_csv['comments_clean2'].apply(lambda x: TreebankWordDetokenizer().detokenize(x))




# =============================================================================
# General sentiment with text blob
#pip install -U textblob
#python -m textblob.download_corpora
# =============================================================================
from textblob import TextBlob
df_csv['comment'] = df_csv['comment'].astype(str)
df_csv['sentiment'] = df_csv.comment.map(lambda text: TextBlob(text).sentiment.polarity)

df_csv['comments_clean'] = df_csv['comments_clean'].astype(str)
df_csv['sentiment_clean'] = df_csv.comments_clean.map(lambda text: TextBlob(text).sentiment.polarity)




# =============================================================================
col_list = ["Numb", "count", "hate_speech", "offensive_language", "neither", "class", "tweet", "comments_clean", "comments_clean2", "sentiment", "sentiment_clean"]
df_csv = df_csv[col_list]
# =============================================================================

#df_csv = df_csv[df_csv.comments_clean.notnull()]
#df = df_csv[df_csv.comment != '[]']

#####saving some of the results
from pandas import DataFrame
import pandas as pd
from pandas import DataFrame

df_csv.to_excel(r'C:/Users/Friendly_Snek_Box/Desktop/Anul 2 - Semestrul 1/Analiza Text/Proiect Pornhub/twt_training_clean.xlsx')





