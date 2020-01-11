# -*- coding: utf-8 -*-
"""
Created on Sat Dec 28 19:31:56 2019

@author: Friendly Snek Box
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 18:24:19 2019

@author: Companion Box
"""

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

# =============================================================================
# file = "C:/Users/Friendly_Snek_Box/Desktop/Anul 2 - Semestrul 1/Analiza Text/Proiect Pornhub/category_root_19_12.xlsx"
# file2 = "C:/Users/Friendly_Snek_Box/Desktop/Anul 2 - Semestrul 1/Analiza Text/Proiect Pornhub/python_test.xlsx"
# 
# 
# =============================================================================


df_csv = pd.read_csv("C:/Users/Friendly_Snek_Box/Desktop/Anul 2 - Semestrul 1/Analiza Text/Proiect Pornhub/category_root_19_12.csv") 


# df = pd.read_excel(file, index_col=0,
#              dtype={'web-scraper-order': str, 'web-scraper-start-url': str, 'category_link': str,
#        'category_link-href': str, 'video_link': str, 'video_link-href': str, 'view_count': float,
#        'likes': float, 'dislikes': float, 'comment': str, 'user': str, 'upvotes': float}) 
# 
# df2 = pd.read_excel(file2, index_col=0,
#              dtype={'web-scraper-order': str, 'video_link': str, 'video_link-href': str, 'view_count': float,
#        'likes': float, 'dislikes': float, 'comment': str, 'user': str, 'upvotes': float})
# =============================================================================

### Make everything lowercase NNN -> nnn
df_csv['comment']  = df_csv['comment'].str.lower()



df_csv['comment'] = df_csv['comment'].astype(str)

#=============================================================================
#  :)) Emojis & Emoticons <3  -  not working yet
# =============================================================================
import re
import sys

# =============================================================================
# pip install emot
# pip search emot 

# emot.download('all')
# import emot
# 
# from emot.emo_unicode import UNICODE_EMO, EMOTICONS
# # Converting emojis to words
# def convert_emojis(text):
#     for emot in UNICODE_EMO:
#         text = text.replace(emot, "_".join(UNICODE_EMO[emot].replace(",","").replace(":","").split()))
#         return text
#     
# # Converting emoticons to words    
# def convert_emoticons(text):
#     for emot in EMOTICONS:
#         text = re.sub(u'('+emot+')', "_".join(EMOTICONS[emot].replace(",","").split()), text)
#         return text
# 
# df_csv['comments_emoticon'] = df_csv['comment'].apply(lambda x: convert_emojis(x))
# df_csv['comments_emoticon_emoji'] = df_csv['comments_emoticon'].apply(lambda x: convert_emoticons(x))
# df_csv.head(10)
# 
# =============================================================================

# =============================================================================
pip install demoji
# =============================================================================

import demoji
demoji.download_codes()


def replaceEmoticons(comment):
    dict_emotion = demoji.findall(comment)
    for key in dict_emotion:
        comment = comment.replace(key," "+dict_emotion[key]+" ")
    return comment;

df_csv['comments_emoticon'] = df_csv['comment'].apply(lambda x: replaceEmoticons(x))

# =============================================================================
# #Good one! Majestic lambda! Trying to remove non-normal characters & Non-English 
# =============================================================================

df_csv['comments_emoticon'] = df_csv['comments_emoticon'].astype(str)


from nltk.tokenize import RegexpTokenizer
tokenizer = RegexpTokenizer(r'[a-zA-Z]+')

df_csv['comments_token'] = df_csv['comments_emoticon'].apply(lambda x: tokenizer.tokenize(x))


# =============================================================================
# remove non-english
#i need to do stemming first because it removes words with plural
# =============================================================================
# =============================================================================
# from nltk.stem import WordNetLemmatizer,PorterStemmer
# from nltk.corpus import stopwords
# import re
# stemmer = PorterStemmer() 
# 
# p = nltk.PorterStemmer()
# def stem(a):
#     a = [p.stem(word) for word in a]
#     return a
# 
# df_csv['comments_stemm'] = df_csv['comment'].apply(lambda x: stem(x))
# 
# =============================================================================

from nltk.corpus import words
words = set(nltk.corpus.words.words())

def remove_noneng(text):  
    text = [w for w in nltk.wordpunct_tokenize(text) if w.lower() in words or not w.isalpha()]
    return text


df_csv['comments_emoticon'] = df_csv['comments_emoticon'].astype(str)    
df_csv['comments_nonenglish'] = df_csv['comments_emoticon'].apply(lambda x: remove_noneng(x))





# =============================================================================
# Lemmatization -  not impressed i would say
# =============================================================================

from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer() 

df_csv['comments_lemm'] = df_csv['comments_emoticon'].apply(lambda x: lemmatizer.lemmatize(x)) 



print(lemmatizer.lemmatize("cars"))
#> are




# =============================================================================
# Spacy lemmatization
# =============================================================================

# Install spaCy (run in terminal/prompt)
import sys
!{sys.executable} -m pip install spacy

# Download spaCy's  'en' Model
!{sys.executable} -m spacy download en










# =============================================================================
# Trying to remove non-normal characters & Non-English 
# =============================================================================
# =============================================================================
# 
# df_csv['tokenized_sentences'] = df_csv.comment.apply(lambda x: word_tokenize(x))
# #saving the tokens without punctuation.
# from nltk.tokenize import RegexpTokenizer
# tokenizer = RegexpTokenizer(r'[a-zA-Z]+')
# 
# df_csv['tokenized_sentences_2'] = df_csv.comment.apply(lambda x: tokenizer.tokenize(x))
# 
# df_csv['tokenized_sentences'] = df_csv['tokenized_sentences'].astype(str)
# 
# 
# =============================================================================
# 

# =============================================================================
# Remove stopwords - it works now!!
# =============================================================================
import nltk
stopword = nltk.corpus.stopwords.words('english')

def remove_stopwords(text):
    text = [word for word in text if word not in stopword]
    return text
    
df_csv['tokenized_sentences_nonstop'] = df_csv['tokenized_sentences'].apply(lambda x: remove_stopwords(x))
df_csv.head(10)


# =============================================================================
# preprocess all wrapped up
# =============================================================================


import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer,PorterStemmer
from nltk.corpus import stopwords
import re
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer() 

def preprocess(sentence):
    sentence=str(sentence)
    sentence = sentence.lower()
    sentence=sentence.replace('{html}',"") 
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, '', sentence)
    rem_url=re.sub(r'http\S+', '',cleantext)
    rem_num = re.sub('[0-9]+', '', rem_url)
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(rem_num)  
    filtered_words = [w for w in tokens if len(w) > 2 if not w in stopwords.words('english')]
    stem_words=[stemmer.stem(w) for w in filtered_words]
    lemma_words=[lemmatizer.lemmatize(w) for w in stem_words]
    return " ".join(filtered_words)


df_csv['comments_processed'] = df_csv['comments_emotion_replaced'].apply(lambda x: preprocess(x)) 





###Implement lemmatization on your corpus.
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer() 
print(lemmatizer.lemmatize(df_csv.tokenized_sentences_2[1]))

counter = 0
lemm_tokens = []
for i in tokens_w:
    lemm_tokens.append(lemmatizer.lemmatize(i))
print(lemm_tokens)
 
#removing stop words from the list
from nltk.corpus import stopwords
stopWords = set(stopwords.words('english'))

wordsFiltered = []
for w in lemm_tokens:
    if w not in stopWords:
        wordsFiltered.append(w)
print(wordsFiltered)










df2.comment.dtype
df['comment'].apply(nltk.word_tokenize)
df['tokenized_sentences'] = df['comment'].apply(nltk.word_tokenize)


df['comment'] = df.apply(lambda row: nltk.word_tokenize(row['comment']), axis=1)



# We can tokenize a sentence using a tokenizer or splitting by space simply.
df['tokenized_sentences_nltk'] = df['comment'].apply(nltk.word_tokenize) # it take some times
df['tokenized_sentences_naive'] = df['comment'].apply(lambda s: s.split(' '))


















