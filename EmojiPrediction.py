# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 11:33:12 2020

@author: Friendly Snek Box
"""


pip install -U scikit-learn
conda install seaborn
conda install matplotlib
conda install sklearn


import sklearn
import seaborn
import matplotlib 



def read_file(file_name): 
    data_list  = []
    with open(file_name, 'r') as f: 
        for line in f: 
            line = line.strip() 
            label = ' '.join(line[1:line.find("]")].strip().split())
            text = line[line.find("]")+1:].strip()
            data_list.append([label, text])
    return data_list


# file_name = "psych.txt"
psych = "C:\\Users\\Friendly_Snek_Box\\Desktop\\Anul 2 - Semestrul 1\\Analiza Text\\Proiect Pornhub\\DeepEmoji\\psych.txt"
psychExp_txt = read_file(psych)

print("The number of instances: {}".format(len(psychExp_txt)))


print("Data example: ")
print(psychExp_txt[0])



import re 
from collections import Counter

def ngram(token, n): 
    output = []
    for i in range(n-1, len(token)): 
        ngram = ' '.join(token[i-n+1:i+1])
        output.append(ngram) 
    return output


def create_feature(text, nrange=(1, 1)):
    text_features = [] 
    text = text.lower() 

    # 1. treat alphanumeric characters as word tokens
    # Since tweets contain #, we keep it as a feature
    # Then, extract all ngram lengths
    text_alphanum = re.sub('[^a-z0-9#]', ' ', text)
    for n in range(nrange[0], nrange[1]+1): 
        text_features += ngram(text_alphanum.split(), n)
    
    # 2. treat punctuations as word token
    text_punc = re.sub('[a-z0-9]', ' ', text)
    text_features += ngram(text_punc.split(), 1)
    
    # 3. Return a dictinaory whose keys are the list of elements 
    # and their values are the number of times appearede in the list.
    return Counter(text_features)


print(create_feature("I love you!"))
print(create_feature(" aly wins the gold!!!!!!  #olympics"))
print(create_feature(" aly wins the gold!!!!!!  #olympics", (1, 2)))



# =============================================================================
# Converting labels
# =============================================================================

def convert_label(item, name): 
    items = list(map(float, item.split()))
    label = ""
    for idx in range(len(items)): 
        if items[idx] == 1: 
            label += name[idx] + " "
    
    return label.strip()


emotions = ["joy", 'fear', "anger", "sadness", "disgust", "shame", "guilt"]

X_all = []
y_all = []
for label, text in psychExp_txt:
    y_all.append(convert_label(label, emotions))
    X_all.append(create_feature(text, nrange=(1, 4)))
    
    
    
print("features example: ")
print(X_all[0])    
    
print("Label example:")
print(y_all[0])




# =============================================================================
# Classifiers
# =============================================================================
from sklearn import model_selection

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = \
    train_test_split(X_all, y_all, test_size = 0.2, random_state = 123)


#####DictVectorizer
from sklearn.metrics import accuracy_score

def train_test(clf, X_train, X_test, y_train, y_test):
    clf.fit(X_train, y_train)
    train_acc = accuracy_score(y_train, clf.predict(X_train))
    test_acc = accuracy_score(y_test, clf.predict(X_test))
    
#     print("Training acc: {}".format(train_acc))
#     print("Test acc    : {}".format(test_acc))
    
    return train_acc, test_acc



from sklearn.feature_extraction import DictVectorizer
vectorizer = DictVectorizer(sparse = True)
X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)



from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

# Classifiers 
svc = SVC()
lsvc = LinearSVC(random_state=123)
rforest = RandomForestClassifier(random_state=123)
dtree = DecisionTreeClassifier()

clifs = [svc, lsvc, rforest, dtree]


# train and test them 
print("| {:25} | {} | {} |".format("Classifier", "Training Accuracy", "Test Accuracy"))
print("| {} | {} | {} |".format("-"*25, "-"*17, "-"*13))
for clf in clifs: 
    clf_name = clf.__class__.__name__
    train_acc, test_acc = train_test(clf, X_train, X_test, y_train, y_test)
    print("| {:25} | {:17.7f} | {:13.7f} |".format(clf_name, train_acc, test_acc))




# =============================================================================
# Model improvement
# =============================================================================

from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

parameters = {'C':[1, 2, 3, 5, 10, 15, 20, 30, 50, 70, 100], 
             'tol':[0.1, 0.01, 0.001, 0.0001, 0.00001]}

lsvc = LinearSVC(random_state=123)
grid_obj = GridSearchCV(lsvc, param_grid = parameters, cv=5)
grid_obj.fit(X_train, y_train)

print("Validation acc: {}".format(grid_obj.best_score_))
print("Training acc: {}".format(accuracy_score(y_train, grid_obj.predict(X_train))))
print("Test acc    : {}".format(accuracy_score(y_test, grid_obj.predict(X_test))))
print("Best parameter: {}".format(grid_obj.best_params_))



# =============================================================================
# Error analysis - confusion matrix
# =============================================================================

from sklearn.metrics import confusion_matrix
matrix = confusion_matrix(y_test, grid_obj.predict(X_test))
print(matrix)


%matplotlib inline
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt

l = ["joy", 'fear', "anger", "sadness", "disgust", "shame", "guilt"]
l.sort()
df_cm = pd.DataFrame(matrix, index = l, columns = l)
plt.figure(figsize = (10,7))
sn.heatmap(df_cm, annot=True, fmt="d")
plt.show()



# Get counts for each label
label_freq = {}
for label, _ in psychExp_txt: 
    label_freq[label] = label_freq.get(label, 0) + 1

# print the labels and their counts in sorted order 
for l in sorted(label_freq, key=label_freq.get, reverse=True):
    print("{:10}({})  {}".format(convert_label(l, emotions), l, label_freq[l]))





# =============================================================================
# Classify with emojis
# =============================================================================
    
emoji_dict = {"joy":"😍", "fear":"😱", "anger":"😠", "sadness":"😢", "disgust":"🤮", "shame":"😳", "guilt":"😞"}

t1 = "Thank you for dinner!"
t2 = "I don't like it"
t3 = "My car skidded on the wet street"
t4 = "My cat died"
t5 = "This is disgusting"
t6 = "Sorry i did that"

texts = [t1, t2, t3, t4, t5, t6]
for text in texts: 
    features = create_feature(text, nrange=(1, 4))
    features = vectorizer.transform(features)
    prediction = grid_obj.predict(features)[0]
    print("{} {}".format(emoji_dict[prediction], text))





def emoji_predict(texts):
    for text in texts: 
        features = create_feature(text, nrange=(1, 4))
        features = vectorizer.transform(features)
        prediction = grid_obj.predict(features)[0]
       # emotion = print("{} {}".format(emoji_dict[prediction]))   
    return emoji_dict[prediction]

df_csv['comments_clean_emoji'] = df_csv['comments_clean2'].apply(lambda x: emoji_predict(x))

df_csv['comments_emoji'] = df_csv['comments_emoticon'].apply(lambda x: emoji_predict(x))


filtered_data = df_csv[df_csv['comments_clean_emoji']=='😍']




# =============================================================================
# Classify each video based on all it's comments
# =============================================================================
col_list2 = ["video_link", "video_link-href", "comments_emoticon", "comments_clean2"]
df_video = df_csv[col_list2]
df_video.is_copy = False
df_video['comments_emoticon'] = df_video['comments_emoticon'].astype(str)

df_video['text'] = df_video[["video_link", "video_link-href", "comments_emoticon"]].groupby(["video_link", "video_link-href"])["comments_emoticon"].transform(lambda x: ','.join(x))
df_video_2 = df_video[["video_link", "video_link-href", "text"]]
df_video_2 = df_video_2[["video_link", "video_link-href", "text"]].drop_duplicates()


from textblob import TextBlob
df_video_2['text'] = df_video_2['text'].astype(str)
df_video_2['sentiment'] = df_video_2.text.map(lambda text: TextBlob(text).sentiment.polarity)

df_video_2['text_emoji'] = df_video_2['text'].apply(lambda x: emoji_predict(x))

from pandas import DataFrame
import pandas as pd
from pandas import DataFrame

df_video_2.to_excel(r'C:/Users/Friendly_Snek_Box/Desktop/Anul 2 - Semestrul 1/Analiza Text/Proiect Pornhub/videos_sentiment.xlsx')

df_csv.to_excel(r'C:/Users/Friendly_Snek_Box/Desktop/Anul 2 - Semestrul 1/Analiza Text/Proiect Pornhub/comments_sentiment_emoji.xlsx')
