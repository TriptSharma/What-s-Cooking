import numpy as np
import re
import nltk
import pandas as pd
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize
from nltk.stem import PorterStemmer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer


df = pd.read_json('train.json')
# testset = pd.read_json('test.json')

# print(df.describe())
# print(df.head(5))
# print(df['cuisine'].unique())

df.ingredients = df.ingredients.astype('str')
# testset.ingredients = testset.ingredients.astype('str')
df.ingredients = df.ingredients.str.replace("["," ")
df.ingredients = df.ingredients.str.replace("]"," ")
df.ingredients = df.ingredients.str.replace("'"," ")
df.ingredients = df.ingredients.str.replace(","," ")
# testset.ingredients = testset.ingredients.str.replace("["," ")
# testset.ingredients = testset.ingredients.str.replace("]"," ")
# testset.ingredients = testset.ingredients.str.replace("'"," ")
# testset.ingredients = testset.ingredients.str.replace(","," ")

# nltk.download(['punkt', 'wordnet'])

df.ingredients = df.ingredients.apply(lambda x: word_tokenize(x))
# testset.ingredients = testset.ingredients.apply(lambda x: word_tokenize(x))


lemmatizer = WordNetLemmatizer()
def preprocess(ingredients):
    ingredients_text = ' '.join(ingredients)
    ingredients_text = ingredients_text.lower()
    ingredients_text = ingredients_text.replace('-', ' ')
    words = []
    for word in ingredients_text.split():
        if re.findall('[0-9]', word): continue
        if len(word) <= 2: continue
        if '’' in word: continue
        word = lemmatizer.lemmatize(word)
        if len(word) > 0: words.append(word)
    return ' '.join(words)

df.ingredients = df.ingredients.apply(preprocess)
# testset.ingredients = testset.ingredients.apply(preprocess)

print(df.ingredients[0])



vect = TfidfVectorizer()
features = vect.fit_transform(df.ingredients)

# testfeatures = vect.transform(testset.ingredients)


encoder = LabelEncoder()
labels = encoder.fit_transform(df.cuisine)

from sklearn.feature_selection import SelectPercentile, SelectKBest, f_classif
from scipy.sparse import csr_matrix

sparse_dataset = csr_matrix(features)
features = sparse_dataset.todense()

#feature scaling
from sklearn import preprocessing
features_scaled = pd.DataFrame(preprocessing.scale(features))


encoder = LabelEncoder()
labels = encoder.fit_transform(df.cuisine)
X_train, X_test, y_train, y_test = train_test_split(features_scaled, labels, test_size=0.2)


selector = SelectKBest(f_classif, k=50)
X_trainK = selector.fit_transform(X_train, y_train)
X_testK = selector.transform(X_test)


import statsmodels.api as sm
from sklearn.metrics import accuracy_score, roc_auc_score

model = sm.GLM(y_train, X_train, family=sm.families.Binomial())
model_results = model.fit()
model_results.predict(X_test)
y_pred_test = model_results.predict(X_test).reshape(-1,)
score = roc_auc_score(y_test, y_pred_test)
            
# y_pred = model_results.predict(X_test)

oof[valid_index] = y_pred_test.reshape(-1,)
scores.append(roc_auc_score(y_test, y_pred_test))

print(oof, prediction, scores)
