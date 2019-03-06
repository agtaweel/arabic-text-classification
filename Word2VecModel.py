import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
#from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import Imputer
import gensim
from gensim.models import word2vec
import random
import numpy as np
import multiprocessing

#nltk.download('stopwords')
cores = multiprocessing.cpu_count()
def read_data(path):

    print('reading data')
    dataset = pd.read_csv(path)
    # Taking care of missing data
    imputer = Imputer(missing_values = 'NaN', strategy = 'most_frequent', axis = 0)
    dataset['target'] = imputer.fit_transform(dataset[['target']]).ravel()
    #text_imputer = Imputer(missing_values = '',strategy='constant',axis=0,fill_value='مازا')
    #dataset.text = text_imputer.fit_transform(dataset.text)
    dataset.text = bag_of_words(dataset.text)
    # Splitting the dataset into the Training set and Test set
    print('splitting data')
    x_train, x_test, y_train, y_test = train_test_split(dataset.text, dataset.target, random_state=0, test_size=0.10)
    #x_train = label_sentences(x_train, 'Train')
    #x_test = label_sentences(x_test, 'Test')
    all_data = dataset.text
    print('data was read')
    return x_train, x_test, y_train, y_test, all_data


def bag_of_words(x):
    corpus = []
    for i in range(0, len(x)):
        page = re.sub('[^\u0627-\u064a]', ' ', str(x[i]))
        #page = page.encode('utf-8').split()
        page = page.split()
        if(len(page)==0):
            page.append('مازا')
        #ps = PorterStemmer()
        #page = [ps.stem(word) for word in page if not word in set(stopwords.words('arabic'))]
        #page = ' '.join(page)
        #page = gensim.utils.simple_preprocess(page)
        corpus.append(page)
        #w2v = word2vec.Word2Vec(min_count=1,window=2, size=300, alpha=0.03, min_alpha=0.0007, workers=cores-1)
        #w2v.build_vocab(corpus)
        #w2v.train(page, epochs=10,total_examples=w2v.corpus_count)
        print((i + 1) / len(x) * 100, '%')
        print(i + 2)
    print (corpus[1919])
    
    print('done')
    return corpus

def train_word2vec(corpus):
    
    w2v = word2vec.Word2Vec(min_count=1,window=2, size=300, alpha=0.03, min_alpha=0.0007, workers=cores-1)
    print('start training word2vec....')
    w2v.build_vocab(corpus)
    print('building vocablaries')
    w2v.train(corpus, epochs=10,total_examples=w2v.corpus_count)
    print('finished trainig')
    #for epoch in range(10):
    #    w2v.train(corpus, total_examples=w2v.corpus_count, epochs=w2v.iter)
    #    # shuffle the corpus
    #    random.shuffle(corpus)
    #    # decrease the learning rate
    #    d2v.alpha -= 0.0002
    #    # fix the learning rate, no decay
    #   
    w2v.save("d2v.model")
    return w2v

def train_classifier(w2v, x_train, y_train):

    x_train = get_vectors(w2v,x_train, len(x_train), 300)
    print('start training data')
    # Fitting Naive Bayes to the Training set
    classifier = GaussianNB()
    print('fitting data')
    classifier.fit(x_train, np.array(y_train))
    print('training finished')
    return classifier


def test_classifier(w2v,classifier, x_test, y_test):
    # Predicting the Test set results
    x_test = get_vectors(w2v, x_test, len(x_test),300)
    y_pred = classifier.predict(x_test)
    print('Getting Confusion Matrix .....')
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    print('testing results: ')

    print('Confusion Matrix goes like => ')
    print(cm)
    print('With accuracy of: ', accuracy*100, '%')
    return accuracy, cm

def get_vectors(word2vec_model, sentences, corpus_size, vectors_size):

    vectors = np.zeros((corpus_size, vectors_size))
    for i in range(0, corpus_size):
        print(i)
        rms =0
        try:
            words = sentences[i]
            for word in range(0, len(words)):
                sentence = sentence + word2vec_model.wv[words[word]]
                rms = rms + word2vec_model.wv[words[word]]*word2vec_model.wv[words[word]]
        except :
            words=['لا']
            rms = 1
            sentence =word2vec_model.wv['لا']
        vectors[i] = sentence/rms
    return vectors


#def label_sentences(corpus, label_type):
#
#    labeled = []
#    for i, v in enumerate(corpus):
#        label = label_type + '_' + str(i)
#        print(i+2)
#        #labeled.append(doc2vec.LabeledSentence(v.split(), [label]))
#        
#    return labeled
#

if __name__ == "__main__":
    x_train, x_test, y_train, y_test, all_data = read_data('arabic_dataset_classifiction.csv')
    w2v = train_word2vec(all_data)
    #print(x_train[3])
    classifier = train_classifier(w2v, x_train, y_train)
    cm , accuracy = test_classifier(w2v,classifier, x_test, y_test)
    print('Confusion Matrix goes like => ')
    print(cm)
    print('With accuracy of: ', accuracy*100, '%')