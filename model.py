import pandas as pd
from bs4 import BeautifulSoup
import requests
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import Imputer

nltk.download('stopwords')


def read_data(path):

    print('reading data')
    dataset = pd.read_csv(path)
    # Taking care of missing data
    imputer = Imputer(missing_values = 'NaN', strategy = 'most_frequent', axis = 0)
    dataset['target'] = imputer.fit_transform(dataset[['target']]).ravel()
    x = dataset.iloc[:, :-1].values
    print('Start bag of words')
    x = bag_of_words(x)
    y = dataset.iloc[:, 1].values
    # Splitting the dataset into the Training set and Test set
    print('splitting data')
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=0, shuffle=True)
    print('data was read')
    return x_train, x_test, y_train, y_test, x

def bag_of_words(x):
    corpus = []
    for i in range(0, len(x)):
        page = re.sub('[^\u0627-\u064a]', ' ', str(x[i]))
        page = page.lower()
        page = page.split()
        ps = PorterStemmer()
        page = [ps.stem(word) for word in page if not word in set(stopwords.words('arabic'))]
        page = ' '.join(page)
        corpus.append(page)
        print((i + 1) / len(x) * 100, '%')
        print(i + 2)
    cv = CountVectorizer(max_features=1500)
    x = cv.fit_transform(corpus).toarray()
    return x

def train_data(x_train, y_train):

    print('start training data')
    # Fitting Naive Bayes to the Training set
    classifier = GaussianNB()
    print('fitting data')
    classifier.fit(x_train, y_train)
    print('training finished')
    return classifier

def get_results(classifier, x_test, y_test):
    # Predicting the Test set results
    y_pred = classifier.predict(x_test)
    print('Getting Confusion Matrix .....')
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    print('testing results: ')
    return cm, accuracy

if __name__ == "__main__":
    x_train, x_test, y_train, y_test,x = read_data('arabic_dataset_classifiction.csv')
    classifier = train_data(x_train, y_train)
    cm , accuracy = get_results(classifier, x_test, y_test)
    print('Confusion Matrix goes like => ')
    print(cm)
    print('With accuracy of: ', accuracy*100, '%')
