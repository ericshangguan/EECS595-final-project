import pandas as pd
import numpy as np
import nltk
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.feature_extraction import DictVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.metrics import classification_report, f1_score
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import SelectKBest, chi2
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize
from gensim.models import Word2Vec
from nltk import pos_tag
# %pip install textblob
from scipy.sparse import hstack
from textblob import TextBlob
import re
import warnings, time
import argparse
from collections import Counter
warnings.filterwarnings('ignore')

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

# train_path = './dataset/semeval2016-task6-trainingdata.txt'
# test_path = './dataset/semeval2016-task6-testdata-gold/SemEval2016-Task6-subtaskA-testdata-gold.txt'

def define_params():
    C = [0.01, 0.1, 1, 10]
    logistic_fit_intercept = [True, False]
    logistic_class_weight = [None, 'balanced']
    logistic_solver = ['lbfgs', 'liblinear', 'newton-cg']
    n_neighbors = [3,4,5,6,7]
    n_estimators = [10,25,50,75,100]
    criterion =['gini', 'entropy']
    max_depth =  [2,3,4,5,6,7,10]
    decision_function_shape = ['ovo', 'ovr']
    kernel = ['linear', 'rbf']
    gamma = ['scale', 'auto', 0.1, 1, 10]
    classifiers = {
        'LogisticRegression': LogisticRegression(max_iter=1000),
        'KNN': KNeighborsClassifier(),
        'RandomForest': RandomForestClassifier(),
        'SVM': SVC(),
        'GradientBoosting': GradientBoostingClassifier()
    }
    params_grid = {
        'LogisticRegression':{
            'classifier__C': C, 
            'classifier__fit_intercept':logistic_fit_intercept, 
            'classifier__class_weight': logistic_class_weight,
            'classifier__solver': logistic_solver
        },
        'KNN': {
            'classifier__n_neighbors':n_neighbors
        },
        'RandomForest': {
            'classifier__n_estimators':n_estimators,
            'classifier__criterion':criterion,
            'classifier__max_depth':max_depth
        },
        'SVM': {
            'classifier__class_weight': ['balanced', None],
            'classifier__decision_function_shape': decision_function_shape,
            'classifier__kernel':kernel,
            'classifier__gamma':gamma
        },
        'GradientBoosting': {
            'classifier__max_depth':max_depth,
            'classifier__n_estimators': n_estimators
        }
    }

    return classifiers, params_grid

def load_data(path):
    df = pd.read_csv(path, sep='\t', encoding='ISO-8859-1')
    return df

def split_data(train, test, name):
    X_train = train[train['Target']==name][['Tweet', 'Target']]
    y_train = train[train['Target']==name]['Stance']
    X_test = test[test['Target']==name][['Tweet', 'Target']]
    y_test = test[test['Target']==name]['Stance']
    return X_train, y_train, X_test, y_test

def report_score(feature_union, pipeline, X_test, y_test):
    # X_test = feature_union.transform(X_test)
    prediction = pipeline.predict(X_test)
    report = classification_report(y_test, prediction, output_dict=True, zero_division=0)
    # print(classification_report(y_test, prediction, zero_division=0))
    f1_favor = report['FAVOR']['f1-score']
    f1_against = report['AGAINST']['f1-score']
    score = (f1_favor + f1_against)/2
    # print("The score of this model is {}.".format(score))

    return score

class TextPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, lower=True, remove_at=True, lemmatize=False, remove_semst=True):
        self.lower = lower
        self.remove_at = remove_at
        self.lemmatize = lemmatize
        self.lemmatizer = WordNetLemmatizer()
        self.remove_semst = remove_semst

    def fit(self, X, y=None):
        return self

    def transform(self, text, y=None):
        processed_texts = []
        # for text in X:
        if self.lower:
            text = text.lower()
        if self.remove_at:
            text = re.sub(r'(@\w+\s?)', '', text)
        if self.remove_semst:
            text = re.sub(r'(\#semst\s?)', '', text)
        if self.lemmatize:
            text = ' '.join([self.lemmatizer.lemmatize(word) for word in text.split()])
            processed_texts.append(text)
        return text
    
def preprocess(text):
    return TextPreprocessor().transform(text)

def transform_all(data):
    return data.apply(preprocess)

class ModifiedTfidfVectorizer(BaseEstimator, TransformerMixin):
    def __init__(self, max_feature):
        self.tfidf = TfidfVectorizer(max_features=max_feature)

    def fit(self, X):
        X = X['Tweet']
        self.tfidf.fit(X)
        return self
    def transform(self, X):
        X = X['Tweet']
        return self.tfidf.transform(X)

class SentimentExtractor(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return np.array([[TextBlob(text).sentiment.polarity, TextBlob(text).sentiment.subjectivity] for text in X['Tweet']])

def train_word2vec(train, k=200):
    tweet = transform_all(train["Tweet"])
    tokenized_tweet = [word_tokenize(sentence) for sentence in tweet]
    wrod2vec_model = Word2Vec(tokenized_tweet, vector_size=k, window=5, min_count=1, workers=4)
    return wrod2vec_model

class Word2VecVectorizer(BaseEstimator, TransformerMixin):
    # Initialize with a pre-trained Word2Vec model
    def __init__(self, model):
        self.word2vec_model = model
        self.vector_size = model.vector_size

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return np.array([
            np.mean([self.word2vec_model.wv[word] for word in doc.split() if word in self.word2vec_model.wv]
                    or [np.random.rand(self.vector_size)], axis=0)
            for doc in X['Tweet']
        ])

class GloVeVectorizer(BaseEstimator, TransformerMixin):
    def __init__(self, glove_path, vector_size=200):
        self.glove_path = glove_path
        self.vector_size = vector_size
        self.embeddings = self.load_glove_embeddings()

    def load_glove_embeddings(self):
        embeddings = {}
        with open(self.glove_path, 'r', encoding='utf-8') as file:
            for line in file:
                values = line.split()
                word = values[0]
                vector = np.asarray(values[1:], dtype='float32')
                embeddings[word] = vector
        return embeddings
    
    def document_vector(self, doc):
        words = doc.split()
        word_vectors = [self.embeddings[word] for word in words if word in self.embeddings]

        if len(word_vectors) == 0:
            return np.zeros(self.vector_size)


        word_mean_vec = np.mean(word_vectors, axis=0)
        # if len(word_mean_vec) != self.vector_size:
        #     print(word_mean_vec)
        return word_mean_vec

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return np.array([self.document_vector(doc) for doc in X['Tweet']])

class NGramVectorizer(BaseEstimator, TransformerMixin):
    def __init__(self, ngram_range_word=(1, 3), ngram_range_char=None, binary=True, k=200):
        self.word_vectorizer = CountVectorizer(ngram_range=ngram_range_word, binary=binary)
        if ngram_range_char is not None:
            self.char_vectorizer = CountVectorizer(analyzer='char', ngram_range=ngram_range_char, binary=binary)
        else:
            self.char_vectorizer = None
        self.selector = SelectKBest(chi2, k=k)
        self.is_fitted = False

    def fit(self, X, y=None):
        return self
    
    def fit_selector(self, X, y):
        word_features = self.word_vectorizer.fit_transform(X['Tweet'])
        if self.char_vectorizer is not None:
            char_features = self.char_vectorizer.fit_transform(X['Tweet'])
            combined_features = hstack([word_features, char_features])
        else:
            combined_features = word_features
        self.selector.fit(combined_features, y)
        self.is_fitted = True

    def transform(self, X):
        if not self.is_fitted:
            raise RuntimeError("You must call fit_selector before calling transform")
        word_features = self.word_vectorizer.transform(X['Tweet'])
        if self.char_vectorizer is not None:
            char_features = self.char_vectorizer.transform(X['Tweet'])
            combined_features = hstack([word_features, char_features])
            selected_features = self.selector.transform(combined_features)
        else:
            selected_features = self.selector.transform(word_features)
        return selected_features

class PosTagVectorizer(BaseEstimator, TransformerMixin):
    def __init__(self, pos_tag_list):
      self.pos_tags = pos_tag_list
    def fit(self, X, y=None):
        return self

    def transform(self, X):
      transformed = []
      for tweet in X['Tweet']:
        tag_counts = {tag: 0 for tag in self.pos_tags}
        counts = Counter(tag for word, tag in pos_tag(word_tokenize(tweet)))
        for key, value in counts.items():
          if key in tag_counts.keys():
            tag_counts[key] = value
        transformed.append(tag_counts)
      return transformed

class TargetPresence_Single(BaseEstimator, TransformerMixin):
    def __init__(self, target_words_dict):
        self.target_words_dict = target_words_dict

    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        features = []

        for _, row in X.iterrows():
            tweet = row['Tweet']
            target = row['Target']
            related_words = self.target_words_dict.get(target, [])

            # Check for the presence of each related word in the tweet
            presence = [int(word in tweet) for word in related_words]
            features.append(presence)

        return np.array(features)

class TargetPresence(BaseEstimator, TransformerMixin):
    def __init__(self, target_words_list):
        self.target_words_list = target_words_list

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        features = []
        for tweet in X['Tweet']:
            presence = [int(word in tweet.split()) for word in self.target_words_list]
            features.append(presence)

        return np.array(features)

def classifier_grid(feature_union, classify_pipeline, classifiers, params_grid, X_train, y_train, X_test, y_test):
    best_score = 0
    best_classifier = None
    best_classifier_name = ''
    X_train_transformed = feature_union.fit_transform(X_train)
    X_test_transformed = feature_union.transform(X_test)
    for classifier_name, model in classifiers.items():
        classify_pipeline.set_params(classifier=model)
        grid_search = RandomizedSearchCV(classify_pipeline, param_distributions=params_grid[classifier_name], cv=5, verbose=0, random_state=595, n_jobs=-1)
        start = time.time()
        grid_search.fit(X_train_transformed, y_train)
        end = time.time()
        # print('The best parameter for {} is {}.'.format(classifier_name, grid_search.best_params_))
        # print("Grid Search for Model {} needs {} seconds.".format(classifier_name, end-start))
        # print("The score for {} is {:.2f}.".format(classifier_name, grid_search.best_score_))
        best_model = grid_search.best_estimator_
        score = report_score(feature_union, best_model, X_test_transformed, y_test)
        
        if score > best_score:
            best_score = score
            best_classifier = best_model
            best_classifier_name = classifier_name
        
    return feature_union, best_classifier, best_classifier_name, best_score

def find_best(feature_union, pipeline, train, test, name, classifiers, params_grid):
    X_train, y_train, X_test, y_test = split_data(train, test, name)
    feature_extraction, best_classifier, classifier_name, score = classifier_grid(feature_union, pipeline, classifiers, params_grid, X_train, y_train, X_test, y_test)
    print("The best model for {} is {} with {}".format(name, classifier_name, best_classifier.get_params()))
    # print("The avg F1-score is ", score)
    return feature_extraction, best_classifier, classifier_name, score

def train_test_all(train, test, feature_union, classify_pipeline, classifiers, params_grid):
    trained_classifiers = {}
    trained_feature_extraction = {}
    for name in train["Target"].unique():
        feature_extraction, classifier, classifier_name, score = find_best(feature_union, classify_pipeline, train, test, name, classifiers, params_grid)
        print("The best classifier for {} is {} with the average of F1 as {}.".format(name, classifier_name, score))
        trained_classifiers[name] = classifier
        trained_feature_extraction[name] = feature_extraction
    return trained_feature_extraction, trained_classifiers

def test_all(test_data, trained_feature_extraction, classifiers):
    predictions = pd.DataFrame(index=test_data.index)

    for target in classifiers:
        # Select the test data for the current target
        target_data = test_data[test_data['Target'] == target][['Tweet', 'Target']]

        if not target_data.empty:
            classifier = classifiers[target]
            feature_union = trained_feature_extraction[target]
            transformed_features = feature_union.transform(target_data)
            target_predictions = classifier.predict(transformed_features)
            predictions.loc[target_data.index, 'Prediction'] = target_predictions

    print(classification_report(test_data['Stance'], predictions))
    report = classification_report(test_data['Stance'], predictions, output_dict=True, zero_division=0)
    f1_favor = report['FAVOR']['f1-score']
    f1_against = report['AGAINST']['f1-score']
    score = (f1_favor + f1_against)/2
    print("The average F1-score for total test dataset is ", score)
    
    return score, predictions

def train_test_single(train, test, feature_union, pipeline, classifiers, params_grid):
    X_train = train[['Tweet', 'Target']]
    y_train = train['Stance']
    X_test = test[['Tweet', 'Target']]
    y_test = test['Stance']
    feature_extarction, best_classifier, classifier_name, score = classifier_grid(feature_union, pipeline, classifiers, params_grid, X_train, y_train, X_test, y_test)
    print("The best model for considering all targets is {} with {}".format(classifier_name, best_classifier.get_params()))
    print("The avg F1-score is ", score)

def create_feature_union(args):
    transformers = []

    if args.tfidf:
        transformers.append(('tfidf', ModifiedTfidfVectorizer(max_feature=args.tfidf_num)))

    if args.sentiment:
        transformers.append(('sentiment', SentimentExtractor()))
    
    if args.bigram:
        train = load_data(args.train_path)
        train['Tweet'] = transform_all(train['Tweet'])
        X_train = train[['Tweet', 'Target']]
        y_train = train['Stance']
        bigram_transformer = NGramVectorizer(ngram_range_word=(2,2), k=args.k)
        bigram_transformer.fit_selector(X_train, y_train)
        transformers.append(('bigram', bigram_transformer))

    if args.trigram:
        train = load_data(args.train_path)
        train['Tweet'] = transform_all(train['Tweet'])
        X_train = train[['Tweet', 'Target']]
        y_train = train['Stance']
        trigram_transformer = NGramVectorizer(ngram_range_word=(3,3), k=args.k)
        trigram_transformer.fit_selector(X_train, y_train)
        transformers.append(('trigram', trigram_transformer))

    if args.ngram:
        train = load_data(args.train_path)
        train['Tweet'] = transform_all(train['Tweet'])
        X_train = train[['Tweet', 'Target']]
        y_train = train['Stance']
        ngram_transformer = NGramVectorizer(ngram_range_word=(1,3), ngram_range_char=(2, 5), k=args.k)
        ngram_transformer.fit_selector(X_train, y_train)
        transformers.append(('ngram', ngram_transformer))

    if args.word2vec:
        train = load_data(args.train_path)
        word2vec_model = train_word2vec(train)
        word2vec_vectorizer = Word2VecVectorizer(word2vec_model)
        transformers.append(('word2vec', word2vec_vectorizer))
    
    if args.glove:
        glove_vectorizer = GloVeVectorizer(glove_path=args.glove_path, vector_size=200)
        transformers.append(('glove', glove_vectorizer))

    if args.target:
        target_words = {
            'Atheism': ['atheism', 'god'],
            'Hillary Clinton': ['hillary', 'clinton'],
            'Climate Change is a Real Concern': ['climate'],
            'Feminist Movement': ['feminism', 'feminist', 'female', 'woman'],
            'Legalization of Abortion': ['abortion', 'prolife', 'youth']
        }

        unique_words = set()
        for words in target_words.values():
            unique_words.update(words)
        target_words = list(unique_words)
        transformers.append(('target_presence', TargetPresence(target_words)))

    if args.pos_tag:
        pos_tag_list = ['CC','CD','DT','EX','FW','IN','JJ','JJR','JJS','MD','NN','NNP','NNS','PDT','POS','PRP','PRP$','RB','RBR','RBS','RP','SYM','TO','UH','VB','VBD','VBG','VBN','VBP','VBZ','WDT','WP','WRB']
        transformers.append(('pos_tag', Pipeline([
            ('pos_extractor', PosTagVectorizer(pos_tag_list)),
            ('vectorizer', DictVectorizer())
        ])))

    return FeatureUnion(transformers)

def create_pipeline(args):

    feature_union = create_feature_union(args)
    pipeline = Pipeline([
        ('classifier', None)
    ])

    return feature_union, pipeline

def main(args):
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")
    train = load_data(args.train_path)
    test = load_data(args.test_path)
    train['Tweet'] = transform_all(train['Tweet'])
    test['Tweet'] = transform_all(test['Tweet'])
    feature_union, classify_pipeline = create_pipeline(args)
    classifiers, params_grid = define_params()
    start = time.time()
    trained_feature_extraction, trained_classifiers = train_test_all(train, test, feature_union, classify_pipeline, classifiers, params_grid)
    end = time.time()
    print(end-start, "for train_test_all.")
    score, predictinos = test_all(test, trained_feature_extraction, trained_classifiers)
    if args.train_single:
        train_test_single(train, test, feature_union, classify_pipeline, classifiers, params_grid)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_path', required=True, help='Path to the training dataset')
    parser.add_argument('--test_path', required=True, help='Path to the test dataset')
    parser.add_argument('--tfidf', action='store_true', help='Include TF-IDF features')
    parser.add_argument('--tfidf_num', type=int, default=1000, help='the number of max feature')
    parser.add_argument('--word2vec', action='store_true', help='Include word2vec model')
    parser.add_argument('--glove_path', required=False, help='the path of glove embedding')
    parser.add_argument('--glove', action='store_true', help='Include GloVE embedding')
    parser.add_argument('--target', action='store_true', help='Include target presence')
    parser.add_argument('--pos_tag', action='store_true', help='Include pos tag extraction')
    parser.add_argument('--sentiment', action='store_true', help='Include word sentiment')
    parser.add_argument('--ngram', action='store_true', help='Include word N-gram')
    parser.add_argument('--k', default=200, help='the vector size of ngram')
    parser.add_argument('--bigram', action='store_true', help='Include bigram')
    parser.add_argument('--trigram', action='store_true', help='Include trigram')

    parser.add_argument('--train_single', action='store_true', help='If we want to train the model without splitting target subset')
    args = parser.parse_args()
    main(args)