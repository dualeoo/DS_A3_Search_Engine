import argparse
import logging
from collections import defaultdict
from pathlib import Path
from pprint import pprint
from time import time

import numpy as np
from gensim import corpora, models, similarities
from sklearn import metrics
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

NO_ABOVE = 0.5

NO_BELOW = 5

NUM_SEARCH_RESULTS_TAKE = 10

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

NUM_TOPICS = 10000

DATA_PATH = 'data'

DATASET_NAME = '20newsgroups'

CORE_PATH = (DATA_PATH, DATASET_NAME)

LSI_MODEL_PATH = '%s/%s.lsi' % CORE_PATH

CORPUS_PATH = '%s/%s.mm' % CORE_PATH

NEWSGROUP_DICT_PATH = '%s/%s.dict' % CORE_PATH

TFID_MODEL_PATH = '%s/%s.tfid' % CORE_PATH

INDEX_PATH = '%s/%s.index' % CORE_PATH

FORCE_RECREATE_TFID_MODEL = False
FORCE_RECREATE_LSI_MODEL = False
FORCE_RECREATE_INDEX = False


def create_data_path_if_not_exit():
    data_dir_path = Path(DATA_PATH)
    if not data_dir_path.exists():
        data_dir_path.mkdir()


def load_newsgroup_dict() -> corpora.dictionary:
    dictionary = corpora.Dictionary()
    return dictionary.load(NEWSGROUP_DICT_PATH)


def show_top10(classifier, vectorizer, categories):
    feature_names = np.asarray(vectorizer.get_feature_names())
    for i, category in enumerate(categories):
        top10 = np.argsort(classifier.coef_[i])[-10:]
        print("%s: %s" % (category, " ".join(feature_names[top10])))


def convert_data_to_dict():
    dictionary = corpora.Dictionary(texts)
    return dictionary


def preprocess_dict(dictionary):
    dictionary.filter_n_most_frequent(5)
    # fixme, when NO_BELOW or NO_ABOVE change, this will not automatically recreate dict
    dictionary.filter_extremes(no_below=NO_BELOW, no_above=NO_ABOVE)
    return dictionary


def get_data():
    return [[word for word in document.lower().split()] for document in newsgroups_train.data]


def calculate_token_freq():
    frequency = defaultdict(int)
    for text in texts:
        for token in text:
            frequency[token] += 1
    return frequency


def grid_search():
    pipeline = Pipeline([
        ('vect', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('clf', SGDClassifier()),
    ])
    parameters = {
        'vect__max_df': (0.5, 0.75, 1.0),
        # 'vect__max_features': (None, 5000, 10000, 50000),
        'vect__ngram_range': ((1, 1), (1, 2)),  # unigrams or bigrams
        # 'tfidf__use_idf': (True, False),
        # 'tfidf__norm': ('l1', 'l2'),
        'clf__alpha': (0.00001, 0.000001),
        'clf__penalty': ('l2', 'elasticnet'),
        # 'clf__n_iter': (10, 50, 80),
    }

    grid_search = GridSearchCV(pipeline, parameters, n_jobs=-1, verbose=1)

    print("Performing grid search...")
    print("pipeline:", [name for name, _ in pipeline.steps])
    print("parameters:")
    pprint(parameters)
    t0 = time()
    grid_search.fit(newsgroups_train.data, newsgroups_train.target)
    print("done in %0.3fs" % (time() - t0))
    print()

    print("Best score: %0.3f" % grid_search.best_score_)
    print("Best parameters set:")
    best_parameters = grid_search.best_estimator_.get_params()
    for param_name in sorted(parameters.keys()):
        print("\t%s: %r" % (param_name, best_parameters[param_name]))


def scikit_learn_tut():
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(newsgroups_train.data)
    print("The shape of the vectors = {}".format(vectors.shape))
    newsgroups_test = fetch_20newsgroups(subset='test', remove=('headers', 'footers', 'quotes'))
    vectors_test = vectorizer.transform(newsgroups_test.data)
    clf = MultinomialNB(alpha=.01)
    clf.fit(vectors, newsgroups_train.target)
    pred = clf.predict(vectors_test)
    prediction_score = metrics.f1_score(newsgroups_test.target, pred, average='macro')
    print("The score of the prediction = {}".format(prediction_score))
    show_top10(clf, vectorizer, newsgroups_train.target_names)


def load_dict():
    global corpora_dict
    logging.info("*** Start loading dictionary ***")
    dict_path = Path(NEWSGROUP_DICT_PATH)
    if dict_path.exists():
        corpora_dict = load_newsgroup_dict()
    else:
        # frequency = calculate_token_freq(texts)
        # texts = [[token for token in text if frequency[token] > 1]
        #          for text in texts]
        corpora_dict = convert_data_to_dict()
        # TODOx look inside
        corpora_dict = preprocess_dict(corpora_dict)
        corpora_dict.save(NEWSGROUP_DICT_PATH)


def load_corpus():
    global corpus
    logging.info("*** Start loading corpus ***")
    corpus_path = Path(CORPUS_PATH)
    if corpus_path.exists():
        corpus = corpora.MmCorpus(CORPUS_PATH)
    else:
        corpus = [corpora_dict.doc2bow(text) for text in texts]
        corpora.MmCorpus.serialize(CORPUS_PATH, corpus)


def load_lsi_model():
    global lsi
    logging.info("*** Start loading LSI model ***")
    lsi_model_path = Path(LSI_MODEL_PATH)
    if lsi_model_path.exists() and not FORCE_RECREATE_LSI_MODEL:
        lsi = models.LsiModel.load(LSI_MODEL_PATH)
    else:
        lsi = models.LsiModel(corpus_tfidf, id2word=corpora_dict,
                              num_topics=NUM_TOPICS)  # initialize an LSI transformation
        lsi.save(LSI_MODEL_PATH)


def load_tfid_model():
    global tfidf
    logging.info("*** Start loading tfid model ***")
    tfid_model_path = Path(TFID_MODEL_PATH)
    if tfid_model_path.exists() and not FORCE_RECREATE_TFID_MODEL:
        tfidf = models.TfidfModel.load(TFID_MODEL_PATH)
    else:
        tfidf = models.TfidfModel(corpus)
        tfidf.save(TFID_MODEL_PATH)


def load_model():
    global newsgroups_train, texts, corpus_tfidf, corpus_lsi
    logging.info("*** Start loading model ***")
    create_data_path_if_not_exit()
    newsgroups_train = fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'))
    texts = get_data()
    load_dict()
    load_corpus()
    load_tfid_model()
    corpus_tfidf = tfidf[corpus]
    load_lsi_model()
    corpus_lsi = lsi[corpus_tfidf]  # create a double wrapper over the original corpus: bow->tfidf->fold-in-lsi


def process_args():
    global query
    logging.info("Start processing args")
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", )
    args = parser.parse_args()
    query = args.query


def load_index():
    global index
    logging.info("*** Start loading index ***")
    index_path = Path(INDEX_PATH)
    if index_path.exists() and not FORCE_RECREATE_LSI_MODEL and not FORCE_RECREATE_INDEX:
        index = similarities.MatrixSimilarity.load(INDEX_PATH)
    else:
        # fixme the case when NUM_TOPICS change
        index = similarities.MatrixSimilarity(corpus_lsi, num_features=NUM_TOPICS)
        index.save(INDEX_PATH)


def project_query_to_lsi_space():
    global vec_lsi
    logging.info("*** Start project query to LSI space ***")
    vec_bow = corpora_dict.doc2bow(query.lower().split())
    vec_lsi = lsi[vec_bow]


def find_top_n_results():
    global top_results
    sims = index[vec_lsi]
    sims = sorted(enumerate(sims), key=lambda item: -item[1])
    top_results = sims[slice(0, NUM_SEARCH_RESULTS_TAKE)]


def find_top_n_articles():
    global top_articles
    logging.info("*** Start finding top articles ***")
    find_top_n_results()
    top_articles = []
    for result in top_results:
        article_id = result[0]
        article_score = result[1]

        article = newsgroups_train.data[article_id]
        top_articles.append((article, article_score))


if __name__ == '__main__':
    process_args()
    load_model()
    load_index()

    if query:
        project_query_to_lsi_space()
        find_top_n_articles()
        # pprint(top_articles)
        pass
