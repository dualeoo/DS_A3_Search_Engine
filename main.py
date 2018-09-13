import argparse
import logging
from pathlib import Path
from pprint import pprint

from gensim import corpora, models, similarities
from sklearn.datasets import fetch_20newsgroups

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


def create_data_path_if_not_exit():
    data_dir_path = Path(DATA_PATH)
    if not data_dir_path.exists():
        data_dir_path.mkdir()


def load_newsgroup_dict() -> corpora.dictionary:
    dictionary = corpora.Dictionary()
    return dictionary.load(NEWSGROUP_DICT_PATH)


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
    if lsi_model_path.exists():
        lsi = models.LsiModel.load(LSI_MODEL_PATH)
    else:
        lsi = models.LsiModel(corpus_tfidf, id2word=corpora_dict,
                              num_topics=NUM_TOPICS)  # initialize an LSI transformation
        lsi.save(LSI_MODEL_PATH)


def load_tfid_model():
    global tfidf
    logging.info("*** Start loading tfid model ***")
    tfid_model_path = Path(TFID_MODEL_PATH)
    if tfid_model_path.exists():
        tfidf = models.TfidfModel.load(TFID_MODEL_PATH)
    else:
        tfidf = models.TfidfModel(corpus)
        tfidf.save(TFID_MODEL_PATH)


def load_model():
    global newsgroups_train, texts, corpus_tfidf, corpus_lsi
    logging.info("*** Start loading model ***")

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
    if index_path.exists():
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
    create_data_path_if_not_exit()
    load_model()
    load_index()

    if query:
        project_query_to_lsi_space()
        find_top_n_articles()
        pprint(top_articles)
        pass
