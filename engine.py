import argparse
import collections
import logging
from abc import ABCMeta, abstractmethod
from pathlib import Path
from pprint import pprint
from typing import List, Tuple

from gensim import corpora, models, similarities
from sklearn.datasets import fetch_20newsgroups

import config

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


class IQueryable:
    __metaclass__ = ABCMeta

    @abstractmethod
    def find_top_n_articles(self, query: str) -> List[Tuple[str, float]]: raise NotImplementedError


class LoadModel(IQueryable):
    def create_data_path_if_not_exit(self):
        data_dir_path = Path(self.data_path)
        if not data_dir_path.exists():
            data_dir_path.mkdir()

    @staticmethod
    def get_data(newsgroups_train) -> List[List[str]]:
        return [[word for word in document.lower().split()] for document in newsgroups_train.data]

    def __init__(self,
                 newsgroup_dict_path: str = config.NEWSGROUP_DICT_PATH,
                 no_below: int = config.NO_BELOW,
                 no_above: float = config.NO_ABOVE,
                 corpus_path: str = config.CORPUS_PATH,
                 lsi_model_path: str = config.LSI_MODEL_PATH,
                 num_topics: int = config.NUM_TOPICS,
                 tfid_model_path: str = config.TFID_MODEL_PATH,
                 to_remove_from_train: tuple = ('headers', 'footers', 'quotes'),
                 index_path=config.INDEX_PATH,
                 num_result_to_take=config.NUM_SEARCH_RESULTS_TAKE,
                 data_path=config.DATA_PATH) -> None:
        self.data_path = data_path
        self.num_result_to_take = num_result_to_take
        self.index_path = index_path
        self.query_arg = config.QUERY_ARG
        self.to_remove_from_train = to_remove_from_train
        self.tfid_model_path = tfid_model_path
        self.num_topics = num_topics
        self.lsi_model_path = lsi_model_path
        self.corpus_path = corpus_path
        self.no_above = no_above
        self.no_below = no_below
        self.newsgroup_dict_path = newsgroup_dict_path
        self.create_data_path_if_not_exit()

        logging.info("*** Start loading model ***")
        self.newsgroups_train = fetch_20newsgroups(subset='train', remove=self.to_remove_from_train)
        self.texts = self.get_data(self.newsgroups_train)
        self.corpora_dict = self.load_dict()
        self.corpus = self.load_corpus(self.corpora_dict, self.texts)
        self.tfidf = self.load_tfid_model(self.corpus)
        self.corpus_tfidf = self.tfidf[self.corpus]
        self.lsi = self.load_lsi_model(self.corpus_tfidf, self.corpora_dict)
        self.corpus_lsi = self.lsi[self.corpus_tfidf]
        self.index = self._load_index()

    def load_dict(self) -> corpora.Dictionary:
        logging.info("*** Start loading dictionary ***")
        dict_path = Path(self.newsgroup_dict_path)
        if dict_path.exists():
            corpora_dict = self.load_newsgroup_dict()
        else:
            # frequency = calculate_token_freq(texts)
            # texts = [[token for token in text if frequency[token] > 1]
            #          for text in texts]
            corpora_dict = self.convert_data_to_dict(self.texts)
            # TODOx look inside
            corpora_dict = self.preprocess_dict(corpora_dict)
            corpora_dict.save(self.newsgroup_dict_path)
        return corpora_dict

    def load_newsgroup_dict(self):
        # TODOx decouple
        dictionary = corpora.Dictionary()
        return dictionary.load(self.newsgroup_dict_path)

    @staticmethod
    def convert_data_to_dict(texts) -> corpora.Dictionary:
        # TODOx decouple
        dictionary = corpora.Dictionary(texts)
        return dictionary

    def preprocess_dict(self, dictionary: corpora.Dictionary) -> corpora.Dictionary:
        # TODOx decouple
        dictionary.filter_n_most_frequent(5)
        # fixmeX when NO_BELOW or NO_ABOVE change, this will not automatically recreate dict
        dictionary.filter_extremes(no_below=self.no_below, no_above=self.no_above)
        return dictionary

    def load_corpus(self, corpora_dict: corpora.Dictionary, texts: List[List[str]]) -> List[List[Tuple[int, int]]]:
        logging.info("*** Start loading corpus ***")
        corpus_path = Path(self.corpus_path)
        if corpus_path.exists():
            corpus: List[List[Tuple[int, int]]] = corpora.MmCorpus(self.corpus_path)
        else:
            corpus = [corpora_dict.doc2bow(text) for text in texts]
            corpora.MmCorpus.serialize(self.corpus_path, corpus)
        return corpus

    def load_lsi_model(self, corpus_tfidf, corpora_dict):
        logging.info("*** Start loading LSI model ***")
        lsi_model_path = Path(self.lsi_model_path)
        if lsi_model_path.exists():
            lsi = models.LsiModel.load(self.lsi_model_path)
        else:
            lsi = models.LsiModel(corpus_tfidf, id2word=corpora_dict,
                                  num_topics=self.num_topics)  # initialize an LSI transformation
            lsi.save(self.lsi_model_path)
        return lsi

    def load_tfid_model(self, corpus: List[List[Tuple[int, int]]]) -> models.tfidfmodel.TfidfModel:
        logging.info("*** Start loading tfid model ***")
        tfid_model_path = Path(self.tfid_model_path)
        if tfid_model_path.exists():
            tfidf = models.TfidfModel.load(self.tfid_model_path)
        else:
            tfidf = models.TfidfModel(corpus)
            tfidf.save(self.tfid_model_path)
        return tfidf

    def _load_index(self):
        logging.info("*** Start loading index ***")
        index_path = Path(self.index_path)
        if index_path.exists():
            index = similarities.MatrixSimilarity.load(self.index_path)
        else:
            # fixmeX the case when NUM_TOPICS change
            index = similarities.MatrixSimilarity(self.corpus_lsi, num_features=self.num_topics)
            index.save(self.index_path)
        return index

    def project_query_to_lsi_space(self, query: str):
        logging.info("*** Start project query to LSI space ***")
        vec_bow = self.corpora_dict.doc2bow(query.lower().split())
        vec_lsi = self.lsi[vec_bow]
        return vec_lsi

    def _find_top_n_results(self, index, vec_lsi):
        sims = index[vec_lsi]
        sims = sorted(enumerate(sims), key=lambda item: -item[1])
        top_results = sims[slice(0, self.num_result_to_take)]
        return top_results

    def find_top_n_articles(self, query: str):
        logging.info("*** Start finding top articles ***")
        top_results = self._find_top_n_results(self.index, self.project_query_to_lsi_space(query))
        top_articles = []
        for index, result in enumerate(top_results):
            article_id = result[0]
            article_score = result[1]
            if index == 0 and article_score == 0:
                break
            article = self.newsgroups_train.data[article_id]
            top_articles.append((article, float(article_score)))
        return top_articles


Argument = collections.namedtuple("Argument", "query debug port host")


def argument_to_string(argument: Argument):
    return "query=%s\tdebug=%s\tport=%s\thost=%s" % (argument.query, argument.debug, argument.port, argument.host)


def process_args() -> Argument:
    logging.info("Start processing args")
    parser = argparse.ArgumentParser()
    parser.add_argument(config.QUERY_ARG)
    parser.add_argument(config.DEBUG_ARG, default=False, action='store_true')
    parser.add_argument(config.PORT_ARG, default=config.DEFAULT_PORT)
    parser.add_argument(config.HOST_ARG, default=config.DEFAULT_HOST)
    args = parser.parse_args()
    query = args.query
    debug = args.debug
    port = args.port
    host = args.host
    argument = Argument(query=query, debug=debug, port=port, host=host)
    logging.info('Argument used in this run = "%s"' % argument_to_string(argument))
    return argument


if __name__ == '__main__':
    load_model = LoadModel()
    args = process_args()

    if args.query:
        top_articles = load_model.find_top_n_articles(args.query)
        pprint(top_articles)
        pass
