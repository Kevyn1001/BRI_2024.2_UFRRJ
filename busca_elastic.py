from dotenv import dotenv_values
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
import re
import timeit
import matplotlib.pyplot as plt
import numpy as np
import time
from collections import Counter
from nltk.corpus import stopwords
import nltk

INDEX_NAME = 'cran'

config = dotenv_values('.env')

es_client = Elasticsearch(
    config['elasticsearch_uri'],
    api_key=config['api_key']
)

class Text:
    def __init__(self, original):
        result = re.split(r'.T|.A|.B|.W', original.replace('\n', ' '))
        self.index, self.title, self.author, self.bibliography, self.body, *_ = result

    def source_dict(self):
        source = self.__dict__.copy()
        source.pop('index')
        return source

    def to_index_dict(self, index_name):
        return {
            '_op_type': 'index',
            '_index': index_name,
            '_id': int(self.index),
            '_source': self.source_dict()
        }

class Query:
    def __init__(self, text):
        result = text.split('\n.W\n')
        self.index, self.body = map(lambda x: x.strip().replace('\n', ' '), result)

def parse_queries(filename):
    queries = []
    with open(filename, 'r') as file:
        txt = file.read()
        txt = txt.split('.I')[1:]
        queries = list(map(lambda x: Query(x), txt))
    return queries

def parse_text(filename):
    words = []
    with open(filename, 'r') as file:
        txt = file.read()
        txt = txt.split('.I')[1:]
        words = list(map(lambda x: Text(x), txt))
    return words

def parse_to_bulk(filename, index_name):
    return list(map(lambda x: x.to_index_dict(index_name), parse_text(filename)))

def get_ordered_relevant_searches(filename):
    query_relations = {}
    with open(filename, 'r') as file:
        txt = file.read()
        txt = txt.strip().split('\n')
        for i in txt:
            query, abstract, score = map(
                lambda x: int(x),
                filter(lambda x: len(x) > 0, i.strip().split(' '))
            )
            if query - 1 not in query_relations:
                query_relations[query - 1] = [(abstract, score)]
            else:
                query_relations[query - 1].append((abstract, score))
    for i in query_relations:
        query_relations[i].sort(key=lambda x: x[1])
    return query_relations

def search_results(index_name, client: Elasticsearch, queries: list[Query], limits: list[int], fields: list[str]):
    results_dict = {}
    for i, (limit, query) in enumerate(zip(limits, queries)):
        body = {
            'query': {
                "multi_match": {
                    "query": query.body,
                    "fields": fields
                }
            },
            'size': max(limit, 10)
        }
        response = client.search(index=index_name, body=body)
        results_dict[i] = list(
            map((lambda x: (int(x['_id']), x['_score'])),
                response['hits']['hits']))
    return results_dict

def search_results_one_field(index_name, client: Elasticsearch, queries: list[Query], limits: list[int], field: str):
    results_dict = {}
    for i, (limit, query) in enumerate(zip(limits, queries)):
        response = client.search(index=index_name,
                                 size=max(limit, 10),
                                 query={
                                     "match": {
                                         field: query.body,
                                     },
                                 })
        results_dict[i] = list(
            map((lambda x: (int(x['_id']), x['_score'])),
                response['hits']['hits']))
    return results_dict

def precision_at_k(answer, relevant, k=None):
    if k is None or k > len(answer) or k > len(relevant):
        k = min(len(answer), len(relevant))
    result = len(set(answer[:k]) & set(relevant)) / k if k != 0 else 0
    return result

def recall_at_k(answer, relevant, k=None):
    if k is None or k > len(answer) or k > len(relevant):
        k = min(len(answer), len(relevant))
    d = len(relevant)
    return len(set(answer[:k]) & set(relevant)) / d if d != 0 else 0

def all_results_by_func(answer_results_dict: dict, relevant_results_dict: dict, func, k: int = None):
    size = len(relevant_results_dict)
    results = np.zeros(size)
    for i, (answer, relevant) in enumerate(zip(answer_results_dict.values(), relevant_results_dict.values())):
        answer = list(map(lambda x: x[0], answer))
        relevant = list(map(lambda x: x[0], relevant))
        results[i] = func(answer, relevant, k)
    return results

def plot_results(answer_results_dict, relevant_results_dict, func, k_s=range(1, 10 + 1), title: str = ''):
    results = list(
        map(
            lambda k: all_results_by_func(answer_results_dict, relevant_results_dict, func, k).mean(),
            k_s))
    plt.plot(k_s, results, marker='o')
    plt.xlabel('K')
    plt.ylabel(f'{func.__name__} mean')
    plt.title(title)
    plt.show()

queries = parse_queries('cran/cran.qry')
words = parse_to_bulk('cran/cran.all.1400', INDEX_NAME)
relevant_dict = get_ordered_relevant_searches('cran/cranqrel')

nltk.download('stopwords')

def calculate_word_frequencies(filename):
    with open(filename, 'r') as file:
        text = file.read().lower()
        words = re.findall(r'\b\w+\b', text)
        return Counter(words)

def plot_word_distribution(word_freq):
    sorted_freq = sorted(word_freq.values(), reverse=True)
    plt.plot(sorted_freq, marker='o')
    plt.xlabel('Ranking das Palavras')
    plt.ylabel('Frequência')
    plt.title('Distribuição das Palavras no Corpus')
    plt.yscale('log')
    plt.xscale('log')
    plt.show()

def analyze_corpus(word_freq):
    vocab_size = len(word_freq)
    print(f'Tamanho do vocabulário: {vocab_size}')
    total_words = sum(word_freq.values())
    print(f'Total de palavras na coleção: {total_words}')
    most_common = word_freq.most_common(10)
    print('10 palavras mais frequentes e suas frequências:')
    for word, freq in most_common:
        print(f'{word}: {freq}')
    least_common = list(filter(lambda x: x[1] == 1, word_freq.items()))[:10]
    print('10 palavras menos frequentes e suas frequências:')
    for word, freq in least_common:
        print(f'{word}: {freq}')
    stop_words = set(stopwords.words('english'))
    stopword_freq = {word: freq for word, freq in word_freq.items() if word in stop_words}
    most_common_stopwords = Counter(stopword_freq).most_common(10)
    least_common_stopwords = list(filter(lambda x: x[1] == 1, stopword_freq.items()))[:10]
    print('Stopwords mais frequentes:')
    for word, freq in most_common_stopwords:
        print(f'{word}: {freq}')
    print('Stopwords menos frequentes:')
    for word, freq in least_common_stopwords:
        print(f'{word}: {freq}')
    print('\nAs stopwords, sendo palavras comuns e geralmente irrelevantes (como "the", "and"), podem aumentar o ruído nas buscas. '
          'Removê-las pode melhorar a precisão dos resultados, especialmente em consultas curtas.')

word_frequencies = calculate_word_frequencies('cran/cran.all.1400')
plot_word_distribution(word_frequencies)
analyze_corpus(word_frequencies)

def measure_preprocessing_time(filename):
    start_time = timeit.default_timer()
    words = parse_text(filename)
    end_time = timeit.default_timer()
    return (end_time - start_time), words

def measure_full_indexing_time(filename, index_name, client):
    start_time = timeit.default_timer()
    words = parse_text(filename)
    bulk_data = list(map(lambda x: x.to_index_dict(index_name), words))
    client.indices.create(index=index_name, ignore=400)
    bulk(client=client, actions=bulk_data)
    while int(client.cat.indices(index=index_name, format='json')[0]['docs.count']) != len(words):
        time.sleep(0.2)
    end_time = timeit.default_timer()
    return (end_time - start_time)

filename = 'cran/cran.all.1400'
preprocessing_time, processed_words = measure_preprocessing_time(filename)
print(f"Tempo médio de pré-processamento dos documentos: {preprocessing_time:.2f}s")
full_time = measure_full_indexing_time(filename, INDEX_NAME, es_client)
print(f"Tempo médio de pré-processamento, representação e indexação: {full_time:.2f}s")
es_client.indices.delete(index=INDEX_NAME, ignore=[400, 404])
t0 = timeit.default_timer()
es_client.indices.create(index=INDEX_NAME)
bulk(client=es_client, actions=words)
while int(es_client.cat.indices(index=INDEX_NAME, format='json')[0]['docs.count']) != len(words):
    time.sleep(0.2)
t1 = timeit.default_timer()
print(f'TEMPO DE INDEXAÇÃO ELASTICSEARCH = {(t1 - t0):.2f}s')

limits = list(map(lambda x: len(x), relevant_dict.values()))
t0 = timeit.default_timer()
results_1 = search_results(INDEX_NAME, es_client, queries, limits, ["title", "author", "body"])
t1 = timeit.default_timer()
print(f'TEMPO DA BUSCA 1 ELASTICSEARCH = {(t1 - t0):.2f}s')
t0 = timeit.default_timer()
results_2 = search_results_one_field(INDEX_NAME, es_client, queries, limits, "body")
t1 = timeit.default_timer()
print(f'TEMPO DA BUSCA 1 ELASTICSEARCH = {(t1 - t0):.2f}s')

K_S = range(1, 11)
plot_results(results_1, relevant_dict, precision_at_k, k_s=K_S, title='Média de Precision@k da busca 1 x k (Elasticsearch)')
plot_results(results_1, relevant_dict, recall_at_k, k_s=K_S, title='Média de Recall@k da busca 1 x k (Elasticsearch)')
plot_results(results_2, relevant_dict, precision_at_k, k_s=K_S, title='Média de Precision@k da busca 2 x k (Elasticsearch)')
plot_results(results_2, relevant_dict, recall_at_k, k_s=K_S, title='Média de Recall@k da busca 2 x k (Elasticsearch)')

es_client.indices.delete(index=INDEX_NAME)
