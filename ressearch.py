#from __future__ import print_function
import sys
import arxiv
import pandas as pd
import numpy as np # for array manipulation
import json # for reading in Data
from itertools import islice # for slicing and dicing JSON records
import os # for getting the filepath information
import re # to identify characters that are to be removed
import nltk # for preprocessing of textual data
from nltk.corpus import stopwords # for removing stopwords
from nltk.tokenize import word_tokenize # for tokenizing text
from nltk.stem import WordNetLemmatizer # for lemmatizing text
from sklearn.feature_extraction.text import TfidfVectorizer # for featurizing text
from sklearn.metrics.pairwise import cosine_similarity # for getting similarity score
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA #for dimensionality reduction
from sklearn.cluster import KMeans #for clustering
from sklearn.manifold import TSNE
import plotly.express as px 
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
import plotly.io as pio
from nltk.tokenize import sent_tokenize, word_tokenize
nltk.download('punkt')
from nltk.corpus import stopwords
nltk.download('stopwords')
from nltk.tokenize import word_tokenize
from gensim.parsing.preprocessing import remove_stopwords 
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
#from bertopic import BERTopic
from umap import UMAP
#import kaleido
import plotly
import plotly.express as px

#@title Imports related to arXiv search
import logging
import time
import re
import feedparser

try:
    # Python 2
    from urllib import urlencode
    from urllib import urlretrieve
except ImportError:
    # Python 3
    from urllib.parse import urlencode
    from urllib.request import urlretrieve

logger = logging.getLogger(__name__)


class Search(object):
    """
    Class to search and download abstracts from the arXiv
    Args:
        query (string):
        id_list (list): List of arXiv object IDs.
        max_results (int): The maximum number of abstracts that should be downloaded. Defaults to
            infinity, i.e., no limit at all.
        start (int): The offset of the first returned object from the arXiv query results.
        sort_by (string): The arXiv field by which the result should be sorted.
        sort_order (string): The sorting order, i.e. "ascending", "descending" or None.
        max_chunk_results (int): Internally, a arXiv search query is split up into smaller
            queries that download the data iteratively in chunks. This parameter sets an upper
            bound on the number of abstracts to be retrieved in a single internal request.
        time_sleep (int): Time (in seconds) between two subsequent arXiv REST calls. Defaults to
            :code:`3`, the recommendation of arXiv.
        prune (bool): Whether some of the values in each response object should be dropped.
            Defaults to True.
    """

    root_url = 'http://export.arxiv.org/api/'
    prune_keys = [
        'updated_parsed',
        'published_parsed',
        'arxiv_primary_category',
        'summary_detail',
        'author',
        'author_detail',
        'links',
        'guidislink',
        'title_detail',
        'tags',
        'id']

    def __init__(self, query=None, id_list=None, max_results=None, start=0, sort_by=None,
                 sort_order=None, max_chunk_results=None, time_sleep=3, prune=True):

        self.query = query
        self.id_list = id_list
        self.sort_by = sort_by
        self.sort_order = sort_order
        self.max_chunk_results = max_chunk_results
        self.time_sleep = time_sleep
        self.prune = prune
        self.max_results = max_results
        self.start = start

        if not self.max_results:
            logger.info('max_results defaulting to inf.')
            self.max_results = float('inf')

    def _get_url(self, start=0, max_results=None):

        url_args = urlencode(
            {
                "search_query": self.query,
                "id_list": self.id_list,
                "start": start,
                "max_results": max_results,
                "sortBy": self.sort_by,
                "sortOrder": self.sort_order
            }
        )

        return self.root_url + 'query?' + url_args

    def _parse(self, url):
        """
        Downloads the data provided by the REST endpoint given in the url.
        """
        result = feedparser.parse(url)

        if result.get('status') != 200:
            logger.error(
                "HTTP Error {} in query".format(result.get('status', 'no status')))
            return []
        return result['entries']

    def _prune_result(self, result):
        """
        Deletes some of the keys from the downloaded result.
        """

        for key in self.prune_keys:
            try:
                del result['key']
            except KeyError:
                pass

        return result

    def _process_result(self, result):

        # Useful to have for download automation
        result['pdf_url'] = None
        for link in result['links']:
            if 'title' in link and link['title'] == 'pdf':
                result['pdf_url'] = link['href']
        result['affiliation'] = result.pop('arxiv_affiliation', 'None')

        result['arxiv_url'] = result.pop('link')
        result['title'] = result['title'].rstrip('\n')
        result['summary'] = result['summary'].rstrip('\n')
        result['authors'] = [d['name'] for d in result['authors']]
        if 'arxiv_comment' in result:
            result['arxiv_comment'] = result['arxiv_comment'].rstrip('\n')
        else:
            result['arxiv_comment'] = None
        if 'arxiv_journal_ref' in result:
            result['journal_reference'] = result.pop('arxiv_journal_ref')
        else:
            result['journal_reference'] = None
        if 'arxiv_doi' in result:
            result['doi'] = result.pop('arxiv_doi')
        else:
            result['doi'] = None

        if self.prune:
            result = self._prune_result(result)

        return result

    def _get_next(self):

        n_left = self.max_results
        start = self.start

        while n_left > 0:

            if n_left < self.max_results:
                logger.info('... play nice on the arXiv and sleep a bit ...')
                time.sleep(self.time_sleep)

            logger.info('Fetch from arxiv ({} results left to download)'.format(n_left))
            url = self._get_url(
                start=start,
                max_results=min(n_left, self.max_chunk_results))

            results = self._parse(url)

            # Update the entries left to download
            n_fetched = len(results)
            logger.info('Received {} entries'.format(n_fetched))

            if n_fetched == 0:
                logger.info('No more entries left to fetch.')
                logger.info('Fetching finished.')
                break

            # Update the number of results left to download
            n_left = n_left - n_fetched
            start = start + n_fetched

            # Process results
            results = [self._process_result(r) for r in results if r.get("title", None)]

            yield results

    def download(self, iterative=False):
        """
        Triggers the download of the result of the given search query.
        Args:
            iterative (bool): If true, then an iterator is returned, which allows to download the
                data iteratively. Otherwise, all the data is fetched first and then returned.
        Returns:
            iterable: Either a list or a general iterator holding the result of the search query.
        """
        logger.info('Start downloading')
        if iterative:

            logger.info('Build iterator')

            def iterator():
                logger.info('Start iterating')
                for result in self._get_next():
                    for entry in result:
                        yield entry
            return iterator
        else:
            results = list()
            for result in self._get_next():
                # Only append result if title is not empty
                results = results + result
            return results


def query(query="", id_list=[], prune=True, max_results=None, start=0, sort_by="relevance",
          sort_order="descending", max_chunk_results=1000, iterative=False,
          time_sleep=3):
    """
    See :py:class:`arxiv.Search` for a description of the parameters.
    """

    search = Search(
        query=query,
        id_list=','.join(id_list),
        sort_by=sort_by,
        sort_order=sort_order,
        prune=prune,
        max_results=max_results,
        start = start,
        max_chunk_results=max_chunk_results, 
        time_sleep=time_sleep)
    
    

    return search.download(iterative=iterative)


def slugify(obj):
    # Remove special characters from object title
    filename = '_'.join(re.findall(r'\w+', obj.get('title', 'UNTITLED')))
    # Prepend object id
    filename = "%s.%s" % (obj.get('pdf_url').split('/')[-1], filename)
    return filename


def download(obj, dirpath='./', slugify=slugify, prefer_source_tarfile=False):
    """
    Download the .pdf corresponding to the result object 'obj'. If prefer_source_tarfile==True, download the source .tar.gz instead.
    """
    if not obj.get('pdf_url', ''):
        print("Object has no PDF URL.")
        return
    if dirpath[-1] != '/':
        dirpath += '/'
    if prefer_source_tarfile:
        url = re.sub(r'/pdf/', "/src/", obj['pdf_url'])
        path = dirpath + slugify(obj) + '.tar.gz'
    else:
        url = obj['pdf_url']
        path = dirpath + slugify(obj) + '.pdf'

    urlretrieve(url, path)
    return path

#@title Imports related to semantic_scholar search

import requests
from tenacity import (retry,
                      wait_fixed,
                      retry_if_exception_type,
                      stop_after_attempt)

API_URL = 'https://api.semanticscholar.org/v1'


def semantic_paper(id, timeout=2, include_unknown_references=False) -> dict:

    '''Paper lookup
    :param str id: S2PaperId, DOI or ArXivId.
    :param float timeout: an exception is raised
        if the server has not issued a response for timeout seconds
    :param bool include_unknown_references :
        (optional) include non referenced paper.
    :returns: paper data or empty :class:`dict` if not found.
    :rtype: :class:`dict`
    '''

    data = __get_data('paper', id, timeout, include_unknown_references)

    return data


def semantic_author(id, timeout=2) -> dict:

    '''Author lookup
    :param str id: S2AuthorId.
    :param float timeout: an exception is raised
        if the server has not issued a response for timeout seconds
    :returns: author data or empty :class:`dict` if not found.
    :rtype: :class:`dict`
    '''

    data = __get_data('author', id, timeout)

    return data


def __get_data(method, id, timeout, include_unknown_references=False) -> dict:

    '''Get data from Semantic Scholar API
    :param str method: 'paper' or 'author'.
    :param str id: id of the correponding method
    :param float timeout: an exception is raised
        if the server has not issued a response for timeout seconds
    :returns: data or empty :class:`dict` if not found.
    :rtype: :class:`dict`
    '''

    data = {}
    method_types = ['paper', 'author']
    if method not in method_types:
        raise ValueError(
            'Invalid method type. Expected one of: {}'.format(method_types))

    url = '{}/{}/{}'.format(API_URL, method, id)
    if include_unknown_references:
        url += '?include_unknown_references=true'
    r = requests.get(url, timeout=timeout)

    if r.status_code == 200:
        data = r.json()
        if len(data) == 1 and 'error' in data:
            data = {}
    elif r.status_code == 429:
        raise ConnectionRefusedError('HTTP status 429 Too Many Requests.')

    return data

#@title More imports related to semantic scholar search
# https://api.semanticscholar.org/
# semantic scholar search by arxiv id then return references
def semantic_recursive(max_level, recursive_list, titles_list, 
paper_data, curr_level):
  time.sleep(3)
  # verify the keys are appropriate before appending
  try:
    paper = semantic_paper('{}'.format(paper_data['paperId']), timeout=2)
    paper['no_citations'] = len(paper['citations'])
    recursive_list.append(paper)
    titles_list.append(paper['title'])
    print(curr_level, paper['title'])
    #print(paper['title'])

    if curr_level < max_level:
      for ref in paper['references']:
        if any(keyword in ref['title'].lower() for keyword in keyword_list_1):
          if any(keyword in ref['title'].lower() for keyword in keyword_list_2):
            if (ref['title'] not in titles_list):
              semantic_recursive(max_level=max_level, recursive_list=recursive_list, 
              titles_list=titles_list, paper_data=ref, curr_level=curr_level+1)
  except:
    print('Paper cannot be retrieved in proper format. Skipping to next.')

def extract_authors(df):

  authors_list_namesonly = []
  for authors_list in df['authors']:
    curr_paper_authors_list = []
    
    for author in authors_list:
      curr_paper_authors_list.append(author['name'])
      
    curr_paper_authors = ', '.join(curr_paper_authors_list)
    authors_list_namesonly.append(curr_paper_authors)
  
  return authors_list_namesonly


'''if __name__ == "__main__":
    #title = 'fine_grained'
    #keyword_list_1 = ['storage', 'in-storage', 'on-storage', 'near-storage', 'in storage', 'on storage' ,'near storage', 'ssd', 'solid state drive', 'flash']
    #keyword_list_2 = ['compute', 'computing', 'processing', 'training', 'acceleration', 'inference', 'transformer', 'attention', 'self-attention']
    #keyword_list_1 = ['weakly supervised', 'weakly-supervised', 'wsod', 'weak supervision', 'image-level supervision', 'weak label', 'weakly labeled']
    #keyword_list_2 = ['detect', 'localization' , 'recognition', 'segment', 'network']

    # keyword_list = ['fine-grained image', 'fine grained image', 'fine-grained visual', 'fine grained visual', 'fine-grained object', 'fine grained object', 'cascaded image', 'hierarchical image', 'cascaded visual', 'hierarchical visual']
    keyword_list_1 = ['AIDS', 'HIV', 'Tumour']
    keyword_list_2 = []
    # basic search query style
    search_query_basic_1 = ' OR '.join(keyword_list_1)
    search_query_basic_2 = ' OR '.join(keyword_list_2)
    search_query_basic = '(' + search_query_basic_1 + ') AND ('+ search_query_basic_2 + ')'
    #print(search_query_basic)

    # arxiv queries
    # https://arxiv.org/help/api/basics
    # https://arxiv.org/help/api/user-manual
    # https://github.com/lukasschwab/arxiv.py
    # https://github.com/titipata/arxivpy/wiki
    # ti: title, abs: abstract
    search_query_arxiv_1 = 'ti:%22' + 'abs:%22' + r'%22%20OR%20ti:%22'.join(keyword_list_1) + '%22'
    search_query_arxiv_2 = 'ti:%22' + 'abs:%22' + r'%22%20%20OR%20ti:%22'.join(keyword_list_2) + '%22'
    search_query_arxiv = '%28' + search_query_arxiv_1 + r'%29%20AND%20%28'+ search_query_arxiv_2 + '%29'
    #print(search_query_arxiv)
    
    result_raw_arxiv = query(
        query="{}".format(search_query_arxiv),
        max_chunk_results=2000,
        max_results=None,
        iterative=False,
        prune=True,
        sort_by="submittedDate",
        sort_order='descending',
        time_sleep=2
    )
    result_df_arxiv = pd.DataFrame(result_raw_arxiv)

    result_df_arxiv = result_df_arxiv[['id', 'published', 'title', 'authors', 'summary']]

    #print(result_df_arxiv)

    temp = result_df_arxiv.copy()

    result_df_arxiv['paperId'] = 'arXiv:' + result_df_arxiv['id'].str.split('abs/').str[1].str.split('v').str[0].astype(str)
    result_df_arxiv = result_df_arxiv[~result_df_arxiv['title'].str.contains("animal", case=False)]
    #result_df_arxiv = result_df_arxiv

    #result_df_arxiv.to_csv('prelim_results_arxiv_{}.csv'.format(title), index=False)
    #print(len(result_df_arxiv))
    #result_df_arxiv.head()
    
    abst = result_df_arxiv['summary'].tolist()
    #col_one_list = df['one'].tolist()

    id = result_df_arxiv['id'].str.split('abs/').str[1].str.split('v').str[0].astype(str)
    date = result_df_arxiv['published']
    
    #print(len(abst))
    

    sentences = [sent_tokenize(a) for a in abst]
    sentences = [sentence for doc in sentences for sentence in doc]
    

    stop_words = set(stopwords.words('english'))
    
    filtered = []
    for sen in abst:
        temp = sen.split(" ")
        t = []
        for i in temp:
            if i not in stop_words:
                t.append(i)
        filtered.append(' '.join(t))
    
    sentences_filtered = []
    
    for i in filtered:
        temp = re.sub('\n', "", i)

        filtered_words = remove_stopwords(temp)
        sentences_filtered.append(re.sub('(?<=\D)[.,]|[.,](?=\D)', '', filtered_words))
        
    
    topic_model = BERTopic(verbose=True, embedding_model="paraphrase-MiniLM-L12-v2", min_topic_size=60)
    topics, _ = topic_model.fit_transform(sentences_filtered)
    
    sentence_model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = sentence_model.encode(sentences_filtered, show_progress_bar=True)
    

    # Train BERTopic
    topic_model_1 = BERTopic().fit(sentences_filtered, embeddings)
    
    #print(topics)
    #topic_model.update_topics(sentences_filtered, n_gram_range=(1, 5))
    #fig = topic_model.visualize_topics()
    fig1 = topic_model_1.visualize_barchart(top_n_topics=9, height=700)
    plotly.offline.plot(fig1, filename='/Users/akshit/fig1.html')

    # Run the visualization with the original embeddings
    #topic_model_1.visualize_documents(sentences_filtered, embeddings=embeddings)

    # Reduce dimensionality of embeddings, this step is optional but much faster to perform iteratively:
    reduced_embeddings = UMAP(n_neighbors=10, n_components=2, min_dist=0.0, metric='cosine').fit_transform(embeddings)
    fig2 = topic_model_1.visualize_documents(sentences_filtered, reduced_embeddings=reduced_embeddings) 
    plotly.offline.plot(fig2, filename='/Users/akshit/fig2.html')   
    
    #topics_over_time = topic_model_1.topics_over_time(sentences_filtered, date)
    
    fig3 = topic_model_1.visualize_heatmap()
    plotly.offline.plot(fig3, filename='/Users/akshit/fig3.html')
'''