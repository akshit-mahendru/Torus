#----------------------Imports-----------------------------------
from flask import Blueprint, render_template,  request, jsonify, url_for
import re
from bs4 import BeautifulSoup
import base64
from ressearch import *
#import research.py (rename to model)

views = Blueprint(__name__, "views")

@views.route("/home")
def home():
#Home page
    return render_template("index.html")

@views.route("/visualise")
def visualisations():
#where we actually dsipaly the visualtions
    return render_template("visualise.html")

def sanitize_input(user_input):
    # Split the input by commas and remove any leading/trailing whitespaces and consecutive black spaces
    parts = [part.strip() for part in re.split(r',\s*', user_input) if part.strip()]
    # Remove duplicates using a set
    unique_parts = list(set(parts))
    return unique_parts    

@views.route("/process_text", methods=['POST'])
def process_text():
    data = request.get_json()
    text = data['text']
    response = sanitize_input(text)
    #Passs this response into research
    keyword_list_1 = response
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
    plotly.offline.plot(fig1, filename='/Users/akshit/Downloads/Durhack/static/fig1.html')

    # Run the visualization with the original embeddings
    #topic_model_1.visualize_documents(sentences_filtered, embeddings=embeddings)

    # Reduce dimensionality of embeddings, this step is optional but much faster to perform iteratively:
    reduced_embeddings = UMAP(n_neighbors=10, n_components=2, min_dist=0.0, metric='cosine').fit_transform(embeddings)
    fig2 = topic_model_1.visualize_documents(sentences_filtered, reduced_embeddings=reduced_embeddings) 
    plotly.offline.plot(fig2, filename='/Users/akshit/Downloads/Durhack/static/fig2.html')   
    
    #topics_over_time = topic_model_1.topics_over_time(sentences_filtered, date)
    
    fig3 = topic_model_1.visualize_heatmap()
    plotly.offline.plot(fig3, filename='/Users/akshit/Downloads/Durhack/static/fig3.html')
    
    
    
    #It will generate graphs
    #On detection of these graphs, by sending jsonify to get the ipython not hidden, will do with css opacity on div over whole thing
    #Would be good, delete the graphs upon finishing
