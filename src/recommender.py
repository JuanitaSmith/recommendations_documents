import logging
import re
import warnings
from pickle import dump, load
from typing import Dict, List, Set

import numpy as np
import pandas as pd

import nltk
nltk.download('averaged_perceptron_tagger_eng')
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from pandas import DataFrame
from scipy.sparse.linalg import svds
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src import path_log, path_tfidf, path_tfidf_df

# suppress warnings
warnings.filterwarnings('ignore')


class Recommender:

    def __init__(self, activate_logger=True):

        self.tfidf_vectorizer = None
        self.tfidf_df = pd.DataFrame()
        self.df_content = pd.DataFrame()
        self.df_interactions = pd.DataFrame()

        # activate logging
        print('Activating logger...')
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(filename=path_log,
                            format='%(asctime)s %(levelname)-8s %(message)s',
                            filemode='w',
                            level=logging.INFO,
                            datefmt='%Y-%m-%d %H:%M:%S')

    def get_data(
            self,
            filename_content: str,
            filename_interactions: str):
        """
        Load csv files into a pandas dataframe

        Input:
        filename_content (str): path to content csv file
        filename_interactions (str): path to user item interactions csv file

        Output:
        df (DataFrame): pandas dataframe of user document interactions
        df_article_content (DataFrame): pandas dataframe with article contents
        """

        print('Getting data...')
        self.logger.info(f'Loading {filename_content}...')
        self.logger.info(f'Loading {filename_interactions}...')

        dtype_dict = {'article_id': int}
        self.df_interactions = pd.read_csv(
            filename_interactions,
            dtype=dtype_dict)

        self.df_content = pd.read_csv(
            filename_content,
            index_col='article_id')

        del self.df_interactions['Unnamed: 0']
        del self.df_content['Unnamed: 0']

        self.logger.info('Article contents loaded with shape {}'.format(
            self.df_content.shape))
        self.logger.info('User interactions loaded with shape {}'.format(
            self.df_interactions.shape))

    def clean_contents(self):
        """ Clean article contents dataset """

        print('Cleaning article contents...')

        # remove rows with duplicate indexes
        self.df_content = (
            self.df_content)[~self.df_content.index.duplicated(keep='last')]

        # drop documents without a body or description
        self.df_content.dropna(
            subset=['doc_description', 'doc_body'],
            inplace=True)

        # drop doc_status as it has only 1 unique value
        self.df_content = self.df_content.drop('doc_status', axis=1)

        # for content-based recommendations, merge title, description and body
        self.df_content['doc_body_all'] = (
            self.df_content[self.df_content.columns[:]].apply(
                lambda x: ','.join(x.astype(str)), axis=1))

        self.logger.info(
            'Content dataset cleaned, shape {} and columns {}'.format(
                self.df_content.shape,
                self.df_content.columns.tolist()))

    def clean_interactions(self):
        """ Clean interaction dataset by converting email to userid"""

        print('Cleaning interactions...')

        # delete interactions not present in content dataset
        content_ids = self.df_content.index.tolist()
        self.df_interactions = self.df_interactions[
            self.df_interactions['article_id'].isin(content_ids)]

        email_encoded = self._email_mapper(self.df_interactions)
        del self.df_interactions['email']
        self.df_interactions['user_id'] = email_encoded

        self.logger.info(
            'Interaction dataset cleaned, shape {} an columns {}'.format(
                self.df_interactions.shape,
                self.df_interactions.columns.tolist()))

    def _email_mapper(self, df: DataFrame) -> List[int]:
        """ Convert column `email` to `user_id` using incremental integers """

        coded_dict = dict()
        counter = 1
        email_encoded = []

        for val in df['email']:
            if val not in coded_dict:
                coded_dict[val] = counter
                counter += 1

            email_encoded.append(coded_dict[val])

        return email_encoded

    def get_contents(self, df: DataFrame) -> Dict:
        """
        Merge top articles with all details of df_contents

        To print document cards and their contents in the webapp, we need all
        information about each article we want to recommend

        INPUT:
        df (pd.DataFrame):
        recommendations with at least a column 'article_id',
        and 'num_interactions' for the count interactions of the article.

        OUTPUT:
        df (dictionary):
        key is index in sequential order of importance
        values contain columns:
        count (int): number of interactions of the article
        article_id (int): id of the article
        doc_full_title (str): full title of the article
        doc_description (str): description of the article
        doc_body (str) main text of the article
        """

        top_articles = df.merge(
            self.df_content,
            how='left',
            left_on='article_id',
            right_index=True)

        top_articles.fillna(0, inplace=True)

        top_articles['num_interactions'] = (
            top_articles['num_interactions'].astype(int))

        # reset index to display documents in order of most interactions
        top_articles.reset_index(drop=False, inplace=True)
        top_articles = top_articles.to_dict(orient='index')

        return top_articles

    def get_top_articles(self, n: int) -> Dict:
        """
        Use ranked-based recommendation to find the top n articles
        from interactions and return all its contents

        INPUT:
        n - (int) the number of top articles to return

        OUTPUT:
        top_articles - (dict) Dict containing top n articles with their content
        """

        top_articles = self.df_interactions.article_id.value_counts()
        top_articles = top_articles.to_frame()[:n]
        top_articles.columns = ['num_interactions']

        # add contents to our recommendations
        top_articles = self.get_contents(top_articles)

        # Return the top article
        return top_articles

    def create_user_item_matrix(self):
        """
        INPUT:
        df - pandas dataframe with article_id, title, user_id columns

        OUTPUT:
        user_item - user item matrix

        Description:
        Return a matrix with user ids as rows and article ids on the columns
        with 1 values where a user interacted with an article and a 0 otherwise
        """

        user_item = pd.crosstab(
            self.df_interactions.user_id,
            self.df_interactions.article_id)

        user_item = user_item.where(user_item == 0, 1)

        return user_item

    def get_similar_users(self, user_id, user_item):
        """
        Get most similar users to the input user id, using dot product similarity

        Sort users by the highest similarity score first,
        and most interactions second

        INPUT:
        user_id - (int)
        user_item - (pandas dataframe) matrix of users by articles:
                1's when a user has interacted with an article, 0 otherwise


        OUTPUT:
        neighbors_df - (pandas dataframe) a dataframe with:
                        user_id - a neighbor user_id as index
                        similarity -
                        dot similarity between input users and neighbor user id
                        num_interactions -
                        the number of interactions for neighbor user id

        Other Details -
        sort the neighbors_df by the similarity and then by the
        number of interactions

        """

        # List of user id's that read similar documents to the requested user
        # Compute the similarity of each user to the input user id
        input_user_series = user_item.loc[user_id]
        similarity = input_user_series.dot(np.transpose(user_item))

        # sort user_ids by top interactions
        top_user_interactions = self.df_interactions.user_id.value_counts()

        # merge and sort similarities and interactions
        neighbors_df = pd.concat([similarity, top_user_interactions], axis=1)
        neighbors_df.columns = ['similarity', 'num_interactions']
        neighbors_df.drop(user_id, axis=0, inplace=True)
        neighbors_df.sort_values(by=['similarity', 'num_interactions'],
                                 ascending=False, inplace=True)

        return neighbors_df

    def user_user_recommendations(self, user_id, top_n=10):
        """
        Use user-user collaborative filtering to make document recommendations

        Loops through the users based on closeness to the input user_id
        For each user - finds top n articles the user hasn't seen before and
        provide them as recommendations.

        If a user had read 2 articles or fewer, switch to content-based
        recommendations instead.
        Use the content titles as search term.

        Notes:
        * Choose the users that have the most total article interactions
        before choosing those with fewer article interactions.

        * Choose articles with the articles with the most total interactions
        before choosing those with fewer total interactions.

        INPUT:
        user_id - (int) a user id
        m - (int) the number of recommendations you want for the user

        OUTPUT:
        top_articles - (dict) Dict containing top n articles with their content
        """

        search_text = ''

        # create a user-item matrix
        user_item_matrix = self.create_user_item_matrix()

        # get all the documents input user has read
        docs_read = self.get_documents_read(user_id)

        # if user read 2 documents or fewer, rather do a content-based search
        # using the titles of the documents as a search term
        if len(docs_read) <= 2:
            content_read = self.df_content.loc[
                list(docs_read), 'doc_full_name'].to_frame()
            search_text = re.sub(
                '\s+\.',
                '.',
                '. \n'.join(content_read['doc_full_name']))

            top_articles = self.make_content_recommendations(
                search_text,
                user_id=user_id,
                top_n=top_n
            )

            self.logger.info('User {} has <= 3 interactions, '
                             'switching to content based '
                             'recommendations with text "{}"'.format(
                user_id, search_text))
        else:
            # get the most similar user ids to the input user id
            neighbors_df = self.get_similar_users(
                user_id,
                user_item=user_item_matrix
            )

            # select the top 5 neighbors
            nearest_user_id = neighbors_df[:5].index.values.tolist()

            # reduce the user_item matrix to only nearest neighbors
            neighbors_docs = user_item_matrix.reindex(nearest_user_id)

            # get the documents read the most by neighbors
            # with the highest score
            neighbors_docs = neighbors_docs.sum().sort_values(
                ascending=False)

            # merge and sort similarities and interactions
            top_article_interactions = (
                self.df_interactions.article_id.value_counts())
            similar_articles = pd.concat(
                [neighbors_docs,
                 top_article_interactions],
                axis=1)
            similar_articles.columns = ['similarity', 'num_interactions']
            similar_articles = similar_articles.drop(
                docs_read,
                errors='ignore')
            similar_articles.sort_values(by=['similarity', 'num_interactions'],
                                         ascending=False, inplace=True)

            # add contents to our recommendations
            top_articles = self.get_contents(similar_articles[:top_n])

        return top_articles, search_text

    def tokenize(self, text):
        """ Summarize text into words

        Most important functions:
        - Summarize url links starting with http or www to a common phrase 'url
        - Summarize email addresses to a common phrase 'email'
        - Get rid of new lines `\n'
        - Remove all words that are just numbers
        - Remove all words that contain numbers
        - Cleanup basic punctuation like '..', '. .'
        - Remove punctuation
        - Remove words that are just 1 character long after removing punctuation
        - Use lemmatization to bring words to the base

        INPUT:
            text: string, Text sentences to be split into words

        OUTPUT:
            clean_tokens: list, List containing most crucial words
        """

        # Replace urls starting with 'https' with placeholder
        url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        # replace urls with a common keyword
        detected_urls = re.findall(url_regex, text)
        for url in detected_urls:
            text = text.replace(url, 'url')

        # Replace urls starting with 'www' with placeholder
        url_regex = 'www.(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        detected_urls = re.findall(url_regex, text)
        for url in detected_urls:
            text = text.replace(url, 'url')

        # replace emails with placeholder
        email_regex = '([A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,})'
        detected_emails = re.findall(email_regex, text)
        for email in detected_emails:
            text = text.replace(email, 'email')

        # replace newlines, which can negatively affect performance.
        text = text.replace("\n", " ")
        text = text.replace("\r", " ")
        text = text.replace("..", ".")
        text = text.replace(". .", ".")
        text = text.replace(" ,.", ".")

        text = re.sub(r'\s+', ' ', text).strip()

        # normalize text by removing punctuation, remove case and strip spaces
        text = re.sub(r"[^a-zA-Z0-9]", " ", text)
        text = text.lower().strip()

        # remove numbers
        text = re.sub(r'\d+', '', text)

        #  split sentence into words
        tokens = word_tokenize(text)

        # Remove stopwords, e.g. 'the', 'a',
        tokens = [w for w in tokens if w not in stopwords.words("english")]

        # take words to their core, e.g., children to child
        lemmatizer = WordNetLemmatizer()
        clean_tokens = []
        for tok in tokens:
            clean_tok = lemmatizer.lemmatize(tok, wordnet.VERB)
            # ignore tokens that have only 1 character or contains numbers
            if len(clean_tok) >= 2 & clean_tok.isalpha():
                clean_tokens.append(clean_tok)

        return clean_tokens

    def create_word_count_matrix(self, column='doc_body_all'):
        """
        Create a word count matrix for a dataframe column containing text

        Input:
        df: dataframe containing document texts, with index 'article_id'
        column: string -> column to convert to word counts

        Output:
        tfidf_df: pandas dataframe containing the word count matrix
        tfidf_vectorizer: instance of object TfidfVectorizer

        """

        # Create a word count matrix by article
        tfidf_vectorizer = TfidfVectorizer(min_df=3,
                                           max_df=0.6,
                                           tokenizer=self.tokenize,
                                           token_pattern=None,
                                           max_features=5000)

        vectorized_data = tfidf_vectorizer.fit_transform(
            self.df_content[column])

        tfidf_df = pd.DataFrame(
            vectorized_data.toarray(),
            columns=tfidf_vectorizer.get_feature_names_out(),
            index=self.df_content.index)

        return tfidf_df, tfidf_vectorizer

    def get_documents_read(self, user_id: str) -> Set:
        """
        Get articles a user has read already

        INPUT:
        user_id: int -> Id of a user

        OUTPUT: Set -> article_ids user has read
        """

        docs_read = (self.df_interactions[
                         self.df_interactions['user_id'] == user_id][
                         'article_id']
                     .tolist())

        return set(docs_read)

    def get_user_interests(self, user_id: str, top_n: int = 10) -> List:
        """
        Get a list of keywords to describe what a user is most interested in

        INPUT:
        user_id: int -> Id of a user
        top-n: int -> Number of most used keywords to return

        OUTPUT:
        top_keywords: list, List containing most used nouns and verbs
        """

        # get all the documents titles a user has read
        docs_read = (self.df_interactions[
                         self.df_interactions['user_id'] == user_id][
                         'title']
                     .tolist())

        # ids_read = self.get_documents_read(user_id)
        # docs_read = self.df_content[
        #     self.df_content.index.isin(ids_read)]['doc_description'].tolist()

        # keep only keywords that are nouns and verbs
        keywords = []
        for title in docs_read:
            keyword = self.tokenize(title)
            pos_tags = nltk.pos_tag(keyword)
            for tag in pos_tags:
                if tag[1] in ['NN', 'NNP', 'NNS', 'VB']:
                    keywords.append(tag[0])

        # get the top n keywords
        top_keywords = pd.value_counts(keywords)[:top_n].index.tolist()

        return top_keywords

    def make_content_recommendations(
            self,
            input_search_text,
            user_id,
            top_n=10):
        """
        Content-based recommendations based on text-based similarity

        User input any search text

        INPUT:
        input_search_text: string, any text a user input to search for documents
        user_id: integer, id of the user we make recommendations for

        OUTPUT:
        content_ids: list, List containing article_id recommendations
        content_descriptions: list, List containing article descriptions
        """

        # create a word vector only once

        top_articles = {}

        # execute NLP only once as run time is slow
        if self.tfidf_df.shape[0] == 0:
            self.load_tfidf_vectorizer()

        # convert input text to the fitted tfidf model
        input_search_text = (
            self.tfidf_vectorizer.transform([input_search_text]))

        # find similarity between input text to each document
        similarity = cosine_similarity(input_search_text, self.tfidf_df)
        cosine_similarity_df = pd.DataFrame(
            similarity,
            columns=self.tfidf_df.index)

        # get the most similar records
        cosine_similarity_df = cosine_similarity_df.loc[0].sort_values(
            ascending=False)

        # remove documents the user has read already
        # docs_read = (self.df_interactions[
        #                  self.df_interactions['user_id'] == user_id][
        #                  'article_id']
        #              .tolist())
        docs_read = self.get_documents_read(user_id)
        cosine_similarity_df.drop(docs_read, inplace=True, errors='ignore')

        # make sure similarity scores are above 0,
        # as otherwise no matches were found
        cosine_similarity_df = cosine_similarity_df[cosine_similarity_df > 0]
        if cosine_similarity_df.shape[0] > 0:
            # merge and sort similarities and interactions
            top_article_interactions = (
                self.df_interactions.article_id.value_counts())

            similar_articles = pd.concat(
                [cosine_similarity_df, top_article_interactions],
                axis=1)

            similar_articles.columns = ['similarity', 'num_interactions']
            similar_articles.sort_values(by=['similarity', 'num_interactions'],
                                         ascending=False, inplace=True)

            # add contents to our recommendations
            top_articles = self.get_contents(similar_articles[:top_n])

        return top_articles

    def matrix_factorization(self, user_id, top_n=10):
        """
        Use SVD matrix factorization to get recommendations

        INPUT: user_id: integer, id of the user we make recommendations for
        top_n: int -> Number of documents to recommend

        OUTPUT:
        prediction_df -> DataFrame containing recommendations

        """
        # create a user-item matrix
        user_item_matrix = self.create_user_item_matrix()

        # Decompose the matrix using training dataset with SVD with k latent features
        U, sigma, Vt = svds(np.array(user_item_matrix), k=400)

        # correct shape of s (latent features)
        sigma = np.diag(sigma)

        # predict which documents users will enjoy using the dot product
        prediction = np.abs(np.around(np.dot(np.dot(U, sigma), Vt), 0))
        prediction_df = pd.DataFrame(prediction,
                                     columns=user_item_matrix.columns,
                                     index=user_item_matrix.index)

        # filter predictions to input user
        prediction_df = prediction_df.loc[user_id].sort_values(ascending=False)

        # keep records where the predicted user will read the document
        prediction_df = prediction_df[prediction_df > 0]

        # Get and remove documents the user read already
        docs_read = self.get_documents_read(user_id)
        prediction_df.drop(docs_read, inplace=True, errors='ignore')

        if prediction_df.shape[0] > 0:

            # merge and sort predictions and interactions
            top_article_interactions = (
                self.df_interactions.article_id.value_counts())

            similar_articles = pd.concat(
                [prediction_df, top_article_interactions],
                axis=1,
                join='inner')

            similar_articles.columns = ['similarity', 'num_interactions']
            similar_articles.sort_values(by=['similarity', 'num_interactions'],
                                         ascending=False, inplace=True)

            # add contents to our recommendations
            top_articles = self.get_contents(similar_articles[:top_n])

        else:
            print("No recommendations found")
            self.logger.info("No recommendations found")

        return prediction_df

    def generate_tfidf_vectorizer(self):
        """ Generate and save tfidf vector object and matrix"""

        # create word count matrix
        self.logger.info('Generating tfidf word count matrix and model...')
        print('Generating tfidf word count matrix and model...')
        self.tfidf_df, self.tfidf_vectorizer = (
            self.create_word_count_matrix(column='doc_body_all'))

        # save to file
        if self.tfidf_df.shape[0] > 0:
            print('Saving model to path {}'.format(path_tfidf))
            self.logger.info('Saving model to path {}'.format(path_tfidf))
            dump(self.tfidf_vectorizer, open(path_tfidf, 'wb'))

            print('Saving matrix to path {}'.format(path_tfidf_df))
            self.logger.info('Saving matrix to path {}'.format(path_tfidf_df))
            self.tfidf_df.to_parquet(path_tfidf_df)

    def load_tfidf_vectorizer(self):
        """ Load tfidf vectorizer and matrix

        As the tfidf vector takes a long time
        to run due to the custom tokenizer,
        the tokenizer object and matrix are loaded from disk if it exists.
        If it does not exist,
        the recommender will create it the first time it runs.

        Alternatively,
        a separate script `src/nlp_preprocessing.py` can be scheduled
        to run periodically to incorporate new users and articles.
        """

        try:
            self.logger.info('Loading tfidf vectorizer...')
            self.tfidf_df = pd.read_parquet(path_tfidf_df)
            self.tfidf_vectorizer = load(open(path_tfidf, 'rb'))
        except FileNotFoundError:
            self.generate_tfidf_vectorizer()
