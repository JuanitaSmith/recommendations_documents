# script will create tfidf object and data matrix
# as it takes too long for to do during the webapp experience.
# IMPORTANT run from the main project root `python src/nlp.preprocessing.py`

from src import Recommender
from src import path_articles, path_interactions

if __name__ == '__main__':

    # instantiate recommender, load and clean the data
    r = Recommender()
    r.get_data(path_articles, path_interactions)
    r.clean_contents()
    r.clean_interactions()
    r.generate_tfidf_vectorizer()
