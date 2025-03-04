# simple program to validate recommendations based on matrix factorization
# to run, go tthe main root folder and type ` python temp2.py --user_id 2`

from src import Recommender
from src import path_articles, path_interactions
import argparse

def parse_args():
    """ default arguments """
    parser = argparse.ArgumentParser(
        prog="Document Recommender",
        description="Run IBM Watson Document Recommender")
    parser.add_argument('--user_id',
                        type=int,
                        default=1528,
                        required=True,
                        help='enter user id, e.g. 1528, 2')
    return parser.parse_args()


if __name__ == "__main__":

    user_id = parse_args().user_id

    r = Recommender()
    r.get_data(path_articles, path_interactions)
    r.clean_contents()
    r.clean_interactions()

    # number of
    print('User have read {} documents'.format(
        r.df_interactions.user_id.value_counts()[user_id]))

    recommendations = r.matrix_factorization(user_id=user_id)

    print('Number of additional documents to read: {}'.format(
        len(recommendations)))

