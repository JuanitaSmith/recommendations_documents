"""
Unit tests for run.py

To run this test, use the command `python -m unittest tests.test_recommender` in the terminal
"""

import unittest
from src import path_articles, path_interactions, Recommender


class TestRecommender(unittest.TestCase):
    def setUp(self):

        # instantiating this class will load and clean the data
        self.r = Recommender()
        self.r.get_data(path_articles, path_interactions)
        self.r.clean_contents()
        self.r.clean_interactions()

    def test_load_data(self):

        self.assertEqual(self.r.df_interactions.shape[1],
                         3,
                         'Interaction dataset should have 3 columns')
        self.assertEqual(self.r.df_content.shape[1],
                         4,
                         'Content dataset should have 5 columns')

    def test_clean_data(self):

        # make sure contents contain no duplicate indexes
        self.assertEqual(
            first=self.r.df_content.index.duplicated().any(),
            second=False,
            msg='Contents have duplicate indexes'
        )

        # make sure contents have no null values
        self.assertEqual(
            first=self.r.df_content.isnull().sum().any(),
            second=False,
            msg='Contents should have no null values'
        )

        self.assertIn(
            'user_id',
            self.r.df_interactions.columns.tolist(),
            'Interaction dataset should have column "user_id"')