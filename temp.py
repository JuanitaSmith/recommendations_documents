from src import recommender

r = recommender.Recommender()

# temp = r.make_content_recommendations('xgboost', 10)

# print(r.tokenize('I love Ian Smith 999'))

tfidf_df, tfidf_vectorizer = r.create_word_count_matrix(
            column='doc_body')