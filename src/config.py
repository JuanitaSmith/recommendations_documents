import os

cwd = os.getcwd()
print(cwd)

# FOLDER NAMES
folder_data = 'data'
folder_raw = 'data'
folder_clean = 'data/clean'
folder_embeddings = 'data/embeddings'
folder_logs = 'logs'
folder_scripts = 'src'
folder_models = 'models'

# LOG NAMES
filename_log_process_data = 'log.log'

# FILE/TABLE NAMES
filename_articles = 'articles_community.csv'
filename_interactions = 'user_item_interactions.csv'
filename_tfidf = 'tfidf.pkl'
filename_tfidf_df = 'tfidf_df.parquet'

# FILE PATHS
path_log = os.path.join(folder_logs, filename_log_process_data)
path_articles = os.path.join(folder_raw, filename_articles)
path_interactions = os.path.join(folder_raw, filename_interactions)
path_tfidf = os.path.join(folder_models, filename_tfidf)
path_tfidf_df = os.path.join(folder_models, filename_tfidf_df)

# make a folder if it does not exist
if not os.path.exists(folder_models):
    os.makedirs(folder_models)

if not os.path.exists(folder_logs):
    os.makedirs(folder_logs)