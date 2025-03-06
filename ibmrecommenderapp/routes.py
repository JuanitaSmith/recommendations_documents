from ibmrecommenderapp import app
from flask import render_template, request
from src import path_articles, path_interactions, Recommender

r = Recommender()

data_loaded = False

@app.before_request
def firstRun():
    """ Load and clean the data only once """
    global data_loaded
    if not data_loaded:
        r.get_data(path_articles, path_interactions)
        r.clean_contents()
        r.clean_interactions()
        data_loaded = True

# main webpage receiving user input
@app.route('/')
@app.route('/index')
def index():

    return render_template(
        'input.html',
    )

# web page that receives user input and make recommendations
@app.route('/recommender', methods=['POST'])
def recommender():

    keywords = []

    # get user id from form
    inputUserId = int(request.form.get('inputUserId'))
    if not inputUserId:
        inputUserId = request.args.get('inputUserId')
    print('User id received is {}'.format(inputUserId))
    r.logger.info('User id received is {}'.format(inputUserId))

    # get search text
    search_text = request.form.get('search_input')
    print('Search text received is {}'.format(search_text))
    r.logger.info('Search text received is {}'.format(search_text))

    # Decide which type of recommendation to do:
    # 1) If a search text is entered, do a content-based recommendation
    # 2) If interactions for user id 2 exist, do collaborative filtering
    # 3) If the user is a new user without interactions,
    # do ranked-based recommendation

    articles = {}
    if search_text:
        text = 'Content-based recommendation triggered with text {}'
        r.logger.info(text.format(search_text))
        articles = r.make_content_recommendations(
            search_text,
            user_id=inputUserId,
            top_n=15
        )
        if len(articles) == 0:
            recommender_comment =(
                "No documents found for search '{}'".format(search_text))
        else:
            recommender_comment = (
                "Content based recommendation using '{}'".format(search_text))
    else:
        user_exist = (r.df_interactions['user_id'] == inputUserId).any()
        if user_exist:
            print('Interactions for user {} exists'.format(inputUserId))
            r.logger.info(
                'Interactions for user {} exists'.format(inputUserId))

            text = 'Collaborative filtering recommendation was triggered'
            r.logger.info(text)

            articles, search_text = r.user_user_recommendations(
                user_id=inputUserId,
                top_n=15
            )
            if search_text:
                recommender_comment = (
                    "Content based search using '{}'".format(search_text))
            else:
                keywords = r.get_user_interests(
                    user_id=inputUserId,
                    top_n=15)
                recommender_comment = (
                    'What users with similar tastes are reading')
        else:
            print('No Interactions for user {} exists'.format(inputUserId))
            r.logger.info(
                'No Interactions for user {} exists'.format(inputUserId))
            text = 'Ranked-based recommendation was triggered'
            r.logger.info(text)
            articles = r.get_top_articles(n=15)
            recommender_comment = ("As it's your first visit, "
                                   "we recommend these most popular documents")

    nav_texts = " - Recommendations for UserID"

    return render_template(
        'recommender.html',
        inputUserId=inputUserId,
        recommender_comment=recommender_comment,
        nav_texts=nav_texts,
        articles=articles,
        keywords=keywords)


@app.route('/view', methods=['POST'])
def view():
    print('View button is clicked')
    r.logger.info('View button is clicked')

    content = None

    inputUserId = request.args.get('inputUserId')
    print('user id received is {}'.format(inputUserId))

    article_id = int(request.form.get('view_button'))
    print('Article id view requested: {}'.format(article_id))

    if article_id:
        cols = ['doc_full_name', 'doc_description', 'doc_body']
        content = r.df_content[r.df_content.index == article_id][cols]
        content = content.reset_index(drop=False).to_dict(orient='index')[0]

    return render_template(
        'display_document.html',
        content=content)
