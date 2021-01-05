from recommenders import *

def test_recommender():

    movies_data: DataFrame = load_movie_data()
    newRec: Recommender = Recommender(movies_data, "userId", "movieId", "title", "rating", "metadata")
    my_recs: DataFrame = newRec.recommend_items("Toy Story (1995)", "cos", "content")
    print(newRec.prune_recommended(my_recs, 10, 50))

