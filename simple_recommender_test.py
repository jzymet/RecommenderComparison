from simple_recommender import *
from sklearn.metrics import average_precision_score

def test_ratings_data():

    movies_data: DataFrame = load_movie_data()
    newRec: Recommender = Recommender(movies_data, "userId", "movieId", "title", "rating", "metadata")
    recommendation_count: int = 10
    my_recs = newRec.recommend_items("Toy Story (1995)", recommendation_count, "cos", "content")
    print(my_recs)

    #print(movies_data.average_precision(my_recs, user_0_hits))
    #print(movies_data.MAP_at_k(10))
    
    #relevant_counts: DataFrame = {}
    #relevant_counts["Sums"] = movies_data._held_out_matrix.sum(axis = 1)
