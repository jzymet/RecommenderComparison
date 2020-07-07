from simple_recommender import *
from sklearn.metrics import average_precision_score

def test_ratings_data():

    movies_data: DataFrame = load_movie_data(content = True)
    recommendation_count: int = 10
    my_recs = movies_data.recommend_items("Toy Story (1995)", recommendation_count)
    #user_0_hits = movies_data._held_out_matrix.iloc[0]
    print(my_recs)

    #print(movies_data.average_precision(my_recs, user_0_hits))
    #print(movies_data.MAP_at_k(10))
    
    #relevant_counts: DataFrame = {}
    #relevant_counts["Sums"] = movies_data._held_out_matrix.sum(axis = 1)
