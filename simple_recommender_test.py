from simple_recommender import *

def test_ratings_data():

    movies_data = RatingsData(load_dataframe("ratings.csv"), "userId", "movieId", "title", "rating")
    movies_data.recommend_items("Toy Story (1995)", 10)
    movies_data.recommend_items("Toy Story (1995)", 10, "cos")
    
    
