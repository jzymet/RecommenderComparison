from simple_recommender import *

def test_ratings_data(title: str, top_n: int):

    movies_data = RatingsData(load_dataframe("ratings.csv"), "userId", "title", "rating")
    movies_data.recommend_items(title, top_n)
    
    
