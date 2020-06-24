from simple_recommender import *
import ml_metrics as metrics

def test_ratings_data():

    movies_data = RatingsData(load_dataframe("ratings.csv"), "userId", "movieId", "title", "rating", False)
    movies_data.recommend_items("Toy Story (1995)", 10)
    movies_data.recommend_items("Toy Story (1995)", 10, "cos")
    
    
