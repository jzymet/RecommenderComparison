import pandas as pd 
from pandas import DataFrame
import numpy as np
import warnings
warnings.filterwarnings('ignore')

DEFAULT_MOVIE_CSV = "ratings.csv"

def load_dataframe(path: str = DEFAULT_MOVIE_CSV) -> pd.DataFrame:
    """load pandas dataframe and return it"""
    
    df: DataFrame = pd.read_csv(path)
    titles: DataFrame = pd.read_csv("movies.csv")
    df = pd.merge(df, titles, on='movieId')
    df.head() # print first six rows of df
    
    return df

class RatingsData:
    """object for storing data for user ratings of items"""

    def __init__(self, df: DataFrame, user_id_col: str, title_col: str, rating_col: str):
        """
        :param df: dataframe
        :param user_id_col: column title for user ID
        :param item_id_col: column title for item ID
        :param rating_col: column title for ratings
        """
        #:param item_id_2_names: dictionary indexing item IDs to their names
        
        self._df: DataFrame = df
        self._user: str = user_id_col
        self._title: str = title_col
        self._rating: str = rating_col

        #matrix with users as rows and items as cols, ratings as values
        self._item_matrix = self._df.pivot_table(index = self._user, columns = 'title', values = 'rating')

        #matrix with items as rows and with title, mean ratings, and rating count as cols
        self._ratings = DataFrame(self._df.groupby('title')['rating'].mean())
        self._ratings['number_of_ratings'] = self._df.groupby('title')['rating'].count()
        
    def recommend_items(self, item_name: str, top_n: int = 5) -> None:
        """
        prints top_n recommended movies having at least 100 ratings, based on given item
        :param item_name: name of item to use a seed
        :param top_n: number of recommended items to print
        """
        print(self.correlations(item_name, 50).sort_values(by = 'Correlation', ascending=False).head(top_n))

    def correlations(self, item_name: str, minimum_ratings: int) -> pd.Series:
        """returns an item's correlation with all other item having at least minimum_ratings rating; drops missing values
        :param item_id: number to which an item is indexed
        :param minimum_ratings: minimum number of ratings an item must have
        """
        #but the cold start problem!

        #vector of ratings for given item
        item_matrix: DataFrame = self._df.pivot_table(index = self._user, columns = self._title, values = self._rating)
        item_ratings = item_matrix[item_name]

        #matrix of correlations between given item and other items, minus missing values
        similar_to_item = item_matrix.corrwith(item_ratings)
        similarity_vector = DataFrame(similar_to_item, columns = ['Correlation'])
        similarity_vector.dropna(inplace = True)

        #same matrix, but subtracting all items with fewer than minimum_ratings
        similarity_vector = similarity_vector.join(self._ratings['number_of_ratings'])
        similarity_vector = similarity_vector[similarity_vector['number_of_ratings'] > minimum_ratings]
        
        return similarity_vector
        
