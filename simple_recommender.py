import pandas as pd 
from pandas import DataFrame
import numpy as np
from scipy.spatial import distance
from sklearn.metrics import mean_squared_error
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

    def __init__(self, df: DataFrame, user_id_col: str, item_id_col: str,  title_col: str, rating_col: str):
        """
        :param df: dataframe
        :param user_id_col: column title for user ID
        :param item_id_col: column title for item ID
        :param rating_col: column title for ratings
        """
        
        self._df: DataFrame = df
        self._user: str = user_id_col
        self._item: str = item_id_col
        self._title: str = title_col
        self._rating: str = rating_col

        self._title_to_item = {}
        for i in range(1, len(self._df)):
            self._title_to_item[self._df[self._title][i]] = self._df[self._item][i]

        self._item_to_title = {}
        for i in range(1, len(self._df)):
            self._item_to_title[self._df[self._item][i]] = self._df[self._title][i]

        #matrix of user ratings of each item
        self._item_matrix: DataFrame = self._df.pivot_table(index = self._user, columns = self._item, values = self._rating)

        #list of titles for each item in item matrix
        self._title_list = []
        for col in self._item_matrix.columns:
            self._title_list.append(self._item_to_title[col])

        #matrix with items as rows and with mean ratings and rating count as cols
        self._ratings = DataFrame(self._df.groupby(self._item)['rating'].mean())
        self._ratings['Number_of_ratings'] = self._df.groupby(self._item)['rating'].count()
        
    def recommend_items(self, item_name: str, top_n: int = 5, distance_metric: str = "corr") -> None:
        """
        prints top_n recommended movies having at least 100 ratings, based on given item
        :param item_name: name of item to use a seed
        :param top_n: number of recommended items to print
        :param distance_metric: "corr" for Pearson's r, "cos" for cosine similarity
        """
        if distance_metric == "corr": dist = self._correlations
        elif distance_metric == "cos": dist = self._cosine_similarities
        else: raise ValueError("The distance heuristic must be 'corr', for correlation, or 'cos', for cosine similarity.")
    
        print(dist(item_name, 50).sort_values(by = 'Similarity', ascending = False).head(top_n))

    def _correlations(self, item_name: str, minimum_ratings: int) -> pd.Series:
        """returns an item's correlation with all other item having at least minimum_ratings rating; drops missing values
        :param item_id: number to which an item is indexed
        :param minimum_ratings: minimum number of ratings an item must have
        """
        #but the cold start problem!

        #vector of ratings for given item
        item_ratings = self._item_matrix[self._title_to_item[item_name]]

        #matrix of correlations between given item and other items, minus missing values
        similarity_vector = DataFrame({'Similarity': self._item_matrix.corrwith(item_ratings), 'Title': self._title_list, 'Ratings_count': self._ratings['Number_of_ratings']})
        similarity_vector.dropna(inplace = True)

        #same matrix, but subtracting all items with fewer than minimum_ratings
        similarity_vector = similarity_vector[similarity_vector['Ratings_count'] > minimum_ratings]
        
        return similarity_vector
        
    def _cosine_similarities(self, item_name: str, minimum_ratings: int) -> pd.Series:
        """returns an item's correlation with all other item having at least minimum_ratings rating; drops missing values
        :param item_id: number to which an item is indexed
        :param minimum_ratings: minimum number of ratings an item must have
        """
        #but the cold start problem!

        #vector of ratings for given item, with empty cells filled in with zero, and then with centering applied
        self._item_matrix.fillna(0, inplace = True)
       #for i in range(1, len(item_matrix.loc[:, 1])+1):
       #        self._item_matrix.loc[i, :] = item_matrix.loc[i, :] - item_matrix.loc[i, :].mean()
       #print(self._item_matrix)
       
        item_ratings = self._item_matrix[self._title_to_item[item_name]]

        #matrix of cosine similarities between given item and other items
        similarity_list = []
        for col in self._item_matrix.columns:
            similarity_list.append(1 - distance.cosine(item_ratings, self._item_matrix.loc[:, col]))
        similarity_vector = DataFrame({'Similarity': similarity_list, 'Title': self._title_list, 'Ratings_count': self._ratings['Number_of_ratings']})

        #same matrix, but subtracting all items with fewer than minimum_ratings
        similarity_vector = similarity_vector[similarity_vector['Ratings_count'] > minimum_ratings]
        
        return similarity_vector
