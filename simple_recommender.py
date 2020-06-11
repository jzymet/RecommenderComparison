import pandas as pd 
from pd import DataFrame
import numpy as np
import warnings
warnings.filterwarnings('ignore')

DEFAULT_MOVIE_CSV = "ratings.csv"

def load_dataframe(path: str = DEFAULT_MOVIE_CSV) -> pd.DataFrame:
    """load pandas dataframe and return it"""
    
    df: DataFrame = pd.read_csv(path, sep='\t')
    df.head() # print first six rows of df
    
    return df

class RatingsData:
    """object for storing data for user ratings of items"""

    def __init__(self, df: DataFrame, user_id_col: str, item_id_col: str, rating_col: str):
        """param: ...""" #TODO: write docstring

        self._df: DataFrame = df
        self._user: str = user_id_col
        self._item: str = item_id_col
        self._rating: str = rating_col

        assert self._df[self._item].dtype is np.int64
        assert self._df[self._user].dtype is np.int64
        assert self._df[self._rating].dtype is np.float64

    def recommend_items(self, item_name: str, top_n: int = 5) -> None:
        """
        prints top n recommendations based on item
        :param item_name: name of item to use a seed
        :param top_n: number of recommended items to print
        """
    
        return #TODO: implement

    def correlations(self, item_id: int) -> pd.Series:
        """TODO: write docstring"""

        return #TODO: implement
