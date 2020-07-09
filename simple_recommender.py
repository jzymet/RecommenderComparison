from typing import List
from functools import lru_cache
import pandas as pd 
from pandas import DataFrame
import numpy as np
from scipy.spatial import distance
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import warnings
warnings.filterwarnings('ignore')

class CollaborativeRecommender:
    """object for storing data for user ratings of items"""

    def __init__(self, df: DataFrame, user_id_col: str, item_id_col: str,  title_col: str, rating_col: str, binary: bool = True):
        """
        :param df: dataframe
        :param user_id_col: column title for user ID
        :param item_id_col: column title for item ID
        :param rating_col: column title for ratings
        :param binary: whether the user data are binary (e.g., purchase data) or not (e.g., ratings data)
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

        #splitting the user-ratings matrix into training (90%) and test (10%)
        held_out_count = int(len(self._item_matrix)/10)
        self._held_out_matrix: DataFrame = self._item_matrix.loc[:held_out_count, :]
        self._item_matrix: DataFrame = self._item_matrix.loc[(held_out_count+1):len(self._item_matrix), :]
        #fill missing values with average rating
        self._item_matrix.fillna(2.5, inplace = True)
       #for i in range(1, len(item_matrix.loc[:, 1])+1):
              #self._item_matrix.loc[i, :] = item_matrix.loc[i, :] - item_matrix.loc[i, :].mean()
       #print(self._item_matrix)

        ###if ratings data, convert held out users' ratings to 1's if rating is greater than 4, else convert to 0 
        if not binary:
            self._held_out_matrix = pd.DataFrame(np.where(self._held_out_matrix.values >= 4, 1, 0), self._held_out_matrix.index)
            self._held_out_matrix.columns = self._item_matrix.columns 
        
        #list of titles for each item in item matrix
        self._title_list = []
        for col in self._item_matrix.columns:
            self._title_list.append(self._item_to_title[col])

        #matrix with items as rows and with mean ratings and rating count as cols
        self._ratings = DataFrame(self._df.groupby(self._item)['rating'].mean())
        self._ratings['Number_of_ratings'] = self._df.groupby(self._item)['rating'].count()

    @lru_cache(maxsize = 1000)
    def recommend_items(self, item_name: str, top_n: int = 5, distance_metric: str = "corr") -> pd.DataFrame:
        """
        prints top_n recommended movies having at least 100 ratings, based on given item
        :param item_name: name of item to use a seed
        :param top_n: number of recommended items to print
        :param distance_metric: "corr" for Pearson's r, "cos" for cosine similarity
        """
        if distance_metric == "corr": dist = self._correlations
        elif distance_metric == "cos": dist = self._cosine_similarities
        else: raise ValueError("The distance heuristic must be 'corr', for correlation, or 'cos', for cosine similarity.")

        recommendations = dist(item_name, 50).sort_values(by = 'Similarity', ascending = False).head(top_n)
        return recommendations

    def _correlations(self, item_name: str, minimum_ratings: int = 0) -> pd.Series:
        """returns an item's correlation with all other items having at least minimum_ratings rating; drops missing values
        :param item_id: number to which an item is indexed
        :param minimum_ratings: minimum number of ratings an item must have
        """
        #but the cold start problem!

        #vector of ratings for given item
        item_ratings = self._item_matrix[self._title_to_item[item_name]]
        if (item_ratings == 2.5).all(): #check this over
            similarity_vector = DataFrame({'Title': self._title_list, 'Ratings_count': self._ratings['Number_of_ratings']})
            similarity_vector['Similarity'] = pd.Series([0 for x in range(len(similarity_vector.index))], index=similarity_vector.index)
        else:
        #matrix of correlations between given item and other items, minus missing values
            similarity_vector = DataFrame({'Similarity': self._item_matrix.corrwith(item_ratings), 'Title': self._title_list, 'Ratings_count': self._ratings['Number_of_ratings']})
        #similarity_vector.dropna(inplace = True)

        #same matrix, but subtracting all items with fewer than minimum_ratings
        similarity_vector = similarity_vector[similarity_vector['Ratings_count'] > minimum_ratings]
        
        return similarity_vector

    def _cosine_similarities(self, item_name: str, minimum_ratings: int) -> pd.Series:
        """returns an item's correlation with all other item having at least minimum_ratings rating; drops missing values
        :param item_id: number to which an item is indexed
        :param minimum_ratings: minimum number of ratings an item must have
        """
        #but the cold start problem!

        item_ratings = self._item_matrix[self._title_to_item[item_name]]
        if (item_ratings == 2.5).all():
            similarity_vector = DataFrame({'Title': self._title_list, 'Ratings_count': self._ratings['Number_of_ratings']})
            similarity_vector['Similarity'] = pd.Series([0 for x in range(len(similarity_vector.index))], index=similarity_vector.index)
        else:    
        #matrix of cosine similarities between given item and other items
            similarity_list = []
            for col in self._item_matrix.columns:
                similarity_list.append(1 - distance.cosine(item_ratings, self._item_matrix.loc[:, col]))
            similarity_vector = DataFrame({'Similarity': similarity_list, 'Title': self._title_list, 'Ratings_count': self._ratings['Number_of_ratings']})

        #same matrix, but subtracting all items with fewer than minimum_ratings
        similarity_vector = similarity_vector[similarity_vector['Ratings_count'] > minimum_ratings]
        
        return similarity_vector

    def MAP_at_k(self, k: int, dist_metr: str = "corr") -> float: 
        """returns MAP@k metric over all user-reclist pairs* 
        :param k: recommendation list length
        """
        userAPs = []
        for _, current_user in self._held_out_matrix.iterrows():
            APs = []
            print("Current User: %s" % current_user)
            for movId, val in current_user.iteritems():
                if val == 1:
                    print("Seed for recommendation: %s" % self._item_to_title[movId])
                    recommendations = self.recommend_items(self._item_to_title[movId], k)
                    print("Precision: %s" % self.average_precision(recommendations, current_user))
                    APs.append(self.average_precision(recommendations, current_user))
            userAPs.append(sum(APs)/k)
        print(userAPs)
        MAP = sum(userAPs)/len(self._held_out_matrix)
        return MAP
                
    def average_precision(self, recs: DataFrame, user_Data: DataFrame) -> float:
        """returns average precision for particular user-reclist pair
        :param recs: dataframe of recommendations
        :param user_Data: the Series of user ratings for the pertinent movie
        """

        user = user_Data
        precisions = []
        for i in range(1, len(recs)+1):
            top_i_recs: List[ints] = recs.index.values[:i]
            precision: float = sum(user[movId] for movId in top_i_recs)/i
            precisions.append(precision)
        a_prec = sum(precisions)/len(recs)
        return a_prec

class ContentRecommender:
    """object for storing SVD embeddings of TD-IDF values of metadata per item"""

    def __init__(self, tdidf_SVD_output: DataFrame):
        """
        :param df: dataframe
        """
        self._item_matrix = tdidf_SVD_output

    @lru_cache(maxsize = 1000)
    def recommend_items(self, item_name: str, top_n: int = 5, distance_metric: str = "corr") -> pd.DataFrame:
        """
        prints top_n recommended movies having at least 100 ratings, based on given item
        :param item_name: name of item to use a seed
        :param top_n: number of recommended items to print
        :param distance_metric: "corr" for Pearson's r, "cos" for cosine similarity
        """
        if distance_metric == "corr": dist = self._correlations
        elif distance_metric == "cos": dist = self._cosine_similarities
        else: raise ValueError("The distance heuristic must be 'corr', for correlation, or 'cos', for cosine similarity.")

        recommendations = dist(item_name, 50).sort_values(by = 'Similarity', ascending = False).head(top_n)
        return recommendations

    def _correlations(self, item_name: str, minimum_ratings: int = 0) -> pd.Series:
        """returns an item's correlation with all other items having at least minimum_ratings rating; drops missing values
        :param item_id: number to which an item is indexed
        :param minimum_ratings: minimum number of ratings an item must have
        """
        #but the cold start problem!

        #vector of ratings for given item
        item_values = self._item_matrix[item_name]
        #matrix of correlations between given item and other items, minus missing values
        similarity_vector = DataFrame({'Similarity': self._item_matrix.corrwith(item_values)})
        
        return similarity_vector

    def _cosine_similarities(self, item_name: str, minimum_ratings: int) -> pd.Series:
        """returns an item's correlation with all other item having at least minimum_ratings rating; drops missing values
        :param item_id: number to which an item is indexed
        :param minimum_ratings: minimum number of ratings an item must have
        """
        #but the cold start problem!

        #vector of ratings for given item
        item_values = self._item_matrix[item_name]  
        #matrix of cosine similarities between given item and other items
        similarity_list = []
        for col in self._item_matrix.columns:
            similarity_list.append(1 - distance.cosine(item_values, self._item_matrix.loc[:, col]))
        similarity_vector = DataFrame({'Similarity': similarity_list, 'Title': self._item_matrix.columns})
        
        return similarity_vector

    def MAP_at_k(self, k: int, dist_metr: str = "corr") -> float: 
        """returns MAP@k metric over all user-reclist pairs* 
        :param k: recommendation list length
        """
        userAPs = []
        for _, current_user in self._held_out_matrix.iterrows():
            APs = []
            print("Current User: %s" % current_user)
            for movId, val in current_user.iteritems():
                if val == 1:
                    print("Seed for recommendation: %s" % self._item_to_title[movId])
                    recommendations = self.recommend_items(self._item_to_title[movId], k)
                    print("Precision: %s" % self.average_precision(recommendations, current_user))
                    APs.append(self.average_precision(recommendations, current_user))
            userAPs.append(sum(APs)/k)
        print(userAPs)
        MAP = sum(userAPs)/len(self._held_out_matrix)
        return MAP
                
    def average_precision(self, recs: DataFrame, user_Data: DataFrame) -> float:
        """returns average precision for particular user-reclist pair
        :param recs: dataframe of recommendations
        :param user_Data: the Series of user ratings for the pertinent movie
        """

        user = user_Data
        precisions = []
        for i in range(1, len(recs)+1):
            top_i_recs: List[ints] = recs.index.values[:i]
            precision: float = sum(user[movId] for movId in top_i_recs)/i
            precisions.append(precision)
        a_prec = sum(precisions)/len(recs)
        return a_prec

def load_movie_data(ratings_data: str = "ratings.csv", movies_data: str = "movies.csv", tags_data: str = "tags.csv", content: bool = False):
    """loads and combines movie-related datasets (ratings, titles, tags) from the recommender folder, feeds them into RatingsData object        :param ratings_data: .csv file of movie ratings
    :param movies_data: .csv file of movie titles
    :param tags_data: csv file of movie tags
    """
    
    ratings: DataFrame = pd.read_csv(ratings_data)
    ratings.drop(['timestamp'], 1, inplace=True)
    
    titles: DataFrame = pd.read_csv(movies_data)

    ratings_with_titles: DataFrame = pd.merge(ratings, titles, on = "movieId")

    if not content: return CollaborativeRecommender(ratings_with_titles, "userId", "movieId", "title", "rating", False)
        
    else:
        #gets tags for a movie if there are any, and then creates a metadata column with tags/genre information for each movie
        tags: DataFrame = pd.read_csv(tags_data)
        tags.drop(['timestamp'], 1, inplace = True)
        full_movie_dataset: DataFrame = pd.merge(ratings_with_titles, tags, on = ["userId", "movieId"], how = "left")
        full_movie_dataset.fillna("", inplace = True)
        full_movie_dataset = full_movie_dataset.groupby('movieId')['tag'].apply(lambda x: "%s" % ' '.join(x))
        full_movie_dataset = pd.merge(ratings_with_titles, full_movie_dataset, on = "movieId", how = "left")
        full_movie_dataset['metadata'] = full_movie_dataset[["tag", "genres"]].apply(lambda x: ' '.join(x), axis = 1)
        full_movie_dataset = full_movie_dataset[["title", "metadata"]]
        full_movie_dataset.drop_duplicates(subset = "title", inplace = True)

        tfidf = TfidfVectorizer(stop_words = "english")
        tfidf_matrix = tfidf.fit_transform(full_movie_dataset['metadata'])
        tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), index = full_movie_dataset.index.tolist())
        
        svd = TruncatedSVD(n_components = 200)
        latent_matrix_1 = svd.fit_transform(tfidf_df)
        latent_matrix: DataFrame = pd.DataFrame(latent_matrix_1, index = full_movie_dataset.title.tolist()).transpose()

        return ContentRecommender(latent_matrix)
