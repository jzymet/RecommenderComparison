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
import io
import requests

class Recommender:
    
    """

    Class containing collaborative, content, hybrid weighted-average, and hybrid switching recommenders as subclasses.
    
    """

    
    def __init__(self, inputdata: DataFrame, user_id_col: str, item_id_col: str,  title_col: str, rating_col: str = "rating", metadata_col: str = "metadata", binary: bool = False):
        
        """

        :param inputdata: dataframe
        :param user_id_col: column title for user ID
        :param item_id_col: column title for item ID
        :param rating_col: column title for ratings
        :param metadata_col: column title for metadata
        :param binary: whether the user data are binary (e.g., purchase data) or not (e.g., ratings data)

        (Why parameters for all of the input column names? Because of an ongoing effort on my part to code this as a general *item* recommender, rather than as a movie recommender;          hence should accept any dataframe (movie ratings, item purchases, etc.) having the appropriate columns and whose names are input by the user.)

        """     
        ###TO DO: clarify argument structure of/clean up/comment on average precision and MAP methods. break MAP_at_k into subfunctions, one for recording the various AP@k's and one for actually taking MAP@k
        ###TO DO: "content ratings"/"collaborative ratings"?
        ###TO DO: align outputs of different recommenders
        
        self.inputdata: DataFrame = inputdata
        
        self.user: str = user_id_col
        
        self.item: str = item_id_col
        
        self.title: str = title_col
        
        self.rating: str = rating_col
        
        self.metadata: str = metadata_col

        self.binary: bool = binary

        
        #computes matrix of user ratings of each item, and splits this matrix into training and test using _split_item_matrix()
        
        self.item_matrix: DataFrame = self.inputdata.pivot_table(index = self.user, columns = self.title, values = self.rating)

        split_item_matrices: [DataFrame, DataFrame] = self._split_item_matrix()

        self.item_matrix_training: DataFrame = split_item_matrices[0]

        self.item_matrix_test: DataFrame = split_item_matrices[1]


        #fill missing values in training set with median rating. FOR FUTURE: replace with a better strategy? e.g., estimate the missing numbers.
        
        self.item_matrix_training.fillna(2.5, inplace = True)

        
        #matrix with items as rows and with mean ratings and rating count as cols
        
        self.ratings = DataFrame(self.inputdata.groupby(self.title)['rating'].mean())
        
        self.ratings['Number_of_ratings'] = self.inputdata.groupby(self.title)['rating'].count()

        
        #instantitates matrices of latent features used in content recommendation by applying _create_latent_feature_matrix()
        
        latent_content_feature_matrices: [DataFrame, DataFrame] = self._create_latent_feature_matrices()

        self.latent_content_features: DataFrame = latent_content_feature_matrices[0]

        self.content_ratings: DataFrame = latent_content_feature_matrices[1]

        
    def _split_item_matrix(self, held_out_percentage: float = 0.1) -> [DataFrame, DataFrame]:

        """
        
        Splits the user-ratings matrix into training and test, with default 90%/10% split.
        
        :param held_out_percentage: percentage of rows held out from training set

        """
        
        
        #check that the user inputted a valid percentage for holdout
        
        assert 0 <= held_out_percentage <= .95, 'Percentage held out must be between 0 and 0.95 (inclusive).'


        #split item matrix into training and test
        
        held_out_count: int = int(held_out_percentage*len(self.item_matrix))
        
        item_matrix_test: DataFrame = self.item_matrix.loc[:held_out_count, :]

        item_matrix_training: DataFrame = self.item_matrix.loc[(held_out_count+1):len(self.item_matrix), :]
        

        #if dealing with ratings data, convert held out users' ratings to 1's if rating is greater than or equal to 4, else convert to 0
        #the newly converted 1's will define the seeds to be used in model validation
        #FOR FUTURE: how do these architectures perform with other cutoffs (e.g., >= 3)?

        if not self.binary:
            
            item_matrix_test = DataFrame(np.where(item_matrix_test.values >= 4, 1, 0), item_matrix_test.index)
            
            item_matrix_test.columns = self.item_matrix.columns

            
        return [item_matrix_training, item_matrix_test]

        
    def _create_latent_feature_matrices(self, n_dimensions: int = 200) -> [DataFrame, DataFrame]:

        """

        To compute content-based recommendations, create a matrix of 200 latent features based on metadata... 
        - by applying TF-IDF to the metadata,
        - and then Truncated Singular Value Decomposition to the resulting vectors, reducing their dimensionality so as to speed up later computation.

        Accompany with matrix tracking number of ratings.

        :param n_comps: number of dimensions of the reduced vectors


        """

        #FOR FUTURE: if you plot content-based recommender's MAP against n_dimensions hyperparameter, is there an elbow at 200? seems so far that performance doesn't improve after 200, but perhaps we can go lower.

        
        #create metadata matrix

        metadata_matrix: DataFrame = self.inputdata[[self.title, self.metadata]]
        
        metadata_matrix.drop_duplicates(subset = self.title, inplace = True)

        
        #apply TF-IDF to metadata matrix, creating large vector of values associated with each movie
        
        tfidf = TfidfVectorizer(stop_words = "english")
        
        tfidf_matrix = tfidf.fit_transform(metadata_matrix[self.metadata])
        
        tfidf_df: DataFrame = DataFrame(tfidf_matrix.toarray(), index = metadata_matrix.index.tolist())

        
        #to speed up similarity computation during content recommendation, reduce dimensionality of TF-IDF vectors to 200 latent features by applying Truncated Singular Value Decomposition
        
        svd = TruncatedSVD(n_components = n_dimensions)
        
        latent_matrix_raw: DataFrame = svd.fit_transform(tfidf_df)
        
        latent_content_matrix: DataFrame = DataFrame(latent_matrix_raw, index = metadata_matrix[self.title].tolist()).transpose()

        content_ratings: pd.Series = pd.Series(self.ratings['Number_of_ratings'][x] for x in latent_content_matrix.columns)
        
        return [latent_content_matrix, content_ratings]


    def corr(self, item_name: str, comparanda: DataFrame, ratings: DataFrame) -> pd.Series:

        """

        Returns correlations between a supplied vector of a given item and those of all other items.

        :param item_name: name of item
        :param comparanda: DataFrame consisting of numerical vectors each associated to a particular item (e.g., item_matrix_training if we're comparing user ratings of differnet items, and latent_content_features if we're comparing latent content features of different items)
        :param ratings: architecture-specific ratings

        """

        
        #vector of ratings for input item
        
        item_ratings = self.comparanda[item_name]
        

        #checks if entire item_ratings vector originated with empty values (i.e., every value is 2.5 at this point), set all values of output to 0 if so
        
        if (item_ratings == 2.5).all(): 
            
            similarity_vector = DataFrame({'Title': comparanda.columns, 'Ratings_count': ratings})
            
            similarity_vector['Similarity'] = pd.Series([0 for x in range(len(similarity_vector.index))], index = similarity_vector.index)

        #else compute matrix of correlations between given item and other items, minus missing values
        
        else:

            similarity_vector = DataFrame({'Title': comparanda.columns, 'Similarity': comparanda.corrwith(item_ratings), 'Ratings_count': ratings})

            similarity_vector = similarity_vector.sort_values(by = ["Similarity", "Ratings_count"], ascending = False)

            similarity_vector = similarity_vector[1:]
            
        
        return similarity_vector

    
    def cosine(self, item_name: str, comparanda: DataFrame, ratings: DataFrame) -> pd.Series:

        """

        Returns cosine similarities between a supplied vector of a given item and those of all other items.

        :param item_name: name of item
        :param comparanda: DataFrame consisting of numerical vectors each associated to a particular item (e.g., item_matrix_training if we're comparing user ratings of differnet items, and latent_content_features if we're comparing latent content features of different items)
        :param ratings: architecture-specific ratings

        """       

        
        #vector of ratings for input item
        
        item_ratings = comparanda[item_name]


        #checks if entire item_ratings vector originated with empty values (i.e., every value is 2.5 at this point), set all values of output to 0 if so
        
        if (item_ratings == 2.5).all():
            
            similarity_vector = DataFrame({'Title': comparanda.columns, 'Ratings_count': ratings})
            
            similarity_vector['Similarity'] = pd.Series([0 for x in range(len(similarity_vector.index))], index = similarity_vector.index)
            
        #else compute matrix of correlations between given item and other items, minus missing values

        else:

            similarity_list = []
            
            for col in comparanda.columns:
                
                similarity_list.append(1 - distance.cosine(item_ratings, comparanda.loc[:, col]))
                
            similarity_vector = DataFrame({'Title': comparanda.columns, 'Similarity': similarity_list, 'Ratings_count': ratings})
            
            similarity_vector = similarity_vector.sort_values(by = ["Similarity", "Ratings_count"], ascending = False)
            
            similarity_vector = similarity_vector[1:]

            
        return similarity_vector
    
    
    def recommend_items(self, item_name: str, similarity_metric: str = "cos", approach: str = "collab", weight: float = 0.5, cutoff: int = 100) -> DataFrame:

        """

        Returns ordered list of recommended items based on a given seed item, algorithm (collaborative filter, content-based, etc.), and distance metric.

        :param item_name: name of item to use a seed
        :param similarity_metric: method by which similarity is calculated
        :param approach: type of recommender used
        :param weight: weight to be fed into hybrid weighted-average recommender
        :param cutoff: switchpoint to be fed into hybrid switching recommender

        """

            
        if approach == "collab":
                
            return CollaborativeRecommender._recommend_items(self, item_name, similarity_metric)

        elif approach == "content":
                
            return ContentRecommender._recommend_items(self, item_name, similarity_metric)

        elif approach == "weighted":

            return WeightedRecommender._recommend_items(self, item_name, weight, similarity_metric)
            
        elif approach == "switch":
            
            return SwitchRecommender._recommend_items(self, item_name, cutoff, similarity_metric)
            
        else: raise ValueError("Recommendation algorithm must be 'collab' for collaborative, 'content' for content, 'weighted' for weighted, or 'switch' for switch.")

        
    def prune_recommended(self, similarity_vector: DataFrame, top_n: int = 10,  minimum_ratings: int = 0) -> DataFrame:

        """

        Takes an ordered list of recommendations, optionally removes any movies having fewer than some given minimum number of ratings, and returns only the top_n items on        that list.

        :param similarity_vector: list of recommendations
        :param top_n: number of movies to return
        :param minimum_ratings: minimum number of ratings needed to be returned

        """
        
        new_similarity_vector: DataFrame = similarity_vector[similarity_vector['Ratings_count'] > minimum_ratings]
        
        return new_similarity_vector.head(top_n)

    @lru_cache(maxsize = 1000)
    def MAP_at_k(self, k: int = 10, minratings: int = 0, maxratings: int = 610, dist_metr: str = "cos", approach: str = "collab"): 
        """writes AP@k scores to csv over all user-reclist pairs, and returns MAP@k metric* 
        :param k: recommendation list length
        :param minratings: minimum number of ratings to be included
        :param maxratings: maximum number of ratings to be included        
        :param dist_metric: method by which similarity is calculated
        :param approach: type of recommender used 
        """
        #TO DO: either remove or explicitly use minratings/maxratings
        #TO DO: separate printing function
        
        APdatapoints = []
        userAPs = []
        usercount: int = 0
        for _, current_user in self._held_out_matrix.iterrows():
            userhas: bool = False
            APs = []
            print("Current User: %s" % current_user)
            for itTitle, val in current_user.iteritems():
                if val == 1 and self._ratings['Number_of_ratings'][itTitle] in range(minratings, maxratings+1):
                    userhas = True
                    print("Seed for recommendation: %s" % itTitle)
                    recommendations = self.recommend_items(itTitle, dist_metr, approach)
                    pruned_recommendations = self.prune_recommended(recommendations, k, 0)
                    APs.append(self.average_precision(pruned_recommendations, current_user))
                    APdatapoints.append([self.average_precision(pruned_recommendations, current_user), self._ratings['Number_of_ratings'][itTitle]])
                    print("Precision: %s" % self.average_precision(self.prune_recommended(recommendations, 10, 0), current_user))
            userAPs.append(sum(APs)/k)
            if userhas:
                usercount = usercount + 1
        with open('scatterpointsweighted.csv','w') as file:
            for x in APdatapoints:
                file.write(str(x[1]))
                file.write(" ")
                file.write(str(x[0]))
                file.write('\n')
        if usercount > 0:
            MAP = sum(userAPs)/usercount
            return [MAP, usercount]
        else: return "None"

        
    def average_precision(self, recs: DataFrame, user_Data: DataFrame) -> float:

        """
        
        Computes the average precision of the recommendation list that was generated in response to a seed item that was liked by a held out user.

        :param recs: dataframe of recommendations
        :param user_Data: series of user ratings for different items

        """

        
        user = user_Data
        
        precisions = []


        #take precision for top recommended item, then top two items, then top three, up to the length of the recommendation list
        
        for i in range(1, len(recs)+1):

            top_i_recs: List[strs] = recs["Title"].values[:i]
            
            precision: float = sum(user[itId] for itId in top_i_recs)/i
            
            precisions.append(precision)


        #average the precisions
            
        a_prec = sum(precisions)/len(recs)

        
        return a_prec

    
class CollaborativeRecommender(Recommender):

    """

    Class defining recommender architecture that recommends items with similar user ratings to a seed item. Inherits attributes from base class.

    """

    
    def _recommend_items(self, seed_item_name: str, similarity_metric: str = "cos"):
    
        """

        Returns list of items ordered based on how similar they are to the seed item, calculated in terms of user ratings -- i.e., by-item collaborative filtering. Allows input for similarity metric.

        :param item_name: name of item
        :param similarity_metric: metric to be used to calculate similarity between vectors (either correlation or cosine simiarity)

        """

        
        #return ordered list of items based on how similar they are to the seed item, for given similarity metric (cosine or corr)
        #raise value error if an appropriate similarity metric is not provided
        
        if similarity_metric == "cos":

            return self.cosine(seed_item_name, self.item_matrix_training, self.ratings['Number_of_ratings'])

        elif similarity_metric == "corr":

            return self.corr(seed_item_name, self.item_matrix_training, self.ratings['Number_of_ratings'])

        else: raise ValueError("The similarity metric must be 'corr', for correlation, or 'cos', for cosine similarity.")   

    
class ContentRecommender(Recommender):

    """

    Class defining recommender architecture that recommends items with similar metadata to a seed item. Inherits attributes from base class.

    """

    
    def _recommend_items(self, seed_item_name: str, similarity_metric: str = "cos"):
    
        """

        Returns list of items ordered based on how similar they are to the seed item, calculated in terms of latent features in the metadata. Allows input for similarity metric.

        :param item_name: name of item
        :param similarity_metric: metric to be used to calculate similarity between vectors (either correlation or cosine simiarity)

        """

        
        #return ordered list of items based on how similar they are to the seed item, for given similarity metric (cosine or corr)
        #raise value error if an appropriate similarity metric is not provided
        
        if similarity_metric == "cos":

            return self.cosine(seed_item_name, self.latent_content_features, self.content_ratings)

        elif similarity_metric == "corr":

            return self.corr(seed_item_name, self.latent_content_features, self.content_ratings)

        else: raise ValueError("The similarity metric must be 'corr', for correlation, or 'cos', for cosine similarity.")

    
class WeightedRecommender(Recommender):

    """

    Class defining recommender architecture that recommends items based on a weighted average between content and collaborative recommendation.

    """
    

    def _recommend_items(self, seed_item_name: str, alpha: float, similarity_metric: str = "cos"):
    
        """

        Returns list of items ordered based on how similar they are to the seed item, calculated by a weighted average between the similarity scores determined by collaborative filtering and those determined by content recommendation. Allows input for similarity metric.

        :param item_name: name of item
        :param alpha: the averaging weight, i.e. how much relative emphasis is put on the collaborative recommender
        :param similarity_metric: metric to be used to calculate similarity between vectors (either correlation or cosine simiarity)

        """

        
        #compute weighted average between similarity values determined by collaborative filtering and those determined by content recommendation;
        #raise value error if an appropriate similarity metric is not provided
        
        if similarity_metric == "cos":

            collabrecs = self.cosine(seed_item_name, self.item_matrix_training, self.ratings['Number_of_ratings']).sort_values(by = "Title", ascending = False)

            print(collabrecs)

            contentrecs = self.cosine(seed_item_name, self.latent_content_features, self.content_ratings).sort_values(by = "Title", ascending = False)

            print(contentrecs)

            weighted_average_recs: DataFrame = DataFrame({'Title': collabrecs["Title"], 'Similarity': alpha*collabrecs["Similarity"] + (1 - alpha)*contentrecs["Similarity"], 'Ratings_count': collabrecs["Ratings_count"]})

            return weighted_average_recs.sort_values(by = ["Similarity", "Ratings_count"], ascending = False) 

        elif similarity_metric == "corr":

            collabrecs = self.corr(seed_item_name, self.item_matrix_training, self.ratings['Number_of_ratings']).sort_values(by = "Title", ascending = False)

            print(collabrecs)

            contentrecs = self.corr(seed_item_name, self.latent_content_features, self.content_ratings).sort_values(by = "Title", ascending = False)

            print(contentrecs)

            weighted_average_recs: DataFrame = DataFrame({'Title': collabrecs["Title"], 'Similarity': alpha*collabrecs["Similarity"] + (1 - alpha)*contentrecs["Similarity"], 'Ratings_count': collabrecs["Ratings_count"]})

            return weighted_average_recs.sort_values(by = ["Similarity", "Ratings_count"], ascending = False) 

        else: raise ValueError("The similarity metric must be 'corr', for correlation, or 'cos', for cosine similarity.")


class SwitchRecommender(Recommender):

    """

    Defines recommender that switches from content to collaborative recommendation if the seed item has a number of ratings above a specified cutoff.

    """

    
    def _recommend_items(self, seed_item_name: str, cutoff: int, similarity_metric: str = "cos"):
    
        """

        Returns list of items ordered based on how similar they are to the seed item, calculated with latent content features if the seed item has fewer number of ratings than the specified cutoff, else calculated by collaborative filtering. Allows input for similarity metric.

        :param item_name: name of item
        :param cutoff: threshold for determining whether to apply content recommendation or collaborative filtering
        :param similarity_metric: metric to be used to calculate similarity between vectors (either correlation or cosine simiarity)

        """

        
        #check if nubmer of ratings of the seed item are above cutoff;
        #if so, compute list by content recommendation; else, collaborative filtering
        #raise value error if an appropriate similarity metric is not provided
        
        if self.ratings["Number_of_ratings"][seed_item_name] > cutoff:

            if similarity_metric == "cos":

                return self.cosine(seed_item_name, self.latent_content_features, self.content_ratings)

            elif similarity_metric == "corr":

                return self.corr(seed_item_name, self.latent_content_features, self.content_ratings)

            else: raise ValueError("The similarity metric must be 'corr', for correlation, or 'cos', for cosine similarity.")

        else:

            if similarity_metric == "cos":

                return self.cosine(seed_item_name, self.item_matrix_training, self.ratings['Number_of_ratings'])

            elif similarity_metric == "corr":

                return self.corr(seed_item_name, self.item_matrix_training, self.ratings['Number_of_ratings'])

            else: raise ValueError("The similarity metric must be 'corr', for correlation, or 'cos', for cosine similarity.")
            
        
def load_movie_data(ratings_data: str = "ratings.csv", movies_data: str = "movies.csv", tags_data: str = "tags.csv"):

    """

    Loads and combines movie-related datasets (ratings, titles, tags) from the recommender folder, feeds them into RatingsData object  
      
    :param ratings_data: .csv file of movie ratings
    :param movies_data: .csv file of movie titles
    :param tags_data: csv file of movie tags

    """
    

    #load different movie datasets
    ratings: DataFrame = pd.read_csv(ratings_data)
    ratings.drop(['timestamp'], 1, inplace = True)
    
    titles: DataFrame = pd.read_csv(movies_data)

    ratings_with_titles: DataFrame = pd.merge(ratings, titles, on = "movieId")

    tags: DataFrame = pd.read_csv(tags_data)
    tags.drop(['timestamp'], 1, inplace = True)

    #join datasets into one: dump genres and tags into metadata, clean dataset
    full_movie_dataset: DataFrame = pd.merge(ratings_with_titles, tags, on = ["userId", "movieId"], how = "left")
    full_movie_dataset.fillna("", inplace = True)
    full_movie_dataset = full_movie_dataset.groupby('movieId')['tag'].apply(lambda x: "%s" % ' '.join(x))
    full_movie_dataset = pd.merge(ratings_with_titles, full_movie_dataset, on = "movieId", how = "left")
    full_movie_dataset['metadata'] = full_movie_dataset[["tag", "genres"]].apply(lambda x: ' '.join(x), axis = 1)
    full_movie_dataset.drop(["tag", "genres"], 1, inplace = True)
    full_movie_dataset.to_csv(r'/Users/jzymet/Desktop/recommender/full_movie_dataset.csv', index = False)

    return full_movie_dataset
