from recommenders import *

#def unittests() -> bool:
    #"""execute all unit tests, printing out the results and returning true only if all tests pass"""

#def inputdata_check() -> bool:
    #"""various checks to ensure that the input dataframe was constructed correctly""" 

#def ratings_matrices_check() -> bool:
    #"""various checks to ensure that the ratings-based dataframes were constructed correctly"""

#def cont_matrix_check() -> bool:
    #"""various checks to ensure that the content dataframe was constructed correctly"""

#def collab_check() -> bool:
    #"""various checks to ensure that the collaborative recommender is outputting the intended recommended lists"""

#def content_check() -> bool:
    #"""various checks to ensure that the content recommender is outputting the intended recommended lists"""

    

#def weighted_check() -> bool:
    #"""various checks to ensure that the weighted recommender is outputting the intended recommended lists"""

#def switch_check() -> bool:
    #"""various checks to ensure that the switch recommender is outputting the intended recommended lists"""

#def AP_check() -> bool:
    #"""various checks to ensure that average precision calculator is working correctly"""

#def MAP_check() -> bool:
    #"""various checks to ensure that MAP calculator is working correctly"""
    #"""blech!"""

#def prune_check() -> bool:
    #"""various checks to ensure that prune_recommended is working correctly"""

def test_ratings_data():

    movies_data: DataFrame = load_movie_data()
    newRec: Recommender = Recommender(movies_data, "userId", "movieId", "title", "rating", "metadata")
    my_recs = newRec.recommend_items("Toy Story (1995)", "cos", "content")
    print(newRec.prune_recommended(my_recs, 10, 50))

    #print(newRec.average_precision(my_recs, user_1_hits))
    #print("MAP for collab, content")
    #metr = newRec.MAP_at_k(10, 0, 610, "cos", "weighted")
    #print(metr)
    #metrs = {}
    #for i in range(1,61):
    #    metr = newRec.MAP_at_k(10, "cos", "content")
   #     metrs[5*i] = metr
    #    print("MAP at ratings count ", 5*i, ": ", metr)
    #for k in metrs:
   #     print(k, ": ", metrs[k])
    
    #relevant_counts: DataFrame = {}
    #relevant_counts["Sums"] = movies_data._held_out_matrix.sum(axis = 1)
