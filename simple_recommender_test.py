from recommenders import *

def test_recommender():

    movies_data: DataFrame = load_movie_data()
    newRec: Recommender = Recommender(movies_data, "userId", "movieId", "title", "rating", "metadata")

    
    ###collaborative-filter recommendations in response to Toy Story, which has many user ratings, as well as Children of the Damned, which has few

    print("Collab: Toy Story")
    print()
    collab_recs_many_seed_ratings: DataFrame = newRec.recommend_items("Toy Story (1995)", "collab", "cos")
    print(newRec.prune_recommended(collab_recs_many_seed_ratings, 10, 50))
    print()

    print("Collab: Children of the Damned")
    print()
    collab_recs_few_seed_ratings: DataFrame = newRec.recommend_items("Children of the Damned (1963)", "collab", "cos")
    print(newRec.prune_recommended(collab_recs_few_seed_ratings, 10, 50))

    print()
    print("-----------")
    print()
    
    ###content-based recommendations in response to Toy Story, which has many user ratings, as well as Children of the Damned, which has few

    print("Content: Toy Story")
    print()
    content_recs_many_seed_ratings: DataFrame = newRec.recommend_items("Toy Story (1995)", "content", "cos")
    print(newRec.prune_recommended(content_recs_many_seed_ratings, 10, 50))
    print()

    print("Content: Children of the Damned")
    print()
    content_recs_few_seed_ratings: DataFrame = newRec.recommend_items("Children of the Damned (1963)", "content", "cos")
    print(newRec.prune_recommended(content_recs_few_seed_ratings, 10, 50))

    print()
    print("-----------")
    print()
    
    
    ###recommendations via hybrid weighted average in response to Toy Story
    ###set weight to 1 and notice that we get the collaborative filter's recommendations; set to 0 and we get content-based recommendations; set at e.g. 0.7 and we see something of a blend
    
    print("Weighted (weight = 1): Toy Story")
    print()
    weighted_recs: DataFrame = newRec.recommend_items("Toy Story (1995)", "weighted", "cos", 1)
    print(newRec.prune_recommended(weighted_recs, 10, 50))
    print()

    print("Weighted (weight = 0): Toy Story")
    print()
    weighted_recs: DataFrame = newRec.recommend_items("Toy Story (1995)", "weighted", "cos", 0)
    print(newRec.prune_recommended(weighted_recs, 10, 50))
    print()
    
    print("Weighted (weight = 0.7): Toy Story")
    print()
    weighted_recs: DataFrame = newRec.recommend_items("Toy Story (1995)", "weighted", "cos", 0.7)
    print(newRec.prune_recommended(weighted_recs, 10, 50))

    print()
    print("-----------")
    print()

    ###recommendations via hybrid content-to-collaborative switch in response to Toy Story, which has many user ratings, and Children of the Damned, which has few
    ###notice that the results for Toy Story are the same as those given by the collaborative filter in response to the same movie seed
    ###but the results for Children of the Damned are the same as those given by the content-based recommender

    print("Switch: Toy Story")
    print()
    switch_recs_many_seed_ratings: DataFrame = newRec.recommend_items("Toy Story (1995)", "switch", "cos")
    print(newRec.prune_recommended(switch_recs_many_seed_ratings, 10, 50))
    print()

    print("Switch: Children of the Damned")
    print()
    switch_recs_few_seed_ratings: DataFrame = newRec.recommend_items("Children of the Damned (1963)", "switch", "cos")
    print(newRec.prune_recommended(switch_recs_few_seed_ratings, 10, 50))

    print()
    print("-----------")
    print()
    
    ###decomment if you want to run validation
    
    #APs: DataFrame = newRec.get_all_AP_at_ks(10, "content", "cos")

    #newRec.AP_at_ks_to_csv(APs)
    #print("MAP: ", newRec.get_MAP_at_k(APs))
 
