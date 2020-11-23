# Comparing hybrid movie recommenders

This was a project that I prepared for admission to a couple data science transitional programs — Insight, in particular.

## Problem

Content recommenders and collaborative recommenders have different strengths/weaknesses, affecting how relevant their recommendations are.
- For example, for products that have lots of user ratings, by-item collaborative filters are quite accurate;
- but for products that have been rated few times, the collaborative filter performs poorly.

Data scientists turn to hybrid recommenders—in which multiple simple recommenders are combined—to surmount weaknesses displayed by any component recommender in isolation.

The data scientists on Towards Data Science often use a weighted average between the collaborative and content-based approaches, without comparing the hybrid to other possibilities.

Amongst a variety of choices for hybrid algorithm, which one works best? In other words, how should two recommenders be combined, given a dataset?

## Solution:

Project compares the performance of four movie recommenders against the MovieLens dataset: 
- Content-based
- By-item collaborative filter
- Hybrid weighted average
- Hybrid content-to-collaborative switch once seed has enough ratings

I find that content recommendation performs better than collaborative on MAP@10 accuracy metric for less-rated movie seeds, while collaborative performs better for often-rated seeds.

I further show that averaging—used by many data scientists on TDS—performs only half as well as the uncommon switch algorithm.
- This calls into question the use of averaging if the goal is accuracy.