# Comparing hybrid movie recommenders

This was a project that I prepared for admission to a few data science transitional programs—the Fall 2020 session of Insight Data Science, in particular. 

I coded up from scratch and compared the performance of four movie recommenders—two simple recommenders, and two hybrid recommenders.

## Problem

Simple recommenders such as content-based recommenders and collaborative filters have different strengths and weaknesses, affecting how relevant their recommendations are to users.

Data scientists have turned to <I>hybrid recommenders</I>, in which multiple simple recommenders are combined, to surmount weaknesses displayed by any component recommender in isolation.

A variety of data scientists, in a number of prominent articles on Towards Data Science, use a weighted average between collaborative and content-based recommendations, without comparing this approach to other hybrid approaches.
- E.g., the less well-known approach of switching between content and collaborative recommendation, which, to my knowledge, has not been discussed on TDS.

<b>Question</b>: Amongst a variety of choices for hybrid algorithm, which one works best? In other words, <i>how</i> should two recommenders be combined, given a dataset?

## Solution

Project codes up from scratch and compares the performance of four movie recommenders against the MovieLens dataset: 
- Content-based
- By-item collaborative filter
- Hybrid weighted average
- Hybrid content-to-collaborative switch once seed has enough ratings

I find that content recommendation performs better than collaborative on the Mean Average Precision @ 10 (MAP@10) accuracy metric for less-rated movie seeds, while collaborative performs better for often-rated seeds.

I further show that averaging—used by many data scientists on TDS—performs only half as well as the uncommon switch algorithm. 
- The point is to call into question the use of the weighted average in light of the switching approach, particularly when the goal is accuracy-based.
- If recommender A outperforms recommender B at one time and B outperforms A at a later time, then the hypothesis that I raise in light of my results is that it's generally better (at least according to MAP) to switch from A to B at the best time rather than to average them with the best weight.
- And, of course, if A always outperforms B, then just go with A.

## Code

See [here](https://github.com/jzymet/recommender/blob/master/recommenders.py).

## Demo

See [here](https://github.com/jzymet/recommender/blob/master/demo.ipynb).