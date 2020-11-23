# Comparing hybrid movie recommenders

This was a project that I prepared for admission to a few data science transitional programs—the Fall 2020 session of Insight Data Science, in particular.

## Problem

Simple recommenders such as content-based recommenders and collaborative filters have different strengths and weaknesses, affecting how relevant their recommendations are to users.

Data scientists have turned to <I>hybrid recommenders</I>, in which multiple simple recommenders are combined, to surmount weaknesses displayed by any component recommender in isolation.

A variety of data scientists, in a number of prominent articles on Towards Data Science, use a weighted average between collaborative and content-based recommendations, without comparing this approach to other hybrid approaches.
- E.g., the less well-known approach of switching between content and collaborative recommendation, which, to my knowledge, has not been discussed on Towards Data Science.

<b>Question</b>: Amongst a variety of choices for hybrid algorithm, which one works best? In other words, <i>how</i> should two recommenders be combined, given a dataset?

## Solution

Project codes up from scratch and compares the performance of four movie recommenders against the MovieLens dataset: 
- Content-based
- By-item collaborative filter
- Hybrid weighted average
- Hybrid content-to-collaborative switch once seed has enough ratings

I find that content recommendation performs better than collaborative on MAP@10 accuracy metric for less-rated movie seeds, while collaborative performs better for often-rated seeds.

I further show that averaging—used by many data scientists on TDS—performs only half as well as the uncommon switch algorithm. This calls into question the use of averaging if the goal is accuracy.

## Demo

See [here](https://github.com/jzymet/recommender/blob/master/demo.ipynb).