# Semantic_Search_Engine

Authors: Deepak Shanmugam, Mohanakrishna V H, Vidya Sri Mani

## Dataset Description

The BBC news website sources 2225 documents which primarily covers five broad fields of interest namely business, entertainment, politics, sports and technology that was published between 2004 and 2005. This corpus collected from this source is used in proceeding tasks.  
Natural Classes: 5 (business, entertainment, politics, sport, tech)  
Link to the Corpus: http://mlg.ucd.ie/datasets/bbc.html

## Problem Description

We have to implement a semantic search engine on a News Corpus, which will produce enhanced search results based on semantics. This can be achieved using various natural language Processing features and techniques. This project has to use a keyword-based strategy.

## Approach

Task 1: Aims at building a corpus obtained from the BBC News website.

Task 2: A keyword search index is created by segmentation, tokenization of corpus and indexing using SOLR.

Task 3: A semantic search index is created by segmentation, tokenization, lemmatization, Part of Speech Tagging, stemming features, syntactically parsing of corpus and indexing using SOLR.

Task 4: Improve the shallow NLP pipeline results using a combination of deeper NLP pipeline features.

## Result and Analysis

We will use rank of the query sentence and Mean Reciprocal Rank(MRR) to analyze our search results. We achieved an accuracy of 63%.

## Technology Used

Python, NLTK, SOLR, PySOLR(Wrapper), Stanford CoreNLP
