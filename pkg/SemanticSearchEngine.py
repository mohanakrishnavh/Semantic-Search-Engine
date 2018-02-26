import collections
import os

from nltk import pos_tag
from nltk.corpus import wordnet as wn
from nltk.parse.corenlp import CoreNLPDependencyParser
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize
import pysolr

from pkg.IndexCreation import IndexCreation


class SemanticSearchEngine:
    
    def getArticleAndWordCount(self, path):
        print("Number of articles:", str(len(os.listdir(path))))
        indexSentenceMap = collections.OrderedDict()
        wordCount = 0
        ic = IndexCreation()
        data = ic.readArticles(path)
        data = ic.removeArticleTitle(data)
        for i in range(0, len(data)):
            for j in range(0, len(data[i])):
                tokenizedWords = word_tokenize(data[i][j])
                index = 'A' + str(i + 1) + 'S' + str(j + 1)
                indexSentenceMap[index] = data[i][j]
                wordCount += len(tokenizedWords)
        print("Number of words in the corpus:", str(wordCount))
        return indexSentenceMap

    def processQueryToExtractWords(self, query):
        return list(set(word_tokenize(query)))
    
    def searchInSolr(self, query, indexSentenceMap):
        solr = pysolr.Solr('http://localhost:8983/solr/task2')
        query = "words:" + " || words:".join(query)
        print("Joined Query: ", query)
        results = solr.search(query)
        print()
        print("Top 10 documents that closely match the query")
        for result in results:
            print(result['id'].ljust(10), indexSentenceMap[result['id']])
    
    def processQueryToDoLemmatization(self, words):
        lemmas = []
        wnl = WordNetLemmatizer()
        for word in words:
            lemmas.append(wnl.lemmatize(word))
        return lemmas
    
    def processQueryToDoImprovedLemmatization(self, posTags):
        lemmas = []
        wnl = WordNetLemmatizer()
        ic = IndexCreation()
        for word, tag in posTags:
            wnTag = ic.getWordnetTag(tag)
            if wnTag is None:
                lemmas.append(wnl.lemmatize(word))
            else:
                lemmas.append(wnl.lemmatize(word, pos=wnTag))
        return lemmas
    
    def processQueryToDoStemming(self, words):
        stems = []
        stemmer = PorterStemmer()
        for word in words:
            stems.append(stemmer.stem(word))
        return stems
    
    def processQueryToDoPOSTagging(self, words):
        posTags = []
        for taggedWord in pos_tag(words):
            posTags.append(taggedWord[1])
        return posTags
    
    def processQueryToDoPOSTaggingWithWords(self, words):
        return pos_tag(words)
    
    def processQueryToExtractHeadWord(self, query):
        dependency_parser = CoreNLPDependencyParser('http://localhost:9000')
        headWord = None
        parsedSentence = list(dependency_parser.raw_parse(query))[0]
        rootValue = list(list(parsedSentence.nodes.values())[0]['deps']['ROOT'])[0]
        for n in parsedSentence.nodes.values():
            if n['address'] == rootValue:
                headWord = n['word']
                break
        return headWord 
    
    def processQueryToExtractImprovisedHeadWord(self, query):
        dependency_parser = CoreNLPDependencyParser('http://localhost:9000')
        headWord = None
        parsedSentence = list(dependency_parser.raw_parse(query))[0]
        rootValue = list(list(parsedSentence.nodes.values())[0]['deps']['ROOT'])[0]
        for n in parsedSentence.nodes.values():
            if n['address'] == rootValue:
                headWord = n['word']
                if len(headWord):
                    _, tag = pos_tag([headWord])[0]
                    wnTag = IndexCreation().getWordnetTag(tag)
                    if wnTag is not None:
                        synset = wn.synsets(headWord, pos=wnTag)
                        if len(synset) > 0:
                            headWord = synset[0].name().split('.')[0]
                break
        return headWord
    
#     def processQueryToExtractHeadWordWSD(self, query, headWord):
#         _, tag = pos_tag([headWord])[0]
#         wsd = headWord
#         wsdSynset = lesk(query, headWord, pos=IndexCreation().getWordnetTagLesk(tag))
#         if len(wsdSynset) > 0:
#             wsd = wsdSynset.name().split('.')[0]
#         return wsd
    
    def processQueryToExtractHypernyms(self, words):
        hypernyms = []
        for word in words:
            synset = wn.synsets(word)
            if len(synset) > 0:
                if len(synset[0].hypernyms()) > 0:
                    hypernyms.append(synset[0].hypernyms()[0].name().split('.')[0])
                else:
                    hypernyms.append(word)
        return hypernyms
    
    def processQueryToExtractImprovisedHypernyms(self, posTags):
        hypernyms = []
        for word, tag in posTags:
            wnTag = IndexCreation().getWordnetTag(tag)
            if wnTag is not None:
                synset = wn.synsets(word, pos=wnTag)
            else:
                synset = wn.synsets(word)
            if len(synset) > 0:
                if len(synset[0].hypernyms()) > 0:
                    hypernyms.append(synset[0].hypernyms()[0].name().split('.')[0])
        return hypernyms
    
    def processQueryToExtractHyponyms(self, words):
        hyponyms = []
        for word in words:
            synset = wn.synsets(word)
            if len(synset) > 0:
                if len(synset[0].hyponyms()) > 0:
                    hyponyms.append(synset[0].hyponyms()[0].name().split('.')[0])
                else:
                    hyponyms.append(word)
        return hyponyms
    
    def processQueryToExtractImprovisedHyponyms(self, posTags):
        hyponyms = []
        for word, tag in posTags:
            wnTag = IndexCreation().getWordnetTag(tag)
            if wnTag is not None:
                synset = wn.synsets(word, pos=wnTag)
            else:
                synset = wn.synsets(word)
            if len(synset) > 0:
                if len(synset[0].hyponyms()) > 0:
                    hyponyms.append(synset[0].hyponyms()[0].name().split('.')[0])
        return hyponyms
    
    def processQueryToExtractMeronyms(self, words):
        meronyms = []
        for word in words:
            synset = wn.synsets(word)
            if len(synset) > 0:
                if len(synset[0].part_meronyms()) > 0:
                    meronyms.append(synset[0].part_meronyms()[0].name().split('.')[0])
                else:
                    meronyms.append(word)
        return meronyms
    
    def processQueryToExtractImprovisedMeronyms(self, posTags):
        meronyms = []
        for word, tag in posTags:
            wnTag = IndexCreation().getWordnetTag(tag)
            if wnTag is not None:
                synset = wn.synsets(word, pos=wnTag)
            else:
                synset = wn.synsets(word)
            if len(synset) > 0:
                if len(synset[0].part_meronyms()) > 0:
                    meronyms.append(synset[0].part_meronyms()[0].name().split('.')[0])
        return meronyms
    
    def processQueryToExtractHolonyms(self, words):
        holonyms = []
        for word in words:
            synset = wn.synsets(word)
            if len(synset) > 0:
                if len(synset[0].part_holonyms()) > 0:
                    holonyms.append(synset[0].part_holonyms()[0].name().split('.')[0])
                else:
                    holonyms.append(word)
        return holonyms
    
    def processQueryToExtractImprovisedHolonyms(self, posTags):
        holonyms = []
        for word, tag in posTags:
            wnTag = IndexCreation().getWordnetTag(tag)
            if wnTag is not None:
                synset = wn.synsets(word, pos=wnTag)
            else:
                synset = wn.synsets(word)
            if len(synset) > 0:
                if len(synset[0].part_holonyms()) > 0:
                    holonyms.append(synset[0].part_holonyms()[0].name().split('.')[0])
        return holonyms
    
    def processQueryToExtractAllFeatures(self, query):
        words = self.processQueryToExtractWords(query)
        lemmas = self.processQueryToDoLemmatization(words)
        stems = self.processQueryToDoStemming(words)
        posTags = self.processQueryToDoPOSTagging(words)
        headWord = self.processQueryToExtractHeadWord(query)
        hypernyms = self.processQueryToExtractHypernyms(words)
        hyponyms = self.processQueryToExtractHyponyms(words)
        meronyms = self.processQueryToExtractMeronyms(words)
        holonyms = self.processQueryToExtractHolonyms(words)
        return [words, lemmas, stems, posTags, headWord, hypernyms, hyponyms, meronyms, holonyms]
    
    def improvisationTask(self, query):
        words = self.processQueryToExtractWords(query)
        posTags = self.processQueryToDoPOSTaggingWithWords(words)
        lemmas = self.processQueryToDoImprovedLemmatization(posTags)
        stems = self.processQueryToDoStemming(words)
        headWord = self.processQueryToExtractImprovisedHeadWord(query)
#         headwsd = self.processQueryToExtractHeadWordWSD(query, headWord)
        hypernyms = self.processQueryToExtractImprovisedHypernyms(posTags)
        hyponyms = self.processQueryToExtractImprovisedHyponyms(posTags)
        meronyms = self.processQueryToExtractImprovisedMeronyms(posTags)
        holonyms = self.processQueryToExtractImprovisedHolonyms(posTags)
        return [words, lemmas, stems, posTags, headWord, hypernyms, hyponyms, meronyms, holonyms]
    
    def searchInSolrWithMultipleFeatures(self, featuresList, indexSentenceMap):
        solr = pysolr.Solr('http://localhost:8983/solr/task3')
        query = []
        if len(featuresList[0]) > 0:
            query.append("words:(" + " ".join(featuresList[0]) + ")")
        if len(featuresList[1]) > 0:
            query.append("lemmas:(" + " ".join(featuresList[1]) + ")")
        if len(featuresList[2]) > 0:
            query.append("stems:(" + " ".join(featuresList[2]) + ")")
        if len(featuresList[3]) > 0:
            query.append("POS:(" + " ".join(featuresList[3]) + ")")
        if len(featuresList[4]) > 0:
            query.append("head:(" + featuresList[4] + ")")
        if len(featuresList[5]) > 0:
            query.append("hypernyms:(" + " ".join(featuresList[5]) + ")")
        if len(featuresList[6]) > 0:
            query.append("hyponyms:(" + " ".join(featuresList[6]) + ")")
        if len(featuresList[7]) > 0:
            query.append("meronyms:(" + " ".join(featuresList[7]) + ")")
        if len(featuresList[8]) > 0:
            query.append("holonyms:(" + " ".join(featuresList[8]) + ")")
        joinedQuery = ' || '.join(item for item in query)
        print("Joined Query: ", joinedQuery)
        results = solr.search(joinedQuery)
        print()
        print("Top 10 documents that closely match the query")
        for result in results:
            print(result['id'].ljust(10), indexSentenceMap[result['id']])
            
    def searchInSolrWithMultipleImprovisedFeatures(self, featuresList, indexSentenceMap):
        solr = pysolr.Solr('http://localhost:8983/solr/task4')
        query = []
        if len(featuresList[0]) > 0:
            query.append("words:(" + " ".join(featuresList[0]) + ")^1.0")
        if len(featuresList[1]) > 0:
            query.append("lemmas:(" + " ".join(featuresList[1]) + ")^10.0")
        if len(featuresList[2]) > 0:
            query.append("stems:(" + " ".join(featuresList[2]) + ")^6.0")
        if len(featuresList[3]) > 0:
            query.append("POSWithWords:(" + " ".join(str(term) for term in featuresList[3]) + ")^1.0")
        if len(featuresList[4]) > 0:
            query.append("head:(" + featuresList[4] + ")^1.0")
        if len(featuresList[5]) > 0:
            query.append("hypernyms:(" + " ".join(featuresList[5]) + ")^7.0")
        if len(featuresList[6]) > 0:
            query.append("hyponyms:(" + " ".join(featuresList[6]) + ")^1.0")
        if len(featuresList[7]) > 0:
            query.append("meronyms:(" + " ".join(featuresList[7]) + ")^1.0")
        if len(featuresList[8]) > 0:
            query.append("holonyms:(" + " ".join(featuresList[8]) + ")^1.0")
        joinedQuery = ' || '.join(item for item in query)
        print("Joined Query: ", joinedQuery)
        results = solr.search(joinedQuery)
        print()
        print("Top 10 documents that closely match the query")
        for result in results:
            print(result['id'].ljust(10), indexSentenceMap[result['id']])
    
    
if __name__ == '__main__':
    sse = SemanticSearchEngine()
    path = '/Users/deepaks/Documents/workspace/Semantic_Search_Engine/Data/'
    inputChoice = input("Enter the option to continue with\n 1. Task2 \n 2. Task3\n 3. Task4\n ") 
    indexSentenceMap = sse.getArticleAndWordCount(path)
    query = input("Enter the input query: ")
    # Task 2
    if inputChoice == "1":
        processedQuery = sse.processQueryToExtractWords(query)
        sse.searchInSolr(processedQuery, indexSentenceMap) 
    # Task 3
    elif inputChoice == "2":
        featuresList = sse.processQueryToExtractAllFeatures(query)
        sse.searchInSolrWithMultipleFeatures(featuresList, indexSentenceMap)
    # Task 4    
    elif inputChoice == "3":
        featuresList = sse.improvisationTask(query)
        sse.searchInSolrWithMultipleImprovisedFeatures(featuresList, indexSentenceMap)
        
