from _functools import reduce
import collections
import csv
import io
import json
import os

from nltk import pos_tag
from nltk import tokenize
from nltk.corpus import wordnet as wn
from nltk.parse.corenlp import CoreNLPDependencyParser
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize
import pysolr

import pandas as pd


class IndexCreation():

    def preprocessCorpus(self, path):
        print("Pre-processing and Tokenizing...")
        data = self.readArticles(path)
        data = self.removeArticleTitle(data)

        indexWordsMap, indexSentenceMap = self.createIndexMap(data)
        with io.open('MainData.csv', 'w', encoding='utf-8', errors='ignore') as f:
            w = csv.writer(f)
            w.writerows(indexWordsMap.items())
        wordsDFrame = pd.DataFrame(list(indexWordsMap.items()), columns=['id', 'words'])

        jsonFileName = 'Task2.json'
        wordsDFrame.to_json(jsonFileName, orient='records')
        return data, indexWordsMap, indexSentenceMap, wordsDFrame, jsonFileName

    def readArticles(self, path):
        data = []
        for f in sorted(os.listdir(path), key=lambda x: int(x.split('.')[0])):
            with io.open(path + f, 'r', encoding='utf-8', errors='ignore') as dataFile:
                data.append(dataFile.read())
        return data

    def removeArticleTitle(self, data):
        for i in range(len(data)):
            sentences = tokenize.sent_tokenize(data.pop(i).strip())
            temp = sentences.pop(0).split('\n\n')
            if len(temp) == 2:
                sentences.insert(0, temp[1])
            data.insert(i, sentences)
        return data

    def createIndexMap(self, data):
        indexWordsMap = collections.OrderedDict()
        indexSentenceMap = collections.OrderedDict()
        for i in range(0, len(data)):
            for j in range(0, len(data[i])):
                index = 'A' + str(i + 1) + 'S' + str(j + 1)
                indexSentenceMap[index] = data[i][j]
                indexWordsMap[index] = list(set(word_tokenize(data[i][j])))
        return indexWordsMap, indexSentenceMap

    def extractFeatures(self, indexWordsMap, indexSentenceMap):
        wordsDFrame = pd.DataFrame(list(indexWordsMap.items()), columns=['id', 'words'])
        indexLemmaMap = self.lemmatizeWords(indexWordsMap)
        lemmaDFrame = pd.DataFrame(list(indexLemmaMap.items()), columns=['id', 'lemmas'])
        indexStemMap = self.stemWords(indexWordsMap)
        stemDFrame = pd.DataFrame(list(indexStemMap.items()), columns=['id', 'stems'])
        indexPOSMap = self.tagPOSWords(indexWordsMap)
        POSDFrame = pd.DataFrame(list(indexPOSMap.items()), columns=['id', 'POS'])
        indexHeadMap = self.findHeadWord(indexSentenceMap)
        HeadDFrame = pd.DataFrame(list(indexHeadMap.items()), columns=['id', 'head'])
        indexHypernymMap = self.extractHypernyms(indexWordsMap)
        HypernymDFrame = pd.DataFrame(list(indexHypernymMap.items()), columns=['id', 'hypernyms'])
        indexHyponymMap = self.extractHyponyms(indexWordsMap)
        HyponymDFrame = pd.DataFrame(list(indexHyponymMap.items()), columns=['id', 'hyponyms'])
        indexMeronymMap = self.extractMeronyms(indexWordsMap)
        MeronymDFrame = pd.DataFrame(list(indexMeronymMap.items()), columns=['id', 'meronyms'])
        indexHolonymMap = self.extractHolonyms(indexWordsMap)
        HolonymDFrame = pd.DataFrame(list(indexHolonymMap.items()), columns=['id', 'holonyms'])
        dfList = [wordsDFrame, lemmaDFrame, stemDFrame, POSDFrame, HeadDFrame, HypernymDFrame, HyponymDFrame, MeronymDFrame, HolonymDFrame]
        finalDFrame = reduce(lambda left, right: pd.merge(left, right, on='id'), dfList)

        jsonFileName = 'Task3.json'
        finalDFrame.to_json(jsonFileName, orient='records')
        return jsonFileName
    
    def extractImprovisedFeatures(self, indexWordsMap, indexSentenceMap):
        wordsDFrame = pd.DataFrame(list(indexWordsMap.items()), columns=['id', 'words'])
        indexPOSWithWordsMap = self.tagPOSWithWords(indexWordsMap)
        POSDFrame = pd.DataFrame(list(indexPOSWithWordsMap.items()), columns=['id', 'POSWithWords'])
        indexLemmaMap = self.improvedLemmatizeWords(indexPOSWithWordsMap)
        lemmaDFrame = pd.DataFrame(list(indexLemmaMap.items()), columns=['id', 'lemmas'])
        indexStemMap = self.stemWords(indexWordsMap)
        stemDFrame = pd.DataFrame(list(indexStemMap.items()), columns=['id', 'stems'])
        indexHeadMap = self.findImprovisedHeadWord(indexSentenceMap)
        HeadDFrame = pd.DataFrame(list(indexHeadMap.items()), columns=['id', 'head'])
#         indexWSDHeadWordMap = self.extractWSDForHeadWord(indexHeadMap, indexSentenceMap)
#         WSDHeadWordDFrame = pd.DataFrame(list(indexWSDHeadWordMap.items()), columns=['id', 'head_wsds'])
        indexHypernymMap = self.extractImprovisedHypernyms(indexWordsMap)
        HypernymDFrame = pd.DataFrame(list(indexHypernymMap.items()), columns=['id', 'hypernyms'])
        indexHyponymMap = self.extractImprovisedHyponyms(indexWordsMap)
        HyponymDFrame = pd.DataFrame(list(indexHyponymMap.items()), columns=['id', 'hyponyms'])
        indexMeronymMap = self.extractImprovisedMeronyms(indexWordsMap)
        MeronymDFrame = pd.DataFrame(list(indexMeronymMap.items()), columns=['id', 'meronyms'])
        indexHolonymMap = self.extractImprovisedHolonyms(indexWordsMap)
        HolonymDFrame = pd.DataFrame(list(indexHolonymMap.items()), columns=['id', 'holonyms'])
        dfList = [wordsDFrame, lemmaDFrame, stemDFrame, POSDFrame, HeadDFrame, HypernymDFrame, HyponymDFrame, MeronymDFrame, HolonymDFrame]
        finalDFrame = reduce(lambda left, right: pd.merge(left, right, on='id'), dfList)

        jsonFileName = 'Task4.json'
        finalDFrame.to_json(jsonFileName, orient='records')
        return jsonFileName

    def lemmatizeWords(self, indexWordsMap):
        print("Lemmatizing...")
        indexLemmaMap = collections.OrderedDict()
        wnl = WordNetLemmatizer()
        for k, v in indexWordsMap.items():
            indexLemmaMap[k] = [wnl.lemmatize(word) for word in v]
        return indexLemmaMap
    
    def improvedLemmatizeWords(self, indexPOSWithWordsMap):
        print("Improvised Lemmatizing...")
        indexLemmaMap = collections.OrderedDict()
        wnl = WordNetLemmatizer()
        for k, v in indexPOSWithWordsMap.items():
            lemmasList = []
            for word, tag in v:
                wnTag = self.getWordnetTag(tag)
                if wnTag is None:
                    lemmasList.append(wnl.lemmatize(word))
                else:
                    lemmasList.append(wnl.lemmatize(word, pos=wnTag))
            indexLemmaMap[k] = lemmasList
        return indexLemmaMap
    
    def getWordnetTag(self, tag):
        if tag.startswith('J'):
            return wn.ADJ
        elif tag.startswith('V'):
            return wn.VERB
        elif tag.startswith('N'):
            return wn.NOUN
        elif tag.startswith('R'):
            return wn.ADV
        else:
            return None

    def stemWords(self, indexWordsMap):
        print("Stemming...")
        indexStemMap = collections.OrderedDict()
        stemmer = PorterStemmer()
        for k, v in indexWordsMap.items():
            indexStemMap[k] = [stemmer.stem(word) for word in v]
        return indexStemMap

    def tagPOSWords(self, indexWordsMap):
        print("POS Tagging...")
        indexPOSMap = collections.OrderedDict()
        for k, v in indexWordsMap.items():
            posTags = []
            for taggedWord in pos_tag(v):
                posTags.append(taggedWord[1])
            indexPOSMap[k] = posTags
        return indexPOSMap
    
    def tagPOSWithWords(self, indexWordsMap):
        print("Improvised POS Tagging...")
        indexPOSWithWordsMap = collections.OrderedDict()
        for k, v in indexWordsMap.items():
            indexPOSWithWordsMap[k] = pos_tag(v)
        return indexPOSWithWordsMap

    def findHeadWord(self, indexSentenceMap):
        print("Head Word Extraction...")
        indexHeadMap = collections.OrderedDict()
        dependency_parser = CoreNLPDependencyParser('http://localhost:9000')
        for k, v in indexSentenceMap.items():
            parsedSentence = list(dependency_parser.raw_parse(v))[0]
            rootValue = list(list(parsedSentence.nodes.values())[0]['deps']['ROOT'])[0]
            for n in parsedSentence.nodes.values():
                if n['address'] == rootValue:
                    indexHeadMap[k] = n['word']
                    break
        return indexHeadMap
    
    def findImprovisedHeadWord(self, indexSentenceMap):
        print("Improvised Head Word Extraction...")
        indexHeadMap = collections.OrderedDict()
        dependency_parser = CoreNLPDependencyParser('http://localhost:9000')
        for k, v in indexSentenceMap.items():
            parsedSentence = list(dependency_parser.raw_parse(v))[0]
            rootValue = list(list(parsedSentence.nodes.values())[0]['deps']['ROOT'])[0]
            for n in parsedSentence.nodes.values():
                if n['address'] == rootValue:
                    headWord = n['word']
                    if len(headWord) > 0:
                        _, tag = pos_tag([headWord])[0]
                        wnTag = IndexCreation().getWordnetTag(tag)
                        if wnTag is not None:
                            synset = wn.synsets(headWord, pos=wnTag)
                            if len(synset) > 0:
                                headWord = synset[0].name().split('.')[0]
                    indexHeadMap[k] = headWord
                    break
        return indexHeadMap

    def extractHypernyms(self, indexWordsMap):
        print("Hypernyms Extraction...")
        indexHypernymMap = collections.OrderedDict()
        for k, v in indexWordsMap.items():
            hypernymList = []
            for word in v:
                synset = wn.synsets(word)
                if len(synset) > 0:
                    if len(synset[0].hypernyms()) > 0:
                        hypernymList.append(synset[0].hypernyms()[0].name().split('.')[0])
                else:
                    hypernymList.append(word)
            indexHypernymMap[k] = hypernymList
        return indexHypernymMap
    
    def extractImprovisedHypernyms(self, indexWordsMap):
        print("Improvised Hypernyms Extraction...")
        indexHypernymMap = collections.OrderedDict()
        for k, v in indexWordsMap.items():
            hypernymList = []
            for word in v:
                _, tag = pos_tag([word])[0]
                wnTag = IndexCreation().getWordnetTag(tag)
                if wnTag is not None:
                    synset = wn.synsets(word, pos=wnTag)
                else:
                    synset = wn.synsets(word)
                if len(synset) > 0:
                    if len(synset[0].hypernyms()) > 0:
                        hypernymList.append(synset[0].hypernyms()[0].name().split('.')[0])
            indexHypernymMap[k] = hypernymList
        return indexHypernymMap

    def extractHyponyms(self, indexWordsMap):
        print("Hyponyms Extraction...")
        indexHyponymMap = collections.OrderedDict()
        for k, v in indexWordsMap.items():
            hyponymList = []
            for word in v:
                synset = wn.synsets(word)
                if len(synset) > 0:
                    if len(synset[0].hyponyms()) > 0:
                        hyponymList.append(synset[0].hyponyms()[0].name().split('.')[0])
                else:
                    hyponymList.append(word)
            indexHyponymMap[k] = hyponymList
        return indexHyponymMap
    
    def extractImprovisedHyponyms(self, indexWordsMap):
        print("Improvised Hyponyms Extraction...")
        indexHyponymMap = collections.OrderedDict()
        for k, v in indexWordsMap.items():
            hyponymList = []
            for word in v:
                _, tag = pos_tag([word])[0]
                wnTag = IndexCreation().getWordnetTag(tag)
                if wnTag is not None:
                    synset = wn.synsets(word, pos=wnTag)
                else:
                    synset = wn.synsets(word)
                if len(synset) > 0:
                    if len(synset[0].hyponyms()) > 0:
                        hyponymList.append(synset[0].hyponyms()[0].name().split('.')[0])
            indexHyponymMap[k] = hyponymList
        return indexHyponymMap

    def extractMeronyms(self, indexWordsMap):
        print("Meronyms Extraction...")
        indexMeronymMap = collections.OrderedDict()
        for k, v in indexWordsMap.items():
            meronymList = []
            for word in v:
                synset = wn.synsets(word)
                if len(synset) > 0:
                    if len(synset[0].part_meronyms()) > 0:
                        meronymList.append(synset[0].part_meronyms()[0].name().split('.')[0])
                else:
                    meronymList.append(word)
            indexMeronymMap[k] = meronymList
        return indexMeronymMap
    
    def extractImprovisedMeronyms(self, indexWordsMap):
        print("Improvised Meronyms Extraction...")
        indexMeronymMap = collections.OrderedDict()
        for k, v in indexWordsMap.items():
            meronymList = []
            for word in v:
                _, tag = pos_tag([word])[0]
                wnTag = IndexCreation().getWordnetTag(tag)
                if wnTag is not None:
                    synset = wn.synsets(word, pos=wnTag)
                else:
                    synset = wn.synsets(word)
                if len(synset) > 0:
                    if len(synset[0].part_meronyms()) > 0:
                        meronymList.append(synset[0].part_meronyms()[0].name().split('.')[0])
            indexMeronymMap[k] = meronymList
        return indexMeronymMap

    def extractHolonyms(self, indexWordsMap):
        print("Holonyms Extraction...")
        indexHolonymMap = collections.OrderedDict()
        for k, v in indexWordsMap.items():
            holonymList = []
            for word in v:
                synset = wn.synsets(word)
                if len(synset) > 0:
                    if len(synset[0].part_holonyms()) > 0:
                        holonymList.append(synset[0].part_holonyms()[0].name().split('.')[0])
                else:
                    holonymList.append(word)
            indexHolonymMap[k] = holonymList
        return indexHolonymMap
    
    def extractImprovisedHolonyms(self, indexWordsMap):
        print("Improvised Holonyms Extraction...")
        indexHolonymMap = collections.OrderedDict()
        for k, v in indexWordsMap.items():
            holonymList = []
            for word in v:
                _, tag = pos_tag([word])[0]
                wnTag = IndexCreation().getWordnetTag(tag)
                if wnTag is not None:
                    synset = wn.synsets(word, pos=wnTag)
                else:
                    synset = wn.synsets(word)
                if len(synset) > 0:
                    if len(synset[0].part_holonyms()) > 0:
                        holonymList.append(synset[0].part_holonyms()[0].name().split('.')[0])
            indexHolonymMap[k] = holonymList
        return indexHolonymMap
    
    # New Features
    def getWordnetTagLesk(self, tag):
        if tag.startswith('J'):
            return 'j'
        elif tag.startswith('V'):
            return 'v'
        elif tag.startswith('N'):
            return 'n'
        elif tag.startswith('R'):
            return 'r'
        else:
            return None
        
#     def extractWSDForHeadWord(self, indexHeadMap, indexSentenceMap):
#         print("Improvised Head-Word Sense Diambiguation..")
#         indexWSDMap = collections.OrderedDict()
#         for k, v in indexHeadMap.items():
#             wsd = v
#             word, tag = pos_tag([v])[0]
#             wsdSynset = lesk(indexSentenceMap[k], word, pos=self.getWordnetTagLesk(tag))
#             if len(wsdSynset) > 0:
#                 wsd = wsdSynset.name().split('.')[0]
#             indexWSDMap[k] = wsd    
#         return indexWSDMap

    def indexFeaturesWithSolr(self, jsonFileName, inputChoice):
        print("Indexing...")
        solr = pysolr.Solr('http://localhost:8983/solr/task' + str(int(inputChoice) + 1))
        solr.delete(q='*:*')
        with open("/Users/deepaks/Documents/workspace/Semantic_Search_Engine/pkg/" + jsonFileName, 'rb') as jsonFile:
            entry = json.load(jsonFile)
        solr.add(entry)


if __name__ == '__main__':
    ic = IndexCreation()
    path = '/Users/deepaks/Documents/workspace/Semantic_Search_Engine/Data/'
    inputChoice = input("Enter the option to continue with\n 1. Task2 \n 2. Task3\n 3. Task4\n ") 
    data, indexWordsMap, indexSentenceMap, wordsDFrame, jsonFileName = ic.preprocessCorpus(path)
    if inputChoice == "1":
        ic.indexFeaturesWithSolr(jsonFileName, inputChoice)
    elif inputChoice == "2":
        jsonFileName = ic.extractFeatures(indexWordsMap, indexSentenceMap)
        ic.indexFeaturesWithSolr(jsonFileName, inputChoice)
    elif inputChoice == "3":
        jsonFileName = ic.extractImprovisedFeatures(indexWordsMap, indexSentenceMap)
        ic.indexFeaturesWithSolr(jsonFileName, inputChoice)
    
