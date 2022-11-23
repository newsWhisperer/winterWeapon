import pandas as pd

from pathlib import Path
import os.path
import io
#import requests
import glob


import nltk
import sklearn
from nltk.corpus import stopwords
from HanTa import HanoverTagger as ht
from textblob_de import TextBlobDE
import math
import re
import random

from sklearn.decomposition import PCA

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm

DATA_PATH = Path.cwd()
if(not os.path.exists(DATA_PATH / 'csv')):
    os.mkdir(DATA_PATH / 'csv')
if(not os.path.exists(DATA_PATH / 'img')):
    os.mkdir(DATA_PATH / 'img')

def getNewsFiles():
    fileName = './csv/news_????_??.csv'
    files = glob.glob(fileName)
    return files  

def getNewsDFbyList(files):    
    newsDF = pd.DataFrame(None)
    for file in files:
        df = pd.read_csv(file, delimiter=',')
        if(newsDF.empty):
            newsDF = df
        else:
            newsDF = pd.concat([newsDF, df])
    newsDF = newsDF.sort_values(by=['published'], ascending=True)        
    return newsDF 

def getNewsDF():
    files = getNewsFiles()
    newsDF = getNewsDFbyList(files)
    return newsDF         

keywordsColorsDF = pd.read_csv(DATA_PATH / 'keywords.csv', delimiter=',')
topicsColorsDF = keywordsColorsDF.drop_duplicates(subset=['topic'])

newsDf = getNewsDF()
print(newsDf)   

language = 'ger'
nltk.download('punkt')
nltk.download('stopwords')
tagger = ht.HanoverTagger('morphmodel_'+language+'.pgz')
german_stop_words = set(stopwords.words('german'))

def generateTokensWithPosition(quote):
    sentences = nltk.sent_tokenize(quote,language='german')   
    for sentence in sentences:
        positionSentence = quote.find(sentence)
        lastWord = None
        tokens = nltk.tokenize.word_tokenize(sentence,language='german') 
        lemmata = tagger.tag_sent(tokens,taglevel = 2)
        for (orig,lemma,gramma) in lemmata:
         if(len(orig)>2):
          if(not orig in german_stop_words):
            positionWord = sentence.find(orig)
            yield [orig, positionSentence+positionWord]
            if(lastWord):
                yield [(lastWord+' '+lemma), positionSentence+positionWord]
            lastWord = lemma 


emptyTopics = {'summary':0}
for index2, column2 in keywordsColorsDF.iterrows():     
    topic = column2['topic']
    emptyTopics[topic] = 0

i=0
topicWordsAbs = {'summaryOfAllWords': emptyTopics.copy()}
for index, column in newsDf.iterrows():
    i += 1
    if(i % 50 == 0):
        print(i)

    quote = str(column.title)+' ' +str(column.description)+' '+str(column.content)
    #quote = str(column.title)+' ' +str(column.description)
    for tokenAndPosition in generateTokensWithPosition(quote):
        token = tokenAndPosition[0]
        tokenPosition = tokenAndPosition[1]
        if(not token in topicWordsAbs):
            topicWordsAbs[token] = emptyTopics.copy()  
        for index2, column2 in keywordsColorsDF.iterrows(): 
            found = 0.0
            if(column2['keyword'] == column['keyword']):
               found = 0.05
            keywords = column2['keyword'].strip("'").split(" ")
            topic = column2['topic']
            for keyword in keywords:
                if(keyword in quote):
                    for keyPosition in [m.start() for m in re.finditer(keyword, quote)]:
                        distance = abs(tokenPosition - keyPosition)
                        factor = math.sqrt(1/(1+distance*0.25))/len(keywords)
                        if(factor>found):
                            found = factor  
            topicWordsAbs[token][topic] += found
            topicWordsAbs[token]['summary'] += found
            topicWordsAbs['summaryOfAllWords'][topic] += found
            topicWordsAbs['summaryOfAllWords']['summary'] += found

overallProbability = emptyTopics.copy()
for topic in overallProbability:
    if(not topic == 'summary'):
        if(topicWordsAbs['summaryOfAllWords']['summary'] > 0):
            overallProbability[topic] = float(topicWordsAbs['summaryOfAllWords'][topic])/float(topicWordsAbs['summaryOfAllWords']['summary']) 

## now increase all counting by sqrt(n), but minimum of overall probability
for word in topicWordsAbs:
    if(word != 'summaryOfAllWords'):
        data = topicWordsAbs[word]
        for topic in overallProbability:   
            if(not topic == 'summary'):
                frac = overallProbability[topic]
                delta = math.sqrt(frac+topicWordsAbs[word][topic])
                topicWordsAbs[word][topic] += delta
                topicWordsAbs['summaryOfAllWords'][topic] += delta
                topicWordsAbs[word]['summary'] += delta
                topicWordsAbs['summaryOfAllWords']['summary'] += delta  

emptyCol = emptyTopics.copy()
emptyCol['word'] = 'oneWord'
topicWordsRel = {}  
for word in topicWordsAbs:
    if(word == 'summaryOfAllWords'):  
        relData = topicWordsAbs[word].copy()
    else:    
        data = topicWordsAbs[word]
        relData = emptyCol.copy()
        relData['word'] = word
        relData['summary'] = topicWordsAbs[word]['summary']
        for topic in data:
            if(not topic in ['word','summary']):
                if(not topicWordsAbs['summaryOfAllWords'][topic] == 0): 
                    if(topicWordsAbs['summaryOfAllWords'][topic]*topicWordsAbs[word]['summary'] > 0): 
                      relValue = topicWordsAbs[word][topic]*topicWordsAbs['summaryOfAllWords']['summary']/(topicWordsAbs['summaryOfAllWords'][topic]*topicWordsAbs[word]['summary'])   #Bayes
                      relData[topic] = math.log(relValue)
    topicWordsRel[word] = relData 
topicWordsRelDF = pd.DataFrame.from_dict(topicWordsRel, orient='index', columns=emptyCol.keys()) 
topicWordsRelDF.to_csv(DATA_PATH / 'csv' / "words_bayes_topic_all.csv", index=True) 

#PCA
numberComponents = 5  #0.5*len(topics), minimum: 4
dfn = topicWordsRelDF.drop(columns = ['word'])

dfn['const0'] = 1.0
pca = PCA(n_components=numberComponents)
pca.fit(dfn)
apca = pca.fit_transform(dfn)
dfpca = pd.DataFrame(apca)
dfpca['word'] = topicWordsRelDF.index
dfpca['summary'] = topicWordsRelDF['summary'].values
dfpca.to_csv(DATA_PATH / "csv" /"words_bayes_topic_pca.csv", index=False)

def combine_hex_values(d):
  d_items = sorted(d.items())
  tot_weight = sum(d.values())
  red = int(sum([int(k[:2], 16)*v for k, v in d_items])/tot_weight)
  green = int(sum([int(k[2:4], 16)*v for k, v in d_items])/tot_weight)
  blue = int(sum([int(k[4:6], 16)*v for k, v in d_items])/tot_weight)
  zpad = lambda x: x if len(x)==2 else '0' + x
  return zpad(hex(red)[2:]) + zpad(hex(green)[2:]) + zpad(hex(blue)[2:])

plt.figure( figsize=(20,15) )
plt.xlim([-4, 4])
plt.ylim([-4, 4])
i=0
for index, column in dfpca.iterrows():
  i += 1
  if(i % 50 == 0):
     print(i)    
  if(not " " in str(column['word'])): 
    maxColor = '#000000'
    nxtColor = '#555555'
    maxprobabiliyty = -15  #log!
    nxtprobabiliyty = -15  

    for index2, column2 in topicsColorsDF.iterrows():
        topic = column2['topic']    
        if(str(column['word']) in topicWordsRelDF[topic]):
            if(topicWordsRelDF[topic][str(column['word'])]> maxprobabiliyty):
                maxprobabiliyty = topicWordsRelDF[topic][str(column['word'])]
                maxColor = column2['topicColor']
    for index2, column2 in topicsColorsDF.iterrows():
        topic = column2['topic']    
        if(str(column['word']) in topicWordsRelDF[topic]):
            if(maxprobabiliyty > topicWordsRelDF[topic][str(column['word'])] > nxtprobabiliyty):
                nxtprobabiliyty = topicWordsRelDF[topic][str(column['word'])]
                nxtColor = column2['topicColor']
    if((maxprobabiliyty < -12) & (nxtprobabiliyty < -12)):
        maxColor = '#555555'
        nxtColor = '#555555'                    
 
    ##maxColor = '#'+combine_hex_values({maxColor: math.exp(maxprobabiliyty) , nxtColor: math.exp(nxtprobabiliyty)})          
         
    x = random.uniform(-0.1, 0.1)+column[2]
    y = random.uniform(-0.1, 0.1)+column[3]
    s = (2+math.sqrt(1+math.sqrt(column['summary'])))
    plt.text(x, y, column['word'], color='#ffffff', fontsize=s, ha='center', va='center', zorder=s-1E-7, fontweight='bold')
    plt.text(x, y, column['word'], color=maxColor, fontsize=s, ha='center', va='center', zorder=s)

colorLeg = list(topicsColorsDF['topicColor'])#.reverse()
colorLeg.reverse()
labelLeg = list(topicsColorsDF['topic'])#.reverse()
labelLeg.reverse()
custom_lines = [plt.Line2D([],[], ls="", marker='.', 
                mec='k', mfc=c, mew=.1, ms=20) for c in colorLeg]
             
leg = plt.legend(custom_lines, labelLeg, 
          loc='center left', fontsize=10, bbox_to_anchor=(0.9, .80))
leg.set_title("Topics", prop = {'size':12}) 

plt.savefig(DATA_PATH / 'img' / 'words_bayes_topic_pca.png', dpi=300)  

