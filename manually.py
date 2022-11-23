import pandas as pd
import io
import os
import sys

from pathlib import Path
import os.path

import aiohttp
import asyncio
import requests
from urllib.parse import urlparse
import json
import time
import smtplib
import random
import hashlib


import datetime
from dateutil import parser
import re

#from bs4 import BeautifulSoup

#from deep_translator import GoogleTranslator
#from deep_translator import single_detection

DATA_PATH = Path.cwd()

keywordsDF = pd.read_csv(DATA_PATH / 'keywords.csv', delimiter=',')  #,index_col='keyword'
keywordsDF['uniqueString'] = keywordsDF['keyword'] + "_" + keywordsDF['language'] + "_" + keywordsDF['topic']
keywordsDF['crc'] = keywordsDF['uniqueString'].apply(
    lambda x: 
        hashlib.sha256(x.encode()).hexdigest()
)
keywordsDF = keywordsDF.sort_values(by=['ratioNew'], ascending=False)  



collectedNews = {}

def addNewsToCollection(data):
    global collectedNews

    year_month = '1970_01'
    pubDate = None
    try:
        pubDate = parser.parse(data['published'])
    except:
        print('date parse error 1')
    if(not pubDate):
      try:
        pubDate = parser.isoparse(data['published'])
      except:
        print('date parse error 2')   
    if(pubDate):
        year_month = pubDate.strftime('%Y_%m')


#    if(not data['language'] in collectedNews):
#        collectedNews[data['language']] = {}
    fileDate = 'news_'+year_month+'.csv'
    if(not fileDate in collectedNews):
        if(os.path.isfile(DATA_PATH / 'csv' / fileDate)):
            #df = pd.read_csv(DATA_PATH / fileDate, delimiter=',' ,index_col='url')
            df = pd.read_csv(DATA_PATH / 'csv' / fileDate, delimiter=',',index_col='index')
            collectedNews[fileDate] = df.to_dict('index')
        else:
            collectedNews[fileDate] = {}
    if(not data['url'] in collectedNews[fileDate]):
        #data = translateNews(data)
        #print(data['en'])
        #data = archiveUrl(data)
        collectedNews[fileDate][data['url']] = data
        return True
    return False

# index,url,valid,domain,title,description,image,published,archive,content,quote,language,keyword
def storeCollection():
    global collectedNews
    cols = ['url','valid','domain','title','description','image','published','archive','content','quote','language','keyword']
    for dateFile in collectedNews:
            df = pd.DataFrame.from_dict(collectedNews[dateFile], orient='index', columns=cols)
            #df.to_csv(DATA_PATH / dateFile, index=True) 
            df.to_csv(DATA_PATH / 'csv' / dateFile, index_label='index') 
    collectedNews = {}


def getDFfromGitHub(url, delimiter=','):
        stream=requests.get(url).content         
        dataframe=pd.read_csv(io.StringIO(stream.decode('utf-8')), delimiter=delimiter)
        dataframe = dataframe.sort_values(by=['published'], ascending=True)
        return dataframe

manualDF = pd.DataFrame(None)
gitNames = ["news_2022_01.csv","news_2022_02.csv","news_2022_03.csv","news_2022_04.csv","news_2022_05.csv","news_2022_06.csv",
            "news_2022_07.csv","news_2022_08.csv","news_2022_09.csv","news_2022_10.csv","news_2022_11.csv","news_2022_12.csv"]
for gitName in gitNames:
  gitUrl = "https://raw.githubusercontent.com/newsWhisperer/winterWeapon/main/csv/" + gitName
  df = getDFfromGitHub(gitUrl) 
   
  if(manualDF.empty):
      manualDF = df
  else:
      manualDF = pd.concat([manualDF, df])
manualDF = manualDF.sort_values(by=['published'], ascending=True)
manualDF['title'] = manualDF['title'].fillna('')
manualDF['description'] = manualDF['description'].fillna('')
print(manualDF)

# keyword
# 

counter = 0
notFoundUrls = []
for index, column in manualDF.iterrows():
    #newData = {'url': column['url'], 'language':'de', 'valid':0, 'quote':'', 
    #           'content':'', 'archive':'', 'title':'','description':'', 'published':'1970-01-01T00:00:00'}
    counter += 1
    if((counter % 100) ==0):
        print(counter)
        storeCollection()
    if(random.random()>0.75):
     newData = column
     #print(column)
     searchQuote = newData['title'] + " " + newData['description']
     foundKeywords = []
     found = False
     for index2, column2 in keywordsDF.iterrows(): 
         keyword = column2['keyword']
         allFound = True
         keywords = keyword.strip("'").split(" ")
         for keyw in keywords:
            allFound = allFound and (keyw in searchQuote)
         if(allFound):
             foundKeywords.append(keyword) 
             found = True
     if(found):
         newData['keyword'] = random.choice(foundKeywords)
         addNewsToCollection(newData)
     
storeCollection()                          
#print(notFoundUrls)
for xx in notFoundUrls:
    print(xx)




