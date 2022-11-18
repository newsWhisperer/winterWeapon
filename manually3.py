import pandas as pd
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

from bs4 import BeautifulSoup

from deep_translator import GoogleTranslator
from deep_translator import single_detection

DATA_PATH = Path.cwd()

keywordsDF = pd.read_csv(DATA_PATH / 'keywords.csv', delimiter=',')  #,index_col='keyword'
keywordsDF['uniqueString'] = keywordsDF['keyword'] + "_" + keywordsDF['language'] + "_" + keywordsDF['topic']
keywordsDF['crc'] = keywordsDF['uniqueString'].apply(
    lambda x: 
        hashlib.sha256(x.encode()).hexdigest()
)
keywordsDF = keywordsDF.sort_values(by=['ratioNew'], ascending=False)  

extractCodes = [
    {'trg':'published', 'tag':'meta', 'att':'property', 'idn':'article:modified_time', 'val':'content'},
    {'trg':'published', 'tag':'time', 'att':'class', 'idn':'atc-MetaTime', 'val':'datetime'},

    {'trg':'published', 'tag':'meta', 'att':'itemprop', 'idn':'dateModified', 'val':'content'},
    {'trg':'published', 'tag':'meta', 'att':'name', 'idn':'buildDate', 'val':'content'},
]

# <time datetime="2018-01-04 08:55">  #no att & idn!

# https://docs.python-requests.org/en/master/user/quickstart/
def extractInfo(url):
    print(url)
    data = {}
    data['url'] = url
    #data['published'] = '1970-01-01T00:00:00'
    domain = urlparse(url).netloc
    data['domain'] = domain
    content = ""
    try:
        page = requests.get(url, timeout=10)
        data['status'] = page.status_code
        if page.status_code == 200:
            content = page.content
    except requests.exceptions.RequestException as e:  # This is the correct syntax
        data['status'] = 999  

    DOMdocument = BeautifulSoup(content, 'html.parser')
    #<html lang="de">
    htmlTag = DOMdocument.html
    if(htmlTag and htmlTag.has_attr('lang')):
        data['language'] = htmlTag['lang']
    if(DOMdocument.title):
        data['title'] = DOMdocument.title.string
    if(DOMdocument.description):
        data['description'] = DOMdocument.description.string
    # og:title
    title = DOMdocument.find('meta', attrs={'property': 'og:title'})
    if(title):
      if('content' in title):  
        data['title'] = title['content']

    title = DOMdocument.find('meta', attrs={'property': 'twitter:title'})
    if(title):
      if('content' in title):  
        data['title'] = title['content']

    image = DOMdocument.find('meta', attrs={'property': 'og:image'})
    if(image):
        if(image.get('content')):
            data['image'] = image.get('content')
    description = DOMdocument.find('meta', attrs={'name': 'description'})
    if(description):
        if(description.get('content')):
           data['description'] = description.get('content') 
        if(description.get('value')):
           data['description'] = description.get('value') 
    description = DOMdocument.find('meta', attrs={'property': 'og:description'})
    if(description):
      if(description.get('content')): 
        data['description'] = description.get('content')
    description = DOMdocument.find('meta', attrs={'property': 'twitter:description'})
    if(description):
      if(description.get('content')): 
        data['description'] = description.get('content')
    #extract date from url
    match = re.search(r'20\d{2}/\d{2}/\d{2}', url)
    if(match):
       data['published'] = match.group().replace('/','-')+'T12:00:00' 

    #extract date from archive
    #s2 = 'https://web.archive.org/web/20170817131154/h'
    #match2 = re.search(r'archive.org/web/(?P<year>\d{4})(?P<month>\d{2})(?P<day>\d{2})', s2)
    #print(match2.groupdict())

    updated = DOMdocument.find('meta', attrs={'property': 'og:updated_time'})
    if(updated):
        if(updated.get('content')):
           data['published'] = updated.get('content')  
    modified = DOMdocument.find('meta', attrs={'name': 'last-modified'})
    if(modified):
        if(modified.get('content')):
           data['published'] = modified.get('content')  
    modified2 = DOMdocument.find('meta', attrs={'property': 'article:modified_time'})
    if(modified2):
        if(modified2.get('content')):
           data['published'] = modified2.get('content')     
    published2 = DOMdocument.find('meta', attrs={'name': 'cXenseParse:publishtime'})
    if(published2):
        if(published2.get('content')):
           data['published'] = published2.get('content')  
    published3 = DOMdocument.find('meta', attrs={'itemprop': 'datePublished'})
    if(published3):
        if(published3.get('content')):
           data['published'] = published3.get('content')
                            
    issued = DOMdocument.find('meta', attrs={'name': 'DC.date.issued'})
    if(issued):
        if(issued.get('content')):
           data['published'] = issued.get('content')
 
    issued2 = DOMdocument.find('meta', attrs={'name': 'dcterms.date'})
    if(issued2):
        if(issued2.get('content')):
           data['published'] = issued2.get('content')             
    published = DOMdocument.find('meta', attrs={'property': 'article:published_time'})
    if(published):
        if(published.get('content')):
           data['published'] = published.get('content')  
    #<meta name="publish-date" content="2022-06-29 10:32:38">
    datum = DOMdocument.find('meta', attrs={'name': 'date'})
    if(datum):
        if(datum.get('content')):
           data['published'] = datum.get('content')  

    for code in extractCodes:
        identity = DOMdocument.find(code['trg'], attrs={code['att']: code['idn']})
        if(identity):
            if(identity.get(code['val'])):
                data[code['trg']] = identity.get(code['val'])         



#get time of archive.org (earliest)
#get time of url:  yyyy/mm/dd


#<div class="date-published">19.03.2017 Stand 19.03.2017, 16:57 Uhr</div>

# script only
# https://www.cbc.ca/news/science/climate-change-india-farmer-suicides-1.4230510
# https://www.washingtonpost.com/news/worldviews/wp/2017/09/08/hurricane-jose-threatens-a-second-blow-to-caribbean-islands-devastated-by-irma/
# https://nationalpost.com/news/world/taiwan-earthquake-rescue-efforts-temporarily-suspended-after-building-tilting-at-45-degree-angle-begins-to-slide



    language = DOMdocument.find("meta",  property="og:locale")
    if(language and ('content' in language)):
        data['language'] = language['content']

    #<meta name="news_keywords" content="sierra leona, inundaciones"/>  
    #<meta property="og:locale" content="es_LA"/>    

    #data['author'] = DOMdocument.find("meta",  property="author")['content'] 
    # keywords content-language og:site_name 
    # twitter:title ...
    # rss-feeds:  <link rel="alternate" type="application/rss+xml" title="xxx" href="https://www.lvz.de/rss/feed/lvz_nachrichten" />
    return data


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



async def saveArchive(saveUrl):
    async with aiohttp.ClientSession() as session:
      async with session.get(saveUrl, timeout=120) as response:
        print("x")   

async def getArchives(urlList):
    async with aiohttp.ClientSession() as session:
      async with session.get(saveUrl) as response:
        print("x")   

def findArchives(articles):
    foundArticles = []
    for article in articles:
        data = extractData(article, language, keyWord) 
        if (dataIsNotBlocked(data)):
            a=1

def archiveUrl(data):
    #timetravelDate = datetime.datetime.strptime(data['published'], '%Y-%m-%d %H:%M:%S').strftime('%Y%m%d')
    #pubDate = datetime.datetime.fromisoformat(data['published'])
    #pubDate = parser.isoparse(data['published'])
    timetravelDate = '19700101'
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
        timetravelDate = pubDate.strftime('%Y%m%d')
    timetravelUrl = 'http://timetravel.mementoweb.org/api/json/'+timetravelDate+'/'+data['url']
    try:
        page = requests.get(timetravelUrl, timeout=30)
        if page.status_code == 200:
            content = page.content
            #print(content)
            if(content):
                #print(content)
                jsonData = json.loads(content)
                if(jsonData and jsonData['mementos']):
                    data['archive'] = jsonData['mementos']['closest']['uri'][0]
                    if('1970-01-01T00:00:00' == data['published']):
                        data['published'] = jsonData['mementos']['closest']['datetime']
                #'closest'
    except:
#    except Exception as e:    
#    except json.decoder.JSONDecodeError as e:    
#    except requests.exceptions.RequestException as e:  
        e = sys.exc_info()[0]
        print("not archived yet")
        saveUrl = 'https://web.archive.org/save/' + data['url'] # archive.org
        #saveUrl = 'https://archive.is/submit/'
        #saveUrl = 'https://archive.ph/submit/'

        ##  pip3 install aiohttp
        try:
           loop = asyncio.get_event_loop()
           loop.run_until_complete(saveArchive(saveUrl))
        except:
           e2 = sys.exc_info()[0]
           print("something more went wrong (timeout/redirect/...)")            

        #async with aiohttp.ClientSession() as session:
        #    async with session.get(saveUrl) as response:
        #        print(await response.status())        
        '''
        try:
            page = requests.get(saveUrl, timeout=240)  # archive.org
            #page = requests.post(saveUrl, data = {'url':data['url']}, timeout=240)
            if page.status_code == 200:
                print('archived!')
        except requests.exceptions.RequestException as e2:
            print("not archivable: " + data['url'])
        '''    
    return data 


manualDF = pd.read_csv(DATA_PATH / "csv" / "olds_winter_weapon.csv", delimiter=',',index_col='index') 
print(manualDF)

# keyword
# 

counter = 0
notFoundUrls = []
for index, column in manualDF.iterrows():
    newData = {'url': column['url'], 'language':'de', 'valid':0, 'quote':'', 
               'content':'', 'archive':'', 'title':'','description':'', 'published':'1970-01-01T00:00:00'}
    counter += 1
    if((counter % 100) ==0):
        print(counter)
        #manualDF.to_csv(DATA_PATH / "csv" / "old_winter_weapon.csv", index_label='index')  
        storeCollection()
    if(counter>0):
     newData = archiveUrl(newData)
     if(len(newData['archive'])>5):
             url = newData['archive']
             harvestData = extractInfo(url)
             newData['domain'] = harvestData['domain']
             if('language' in harvestData):
                 newData['language'] = harvestData['language']
             if('title' in harvestData):
                 newData['title'] = harvestData['title']
             if('description' in harvestData):
                 newData['description'] = harvestData['description']
             if('image' in harvestData):
                 newData['image'] = harvestData['image']
             if('published' in harvestData):
                 newData['published'] = harvestData['published']
     url = column['url']
     harvestData = extractInfo(url)
     newData['domain'] = harvestData['domain']
     if('language' in harvestData):
         newData['language'] = harvestData['language']
     if('title' in harvestData):
        newData['title'] = harvestData['title']
     if('description' in harvestData):
        newData['description'] = harvestData['description']
     if('image' in harvestData):
         newData['image'] = harvestData['image']
     if('published' in harvestData):
         newData['published'] = harvestData['published']


     
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
     else:
         print(["no keyword found", url, newData['title']])
         notFoundUrls.append({'url':url, 'title':newData['title'], 'description':newData['description']})
 
     addNewsToCollection(newData)
     
storeCollection()                          
#print(notFoundUrls)
for xx in notFoundUrls:
    print(xx)




