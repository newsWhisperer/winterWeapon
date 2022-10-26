import mysecrets
import os
import sys

#pip3 install newsapi-python
import pandas as pd

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
import glob

import datetime
from dateutil import parser

DATA_PATH = Path.cwd()

keywordsFields = ["keyword","language","topic","topicColor","keywordColor","limitPages","ratioNew"]
keywordsDF = pd.read_csv(DATA_PATH / 'keywords.csv', delimiter=',')  #,index_col='keyword'
keywordsDF['uniqueString'] = keywordsDF['keyword'] + "_" + keywordsDF['language'] + "_" + keywordsDF['topic']
keywordsDF['crc'] = keywordsDF['uniqueString'].apply(
    lambda x: 
        hashlib.sha256(x.encode()).hexdigest()
)



searchWords = dict(zip(keywordsDF.keyword.values, keywordsDF.language.values))

#print(keywordsDF)
#print(searchWords)
#print(keywordsDF.sample() )


stopDomains = ["www.mydealz.de", "www.techstage.de", "www.nachdenkseiten.de", "www.amazon.de", "www.4players.de", "www.netzwelt.de", "www.nextpit.de",
               "www.mein-deal.com", "www.sparbote.de", "www.xda-developers.com" "www.pcgames.de", "blog.google", "www.ingame.de", "playstation.com",
               "www.pcgameshardware.de", "9to5mac.com", "roanoke.com", "billingsgazette.com", "richmond.com", "www.rawstory.com", "slate.com"
                ]


#https://github.com/theSoenke/news-crawler/blob/master/data/feeds_de.txt

                  
def dataIsNotBlocked(data):
    for blocked in stopDomains: 
        if blocked in data['domain']:
            return False
    return True         

#replace/drop: "https://www.zeit.de/zustimmung?url="  

#get url data (inq)  -> check if keyword in title|description    and url equal
#see 'https://www.stern.de/panorama/weltgeschehen/news-heute---ocean-viking--rettet-mehr-als-40-menschen-aus-dem-mittelmeer-30598826.html'




collectedNews = {}

def addNewsToCollection(data):
    global collectedNews
    pubDate = parser.parse(data['published'])
    fileDate = 'news_'+pubDate.strftime('%Y_%m')+'.csv'
    if(fileDate in collectedNews):
      if(not data['url'] in collectedNews[fileDate]):
        if(not 'archive' in data):
           data = archiveUrl(data)
        collectedNews[fileDate][data['url']] = data
        return True
    return False

def storeCollection():
    global collectedNews
    print("Inside store")
    cols = ['url','valid','domain','title','description','image','published','archive','content','quote','language','keyword']
    for dateFile in collectedNews:
        df = pd.DataFrame.from_dict(collectedNews[dateFile], orient='index', columns=cols)
        #df.to_csv(DATA_PATH / dateFile, index=True) 
        if(not os.path.exists(DATA_PATH / 'csv')):
            os.mkdir(DATA_PATH / 'csv')
        print(["store file: ", dateFile])    
        df.to_csv(DATA_PATH / 'csv' / dateFile, index_label='index') 
    collectedNews = {}

# self.randomWordsDF = pd.DataFrame.from_dict(self.randomWords, orient='index', columns=self.randomBase.keys())  
# self.randomWordsDF.to_csv(DATA_PATH / self.category / "csv" / ("words_random_"+str(self.randomSize)+".csv"), index=True)


#https://web.archive.org/save/https://translate.google.com/translate?sl=de&tl=en&u=
#https://web.archive.org/save/https://translate.google.com/translate?sl=de&tl=en&u=https://www.nikos-weinwelten.de/beitrag/weinbau_reagiert_auf_den_klimawandel_abschied_vom_oechsle_hin_zur_nachhaltigkeit/


# https://docs.aiohttp.org/en/stable/client_reference.html
# 
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

def extractData(article, language, keyWord):
    title = article['title']
    description = article['description']
    url = article['url']
    #later use list...
    url = url.replace('https://www.zeit.de/zustimmung?url=', '')
    url = url.replace('%3A', ':')
    url = url.replace('%2F', '/')                
    domain = urlparse(url).netloc
    image = None
    if('urlToImage' in article): 
        image = article['urlToImage']

    published = '1970-01-01T00:00:00'
    if('publishedAt' in article):    
        published = article['publishedAt']
    content = article['content']
    data = {'url':url, 'valid':0, 'domain':domain,'published':published, 'description':description, 'title':title, 
            'image':image, 'content':content, 'quote':'', 'language': language, 'keyword':keyWord}
    return data  

def filterNewAndArchive(articles, language, keyWord):
    global collectedNews
    newArticles = []
    startTime = time.time()
    for article in articles:
        data = extractData(article, language, keyWord) 
        if (dataIsNotBlocked(data)):
            pubDate = parser.parse(data['published'])
            fileDate = 'news_'+pubDate.strftime('%Y_%m')+'.csv'
            if(not fileDate in collectedNews):
                if(os.path.isfile(DATA_PATH / 'csv' / fileDate)):
                    df = pd.read_csv(DATA_PATH / 'csv' / fileDate, delimiter=',',index_col='index')
                    collectedNews[fileDate] = df.to_dict('index')
                else:
                    collectedNews[fileDate] = {}
            if(not data['url'] in collectedNews[fileDate]):
                data = archiveUrl(data)
                newArticles.append(data)
        if((time.time() - startTime) > 60*20):
            return newArticles        
    return newArticles

def getNewsFiles(state='harvest'):
    fileName = './csv/news_????_??.csv'
    if(state):
        fileName = './csv/news_'+state+'_????_??.csv'
    files = glob.glob(fileName)
    return files  

def getLatestFileAge():
    minAge = 1E6
    now = time.time()
    for fileName in getNewsFiles(state=None):
        print([os.path.getatime(fileName),os.path.getctime(fileName),os.path.getmtime(fileName)])
        modifyDate = os.path.getmtime(fileName)
        fileAge = now-modifyDate
        print(fileAge)
        if(fileAge<minAge):
            minAge = fileAge
    return minAge        


def inqRandomNews():
    apiKey = os.getenv('NEWSAPI_KEY')
    if(apiKey == '1a2b3c4d5e6f7a8b9c0d1e2f3a4b5c6d7'): 
        print('Please set newsapi.org key in file: mysecrets.py');
        return None


    rndKey = keywordsDF.sample()
    #if FoundAny: newLimit = minimum(currPage+1,limitPage)
    #if foundNothing:  newLimit = maximum(1,random.choice(range(currPage-1,limitPage-1)))

    #keyWord = random.choice(searchWords)
    #language = 'de'
    #language = 'en'   
    #language = 'fr' 
    crc = rndKey['crc'].iloc[0]
    keyWord = rndKey['keyword'].iloc[0]
    language = rndKey['language'].iloc[0]
    limitPages = rndKey['limitPages'].iloc[0]
    ratioNew = rndKey['ratioNew'].iloc[0]
    currPage = random.choice(range(1,limitPages+1))  
    newLimit = max(1,random.choice(range(currPage-1,limitPages)))
    currRatio = 0.0
          
    print([keyWord, language])
    if(not 'xx'==language):
        sort = random.choice(['relevancy', 'popularity', 'publishedAt'])
        pageSize = 50
        print('keyword: '+keyWord+'; Page: '+str(currPage))
        # https://newsapi.org/docs/endpoints/everything
        url = ('https://newsapi.org/v2/everything?'+
            #'q="'+keyWord+'"&'
            'q='+keyWord+'&'
            'pageSize='+str(pageSize)+'&'
            'language='+language+'&'
            'page='+str(currPage)+'&'
            'sortBy='+sort+'&'
            'apiKey='+apiKey
            #'excludeDomains=www.zeit.de,www.reuters.com'
            )
            
            # sortBy=relevancy   : relevancy, popularity, publishedAt
        response = requests.get(url)
        response.encoding = response.apparent_encoding
        #print(response.text)
        foundNew = False
        if(response.text):
            jsonData = json.loads(response.text)
            if (('ok'==jsonData['status']) and (jsonData['totalResults']>0)):
              if(len(jsonData['articles']) > 0):
                deltaLimit = 0
                #newLimit = limitPages
                if(len(jsonData['articles']) > 30):
                  deltaLimit += 1  
                  #newLimit = max(currPage+1,limitPages)                
                print('#found Articles: '+str(len(jsonData['articles'])))
                print("archive first")
                newArticles = filterNewAndArchive(jsonData['articles'], language, keyWord)
                print('#new Articles: '+str(len(newArticles)))
                currRatio = len(newArticles)/len(jsonData['articles'])
                if(currRatio>0.5):
                    deltaLimit += 1
                    #newLimit = max(currPage+2,limitPages)
                newlimit =  max(currPage+deltaLimit,limitPages)   
                if(len(newArticles) in [1,2]):     
                    print("sleep")   
                    time.sleep(60)
                print("add to collection")
                for data in newArticles:
                    if (dataIsNotBlocked(data)):                    
                        #print(str(keyWord)+': '+str(title)+' '+str(url))
                        print(["addNewsToCollection: ",data])
                        if(addNewsToCollection(data)):
                            foundNew = True
                            print(["+++added"])  
                        else:
                            print(["---not added"])    
                #print(["collectedNews: ",collectedNews])            
                if(foundNew):         
                    storeCollection()
    #print(rndKey.index)
    #keywordsDF.at[rndKey.index, 'limitPages'] = newLimit    
    keywordsDF.loc[keywordsDF['crc'] == crc, 'limitPages'] = newLimit 
    keywordsDF.loc[keywordsDF['crc'] == crc, 'ratioNew'] = currRatio*0.05+ratioNew*0.95
        
      

#b'{"status":"ok","totalResults":1504,
# "articles":[{"source":{"id":null,"name":"heise online"},
#              "author":"Stefan Krempl",
#              "title":"Wissenschaftler: Klimawandel tr\xc3\xa4gt zum Starkregen bei\xe2\x80\x8b",
#              "description":"Die Wolkenbr\xc3\xbcche mit katastrophalen Folgen geh\xc3\xb6ren so nicht mehr zur \xc3\xbcblichen Wetter-Varianz, meinen Wissenschaftler. Sie fordern einen Umbau der Infrastruktur.",
#              "url":"https://www.heise.de/news/Wissenschaftler-Klimawandel-traegt-zum-Starkregen-bei-6140856.html",
#              "urlToImage":"https://heise.cloudimg.io/bound/1200x1200/q85.png-lossy-85.webp-lossy-85.foil1/_www-heise-de_/imgs/18/3/1/3/9/7/8/0/Ueberschwemmung-c06f751f2932e14b.jpeg",
#              "publishedAt":"2021-07-16T15:06:00Z",
#              "content":"Nach den langen und heftigen Regenf\xc3\xa4llen Mitte der Woche treten immer mehr katastrophale Folgen zutage: Die Zahl der Toten w\xc3\xa4chst, allein in Rheinland-Pfalz und Nordrhein-Westfalen sind \xc3\xbcber 100 Mens\xe2\x80\xa6 [+4840 chars]"}

'''
#time.sleep(80000)
i=1
while True:
###while (i<50):    
  print(i)  
  inqRandomNews()
  i += 1
  #time.sleep(200) # unless drop none-french
  #time.sleep(20)
  time.sleep(1000)
'''
age = getLatestFileAge()
print(age)
if(age>60*60*5*0):
    inqRandomNews()
#keywordsDF = keywordsDF.sort_values(by=['topic','keyword'])
keywordsDF = keywordsDF.sort_values(by=['ratioNew'], ascending=False)
keywordsDF.to_csv(DATA_PATH / 'keywords.csv', columns=keywordsFields,index=False)  
    
