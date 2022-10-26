import pandas as pd

from pathlib import Path
import os.path
import io
#import requests
import glob

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm

DATA_PATH = Path.cwd()
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

# Domains  (floods diagrams)

fig = plt.figure(figsize=(12, 6), constrained_layout=True)
gs = gridspec.GridSpec(1, 2, figure=fig)

newsDf2 = pd.merge(newsDf, keywordsColorsDF, how='left', left_on=['keyword'], right_on=['keyword'])
topicsDF = newsDf2.groupby('topic').count()
topicsDF = topicsDF.drop(columns = ['topicColor'])
topicsDF = pd.merge(topicsDF, topicsColorsDF, how='left', left_on=['topic'], right_on=['topic'])
topicsDF = topicsDF.sort_values('index', ascending=False)
axTopics = plt.subplot(gs[0,0])
axTopics.set_title("Topics", fontsize=24)
plot = topicsDF.plot.pie(y='index', ax=axTopics, colors=topicsDF['topicColor'], labels=topicsDF['topic'],legend=False,ylabel='')
#plot = topicsDF.plot(kind='pie', y='index', ax=axKeywords, colors='#'+keywordsDF['keywordColor'])


keywordsDF = newsDf.groupby('keyword').count()
keywordsDF = pd.merge(keywordsDF, keywordsColorsDF, how='left', left_on=['keyword'], right_on=['keyword'])
keywordsDF = keywordsDF.sort_values('index', ascending=False)
axKeywords = plt.subplot(gs[0,1])
axKeywords.set_title("Keywords", fontsize=24)
plot = keywordsDF.plot.pie(y='index', ax=axKeywords, colors=keywordsDF['keywordColor'], labels=keywordsDF['keyword'],legend=False,ylabel='')
#plot = topicsDF.plot(kind='pie', y='index', ax=axKeywords, colors='#'+keywordsDF['keywordColor'])


plt.savefig(DATA_PATH / 'img' / 'keywords_pie_all.png', dpi=300)
plt.close('all')
"""
fig = plt.figure(figsize=(12, 6), constrained_layout=True)
gs = gridspec.GridSpec(1, 2, figure=fig)

axTopics = plt.subplot(gs[0,0])
axTopics.set_title("Topics", fontsize=24)
colors2021 = filterColors(floodsTopics2021['index'], colorsTopics)
wedges2021, texts2021, auto  =  axTopics.pie(floodsTopics2021['count'], labels=floodsTopics2021['index'], colors=colors2021, 
            autopct=limitAutopct, labeldistance=None, textprops={'fontsize': 12})

leg  = axTopics.legend(wedges2021, floodsTopics2021['index'],
          title="Topics",
          loc="center right",
          fontsize=16,
          bbox_to_anchor=(1, 0, 0.5, 1))
leg.set_title("Topics", prop = {'size':20})              

axKeywords = plt.subplot(gs[0,1])
axKeywords.set_title("Keywords", fontsize=24)
colors2021 = filterColors(floodsTopics2021['index'], colorsTopics)
wedges2021, texts2021, auto  =  axKeywords.pie(floodsTopics2021['count'], labels=floodsTopics2021['index'], colors=colors2021, 
            autopct=limitAutopct, labeldistance=None, textprops={'fontsize': 12})            

leg  = axKeywords.legend(wedges2021, floodsTopics2021['index'],
          title="Topics",
          loc="center right",
          fontsize=16,
          bbox_to_anchor=(1, 0, 0.5, 1))
leg.set_title("Keywords", prop = {'size':20})          

plt.savefig(DATA_PATH / 'img' / 'topics_pie_years.png', dpi=300)
plt.close('all')
"""
