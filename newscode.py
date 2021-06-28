from newsapi import NewsApiClient

import pandas as pd
import re
import os

from gemfunction import build_mind_map,StemmingHelper

from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from textblob import TextBlob
import matplotlib.pyplot as plt

from newsapi import NewsApiClient

from gensim.models import Word2Vec
from collections import Counter

import networkx as nx
import pydot


mynewsapi = NewsApiClient(api_key="")  #Enter the Client API

news_sources = mynewsapi.get_sources()    #create list sources for the URL

for source in news_sources['sources']:    #create list of the source
    print(source['name'])                 #print the source name 

#search tags 
searchtags=["Agricultural Machinery","Auto Industry","Barry Plastic",
            "General Motors","logistics Industry","Pet Food","Steel Prices"]

#stopwords
stopwords=["is","are","an","the","this","abroad","about","above","according","to","across","after","against","ago","ahead","of","along","amidst",
              "among","apart","around","apart","as","aside","at","away","barring","because","before","behind","below",
              "beneath","between","beyound","but","by","concerning","despite","due","in","into","instead","like","hence",
             "like","minus","near","next","past","per","pior","round","off","on","account","behalf","within","without",
              "your","you","my","mine","be","as","we","our","is","for","its","and","or","them","their",'could',"would",
              "through","they","have","has","didn","always","these","another","any","again","some","want","where","there",
              "must","might","were","probably","went","more","that","which","until","than","when","from","other"
             "times","till","to","toward","towards","underneath","long","short","up","upon","via","with","view","did","seeing",
             "was","few","who","was","while","seen","most","already","tell","use","told","being","been","said"
             "made","only","not","used","such","also","all","using"]

def MindMap(sentences,root,title):
    min_count = 1
    size = 100
    window = 3

    model = Word2Vec(sentences, min_count=min_count, size=size, window=window)

    v=model.wv.vocab.keys()
    vocab = list(v)
    #root=vocab[-1]

    g=build_mind_map(model,StemmingHelper, root, vocab)
    
    os.environ["PATH"] += os.pathsep + 'C:\\Program Files\\Graphviz\\bin'

    graph = nx.drawing.nx_pydot.to_pydot(g)

    graph.write_png('img/'+title+"_graph_.png")

    plt.figure(figsize=(20,20))

    nx.draw_networkx(g)

    plt.show()


def cleaning_sentence(sentence):
    # Remove all the special characters
    processed_feature = re.sub(r'\W', ' ', str(sentence))
    
    # remove all single characters
    processed_feature= re.sub(r'\s+[a-zA-Z]\s+', ' ', processed_feature)
    
    #remove all digit in the characters
    processed_feature = re.sub(" \d+", " ", processed_feature)
    
    #Remove single characters from the start
    processed_feature = re.sub(r'\^[a-zA-Z]\s+', ' ', processed_feature) 
    
    # Substituting multiple spaces with single space
    processed_feature = re.sub(r'\s+', ' ', processed_feature, flags=re.I)
    
    # Removing prefixed 'b'
    processed_feature = re.sub(r'^b\s+', '', processed_feature)
    
    # Converting to Lowercase
    processed_feature = processed_feature.lower()
    
    return processed_feature.split(' ') 

for searchtag in searchtags:  #for each search tag in search tags
    
    #add search tag
    all_articles = mynewsapi.get_everything(
        q=searchtag,
        language='en', 
        sort_by='publishedAt',
        from_param="2021-02-10",
        to="2021-03-07",
        page=2
    )
    
    #define empty 
    newssource=[]   
    newsurl=[]
    newstitle=[]
    newarticles=[]
    newspublishdatetime=[]
    sentimentscore=[]
    fulltexts=[]
    combinetext=''

    for article in all_articles['articles']:  #for each article in article list 
        print('Source : ',article['source']['name'])  #print list source
        print('Title : ',article['title'])            #print article list
        print('Description : ',article['description'],'\n\n')  #print description
        
        
        if not(article['description']==None): #print the description
        
            newsurl.append(article['url'])  #append the article  to the list
            newssource.append(article['source']['name'])   #append the source to the list
            newstitle.append(article['title'])            #append the title to the list
            newarticles.append(article['description'])     #append the article description
            
            dt=article['publishedAt']  #published date
            dt=dt.replace('T',' ')     
            dt=dt.replace('Z','')
            newspublishdatetime.append(dt)  #append the date
            
            try:
                response=requests.request("GET",article['url']) #response 

                soup = BeautifulSoup(response.content, 'html.parser')  #soup
                s=soup.find_all("p")                                   #scrap paragraph

                full_text=''                               #full text ' '
                
                for text in s:                             #for each text in scrapped paragraph
                    full_text+=text.get_text()             #extract the full text 
                    
                full_text=''
                full_text_list=[]                          #text list
        
                for text in s:                             #for each text in scrapped paragraph
                   full_text+=text.get_text()              #get text
                   full_text_list.append(text.get_text())  #append the text
            
            except:
                full_text=article['description']           #full text
                
            fulltexts.append(full_text_list)                    #append full text
            
            textB=TextBlob(full_text)                      #using text Blob to compute the sentiment
 
            sentiment=textB.sentiment.polarity             #get the sentimental polarity

            sentimentscore.append(sentiment)               #append the sentimental polarity
            
            combinetext=combinetext + full_text + ' '      #combine the text
    
    #search result in dataframe
    searchresult=pd.DataFrame(data={'Source': newssource,'Title':newstitle, 'Article':  newarticles,'Full Text':fulltexts,'Url':newsurl,
                       'Publish datetime':newspublishdatetime, 'sentiment score':sentimentscore
                       
                      })
    
    #export the the dataframe to search
    searchresult.to_excel(searchtag + '.xlsx',index=0)
    
    #using word cloud on the combined text
    wc=WordCloud(background_color = "black").generate(combinetext)
    plt.imshow(wc, interpolation ="bilinear")
    plt.axis("off")
    plt.tight_layout(pad=0)

    plt.savefig(searchtag+'.jpg')
    plt.show()
    
    counter=0 #set the counter
    for text in fulltexts: #for each text in full texts

        root=searchtag        #for each search tag
        processed_sentences=[[searchtag]]   #create first vocabulary 

        for sentence in text:  #for each sentence in the text

            ptext=cleaning_sentence(sentence)       #clearning the sentences

            topwords=Counter(ptext).most_common(10)  #top words

            for t in topwords:     #for each top words
                ptext=list(filter(lambda a: a!=t[0],ptext))  #extract all the top words

            for p in stopwords:  #for p stop words
                
                if p in ptext:   #if p in ptext
                    ptext=list(filter(lambda a: a!=p,ptext))  #extract all the stop words
                    
            for t in ptext:   #for each t in ptext
                
                if len(t)<3:  #if the length word is less than 3
                    ptext=list(filter(lambda a: a!=t,ptext))

            ptext=list(filter(lambda a: a!=[],ptext))  #remove all empty list
            
            if len(ptext)==0:  #if the ptext is equal to 0
                continue       #continue
            processed_sentences.append(ptext) #append ptext
            
        try:

            MindMap(processed_sentences,root,searchtag+'_'+newsurl[counter][8:15]+"_"+str(counter))
            counter+=1
        except:
            continue
