import ast
import glob
import nltk
from nltk.tokenize import word_tokenize

#d :dictionary of inverted index
#doc contains docid
#content is a list of words 

#-----------------calculate tf---------------
def cal_df(d):
    df={}
    for i in d.keys():
        df[i]=len(d.get(i))
    return df

#--------------calculate idf-----------------
idf={}
def cal_idf(d,content,doc):
   docfreq={}
   doc=str(doc)
   print(doc)
   for i in d.keys():
       count=0
       if(doc in d.get(str(i))):
           if i in content:
               count+=1
       docfreq[doc]=count
       idf[i]=docfreq
   print(idf['son'])
       
with open('Downloads/data/model_queries.txt') as f:
    data = f.read()
d=ast.literal_eval(data)          
path = "Downloads/data/en_BDNews24/1/*"
for filename in glob.glob(path):
    content=""
    content1=""
    #print("------new content----------")
    with open(filename, 'r') as f:
        for line in f:
            #print(str(line))
            content1+=str(line)
        text_start=content1.find('<TEXT>')
        text_end=content1.find('</TEXT>')
        text=content1[text_start+6:text_end]
        content=word_tokenize(text)
        doc_start=content1.find('<DOCNO>')
        doc_end=content1.find('</DOCNO')
        doc=content1[doc_start+7:doc_end]
        df=cal_idf(d,content,doc)
        #print(str(content))

df=cal_df(d)
#df=cal_idf(d,content)
    
    
        
