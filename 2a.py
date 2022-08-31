import glob
import nltk
import pickle
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import math

stop_words = list(stopwords.words('english'))

index={}
count_tf={}
def build_inverted_index(text, docid):
    for w in text:
        count=1
        if w in index.keys():
            if docid not in index[w]:
                index[w].append(docid)
                count_tf[w][docid]=count
                count+=1
            elif docid in index[w]:
                count_tf[w][docid]=count
                count+=1
        else:
            index[w] = [docid]
    print(count_tf)
    # index=set(index)

# --------------stop word removal---------------
def stop_removal(text_tokens):
    stop_text = []
    for w in text_tokens:
        if w.lower() not in stop_words:
            stop_text.append(w)
    return stop_text


# -------------punctuation removal--------------
def punc_removal(stop_text):
    punct_text = []
    for w in stop_text:
        if w.isalpha():
            punct_text.append(w)
    return punct_text


# ------------lemmatization-----------------------
def lemma(punct_text):
    lemm_text = ""
    lemmatizer = WordNetLemmatizer()
    for w in punct_text:
        lemm_text += " " + lemmatizer.lemmatize(w)
    return lemm_text


def calc_df(temp):
    df={}
    for key,value in temp.items():
        df[key]=len(value)
    return df

"""
def calc_idf(temp):
    idf={}
    path = "data/en_BDNews24/1/*"
    count=0
    for filename in glob.glob(path):
        count+=1
    N=count
    print(N)
    for key,value in temp.items():
        idf[key]=math.log10(N/df[key])
    return idf
"""

def calc_tf_idf(temp):
    tf_idf={}
    tf_idf_doc_vec={}
    idf = {}
    path = "en_BDNews24/1/*"
    total_docs = 0
    for filename in glob.glob(path):
        total_docs += 1
    N = total_docs
    for key,value in temp.items():
        templist=[]
        total_values=len(value)
        for i in value:
            text=[]
            folder=i.split(".")
            doc_no=folder[2]
            path="en_BDNews24/1/"+i
            content=""
            f = open(path, 'r')
            for line in f:
                content+= str(line)
                text_start = content.find('<TEXT>')
                text_end = content.find('</TEXT>')
                text = content[text_start + 6:text_end]
                doc_start = content.find('<DOCNO>')
                doc_end = content.find('</DOCNO')
                doc = content[doc_start + 7:doc_end]
                text_tokens = word_tokenize(text)
                stop_text = stop_removal(text_tokens)
                punct_text = punc_removal(stop_text)
                build_inverted_index(punct_text,doc)
                lemm_text = lemma(punct_text)
                text=lemm_text.split(" ")
                count = 0
                wordcount=0
            for word in text:
                wordcount+=1
                if key==word:
                    count += 1
            count/=wordcount-1
            #print("N & df[key] :"+str(N)+" "+str(df[key]))
            idf[key] = math.log10(N / df[key])
            count*=idf[key]
            templist.append(str(count))
                #print(templist)
            tf_idf[key]=templist
            tf_idf_doc_vec[doc]=tf_idf
    #print(tf_idf_doc_vec)
    return tf_idf_doc_vec

#-------------calculate df-idf queries------------
def calc_tf_idf_query(temp):
    tf_idf={}
    tf_idf_vec={}
    idf = {}
    N=1
    print(temp)
    for key,value in temp.items():
        templist=[]
        for i in value:
            content=""
            path = "raw_query.txt"
            f = open(path, 'r')
            for line in f:
              #print(str(line))
              content=str(line)
              if '<num>' in content:
                  id_start=content.find('<num>')
                  id_end=content.find('</num>')
                  query_id=content[id_start+5:id_end]
              if '<title>' in content:
                  text_start=content.find('<title>')
                  text_end=content.find('</title>')
                  query_text=content[text_start+7:text_end]
                  #print(query_text)
                  text_tokens = word_tokenize(query_text)
                  stop_text=stop_removal(text_tokens)  
                  punct_text=punc_removal(stop_text)
                  punct_text=set(punct_text)
                  #print(punct_text)
                  lemm_text=lemma(punct_text)
                  text=lemm_text.split(" ")
                  count = 0
                  wordcount=0
                  for word in text:
                      wordcount+=1
                      if key==word:
                          count += 1
                  count/=wordcount-1
                  #print("N & df[key] :"+str(N)+" "+str(df[key]))
                  idf[key] = math.log10(N / df[key])
                  count*=idf[key]
                  templist.append(str(count))
                  #print(templist)
                  tf_idf[key]=templist
                  tf_idf_vec[query_id]=tf_idf
    #print(tf_idf_vec)
    return tf_idf_vec  
    

#f= open("Downloads/data/model_queries.txt",'r')
filename = 'model_queries_07.pth'
outfile = open(filename, 'rb')
temp=pickle.load(outfile)
df=calc_df(temp)
#print(df)
#tf_idf=calc_tf_idf(temp)
#print(tf_idf)
outfile.close()

filename_2= 'queries_processing_07.pth'
outfile_2= open(filename_2, 'rb')
tempq=pickle.load(outfile_2)
df=calc_df(tempq)
print(df)
tf_idf=calc_tf_idf_query(tempq)
print(tf_idf)
outfile.close()