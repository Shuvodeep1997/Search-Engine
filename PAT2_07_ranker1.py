import glob
import nltk
import pickle
import itertools
import csv
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import math
import numpy as np
import sys

stop_words = list(stopwords.words('english'))
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

#Inc.Itc DOC
def doc_tf_idf1(temp,doc_path):
    doc_tfidf1 = {}
    maxtf={}
    avgtf={}
    ctr=0
    path = doc_path+"/*/*"
    #key_list=temp.keys()
    for filename in glob.glob(path):
        max=0
        avg=0
        ctr+=1
        #print(filename)
        file_start=filename.find('en.')
        #print("start:"+str(file_start))
        doc_id=filename[file_start:]
        print(doc_id + ":" + str(ctr))
        doc_tfidf1[doc_id]={}
        content = ""
        text=[]
        f=open(filename,'r')
        for line in f:
            content += str(line)
        text_start = content.find('<TEXT>')
        text_end = content.find('</TEXT>')
        text = content[text_start + 6:text_end]
        text_tokens = word_tokenize(text)
        stop_text = stop_removal(text_tokens)
        punct_text = punc_removal(stop_text)
        lemm_text = lemma(punct_text)
        text = lemm_text.split(" ")

        for key in text:
            doc_tfidf1[doc_id][key]=0.0
            count = 0
            for word in text:
                if key == word:
                    count += 1
            if count ==0:
                tf_idf=0
            if count!=0:
                avg+=count
                tf_idf = (1.0 + math.log10(count))
                if count>max:
                    max=count
            #print(tf_idf)
            doc_tfidf1[doc_id][key]=tf_idf
        ls=set([entry.lower() for entry in text])
        wordcount=len(ls)-1
        #print(str(avg)+':'+str(wordcount))
        maxtf[doc_id]=max
        if (wordcount != 0):
            avgtf[doc_id] = avg / wordcount
        if (wordcount == 0):
            avgtf[doc_id] = 0
        f.close()
    return doc_tfidf1,maxtf,avgtf

#Lnc.Lpc Doc
def doc_tf_idf2(temp,avgtf):
    doc_tfidf2 = {}
    path = doc_path+"/*/*"
    for filename in glob.glob(path):
        #print(filename)
        file=filename.split("/")
        doc_id=file[1]
        doc_tfidf2[doc_id]={}
        content = ""
        f=open(filename,'r')
        for line in f:
            content += str(line)
        text_start = content.find('<TEXT>')
        text_end = content.find('</TEXT>')
        text = content[text_start + 6:text_end]
        text_tokens = word_tokenize(text)
        stop_text = stop_removal(text_tokens)
        punct_text = punc_removal(stop_text)
        lemm_text = lemma(punct_text)
        text = lemm_text.split(" ")
        #print(text)
        for key in temp.keys():
            doc_tfidf2[doc_id][key]=0.0
            count = 0
            for word in text:
                if key == word:
                    count += 1
            if count ==0:
                tf_idf=0
            if count!=0:
                tf_idf = (1.0 + math.log10(count))/(1.0 + math.log10(avgtf[doc_id]))
            #print(tf_idf)
            doc_tfidf2[doc_id][key]=tf_idf

    return doc_tfidf2

#anc.apc Doc
def doc_tf_idf3(temp,maxtf):
    doc_tfidf3 = {}
    path = doc_path+"/*/*"
    for filename in glob.glob(path):
        #print(filename)
        file=filename.split("/")
        doc_id=file[1]
        doc_tfidf3[doc_id]={}
        content = ""
        f=open(filename,'r')
        for line in f:
            content += str(line)
        text_start = content.find('<TEXT>')
        text_end = content.find('</TEXT>')
        text = content[text_start + 6:text_end]
        text_tokens = word_tokenize(text)
        stop_text = stop_removal(text_tokens)
        punct_text = punc_removal(stop_text)
        lemm_text = lemma(punct_text)
        text = lemm_text.split(" ")
        #print(text)
        for key in temp.keys():
            doc_tfidf3[doc_id][key]=0.0
            count = 0
            for word in text:
                if key == word:
                    count += 1
            if count ==0:
                tf_idf=0
            if count!=0:
                tf_idf = (0.5 + 0.5*count/maxtf[doc_id])
            #print(tf_idf)
            doc_tfidf3[doc_id][key]=tf_idf

    return doc_tfidf3

#Inc.Itc
def query_tf_idf1(temp1,temp2,total_docs):
    query_tfidf1={}
    maxtf = {}
    avgtf = {}
    for key1,value in temp2.items():
        max = 0
        avg = 0
        query_tfidf1[key1] = {}
        text = value.split(" ")
        #print(text)
        for key2 in text:
            query_tfidf1[key1][key2]=0.0
            count = 0
            for word in text:
                if key2 == word:
                    count += 1
            if count ==0:
                tf_idf=0
            if count!=0 and key2 in df.keys():
                avg+=count
                tf_idf = (1.0 + math.log10(count))*(math.log10(total_docs / df[key2]))
                if count>max:
                    max=count
            #print(tf_idf)
            query_tfidf1[key1][key2]=tf_idf
            wordcount = len(text)
            #print(wordcount)
            # print(str(avg)+':'+str(wordcount))
            maxtf[key1] = max
            avgtf[key1] = avg / wordcount

    return query_tfidf1,maxtf,avgtf

#Lnc.Lpc
def query_tf_idf2(temp1,temp2,avgqtf,total_docs):
    query_tfidf2 = {}
    for key1,value in temp2.items():
        query_tfidf2[key1] = {}
        text = value.split(" ")
        for key2 in temp1.keys():
            query_tfidf2[key1][key2]=0.0
            count = 0
            for word in text:
                if key2 == word:
                    count += 1
            if count ==0:
                tf_idf=0
            if count!=0:
                tf_idf = (1.0 + math.log10(count))/(1.0 + math.log10(avgqtf[key1]))
                x=math.log10((total_docs-df[key2])/df[key2])
                tf_idf*=max(0.0,x)
            #print(tf_idf)
            query_tfidf2[key1][key2]=tf_idf

    return query_tfidf2

#anc.apc
def query_tf_idf3(temp1,temp2,maxqtf,total_docs):
    query_tfidf3 = {}
    for key1,value in temp2.items():
        query_tfidf3[key1] = {}
        text = value.split(" ")
        for key2 in temp1.keys():
            query_tfidf3[key1][key2]=0.0
            count = 0
            for word in text:
                if key2 == word:
                    count += 1
            if count ==0:
                tf_idf=0
            if count!=0:
                tf_idf = tf_idf = (0.5 + 0.5*count/maxqtf[key1])
                tf_idf*=max(0.0,math.log10((total_docs-df[key2])/df[key2]))
            #print(tf_idf)
            query_tfidf3[key1][key2]=tf_idf

    return query_tfidf3

# ------cosine similerity-------------
def cosine_similerity(doc_tf_idf,query_tf_idf):
    doc_query_cosine={}
    mul=0
    for key1,value1 in query_tf_idf.items():
        doc_query_cosine[key1]={}
        ls1=[entry for entry in value1.values()]
        p=np.sqrt(np.sum(np.square(ls1)))
        for key2,value2 in doc_tf_idf.items():
                ls2=[entry for entry in value2.values()]
                q=np.sqrt(np.sum(np.square(ls2)))
                mul=0
                for key3,value3 in value1.items():
                    for key4,value4 in value2.items():
                        if(key3==key4):
                            mul+=value3*value4
                cosim=mul/p*q
                print("Query Id : "+str(key1)+" Docment id : "+str(key2))
                doc_query_cosine[key1][key2]=cosim
    return doc_query_cosine
                    
                    

#--top 50 document-------------
def top_50_rank(doc_query_cosine):
    out={}
    for qid,value in doc_query_cosine.items():
        out[qid] = []
        lst = [(key,idf) for key,idf in value.items()]
        lst.sort(key=lambda x:x[1], reverse=True)
        lst = lst[:50]
        out[qid]= [item[0] for item in lst]
    return out

doc_path= sys.argv[1]
filename = sys.argv[2]
outfile = open(filename, 'rb')
temp1=pickle.load(outfile)

#df for all terms
df=calc_df(temp1)
#print(df)

#Doc Schemes
doc_tfidfscheme1,maxtf,avgtf=doc_tf_idf1(temp1,doc_path)
#print(doc_tfidfscheme1)
#print(maxtf)
#print(avgtf)
print("\n********Doc Done************\n")
#doc_tfidfscheme2=doc_tf_idf2(temp1,avgtf,doc_path)
#print(doc_tfidfscheme2)

#doc_tfidfscheme3=doc_tf_idf3(temp1,maxtf,doc_path)
#print(doc_tfidfscheme3)

path = doc_path + "/*/*"
count = 0
for filename in glob.glob(path):
    count += 1
total_docs = count

#Queries Schemes
filename1 = 'queries_processing_07.pth'
outfile1 = open(filename1, 'rb')
temp2 = pickle.load(outfile1)
#print(temp2.keys())

query_tfidfscheme1,maxqtf,avgqtf=query_tf_idf1(temp1,temp2,total_docs)
print(query_tfidfscheme1)
print("\n********Query Done************\n")
#query_tfidfscheme2=query_tf_idf2(temp1,temp2,avgqtf,total_docs)
#print(query_tfidfscheme2)

#query_tfidfscheme3=query_tf_idf3(temp1,temp2,maxqtf,total_docs)
#print(query_tfidfscheme3)

doc_query_cosine = cosine_similerity(doc_tfidfscheme1, query_tfidfscheme1)
out=top_50_rank(doc_query_cosine)
with open('PAT2_07_ranked_list_A.csv', 'w') as csv_file:
    csv_file.write("Q Id,Doc Id\n")
    for qid, val in out.items():
        for i in val:
            csv_file.write(str(qid) + "," + i + "\n")
print("\n********Writing to csv file Done************\n")
# doc_query_cosine2 = cosine_similerity(doc_tfidfscheme2, query_tfidfscheme2)
# out2=top_50_rank(doc_query_cosine2)
# with open('out2.csv', 'w') as csv_file:
#     csv_file.write("Q Id,Doc Id\n")
#     for qid, val in out2.items():
#         for i in val:
#             csv_file.write(str(qid) + "," + i + "\n")
#
# doc_query_cosine3 = cosine_similerity(doc_tfidfscheme3, query_tfidfscheme3)
# out3=top_50_rank(doc_query_cosine3)
# with open('out3.csv', 'w') as csv_file:
#     csv_file.write("Q Id,Doc Id\n")
#     for qid, val in out3.items():
#         for i in val:
#             csv_file.write(str(qid) + "," + i + "\n")

outfile.close()
outfile1.close()
