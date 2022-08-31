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

stop_words = list(stopwords.words('english'))

count_tf = {}
word_tf = {}
index = {}
def build_inverted_index(text, docid):
    
    

    for w in text:
        count = 1
        # print(word_tf)
        # print(index)
        if w in index.keys():
            if docid not in index[w]:
                index[w].append(docid)
                count_tf[w] = count
                word_tf[docid] = count_tf
                count += 1
            elif docid in index[w]:
                q = word_tf[docid]
                p = int(q[w])
                r=p+1
                count_tf[w] = 1+math.log10(r)
                word_tf[docid]=count_tf

        else:
            index[w] = [docid]
            count_tf[w] = count
            word_tf[docid] = count_tf
            count += 1
    print("length od Dictionary : "+str(len(word_tf)))
    return word_tf


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
    df = {}
    for key, value in temp.items():
        df[key] = len(value)
    return df


# Inc.Itc DOC
def doc_tf_idf1(temp):
    doc_tfidf1 = {}
    maxtf = {}
    avgtf = {}
    path = "en_BDNews24/*/*"
    for filename in glob.glob(path):
        max = 0
        avg = 0
        print(filename)
        file = filename.split("/")
        print(file)
        doc_id = file[0]
        doc_tfidf1[doc_id] = {}
        content = ""
        f = open(filename, 'r')
        for line in f:
            content += str(line)
        text_start = content.find('<TEXT>')
        text_end = content.find('</TEXT>')
        text = content[text_start + 6:text_end]
        text_tokens = word_tokenize(text)
        stop_text = stop_removal(text_tokens)
        punct_text = punc_removal(stop_text)
        word_tf = build_inverted_index(punct_text, doc_id)
        lemm_text = lemma(punct_text)
        text = lemm_text.split(" ")
        # for key, value in word_tf.items():
        #     # term_tfidf1[doc_id][key] = 0.0
        #     for doc_id, idf_value in value.items():
        #         p = int(idf_value)
        #         tf_idf = (1.0 + math.log10(p))

        #         doc_tfidf1[doc_id][key] = tf_idf
        ls = set([entry.lower() for entry in text])
        wordcount = len(ls) - 1
        # print(str(avg)+':'+str(wordcount))
        maxtf[doc_id] = max
        if (wordcount != 0):
            avgtf[doc_id] = avg / wordcount
        if (wordcount == 0):
            avgtf[doc_id] = 0
    return word_tf, maxtf, avgtf


# Lnc.Lpc Doc
def doc_tf_idf2(temp, avgtf):
    doc_tfidf2 = {}
    path = "data/en_BDNews24/1/*"
    for filename in glob.glob(path):
        # print(filename)
        file = filename.split("/")
        doc_id = file[1]
        doc_tfidf2[doc_id] = {}
        content = ""
        f = open(filename, 'r')
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
        # print(text)
        for key in temp.keys():
            doc_tfidf2[doc_id][key] = 0.0
            count = 0
            for word in text:
                if key == word:
                    count += 1
            if count == 0:
                tf_idf = 0
            if count != 0:
                tf_idf = (1.0 + math.log10(count)) / (1.0 + math.log10(avgtf[doc_id]))
            # print(tf_idf)
            doc_tfidf2[doc_id][key] = tf_idf

    return doc_tfidf2


# anc.apc Doc
def doc_tf_idf3(temp, maxtf):
    doc_tfidf3 = {}
    path = "data/en_BDNews24/1/*"
    for filename in glob.glob(path):
        # print(filename)
        file = filename.split("/")
        doc_id = file[1]
        doc_tfidf3[doc_id] = {}
        content = ""
        f = open(filename, 'r')
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
        # print(text)
        for key in temp.keys():
            doc_tfidf3[doc_id][key] = 0.0
            count = 0
            for word in text:
                if key == word:
                    count += 1
            if count == 0:
                tf_idf = 0
            if count != 0:
                tf_idf = (0.5 + 0.5 * count / maxtf[doc_id])
            # print(tf_idf)
            doc_tfidf3[doc_id][key] = tf_idf

    return doc_tfidf3


# Inc.Itc
def query_tf_idf1(temp1, temp2):
    query_tfidf1 = {}
    maxtf = {}
    avgtf = {}
    path = "en_BDNews24/*/*"
    total_docs = 0
    for filename in glob.glob(path):
        total_docs += 1
    n = total_docs
    for key1, value in temp2.items():
        max = 0
        avg = 0
        query_tfidf1[key1] = {}
        text = value.split(" ")
        for key2 in temp1.keys():
            query_tfidf1[key1][key2] = 0.0
            count = 0
            for word in text:
                if key2 == word:
                    count += 1
            if count == 0:
                tf_idf = 0
            if count != 0:
                avg += count
                tf_idf = (1.0 + math.log10(count)) * (math.log10(n / df[key2]))
                if count > max:
                    max = count
            # print(tf_idf)
            query_tfidf1[key1][key2] = tf_idf
            ls = set([entry.lower() for entry in text])
            wordcount = len(ls)
            # print(str(avg)+':'+str(wordcount))
            maxtf[key1] = max
            avgtf[key1] = avg / wordcount

    return query_tfidf1, maxtf, avgtf


# Lnc.Lpc
def query_tf_idf2(temp1, temp2, avgqtf):
    query_tfidf2 = {}
    path = "data/en_BDNews24/1/*"
    total_docs = 0
    for filename in glob.glob(path):
        total_docs += 1
    n = total_docs
    for key1, value in temp2.items():
        query_tfidf2[key1] = {}
        text = value.split(" ")
        for key2 in temp1.keys():
            query_tfidf2[key1][key2] = 0.0
            count = 0
            for word in text:
                if key2 == word:
                    count += 1
            if count == 0:
                tf_idf = 0
            if count != 0:
                tf_idf = (1.0 + math.log10(count)) / (1.0 + math.log10(avgqtf[key1]))
                x = math.log10((n - df[key2]) / df[key2])
                tf_idf *= max(0.0, x)
            # print(tf_idf)
            query_tfidf2[key1][key2] = tf_idf

    return query_tfidf2


# anc.apc
def query_tf_idf3(temp1, temp2, maxqtf):
    query_tfidf3 = {}
    path = "data/en_BDNews24/1/*"
    total_docs = 0
    for filename in glob.glob(path):
        total_docs += 1
    n = total_docs
    for key1, value in temp2.items():
        query_tfidf3[key1] = {}
        text = value.split(" ")
        for key2 in temp1.keys():
            query_tfidf3[key1][key2] = 0.0
            count = 0
            for word in text:
                if key2 == word:
                    count += 1
            if count == 0:
                tf_idf = 0
            if count != 0:
                tf_idf = tf_idf = (0.5 + 0.5 * count / maxqtf[key1])
                tf_idf *= max(0.0, math.log10((n - df[key2]) / df[key2]))
            # print(tf_idf)
            query_tfidf3[key1][key2] = tf_idf

    return query_tfidf3


# ------cosine similerity-------------
def cosine_similerity(doc_tf_idf, query_tf_idf):
    doc_query_cosine = {}
    for key1, value1 in query_tf_idf.items():
        doc_query_cosine[key1] = {}
        ls1 = [item for item in value1.values()]
        print(type(ls1))
        for key2, value2 in doc_tf_idf.items():
            cosim = 0
            ls2 = [entry for entry in value2.values()]
            print(type(ls2))
            if np.dot(ls1, ls2) != 0:
                cosim = np.dot(ls1, ls2) / (np.sqrt(np.sum(np.square(ls1))) * np.sqrt(np.sum(np.square(ls2))))
            doc_query_cosine[key1][key2] = cosim

    return doc_query_cosine


# --top 50 document-------------
def top_50_rank(doc_query_cosine):
    out = {}
    for qid, value in doc_query_cosine.items():
        out[qid] = []
        lst = [(key, idf) for key, idf in value.items()]
        lst.sort(key=lambda x: x[1], reverse=True)
        lst = lst[:50]
        out[qid] = [item[0] for item in lst]
    return out


filename = 'model_queries_07.pth'
outfile = open(filename, 'rb')
temp1 = pickle.load(outfile)

# df for all terms
df = calc_df(temp1)
# print(df)

# Doc Schemes
doc_tfidfscheme1, maxtf, avgtf = doc_tf_idf1(temp1)
# print(doc_tfidfscheme1)
# print(maxtf)
# print(avgtf)

# doc_tfidfscheme2=doc_tf_idf2(temp1,avgtf)
# print(doc_tfidfscheme2)

# doc_tfidfscheme3=doc_tf_idf3(temp1,maxtf)
# print(doc_tfidfscheme3)

# Queries Schemes
filename1 = 'queries_processing_07.pth'
outfile1 = open(filename1, 'rb')
temp2 = pickle.load(outfile1)
# print(temp2)
print("\n ********* doc or\n")
query_tfidfscheme1, maxqtf, avgqtf = query_tf_idf1(temp1, temp2)
print(query_tfidfscheme1)

# query_tfidfscheme2=query_tf_idf2(temp1,temp2,avgqtf)
# print(query_tfidfscheme2)

# query_tfidfscheme3=query_tf_idf3(temp1,temp2,maxqtf)
# print(query_tfidfscheme3)

doc_query_cosine = cosine_similerity(doc_tfidfscheme1, query_tfidfscheme1)
out = top_50_rank(doc_query_cosine)
with open('out1.csv', 'w') as csv_file:
    csv_file.write("Q Id,Doc Id\n")
    for qid, val in out.items():
        for i in val:
            csv_file.write(str(qid) + "," + i + "\n")

# doc_query_cosine2 = cosine_similerity(doc_tfidfscheme2, query_tfidfscheme2)
# out2=top_50_rank(doc_query_cosine2)
# with open('out2.csv', 'w') as csv_file:
#     csv_file.write("Q Id,Doc Id\n")
#     for qid, val in out2.items():
#         for i in val:
#             csv_file.write(str(qid) + "," + i + "\n")
# doc_query_cosine3 = cosine_similerity(doc_tfidfscheme3, query_tfidfscheme3)
# out3=top_50_rank(doc_query_cosine3)
# with open('out2.csv', 'w') as csv_file:
#     csv_file.write("Q Id,Doc Id\n")
#     for qid, val in out3.items():
#         for i in val:
#             csv_file.write(str(qid) + "," + i + "\n")
outfile.close()
outfile1.close()