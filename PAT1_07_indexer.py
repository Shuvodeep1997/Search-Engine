"""
    Problem Statement 01 : Building the Index
    Author : Group 07

"""

import sys
import glob
import nltk
import pickle
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

stop_words = list(stopwords.words('english'))

# ----------building inverted indeex-------------

def build_inverted_index(text, docid):
    for w in text:
        if w in index.keys():
            if docid not in index[w]:
                index[w].append(docid)
        else:
            index[w] = [docid]


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




index = {}
path=sys.argv[1]
path = path + "/*/*"
for filename in glob.glob(path):
    content = ""
    # print("------new content----------")
    with open(filename, 'r') as f:
        for line in f:
            # print(str(line))
            content += str(line)
    text_start = content.find('<TEXT>')
    text_end = content.find('</TEXT>')
    doc_start = content.find('<DOCNO>')
    doc_end = content.find('</DOCNO')
    doc = content[doc_start + 7:doc_end]
    doc_id = str(doc)
    # print("-------"+doc_id)
    text = content[text_start + 6:text_end]
    # print("without </text> the story is:"+str(text))
    text_tokens = word_tokenize(text)
    stop_text = stop_removal(text_tokens)
    punct_text = punc_removal(stop_text)
    # print(doc_id)
    lemm_text = lemma(punct_text)
    # print(lemm_text)
    build_inverted_index(punct_text, doc_id)


#Writing the Inverted Index into a file

filename = 'model_queries_07.pth'
outfile = open(filename, 'wb')
pickle.dump(index,outfile,protocol=pickle.HIGHEST_PROTOCOL)
outfile.close()

filename = 'model_queries_07.pth'
outfile = open(filename, 'rb')
temp = pickle.load(outfile)
print(temp)

outfile.close()
