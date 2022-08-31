import csv
import pickle
import numpy as np
import math

def calc_avg_precision(top20docs_new,type1,type2):
 list_10=[]
 list_20=[]
 count10=0
 count20=0
 newkey=list(top20docs_new.values())[0]

 finalnew=list(newkey.values())
 for i in finalnew:
     count10+=1
     if(count10<=type1):
         list_10.append(i)

 for j in finalnew:
  count20 += 1
  if (count20<=type2):
       list_20.append(j)


 final_list10=precision_10(list_10)
 final_list20=precision_20(list_20)
 return final_list10,final_list20

def precision_10(list_10):
    total = 0
    count=0
    result = 0
    relevant = 0

    for i in list_10:
        if (i == 0):
            total = total + 1
        else:
            relevant = relevant + 1
            total = total + 1
            result = result + relevant / total

    if (relevant > 0):
        x = result / relevant

        precision_list_10.append(x)

        return precision_list_10
    if (relevant == 0):
        precision_list_10.append(0)

def precision_20(list_20):
    total = 0
    count=0
    result = 0
    relevant = 0

    for i in list_20:
        if (i == 0):
            total = total + 1
        else:
            relevant = relevant + 1
            total = total + 1
            result = result + relevant / total

    if (relevant > 0):
        x = result / relevant

        precision_list_20.append(x)
        return precision_list_20
    if(relevant==0):
     precision_list_20.append(0)

def mean_avg_precision(result10,result20):
 average_10=sum(result10) / len(result10)
 average_20=sum(result20)/ len(result20)
 print("\n Mean Average precision @10 is  ",average_10)
 print("\n Mean Average precision @20 is  ",average_20)

def calc_ndcg(top20docs_new, type1,type2):
    #Your Code starts here
    ndcg10 = []
    ndcg20 = []
    result10=[]
    result20=[]
    count10 = 0
    count20 = 0
    newkey = list(top20docs_new.values())[0]

    finalnew = list(newkey.values())
    for i in finalnew:
        count10 += 1
        if (count10 <= type1):
            ndcg10.append(i)

    for j in finalnew:
        count20 += 1
        if (count20 <= type2):
            ndcg20.append(j)

    result10=dcg_10(ndcg10)

    result20=dcg_20(ndcg20)


    x1=sum(result10)/len(result10)
    x2=sum(result20)/len(result20)
    ndcg_10.append(x1)
    ndcg_20.append(x2)

def dcg_10(list10):

    dcg10=[]
    sum=list10[0]
    ndcgresult10=[]
    dcg10.append(list10[0])
    total=2
    for i in list10[1:10]:
        y=math.log2(total)
        x=i/y
        total=total+1
        sum+=x
        dcg10.append(sum)
    ndcgresult10=ideal_dcg10(list10,dcg10)
    return ndcgresult10

def ideal_dcg10(list10,dcg10):
    list10.sort(reverse=True)
    ndcg10=[]
    idcg10 = []
    sum = list10[0]

    idcg10.append(list10[0])
    total = 2
    for i in list10[1:10]:
        y = math.log2(total)
        x = i / y
        total = total + 1
        sum += x
        idcg10.append(sum)

    for j in range(len(dcg10)):
        if (idcg10[j] == 0):
            ndcg10.append(0)
        if (idcg10[j] != 0):
         x=dcg10[j]/idcg10[j]
         ndcg10.append(x)

    return ndcg10

def ideal_dcg20(list20,dcg20):
    list20.sort(reverse=True)
    ndcg20=[]
    idcg20 = []
    sum = list20[0]

    idcg20.append(list20[0])
    total = 2
    for i in list20[1:20]:
        y = math.log2(total)
        x = i / y

        total = total + 1
        sum += x
        idcg20.append(sum)
    for j in range(len(dcg20)):
        if (idcg20[j] == 0):
            ndcg20.append(0)
        if (idcg20[j] != 0):
         x=dcg20[j]/idcg20[j]
         ndcg20.append(x)
    return ndcg20

def dcg_20(list20):
     dcg20 = []
     sum = list20[0]
     ndcgresult20 = []
     dcg20.append(list20[0])
     total = 2
     for i in list20[1:20]:
         y = math.log2(total)
         x = i / y

         total = total + 1
         sum += x
         dcg20.append(sum)

     ndcgresult20=ideal_dcg20(list20, dcg20)
     return ndcgresult20

def mean_avg_ndcg(ndcg_10,ndcg_20):
    averagendcg10=sum(ndcg_10)/len(ndcg_10)
    averagendcg20 = sum(ndcg_20) / len(ndcg_20)
    print("Average NDCG @ 10",averagendcg10)
    print("Average NDCG @ 20",averagendcg20)

def get_rank_from_gold_standard(top20docs):
    gold_std_file=open('data/rankedRelevantDocList.csv')
    csvread=csv.reader(gold_std_file)
    key1=list(top20docs.keys())[0]
    for row in csvread:
        for key2 in top20docs[key1].keys():
            if (row[0]==key1) and (row[1]==key2):
                top20docs[key1][key2] = int(row[2])
                break
    return top20docs

def get_doc(query_id,csvreader):
    top20docs = {}
    top20docs[str(query_id)] = {}
    count = 0
    for row in csvreader:
        if count >= 20:
            break
        if row[0] == str(query_id):
            count += 1
            top20docs[row[0]][row[1]] = 0
    return top20docs

query_doc_file=open('PAT2_gno_ranked_list_A.csv')
csvreader=csv.reader(query_doc_file)

filename = 'queries_processing_07.pth'
outfile = open(filename, 'rb')
temp = pickle.load(outfile)

id_list=[key for key in temp.keys()]
avg_precision_10={}
avg_precision_20={}
ndcg_10=[]
ndcg_20=[]
N=1
precision_list_10=[]
precision_list_20=[]

for element in id_list:

       result10= []
       result20 = []
       top20docs_original = get_doc(element, csvreader)

       top20docs_new = get_rank_from_gold_standard(top20docs_original)

       calc_avg_precision(top20docs_new,10,20)
       calc_ndcg(top20docs_new,10,20)



print("Precision @ 10 \n",precision_list_10)
print("\n Precision @ 20 \n",precision_list_20)
print("\n NDCG @ 10 \n",ndcg_10)
print("\n NDCG @ 20 \n",ndcg_20)
mean_avg_precision(precision_list_10,precision_list_20)
mean_avg_ndcg(ndcg_10,ndcg_20)
