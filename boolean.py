import pickle
from nltk.tokenize import word_tokenize
file2 = open('Downloads/data/boolean1_out.txt', 'w')
filename = 'Downloads/data/model_queries_07.pth'
outfile = open(filename, 'rb')
d=pickle.load(outfile)
print(d)
file1=open('Downloads/data/query7.txt', "r")
for line in file1:
  root=[]
  text_tokens = word_tokenize(line)
  #print(text_tokens[2])
  if(text_tokens[2] in d.keys()):
      root=d.get(str(text_tokens[2]))
  for i in range(3,len(text_tokens)):
      if(text_tokens[i] in d.keys()):
          val=d.get(str(text_tokens[i]))
          #print(line+str(root))
          root=[n for n in root if n in val]
  file2.write(text_tokens[0]+":"+str(root)+'\n')
  print("-----query ends----")
#print(root)