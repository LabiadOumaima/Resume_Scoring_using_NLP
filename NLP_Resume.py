#importation des libraries
from docx import Document
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter  
from io import StringIO
from pdfminer.layout import LAParams
from pdfminer.converter import TextConverter
from pdfminer.pdfpage import PDFPage

def convert_docx_to_txt(path):        #Fonction qui convertit du format docx au format txt
    document=Document(path)
    return "\n".join([para.text for para in document.paragraphs])

def convert_pdf_to_txt(path):         # Fonction qui convertit du format pdf au format txt
    rsrcmgr = PDFResourceManager()    #stocker des ressources partagées telles que des polices ou des images
    retstr = StringIO()
    codec = 'utf-8'
    laparams = LAParams()
    device = TextConverter(rsrcmgr, retstr, codec=codec, laparams=laparams)
    fp = open(path, 'rb')
    interpreter = PDFPageInterpreter(rsrcmgr, device) #PDFPageInterpreter traite le contenu de la page et PDFDevice le traduit en fonction de besoins
    password = ""
    maxpages = 0
    caching = True
    pagenos=set()
    for page in PDFPage.get_pages(fp, pagenos, maxpages=maxpages, password=password,caching=caching, check_extractable=True):
        interpreter.process_page(page)
    fp.close()
    device.close()
    string = retstr.getvalue()
    retstr.close()
    return string

def read_file(fileName):            # Fonction qui lit tous les types de fichiers(txt, docx ou pdf)
        extension = fileName.split(".")[-1]   #returne l'extension du fichier qui est le dernier terme dans la liste des noms
        if extension == "txt":
            f = open(fileName, 'r') 
            string = f.read()
            f.close() 
            return string

        elif extension == "docx":
            return convert_docx_to_txt(fileName) 

        elif extension == "pdf":
             return convert_pdf_to_txt(fileName) 
        


#Traitement d'un CV


CV=read_file("CV DATA-28.pdf")
CV
text= CV.lower()
text


#teckonisation


import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokeniz
words=word_tokenize(text)
words


#stopwords


import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words = stopwords.words('french')
print('Il y a {} stopwords'.format(len(stop_words)))
print('Les 10 premiers stopwords sont {}'.format(stop_words[:10]))





import spacy
import fr_core_news_sm
nlp = fr_core_news_sm.load()
def lemmatize(text):
    nlp = fr_core_news_sm.load()
    text = nlp(text)
    text =" ".join ([word.lemma_ if word.lemma_ != '-PRON-' else word.text for word in text])
    return text
print(lemmatize(text))


# In[61]:


import re
def remove_email(text):
    emails = None
    pattern = re.compile(r'\S*@\S*')
    emails = pattern.findall(text)
    for email in emails:
        text = text.replace(email, '')
    return text
remove_email(text)


# In[62]:


def remove_names(text):
    list_names = [ent.text for ent in nlp(text).ents if ent.label_=="PER"]
    #list_names[0:100]
    for n in list_names:
        text=text.replace(n,'')
    return text
remove_names(text)


# In[64]:


def remove_loc(text):
    list_loc = [ent.text for ent in nlp(text).ents if ent.label_=="LOC"]
    for n in list_loc:
        text=text.replace(n,'')
    return text
remove_loc(text)


# In[ ]:


def remove_phone_number(text):
    match = None
    pattern = re.compile(
            r'([+(]?\d+[)\-]?[ \t\r\f\v]*[(]?\d{2,}[()\-]?[ \t\r\f\v]*\d{2,}[()\-]?[ \t\r\f\v]*\d*[ \t\r\f\v]*\d*[ \t\r\f\v]*)')
    match = pattern.findall(text)
    for number in match:
            text= text.replace(number, '')
    return text


# In[ ]:


remove_phone_number(text)


# In[65]:


''.join(i for i in text if not i.isdigit())  #Supression des nombres


# In[97]:


import string
words= word_tokenize(text)
punctuations= list(string.punctuation)+ ["–","…","«","»","✓","◆","•","...","’"]
List=["compétences","alpine","alpe","ae","allemagne","formations","formation","diplômes","parcours","connaissances","qualifications","intérêt","loisirs","expériences","professionnelles","parascolaire","activités parascolaires","langues","lu","écrit","parlé","certificats","certification","projets","adresse","intitulé","présentation","infos","informations","e-mail""numéro","coordonnées","téléphone","jour","mois","année","an","semestre","trimestre","janvier","février","mars","avril","mai","juin","juillet","août","septembre","bouches","octobre","novembre","décembre"]

def clean_cv(text):
    l=text.lower()
    doc1= remove_names(l)
    doc2=remove_loc(doc1)
    doc3= remove_email(doc2)
    doc4=remove_phone_number(doc3)
    doc5= ''.join(i for i in doc4 if not i.isdigit())
    doc6=lemmatize(doc5)
    words = word_tokenize(doc6)
    filtered_text = [word for word in words if 
                (word not in stop_words) and (word not in  punctuations) and (word not in List)] 
    assembled=" ".join(filtered_text)
    a= assembled.join("''")
    return a


# In[67]:


clean_cv(CV)


# In[16]:


from os import listdir
from os.path import isfile, join
onlyfiles = [f for f in listdir('C:/Users/red/Downloads/CVS') if isfile(join('C:/Users/red/Downloads/CVS',f))]

docs = [clean_cv(read_file(onlyfiles[i])) for i in range(50)]


# In[68]:


docs


# In[70]:





# In[71]:





# In[18]:





# In[19]:





# In[79]:


corpus=dict(list(enumerate([docs[i] for i in range(50)])))
terms=dict(list(enumerate([docs[i].split() for i in range(50)])))

from math import log

QUERY_TERMS = ['économie', 'appliquée']

def tf(term, doc, normalize=True):
    doc = doc.lower().split()
    if normalize:
        return doc.count(term.lower()) / float(len(doc))
    else:
        return doc.count(term.lower()) / 1.0


def idf(term, corpus):
    num_texts_with_term = len([True for text in corpus if term.lower()                               in text.lower().split()])
    try:
        return 1.0 + log(float(len(corpus)) / num_texts_with_term)
    except ZeroDivisionError:
        return 1.0

def tf_idf(term, doc, corpus):
    return tf(term, doc) * idf(term, corpus)



# In[80]:


for (k, v) in sorted(corpus.items()):
    print(k, ':', v)
print('\n')


# In[92]:


query_scores=dict(list(enumerate([0 for i in range(50)])))

for term in [t.lower() for t in QUERY_TERMS]:
    for doc in sorted(corpus):
        print('TF({}): {}'.format(doc, term), tf(term, corpus[doc]))
    print('IDF: {}'.format(term, ), idf(term, corpus.values()))
    print('\n')
    for doc in sorted(corpus):
        score = tf_idf(term, corpus[doc], corpus.values())
        print('TF-IDF({}): {}'.format(doc, term), score)
        query_scores[doc] += score
    print('\n')
print("Scores TF-IDF triés par ordre décroissant pour le terme '{}'".format(' '.join(QUERY_TERMS), ))
for (doc, score) in sorted(query_scores.items(),key=lambda score:score[1],reverse=True):
    if score!=0:
        print(doc,score)


# In[ ]:





# In[93]:


import pandas as pd
from sklearn.feature_extraction .text import TfidfVectorizer
vectorizer = TfidfVectorizer()
vectors = vectorizer.fit_transform([clean_cv(text)])
feature_names = vectorizer.get_feature_names()
dense = vectors.todense()
denselist = dense.tolist()
df = pd.DataFrame(denselist, columns=feature_names)


# In[94]:


df


# In[114]:


import pandas as pd
from sklearn.feature_extraction .text import TfidfVectorizer, CountVectorizer

vectorizer = CountVectorizer(ngram_range =(2, 2)) 
X1 = vectorizer.fit_transform([clean_cv(text)])
Bigramms= (vectorizer.get_feature_names()) 
X1.toarray()
vectorizer = TfidfVectorizer(ngram_range = (2, 2)) 
X2 = vectorizer.fit_transform([clean_cv(text)]) 
scores = (X2.toarray()) 


# In[115]:


Bigramms


# In[116]:


X1.toarray()
vectorizer = TfidfVectorizer(ngram_range = (2, 2)) 
X2 = vectorizer.fit_transform([clean_cv(read_file("CV DATA-28.pdf"))]) 
scores = (X2.toarray()) 
  
# Getting top ranking features 
sums = X2.sum(axis = 0) 
data1 = [] 
for col, term in enumerate(features):  
    data1.append( (term, sums[0, col] )) 
    ranking = pd.DataFrame(data1, columns = ['term', 'rank']) 
    ranked_terms = (ranking.sort_values('rank', ascending = False)) 
    print ("\n\nWords : \n", words.head(20)) 


# In[117]:


ranked_terms


# In[119]:


import pandas as pd
from sklearn.feature_extraction .text import TfidfVectorizer
documentA = 'the man went out for a walk'
documentB = 'the children sat around the fire'
bagOfWords= text.split(' ') 
uniqueWords=set(bagOfWords)
numOfWords= dict.fromkeys(uniqueWords,0)
for word in bagOfWords:
    numOfWords[word]+=1

    
def computeTF(wordDict, bagOfWords):
    tfDict = {}
    bagOfWordsCount = len(bagOfWords)
    for word, count in wordDict.items():
        tfDict[word] = count / float(bagOfWordsCount)
    return tfDict
    
tf = computeTF(numOfWords, bagOfWords)

def computeIDF(documents):
    import math
    N = len(documents)
    
    idfDict = dict.fromkeys(documents[0].keys(), 0)
    for document in documents:
        for word, val in document.items():
            if val > 0:
                idfDict[word] += 1
    
    for word, val in idfDict.items():
        idfDict[word] = math.log(N / float(val))
    return idfDict

idfs = computeIDF([numOfWords])

def computeTFIDF(tfBagOfWords, idfs):
    tfidf = {}
    for word, val in tfBagOfWords.items():
        tfidf[word] = val * idfs[word]
    return tfidf

tfidf = computeTFIDF(tf, idfs)
df = pd.DataFrame([tfidf])
print(df)







