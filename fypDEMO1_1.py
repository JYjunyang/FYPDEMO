from tkinter import *
from tkinter import ttk
from tkinter import filedialog
from pandastable import Table
from textblob import TextBlob
from PIL import Image, ImageTk
from gensim.models import Word2Vec
from collections import defaultdict
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
import time
import numpy as np
import tkinter
import pandas as pd
import nltk
import re

#some basic things/imports
window = Tk() #The main window
nltk.download("stopwords")
stopwords = nltk.corpus.stopwords.words("english")
wn = nltk.WordNetLemmatizer() #wn = wordnet
nltk.download('omw-1.4')

#word2vec
def FeaturesExtraction():
    word2vec = Word2Vec(sentences=LemmaColumn,vector_size=100,window=5,min_count=1,workers=8,sg=1)
    b1 = time.time()
    train_time = time.time() - b1
    print(word2vec.wv.most_similar('virus', topn=10))
    
    # most_similar_word_man = word2vec.wv.most_similar(positive='man', topn=10)
    # most_similar_word_virus = word2vec.wv.most_similar(positive='virus', topn=10)
    # most_similar_word_hate = word2vec.wv.most_similar(positive='hate', topn=10)
    # most_similar_word_dead = word2vec.wv.most_similar(positive='dead', topn=10)
    # textWidget1 = Text(theRealTab1)
    # textWidget1.grid(column=0,row=9,pady=5)
    # textWidget1.insert(INSERT, "The top 10 most similar word example after feature extraction "
    #                    +"is applied (COVID): \n")
    # for x in range(len(most_similar_word_corona)):
    #     FEresult = ''.join([i for i in str(most_similar_word_corona[x]) 
    #                         if not i.isdigit() and i != "." and i != "," and i != "(" and i!= ")"])
    #     FEresult += '\n'
    #     textWidget1.insert(END, FEresult)
        
#     for eveSentence in sentences:
#         for eveWords in eveSentence:
#             sentenceAveVec = get_ave_vector(word2vec, eveWords)
#     #region splitting data
#     X_trainval, X_test, y_trainval, y_test = train_test_split(sentenceAveVec,dfOri[['Sentiment']].tolist,
#                                                               random_state=0)
#     #X_train,X_val,y_train,y_val = train_test_split(X_trainval,y_trainval,random_state = 1)
#     best_score = 0.0
#     for gamma in [0.001,0.01,0.1,1,10,100]:
#         for C in [0.001,0.01,0.1,1,10,100]:
#             svm = SVC(gamma=gamma,C=C)
#             scores = cross_val_score(svm,X_trainval,y_trainval,cv=5)
#             score = scores.mean()
#             if score > best_score:
#                 best_score = score
#                 best_parameters = {'gamma':gamma,'C':C}
#     svm = SVC(**best_parameters)
#     svm.fit(X_trainval,y_trainval)
#     test_score = svm.score(X_test,y_test)
#     print("Best score on validation set: {:.2f}".format(best_score))
#     print("Best parameters: {}".format(best_parameters))
#     print("Best score on test set: {:.2f}".format(test_score))

# def get_ave_vector(w2v_model, words):
#     words = [word for word in words if word in w2v_model.vocab]
#     if len(words) >= 1:
#         for word in words:
#             theList = np.mean(w2v_model.vocab[word])
#         return np.mean(theList, axis=0)
#     else:
#         return []
#region window size & position
window.title("Sentiment Analysis Tool")
window.geometry("1000x600")

w=800
h=400

ws = window.winfo_screenwidth()
hs = window.winfo_screenheight()

x = (ws/2) - (w/2)
y = (hs/2) - (h/2)

window.geometry("%dx%d+%d+%d" % (w,h,x,y))
#endregion

#region TAB
tabManager = ttk.Notebook(window)

#Tab1
tab1 = ttk.Frame(tabManager)
tab1.pack(fill=BOTH, expand=1,)

canvasTab1 = tkinter.Canvas(tab1, width=1366, height=768)
canvasTab1.pack(side=LEFT, fill=BOTH, expand=1)

vbar = Scrollbar(tab1,orient=VERTICAL,command=canvasTab1.yview)
vbar.pack(side=RIGHT, fill=Y)

canvasTab1.configure(yscrollcommand=vbar.set)

theRealTab1 = Frame(canvasTab1)
theRealTab1.bind('<Configure>', lambda e: canvasTab1.configure(scrollregion=canvasTab1.bbox('all')))
canvasTab1.create_window((0,0),window=theRealTab1,anchor=NW)
#-----------------------------------------------------------------------------------------------#
#Tab2
tab2 = ttk.Frame(tabManager)
tab3 = ttk.Frame(tabManager)

#Parent adding child widgets
tabManager.add(tab1, text="Data Preprocessing")
tabManager.add(tab2, text="Results")
tabManager.add(tab3, text="Details")
tabManager.grid(sticky="ew")
#endregion

#region title & labels
titleLabeltab1 = Label(theRealTab1, text="Natural Language Data Preprocessing: ",padx=5)
titleLabeltab1.grid(column=0,row=1)
titleLabeltab2 = Label(tab2, text="Results: ",padx=5)
titleLabeltab2.grid(column=0,row=1)
titleLabeltab3 = Label(tab3, text="Details: ",padx=5)
titleLabeltab3.grid(column=0,row=1)

tb1Label = Label(theRealTab1, text="Choose a file for data pre-processing: ")
tb1Label.grid(column=0,row=2,padx=0,pady=0)

TokenizedLabel = Label(theRealTab1, text="Tokenized Results: ")
SWLabel = Label(theRealTab1, text="Stop Words Removed Results:")
LemmaLabel = Label(theRealTab1, text="Lemmatization Results:")

imageData = ImageTk.PhotoImage(Image.open("C://Users//USER//OneDrive//sem5//FYP - 1//PPT_FYP1//server.png").resize((50,50)))
tb1LabelImage = tkinter.Label(theRealTab1,image = imageData)
tb1LabelImage.image = imageData
tb1LabelImage.grid(row = 1,column = 1, columnspan = 7, sticky = E)
theRealTab1.grid_rowconfigure(1)
#endregion

def openExcel():
    global dfOri
    filePath = filedialog.askopenfilename()
    dfOri = pd.read_csv(filePath)
    TableB4Anything = Table(datasetsOriFrame, dataframe=dfOri, width=300)
    TableB4Anything.show()

def TriggerLemma():
    global LemmaColumn
    LemmaColumn = SWcolumn[['full_text']]
    LemmaColumn['full_text'] = LemmaColumn['full_text'].map(lambda x: lemmatization(x))
    tableAfterLemma = Table(datasetsResults3, dataframe=LemmaColumn, width=300, editable=True)
    LemmaLabel.grid(column=7,row=6)
    tableAfterLemma.show()
    
def lemmatization(stopWordsOnly):
    LemmaColumn = [wn.lemmatize(word) for word in stopWordsOnly]
    return LemmaColumn
    
def TriggerStopWords():
    global SWcolumn
    SWcolumn = theColumn[['full_text']]
    SWcolumn['full_text'] = SWcolumn['full_text'].map(lambda x: stopWordsRemoval(x))
    tableAfterSW = Table(datasetsResults2, dataframe=SWcolumn,width=300, editable=True)
    SWLabel.grid(column=4,row=6)
    tableAfterSW.show()
    
def stopWordsRemoval(tokenizedTextOnly):
    SWcolumn = [word for word in tokenizedTextOnly if word not in stopwords]
    return SWcolumn

def TriggerTokenization():
    global theColumn
    theColumn = dfOri[['full_text']]
    theColumn['full_text'] = theColumn['full_text'].map(lambda x: Tokenization(str(x)))
    tableAfterToken = Table(datasetsResults, dataframe=theColumn, width=300, editable=True)
    TokenizedLabel.grid(column=0,row=6)
    tableAfterToken.show()

def Tokenization(text):
    text = text.replace('RT', '')
    text = text.lower()
    text = re.sub(r'[^\w\s]','',text)
    text = re.sub('W*dw','',text)
    Tweets = TextBlob(text)
    TokenizedTweets = Tweets.words
    return TokenizedTweets

#region All Buttons
buttonDir = Button(theRealTab1, wraplength = 70,text="Choose a csv or excel datasets",command=openExcel)#import datasets
buttonDir.grid(column=1,row=2)
buttonTokens = Button(theRealTab1, text="Tokenization", width=9, command=TriggerTokenization)#token
buttonTokens.grid(column=1,row=3,pady=5)
buttonSW = Button(theRealTab1, text="Stop Words Removal", width=18, command=TriggerStopWords)#stopwords
buttonSW.grid(column=4,row=3,pady=5)
buttonLemma = Button(theRealTab1, text="Lemmatization", width=12, command=TriggerLemma)#lemma
buttonLemma.grid(column=7,row=3,pady=5)
buttonFE = Button(theRealTab1, text="Features Extraction", width=16, command=FeaturesExtraction)#Feature Extraction
buttonFE.grid(column=0,row=8,pady=5)
#endregion

#region All Frames
#datasets preview frame
datasetsOriFrame = Frame(theRealTab1)
datasetsOriFrame.grid(column=0,row=5,columnspan=8,padx=1,pady=5,sticky="ew")
theRealTab1.grid_columnconfigure(0)

#datasets results frame
datasetsResults = Frame(theRealTab1)
datasetsResults.grid(column=0,columnspan=2,row=7,padx=1,pady=5)

#datasets stop words frames
datasetsResults2 = Frame(theRealTab1)
datasetsResults2.grid(column=4,row=7,padx=1,pady=5)

#datasets lemmatization frames
datasetsResults3 = Frame(theRealTab1)
datasetsResults3.grid(column=7,row=7,padx=1,pady=5)
#endregion

window.mainloop()