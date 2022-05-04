from tkinter import *
from tkinter import ttk
from tkinter import filedialog
from pandastable import Table
from PIL import Image, ImageTk
from gensim.models import Doc2Vec
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
import gensim
import logging
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

imageData = ImageTk.PhotoImage(Image.open("C://Users//USER//OneDrive//sem5//FYP - 1//PPT_FYP1//server.png").resize((50,50)))
tb1LabelImage = tkinter.Label(theRealTab1,image = imageData)
tb1LabelImage.image = imageData
tb1LabelImage.grid(row = 1,column = 1, columnspan = 7, sticky = E)
theRealTab1.grid_rowconfigure(1)
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

def openExcel():
    global dfOri
    filePath = filedialog.askopenfilename()
    dfOri = pd.read_csv(filePath,squeeze=True)
    # dfOriSentiment = pd.read_csv(filePath, usecols = ['Sentiment'])
    TableB4Anything = Table(datasetsOriFrame, dataframe=dfOri, width=300)
    TableB4Anything.show()
    buttonTokens.grid(column=1,row=3,pady=5)

#region Data Preprocessing
def TriggerCleaning():
    dfOri['full_text'] = dfOri['full_text'].apply(Cleaning)
    dfOri['full_text'] = dfOri['full_text'].apply(nltk.word_tokenize)
    dfOri['full_text'] = dfOri['full_text'].apply(lambda x: [item for item in x if item not in stopwords])
    dfOri['full_text'] = dfOri['full_text'].apply(lemmatization)
    tableAfterToken = Table(datasetsResults, dataframe=dfOri, width=300, editable=True)
    TokenizedLabel.grid(column=0,row=6)
    tableAfterToken.show()
    buttonFE.grid(column=0,row=8,pady=5)
    print(dfOri)
    print(dfOri['Sentiment'])
        #splitting data
    param_grid = {'gamma':[0.001, 0.01, 0.1, 1, 10, 100],'C':[0.001, 0.01, 0.1, 1, 10, 100]}
    grid_search = GridSearchCV(SVC(),param_grid,cv=5,error_score='raise')
    X_train, X_test, y_train, y_test = train_test_split(dfOri,dfOri['Sentiment'],random_state=10)
    grid_search.fit(X_train,y_train)
    print("Best score on validation set: {:.2f}".format(grid_search.score(X_test,y_test)))
    print("Best parameters: {}".format(grid_search.best_params_))
    print("Best score on test set: {:.2f}".format(grid_search.best_score_))
    # best_score = 0.0
    # for gamma in [0.001,0.01,0.1,1,10,100]:
    #     for C in [0.001,0.01,0.1,1,10,100]:
    #         svm = SVC(gamma=gamma,C=C)
    #         scores = cross_val_score(svm,X_trainval,y_trainval,cv=5,error_score='raise')
    #         score = scores.mean()
    #         if score > best_score:
    #             best_score = score
    #             best_parameters = {'gamma':gamma,'C':C}
    # svm = SVC(**best_parameters)
    # svm.fit(X_trainval,y_trainval)
    # test_score = svm.score(X_test,y_test)
    # print("Best score on validation set: {:.2f}".format(best_score))
    # print("Best parameters: {}".format(best_parameters))
    # print("Best score on test set: {:.2f}".format(test_score))

def Cleaning(text):
    text = text.replace('RT', '')
    text = text.lower()
    text = re.sub(r'[^\w\s]','',text)
    text = re.sub('W*dw','',text)
    text = re.sub(r'[0-9]+', '', text)
    return text

def lemmatization(text):
    return [wn.lemmatize(word) for word in text]
#endregion
#region word2vec
def FeaturesExtraction():
    textWidget1 = Text(theRealTab1)
    textWidget1.grid(column=0,row=9,pady=5)
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    document = [gensim.models.doc2vec.TaggedDocument(words=row['full_text'],tags=[index]) for index, row in dfOri.iterrows()]
    b1 = time.time()
    doc2vec = Doc2Vec(documents=document,dm=1,alpha=0.025,min_alpha=0.025,workers=8,negative=5)
    train_time = time.time() - b1
    textWidget1.insert(INSERT, "The training time: {0}".format(train_time))
    # for index, word in enumerate(doc2vec.wv.index_to_key):
    #     if index == 10:
    #         break
    #     print(f"word #{index}/{len(doc2vec.wv.index_to_key)} is {word}")
    #most_similar_word
    most_similar_word_sanitizer = doc2vec.dv.most_similar('sanitizer', topn=10)
    textWidget1.insert(END, "The top 10 most similar word example after feature extraction "
                       +"is applied (sanitizer): \n")
    for x in range(len(most_similar_word_sanitizer)):
        FEresult = ''.join([i for i in str(most_similar_word_sanitizer[x]) 
                            if not i.isdigit() and i != "." and i != "," and i != "(" and i!= ")"])
        FEresult += '\n'
        textWidget1.insert(END, FEresult)

# def get_ave_vector(w2v_model):
#     doc2vecmodel = Doc2Vec(documents=w2v_model,dm=1,alpha=0.025,min_alpha=0.025)
#     for epoch in range(200):
#         if epoch % 20 == 0:
#             print ('Now training epoch %s'%epoch)
#         doc2vecmodel.train(w2v_model)
#         doc2vecmodel.alpha -= 0.002  # decrease the learning rate
#         doc2vecmodel.min_alpha = doc2vecmodel.alpha  # fix the learning rate, no decay
#endregion
#region All Buttons
buttonDir = Button(theRealTab1, wraplength = 70,text="Choose a csv or excel datasets",command=openExcel)#import datasets
buttonDir.grid(column=1,row=2)
buttonTokens = Button(theRealTab1, text="Data Cleaning", width=9, command=TriggerCleaning)#Data Preprocessing
buttonFE = Button(theRealTab1, text="Features Extraction", width=16, command=FeaturesExtraction)#Feature Extraction
#endregion

window.mainloop()