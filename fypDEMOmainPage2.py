#from tkinter.font import BOLD
from tkinter import *
from tkinter import filedialog
#from ttkthemes import ThemedTk
from pandastable import Table
from PIL import Image, ImageTk
from gensim.models import Word2Vec
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.linear_model import LogisticRegression as LR
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from sklearn import metrics
import tkinter.ttk as ttk
import numpy as np
import matplotlib
import gensim
import logging
import time
import tkinter
import textwrap
import pandas as pd
import nltk
import re

class fypMenu:
    window = Tk()
    window.withdraw()    
    style = ttk.Style()
    #region TAB
    tabManager = ttk.Notebook(window)

    #Tab1
    tab1 = Frame(tabManager)
    tab1.grid(row=0,column=0,sticky=NSEW)

    canvasTab1 = tkinter.Canvas(tab1, width=1366, height=768)
    canvasTab1.grid(sticky=W)

    vbar = Scrollbar(tab1,orient=VERTICAL,command=canvasTab1.yview)
    vbar.grid(row=0,column=1,sticky=E)
    vbar.grid(sticky=NS)

    canvasTab1.configure(yscrollcommand=vbar.set)

    theRealTab1 = Frame(canvasTab1)
    theRealTab1.bind('<Configure>', lambda e, canvasTab1 = canvasTab1: canvasTab1.configure(scrollregion=canvasTab1.bbox('all')))
    canvasTab1.create_window((0,0),window=theRealTab1,anchor=NW)
    #-----------------------------------------------------------------------------------------------#
    #Tab2
    tab2 = Frame(tabManager)
    tab2.grid(row=0,column=0,sticky=NSEW)
    
    canvasTab2 = tkinter.Canvas(tab2, width=1366, height=768)
    canvasTab2.grid(sticky=W)
    
    vbar2 = Scrollbar(tab2, orient=VERTICAL, command=canvasTab2.yview)
    vbar2.grid(row=0,column=1,sticky=E)
    vbar2.grid(sticky=NS)
    
    canvasTab2.configure(yscrollcommand=vbar2.set)
    
    theRealTab2 = Frame(canvasTab2)
    theRealTab2.bind('<Configure>', lambda e, canvasTab2 = canvasTab2: canvasTab2.configure(scrollregion=canvasTab2.bbox('all')))
    canvasTab2.create_window((0,0),window=theRealTab2,anchor=NW)
    #-----------------------------------------------------------------------------------------------#
    #Tab3
    tab3 = Frame(tabManager)
    tab3.grid(row=0,column=0,sticky=NSEW)
    
    canvasTab3 = tkinter.Canvas(tab3, width=1366, height=768)
    canvasTab3.grid(sticky=W)
    
    vbar3 = Scrollbar(tab3, orient=VERTICAL, command=canvasTab3.yview)
    vbar3.grid(row=0,column=1,sticky=E)
    vbar3.grid(sticky=NS)
    
    canvasTab3.configure(yscrollcommand=vbar3.set)
    
    theRealTab3 = Frame(canvasTab3)
    theRealTab3.bind('<Configure>', lambda e, canvasTab3 = canvasTab3: canvasTab3.configure(scrollregion=canvasTab3.bbox('all')))
    canvasTab3.create_window((0,0),window=theRealTab3,anchor=NW)
    #-----------------------------------------------------------------------------------------------#
    #Tab3
    tab4 = Frame(tabManager)
    tab4.grid(row=0,column=0,sticky=NSEW)
    
    canvasTab4 = tkinter.Canvas(tab4, width=1366, height=768)
    canvasTab4.grid(sticky=W)
    
    vbar4 = Scrollbar(tab4, orient=VERTICAL, command=canvasTab4.yview)
    vbar4.grid(row=0,column=1,sticky=E)
    vbar4.grid(sticky=NS)
    
    canvasTab4.configure(yscrollcommand=vbar3.set)
    
    theRealTab4 = Frame(canvasTab4)
    theRealTab4.bind('<Configure>', lambda e, canvasTab4 = canvasTab4: canvasTab4.configure(scrollregion=canvasTab4.bbox('all')))
    canvasTab4.create_window((0,0),window=theRealTab4,anchor=NW)
    #Parent adding child widgets
    tabManager.add(tab1, text="Data Preprocessing")
    tabManager.add(tab2, text="Features Extraction")
    tabManager.add(tab3, text="Results")
    tabManager.add(tab4, text="Prediction")
    tabManager.grid(sticky="ew")
    #endregion
    
    #region title & labels
    titleLabeltab1 = Label(theRealTab1, text="Datasets:",padx=5)
    titleLabeltab1.grid(column=0,row=1,sticky=S)
    titleLabeltab2 = Label(theRealTab2, text="Almost there. \nWe need features extraction for better accuracy.",padx=5,justify=LEFT)
    titleLabeltab3 = Label(theRealTab3, text="Details of the result: ",padx=5,justify=LEFT)
    titleLabeltab4 = Label(theRealTab4, text="Prediction Testing: ",padx=5,justify=LEFT)

    tb1Label = Label(theRealTab1, text="Select a dataset CSV file to start sentiment analysis: ")
    tb1Label.grid(column=0,row=2,padx=0,pady=0)

    CleanedLabel = Label(theRealTab1, text="Before we move any further: ")

    imageData = ImageTk.PhotoImage(
        Image.open("C://Users//USER//OneDrive//sem6//FYP2//fypVirtual//fypProject1//server.png").resize((50,50)))
    tb1LabelImage = tkinter.Label(theRealTab1,image = imageData)
    tb1LabelImage.image = imageData
    tb1LabelImage.grid(row = 1,column = 1, columnspan = 7, sticky = E)
    theRealTab1.grid_rowconfigure(1)
    
    imageDataTab2 = ImageTk.PhotoImage(
        Image.open("C://Users//USER//OneDrive//sem6//FYP2//fypVirtual//fypProject1//jigsaw.png").resize((50,50)))
    tb2LabelImage = tkinter.Label(theRealTab2,image = imageDataTab2)
    tb2LabelImage.image = imageDataTab2
    theRealTab2.grid_rowconfigure(0)
    
    tb3Label = Label(theRealTab3, text="It's the final step! Press the button below to get the "
                     +"accuracy of each model:\n Please take note: Data will be splitted into 70% for training, "
                     +"10% for validation and 20% for testing.")
    
    imageDataTab3 = ImageTk.PhotoImage(
        Image.open("C://Users//USER//OneDrive//sem6//FYP2//fypVirtual//fypProject1//result.png").resize((50,50)))
    tb3LabelImage = tkinter.Label(theRealTab3,image = imageDataTab3)
    tb3LabelImage.image = imageDataTab3
    
    tb4Label = Label(theRealTab4, text="Let's input some text and test the prediction of each model!\n"+
                     "Press the button that correspond to each model to test the prediction")
    tb4Label2 = Label(theRealTab4, text="")
    tb4Label3 = Label(theRealTab4, text="Answer: ")
    #endregion

    #region All Frames
    #datasets preview frame
    datasetsOriFrame = Frame(theRealTab1)
    datasetsOriFrame.grid(column=0,row=3,columnspan=8,padx=1,pady=5,sticky="ew")
    theRealTab1.grid_columnconfigure(0)

    #datasets results frame
    datasetsResults = Frame(theRealTab1)
    datasetsResults.grid(column=0,columnspan=2,row=6,padx=1,pady=5)
    
    datasetsFEResults = Frame(theRealTab2)
    datasetsFEResults.grid(row=1, column=0, columnspan=3, padx=1, pady=5)
    #endregion
    
    #the important source here:
    nltk.download("stopwords") #stopwords
    stopwords = nltk.corpus.stopwords.words("english") #stopwords in english
    wn = nltk.WordNetLemmatizer() #wn = wordnet
    nltk.download('omw-1.4') #lemmatize
    
    def buttonCreation(self):
        global theRealTab1, theRealTab2, theRealTab3, theRealTab4
        theRealTab1 = self.theRealTab1
        theRealTab2 = self.theRealTab2
        theRealTab3 = self.theRealTab3
        theRealTab4 = self.theRealTab4
        buttonDir = Button(theRealTab1,text="Select",command=self.openExcel)#import datasets
        buttonDir.grid(column=1,row=2)
        global buttonTokens, buttonFE, buttonResult, buttonPrecision, buttonRecall, buttonF1
        buttonTokens = Button(theRealTab1, text="We need to clean the data first: ", width=25, command=self.TriggerCleaning)#Data Preprocessing
        buttonFE = Button(theRealTab2, text="Features Extraction", width=16, command=self.FeaturesExtraction)#Feature Extraction
        buttonResult = Button(theRealTab3, text='Accuracy', width=16)
        buttonPrecision = Button(theRealTab3, text='Precision', width=16)
        buttonRecall = Button(theRealTab3, text='Recall', width=16)
        buttonF1 = Button(theRealTab3, text='F1', width=16)
        global buttonSVM, buttonKNN, buttonLR, tb4inputLabel
        tb4inputLabel = Text(theRealTab4, width=70)
        # inputText = tb4inputLabel.get("1.0", "end-1c")
        # self.inputText = inputText
        buttonSVM = Button(theRealTab4, text='SVM', width=16, command=lambda: self.SVMpredict())
        buttonKNN = Button(theRealTab4, text='KNN', width=16, command=lambda: self.KNNpredict())
        buttonLR = Button(theRealTab4, text='Logistic Regression', width=16, command=lambda: self.LRpredict())
        
    def windowGeometry(self,window):
        window.deiconify()
        window.title("Sentiment Analysis Tool")
        window.geometry("1000x600")

        w = 800
        h = 400

        ws = window.winfo_screenwidth()
        hs = window.winfo_screenheight()

        x = (ws/2) - (w/2)
        y = (hs/2) - (h/2)

        window.geometry("%dx%d+%d+%d" % (w, h, x, y))
        
        
    def openExcel(self):
        global dfOri
        global dfOriReference
        self.style.configure('rowStyle.Treeview',rowheight = 40)
        tvPreview = ttk.Treeview(self.datasetsOriFrame, style = 'rowStyle.Treeview')
        errorLabel = Label(self.window, text='')
        filePath = filedialog.askopenfilename(initialdir='C:/Users/USER/OneDrive/sem6/FYP2',
                                              title = 'Choose a csv file as dataset',
                                              filetypes = [('CSV Files','.csv')])
        if filePath:
            try:
                filePath = r'{}'.format(filePath)
                dfOriReference = pd.read_csv(filePath,squeeze=True) #only reference, cannot be altered
                dfOri = pd.read_csv(filePath,squeeze=True)
            except ValueError:
                errorLabel.config(text='Invalid format for the dataset chosen,' +
                                ' try another file and see.')
                errorLabel.grid(column=0, row=3)
            except FileNotFoundError:
                errorLabel.config(text='File not found. Please choose the correct file'+
                                  'for the tool')
                errorLabel.grid(column=0, row=3)
        #delete the old dataset        
        tvPreview.delete(*tvPreview.get_children())
        tvPreview['column'] = list(dfOri.columns)
        tvPreview['show'] = 'headings'
        
        #setting the heading
        for column in tvPreview['column']:
            tvPreview.heading(column, text=column)
        
        #setting the row in tree view
        theTVlist = dfOri.to_numpy().tolist()
        for row in theTVlist:
            tvPreview.insert('','end',values=row)
        tvPreview.grid(column=0, row=0)
        self.CleanedLabel.grid(column=0,row=4)
        buttonTokens.grid(column=1,row=4,pady=5)
        
    def TriggerCleaning(self):
        global dfOri
        global titleLabeltab2
        global tb2LabelImage
        tb2LabelImage = self.tb2LabelImage
        titleLabeltab2 = self.titleLabeltab2
        dfOri['full_text'] = dfOri['full_text'].apply(self.Cleaning)
        dfOri['full_text'] = dfOri['full_text'].apply(nltk.word_tokenize)
        dfOri['full_text'] = dfOri['full_text'].apply(self.lemmatization)
        dfOri['full_text'] = dfOri['full_text'].apply(lambda x: [item for item in x if item not in self.stopwords])
        tvPreview2 = ttk.Treeview(self.datasetsResults)
        tvPreview2['column'] = list(dfOri.columns)
        tvPreview2['show'] = 'headings'
        
        #setting the heading
        for column in tvPreview2['column']:
            tvPreview2.heading(column, text=column)
        
        theTVlist = dfOri.to_numpy().tolist()
        for row in theTVlist:
            tvPreview2.insert('','end',values=row)
        tvPreview2.grid(column=0, row=0)
        titleLabeltab2.grid(column=0,row=0)
        tb2LabelImage.grid(row = 0, column = 1, columnspan = 7, sticky=E)
        buttonFE.grid(column=0,row=8,pady=5)
        
    def Cleaning(self,text):
        text = text.replace('RT', '') #To remove the RT
        text = text.lower() #To lowercase
        text = re.sub(r'[^\w\s]', '' , text) #To remove punctuation
        text = re.sub(r'[0-9]', '' , text) #To remove number
        text = re.sub(r'_', '', text) #To remove underscore
        return text

    def lemmatization(self,text):
        return [self.wn.lemmatize(word) for word in text]   
        
    def FeaturesExtraction(self):
        global textWidget1
        global titleLabeltab3
        global tb3Label
        global tb3LabelImage
        global vectors, word2vec, finalw2v
        titleLabeltab3 = self.titleLabeltab3
        tb3Label = self.tb3Label
        tb3LabelImage = self.tb3LabelImage
        
        textWidget1 = Text(theRealTab2)
        textWidget1.grid(column=0,row=9,pady=5)
        logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    
        # #doc2vec model building
        # document = [gensim.models.doc2vec.TaggedDocument(words=row['full_text'],tags=[index]) for index, row in dfOri.iterrows()]
        # b1 = time.time()#time started
        # global doc2vec
        # doc2vec = Doc2Vec(documents=document,dm=1,vector_size=300,window=5,min_count=5,workers=8,negative=5,alpha=0.025,epochs=20)
        # train_time = time.time() - b1#time calculation
        # doc2vec.build_vocab(gensim.models.doc2vec.TaggedDocument(words=row['full_text'],tags=[index]) for index, row in dfOri.iterrows())
        # textWidget1.insert(INSERT, "The training time: {0}".format(train_time))
        
        # #Example sentence used for FE
        # token_example1 = "the virus is horrible".split()
        # token_example1_vector = doc2vec.infer_vector(token_example1)
        # most_similar_word_result1 = doc2vec.dv.most_similar([token_example1_vector], topn=10)
        # #Inserting the top 10
        # textWidget1.insert(END, "\nThe top 10 most similar sentence example after feature extraction "
        #                 +"is applied: \nExample of sentence used: the virus is horrible\n")
        # for result in most_similar_word_result1:
        #     textWidget1.insert(END,dfOriReference.loc[result[0],'full_text'])
        #     textWidget1.insert(END,"\n")
        
        #the new word2vec model path
        OUTPUT_FOLDER = 'C:/Users/USER/OneDrive/sem6/FYP2/'
        
        #splitting the data
        dfOri['Sentiment'] =dfOri['Sentiment'].apply(self.targetTransform)#change the sentiment column first
        X_train, X_test, y_train, y_test = train_test_split(dfOri[['full_text']],dfOri['Sentiment'],test_size=0.20,random_state=1,stratify=dfOri['Sentiment'])
        word2vecFolder = OUTPUT_FOLDER + 'word2vec300' + '.model'
        
        starttime = time.time()
        word2vec = Word2Vec(vector_size=300,window=5,min_count=5,workers=8,sg=1)
        word2vec.build_vocab(dfOri['full_text'])
        word2vec.train(dfOri['full_text'], total_examples=word2vec.corpus_count, epochs=20)
        traintime = time.time() - starttime
        word2vec.save(word2vecFolder)
        finalw2v = Word2Vec.load(word2vecFolder)
        
        #Average the vector for each tweet
        textWidget1.insert(INSERT, "The training time: {0}".format(traintime))
        word2vec_filename = OUTPUT_FOLDER + 'train_review_word2vec.csv'
        with open(word2vec_filename, 'w+') as word2vec_file:
            for index, row in X_train.iterrows():
                model_vector = (np.mean([finalw2v.wv[token] for token in row['full_text'] if token in word2vec.wv.key_to_index], axis=0)).tolist()
                if index == 0:
                    header = ",".join(str(ele) for ele in range(300))
                    word2vec_file.write(header)
                    word2vec_file.write("\n")
                # Check if the line exists else it is vector of zeros
                if type(model_vector) is list:  
                    line1 = ",".join( [str(vector_element) for vector_element in model_vector] )
                else:
                    line1 = ",".join([str(0) for i in range(300)])
                word2vec_file.write(line1)
                word2vec_file.write('\n')
        #Top 10 feature display
        textWidget1.insert(END, "\nThe top 10 features from word2vec:\n")
        index = 0
        for feature in word2vec.wv.key_to_index:
            textWidget1.insert(END, "\n{}. {}\n".format(index+1, feature))
            index+=1
            if index == 10:
                break
        word2vec_filename = r'{}'.format(word2vec_filename)
        word2vecPD = pd.read_csv(word2vec_filename)
        
        #Average for the X_test
        test_features_word2vec = []
        for index, row in X_test.iterrows():
            model_vector = (np.mean([finalw2v.wv[token] for token in row['full_text'] if token in word2vec.wv.key_to_index], axis=0)).tolist()
            if type(model_vector) is list:
                test_features_word2vec.append(model_vector)
            else:
                test_features_word2vec.append(np.array([0 for i in range(300)]))
        
        titleLabeltab3.grid(column=0,row=0)
        tb3Label.grid(column=0,row=1)
        tb3LabelImage.grid(row = 1, column = 1, columnspan = 7, sticky = E)
        buttonResult.grid(column=0, row=2, pady=5)
        buttonResult.config(command = lambda: self.hyperparameterTuningAndEvaluating(word2vecPD, test_features_word2vec, y_train, y_test))
        
    # def splittingData(self):
    #     # dfOri['Sentiment'] =dfOri['Sentiment'].apply(self.targetTransform)
    #     # # doc_vectors = []
    #     # # doc_vectors = [doc2vec.infer_vector(row) for row in dfOri['full_text'].to_list()]
    #     # X_train, X_test, y_train, y_test = train_test_split(vectors,dfOri['Sentiment'],test_size=0.20,random_state=1,stratify=dfOri['Sentiment'])
    #     # self.hyperparameterTuningAndEvaluating(X_train, X_test, y_train, y_test)
    #     return "Nothing"
        
    def hyperparameterTuningAndEvaluating(self, X_train, X_test, y_train, y_test):
        global titleLabeltab4, tb4Label, tb4Label2, tb4Label3, SVMmodel, KNNmodel, LRmodel
        tb4Label = self.tb4Label
        titleLabeltab4 = self.titleLabeltab4
        tb4Label2 = self.tb4Label2
        tb4Label3 = self.tb4Label3
        print('\nResult: ')    
        SVMparam_grid = {'C':[0.001,0.01,0.1,1,10,100]}
        SVMgrid_search = GridSearchCV(SVC(kernel='linear'),param_grid=SVMparam_grid,cv=3,n_jobs=-1,verbose=2)
        SVMgrid_search.fit(X_train, y_train)
        print("Best parameters: {}".format(SVMgrid_search.best_params_))
        print("Best score on train set: {:.2f}".format(SVMgrid_search.best_score_))
        SVMmodel = SVC(kernel='linear')
        SVMmodel.set_params(**SVMgrid_search.best_params_)
        SVMmodel.fit(X_train, y_train)
        SVMpredict = SVMmodel.predict(X_test)
        
        SVMaccuracy = metrics.accuracy_score(y_test,SVMpredict)
        SVMprecision = metrics.precision_score(y_test,SVMpredict,average='macro')
        SVMrecall = metrics.recall_score(y_test,SVMpredict,average='macro')
        SVMf1 = metrics.f1_score(y_test,SVMpredict,average='macro')
        print("Accuracy of SVM: {:.2f}".format(SVMaccuracy))
        print("Precision of SVM: {:.2f}".format(SVMprecision))
        print("Recall of SVM: {:.2f}".format(SVMrecall))
        print("F1 of SVM: {:.2f}".format(SVMf1))
        print()
        
        KNNparam_grid = {'n_neighbors':[4,7,10,13,15],
                         'weights':['uniform','distance'],
                         'metric':['minkowski','euclidean','manhattan','cosine']}
        KNNgrid_search = GridSearchCV(KNN(), KNNparam_grid, cv=5, n_jobs=-1)
        KNNgrid_search.fit(X_train, y_train)
        KNNmodel = KNN(n_jobs=-1)
        KNNmodel.set_params(**KNNgrid_search.best_params_)
        KNNmodel.fit(X_train, y_train)
        KNNpredict = KNNmodel.predict(X_test)
        
        print("Best parameters: {}".format(KNNgrid_search.best_params_))
        print("Best score on train set: {:.2f}".format(KNNgrid_search.best_score_))
        KNNaccuracy = metrics.accuracy_score(y_test,KNNpredict)
        KNNprecision = metrics.precision_score(y_test,KNNpredict,average='macro')
        KNNrecall = metrics.recall_score(y_test,KNNpredict,average='macro')
        KNNf1 = metrics.f1_score(y_test,KNNpredict,average='macro')
        print("Accuracy of KNN: {:.2f}".format(KNNaccuracy))
        print("Precision of KNN: {:.2f}".format(KNNprecision))
        print("Recall of KNN: {:.2f}".format(KNNrecall))
        print("F1 of KNN: {:.2f}".format(KNNf1))
        print()
        
        LRparam_grid = {'penalty':['l1','l2','elasticnet','none'],
                        'C':[0.001, 0.01, 0.1, 1, 10, 100],
                        'solver':['newton-cg','lbfgs','liblinear','sag','saga']}
        LRgrid_search = GridSearchCV(LR(), LRparam_grid, cv=5, n_jobs=-1)
        LRgrid_search.fit(X_train, y_train)
        LRmodel = LR(n_jobs=-1)
        LRmodel.set_params(**LRgrid_search.best_params_)
        LRmodel.fit(X_train, y_train)
        LRpredict = LRmodel.predict(X_test)
        
        print("Best parameters: {}".format(LRgrid_search.best_params_))
        print("Best score on train set: {:.2f}".format(LRgrid_search.best_score_))
        LRaccuracy = metrics.accuracy_score(y_test,LRpredict)
        LRprecision = metrics.precision_score(y_test,LRpredict,average='macro')
        LRrecall = metrics.recall_score(y_test,LRpredict,average='macro')
        LRf1 = metrics.f1_score(y_test,LRpredict,average='macro')
        print("Accuracy of LR: {:.2f}".format(LRaccuracy))
        print("Precision of LR: {:.2f}".format(LRprecision))
        print("Recall of LR: {:.2f}".format(LRrecall))
        print("F1 of LR: {:.2f}".format(LRf1))
        print()
        
        matplotlib.use('TkAgg')
        Accuracy = {
            'SVM': SVMaccuracy,
            'KNN': KNNaccuracy,
            'LR': LRaccuracy
        }
        Precision = {
            'SVM':SVMprecision,
            'KNN':KNNprecision,
            'LR':LRprecision
        }
        Recall = {
            'SVM':SVMrecall,
            'KNN':KNNrecall,
            'LR':LRrecall
        }
        F1 = {
            'SVM':SVMf1,
            'KNN':KNNf1,
            'LR':LRf1
        }
        accuracyGraph_X = Accuracy.keys()
        accuracyGraph_Y = Accuracy.values()
        PrecisionGraph_X = Precision.keys()
        PrecisionGraph_Y = Precision.values()
        RecallGraph_X = Recall.keys()
        RecallGraph_Y = Recall.values()
        F1Graph_X = F1.keys()
        F1Graph_Y = F1.values()
        
        accuracyFigure = Figure(figsize=(6,4),dpi=100)
        precisionFigure = Figure(figsize=(6,4),dpi=100)
        recallFigure = Figure(figsize=(6,4),dpi=100)
        f1Figure = Figure(figsize=(6,4),dpi=100)
        
        #fc = figure canvas
        accuracyFC = FigureCanvasTkAgg(accuracyFigure, theRealTab3)
        precisionFC = FigureCanvasTkAgg(precisionFigure, theRealTab3)
        recallFC = FigureCanvasTkAgg(recallFigure, theRealTab3)
        f1FC = FigureCanvasTkAgg(f1Figure, theRealTab3)
        
        accuracyAxes = accuracyFigure.add_subplot()
        precisionAxes = precisionFigure.add_subplot()
        recallAxes = recallFigure.add_subplot()
        f1Axes = f1Figure.add_subplot()
        
        accuracyAxes.bar(accuracyGraph_X, accuracyGraph_Y)
        accuracyAxes.set_title("Text Classifier Models")
        accuracyAxes.set_ylabel("Accuracy")
        precisionAxes.bar(PrecisionGraph_X, PrecisionGraph_Y)
        precisionAxes.set_title("Text Classifier Models")
        precisionAxes.set_ylabel("Precision")
        recallAxes.bar(RecallGraph_X, RecallGraph_Y)
        recallAxes.set_title("Text Classifier Models")
        recallAxes.set_ylabel("Recall")
        f1Axes.bar(F1Graph_X, F1Graph_Y)
        f1Axes.set_title("Text Classifier Models")
        f1Axes.set_ylabel("F1")
        
        buttonResult.destroy()
        accuracyFC.get_tk_widget().grid(column=1, row=2)
        
        buttonPrecision.grid(column=1, row=3)
        buttonPrecision.config(command=lambda: [buttonPrecision.destroy(),precisionFC.get_tk_widget().grid(column=1, row=3)
                                                ,buttonRecall.grid(column=1, row=4)])
        buttonRecall.config(command=lambda: [buttonRecall.destroy(),recallFC.get_tk_widget().grid(column=1, row=4)
                                             ,buttonF1.grid(column=1, row=5)])
        buttonF1.config(command=lambda: [buttonF1.destroy(),f1FC.get_tk_widget().grid(column=1, row=5)])
        titleLabeltab4.grid(row=0, column=0, pady=5, columnspan=3)
        tb4Label.grid(row=1, column=0, pady=5, columnspan=3)
        tb4inputLabel.grid(row=2, column=0, pady=5, columnspan=3)
        buttonSVM.grid(row=3, column=0, pady=5, padx=10)
        buttonKNN.grid(row=3, column=1, pady=5, padx=10)
        buttonLR.grid(row=3, column=2, pady=5, padx=10)
        tb4Label2.grid(row=4, column=1)
        tb4Label3.grid(row=4, column=0)
        
    def SVMpredict(self):
        text = tb4inputLabel.get("1.0", "end-1c")
        text = text.split()
        textVector = (np.mean([finalw2v.wv[token] for token in text if token in finalw2v.wv.key_to_index], axis=0)).tolist()
        if type(textVector) is list:
            textVector = np.array(textVector)
            textVector = textVector.reshape(1,-1)
            result = SVMmodel.predict(textVector)
        else:
            textVector = np.array([0 for i in range(300)])
            textVector = textVector.reshape(1,-1)
            result = SVMmodel.predict(textVector)
        tb4Label2.config(text=result)
        
    def KNNpredict(self):
        text = tb4inputLabel.get("1.0", "end-1c")
        text = text.split()
        textVector = (np.mean([finalw2v.wv[token] for token in text if token in finalw2v.wv.key_to_index], axis=0)).tolist()
        if type(textVector) is list:
            textVector = np.array(textVector)
            textVector = textVector.reshape(1,-1)
            result = KNNmodel.predict(textVector)
        else:
            textVector = np.array([0 for i in range(300)])
            textVector = textVector.reshape(1,-1)
            result = SVMmodel.predict(textVector)
        tb4Label2.config(text=result)
        
    def LRpredict(self):
        text = tb4inputLabel.get("1.0", "end-1c")
        text = text.split()
        textVector = (np.mean([finalw2v.wv[token] for token in text if token in finalw2v.wv.key_to_index], axis=0)).tolist()
        if type(textVector) is list:
            textVector = np.array(textVector)
            textVector = textVector.reshape(1,-1)
            result = LRmodel.predict(textVector)
        else:
            textVector = np.array([0 for i in range(300)])
            textVector = textVector.reshape(1,-1)
            result = SVMmodel.predict(textVector)
        tb4Label2.config(text=result)
        
    def targetTransform(self,target):
        if target>0:
            target='positive'
        elif target==0:
            target='neutral'
        else:
            target='negative'
        return target