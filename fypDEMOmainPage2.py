from tkinter import *
from tkinter import filedialog
from PIL import Image, ImageTk
from gensim.models import Word2Vec
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.linear_model import LogisticRegression as LR
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from sklearn.manifold import TSNE
from sklearn import metrics
from sklearn.metrics import confusion_matrix
import tkinter.ttk as ttk
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import logging
import time
import tkinter
import pandas as pd
import nltk
import re

class fypMenu:
    window = Tk()
    window.withdraw()    
    style = ttk.Style()
    
    #region TAB
    tabManager = ttk.Notebook(window)
    #-----------------------------------------------------------------------------------------------#
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
    hbar2 = Scrollbar(tab2, orient=HORIZONTAL, command=canvasTab2.xview)
    vbar2.grid(row=0,column=1,sticky=E)
    hbar2.grid(row=1,column=0,columnspan=2,sticky=S)
    vbar2.grid(sticky=NS)
    hbar2.grid(sticky=EW)
    
    canvasTab2.configure(yscrollcommand=vbar2.set)
    canvasTab2.configure(xscrollcommand=hbar2.set)
    
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
    #Tab4
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
    #-----------------------------------------------------------------------------------------------#
    #Tab5
    tab5 = Frame(tabManager)
    tab5.grid(row=0,column=0,sticky=NSEW)
    
    canvasTab5 = tkinter.Canvas(tab5, width=1366, height=768)
    canvasTab5.grid(sticky=W)
    
    vbar5 = Scrollbar(tab5, orient=VERTICAL, command=canvasTab5.yview)
    vbar5.grid(row=0,column=1,sticky=E)
    vbar5.grid(sticky=NS)
    
    canvasTab5.configure(yscrollcommand=vbar5.set)
    
    theRealTab5 = Frame(canvasTab5)
    theRealTab5.bind('<Configure>', lambda e, canvasTab5 = canvasTab5: canvasTab5.configure(scrollregion=canvasTab5.bbox('all')))
    canvasTab5.create_window((0,0),window=theRealTab5,anchor=NW)
    #-----------------------------------------------------------------------------------------------#
    #TabManager adding childs
    tabManager.add(tab1, text="Data Preprocessing")
    tabManager.add(tab2, text="Features Extraction")
    tabManager.add(tab3, text="Results")
    tabManager.add(tab4, text="Prediction")
    tabManager.add(tab5, text="Help")
    tabManager.grid(sticky="ew")
    #endregion
    
    #region title & labels
    #Title Label First
    titleLabeltab1 = Label(theRealTab1, text="Datasets:",padx=5, font=8)
    titleLabeltab1.grid(column=0,row=1,sticky=S)
    titleLabeltab2 = Label(theRealTab2, text="Almost there. \nWe need features extraction for better accuracy.",padx=5,justify=LEFT, font=3)
    titleLabeltab3 = Label(theRealTab3, text="Details of the result: ",padx=5,justify=LEFT, font=8)
    titleLabeltab4 = Label(theRealTab4, text="Prediction Testing: ",padx=5,justify=LEFT, font=8)
    titleLabeltab5 = Label(theRealTab5, text="Guide on Using This Application:", font=("Bernard MT Condensed",25))
    titleLabeltab5.grid(column=0,row=0, padx=5, pady=5)
    #-----------------------------------------------------------------------------------------------#
    tb1Label = Label(theRealTab1, text="Select a dataset CSV file to start sentiment analysis: ", font=8)
    tb1Label.grid(column=0,row=2,padx=0,pady=0)

    CleanedLabel = Label(theRealTab1, text="Before we move any further: ", font=8)
    xTrainLabel = Label(theRealTab1, text="Please state the column for machine learning training", font=6)
    xInputSV = StringVar(theRealTab1, value="Column for machine learning training")
    xInput = Entry(theRealTab1,width=30,font=6,textvariable=xInputSV)
    yTrainLabel = Label(theRealTab1, text="Please state the column that contains sentiment score", font=6)
    yInputSV = StringVar(theRealTab1, value="Column for sentiment score")
    yInput = Entry(theRealTab1,width=30, font=6, textvariable=yInputSV)
    tb1emptyLabel = Label(theRealTab1)
    def xInputSVget(xInput):
        return xInput.get()
    xInputSV.trace('w', xInputSVget(xInput))
    def yInputSVget(yInput):
        return yInput.get()
    yInputSV.trace('w', yInputSVget(yInput))
    
    imageData = ImageTk.PhotoImage(
        Image.open("C://Users//USER//OneDrive//sem6//FYP2//fypVirtual//fypProject1//server.png").resize((50,50)))
    tb1LabelImage = tkinter.Label(theRealTab1,image = imageData)
    tb1LabelImage.image = imageData
    tb1LabelImage.grid(row = 1,column = 1, sticky = E)
    #-----------------------------------------------------------------------------------------------#
    imageDataTab2 = ImageTk.PhotoImage(
        Image.open("C://Users//USER//OneDrive//sem6//FYP2//fypVirtual//fypProject1//jigsaw.png").resize((50,50)))
    tb2LabelImage = tkinter.Label(theRealTab2,image = imageDataTab2)
    tb2LabelImage.image = imageDataTab2
    
    tb2Label1 = Label(theRealTab2, text="Your preferred test size: \n*Dataset splitting will start from here,\n"+
                      " the training and validation set will be adjusted accordingly.", font=8)
    tb2Label2= Label(theRealTab2, text="Word2Vec Vectors Visualization", font=8)
    tb2InputSV = StringVar(theRealTab2, value="20% for default(Write 20 for 20%)")
    tb2Input = Entry(theRealTab2, textvariable=tb2InputSV, font=8, width=30)
    def tb2InputSVget(tb2Input):
        return tb2Input.get()
    tb2InputSV.trace('w',tb2InputSVget(tb2Input))
    #-----------------------------------------------------------------------------------------------#
    tb3Label = Label(theRealTab3, text="It's the final step! Press the button below to get the "
                     +"accuracy of each model:\n Please take note: Data will be split according to the test size "
                     +"you have set previously.\n(Validation set will be adjusted accordingly from the training set)", font=8)
    
    imageDataTab3 = ImageTk.PhotoImage(
        Image.open("C://Users//USER//OneDrive//sem6//FYP2//fypVirtual//fypProject1//result.png").resize((50,50)))
    tb3LabelImage = tkinter.Label(theRealTab3,image = imageDataTab3)
    tb3LabelImage.image = imageDataTab3
    #-----------------------------------------------------------------------------------------------#
    tb4Label = Label(theRealTab4, text="Let's input some text and test the prediction of each model!\n"+
                     "Press the button that correspond to each model to test the prediction", font=8)
    tb4Label2 = Label(theRealTab4, text="", font=8)
    tb4Label3 = Label(theRealTab4, text="Answer: ", font=8)
    tb4EmptyLabel = Label(theRealTab4, text="")
    #-----------------------------------------------------------------------------------------------#
    GuideOrHelp = ("To start sentiment analysis, first we need to select a dataset from local storage first. "
                   "Go to the first tab of the application, press the 'Select' button to select\n a dataset. "
                   "Make sure the selected dataset is a file in 'CSV' format and it should consists Tweet "
                   "data for the machine learning model to train and sentiment \nscore as the labels for the "
                   "models to learn.\n\n"
                   "Next, write down the column names that contain the Tweet data for the machine learning "
                   "training and sentiment score to learn.\n"
                   "Please take note you still can change the dataset if the data-preprocessing is not yet "
                   "started.\n\n"
                   "Moving on to the next step, we will start clean the dataset from the noises and extra "
                   "information from it. Press the 'We need to clean the data first:' to start \nthe data"
                   "-preprocessing process.\n\n"
                   "Congrats! You have just finished the data-preprocessing process. However, machine learning model "
                   "only take numerical value to carry out classification \nprocess. To achive this, the "
                   "dataset used has to undergo features extraction provided by Word2Vec algorithm. Go to 'Features Extraction'"
                   " tab and press \nthe 'Feature Extraction' button within it to start this process.\n"
                   "Please take note where preferred test size has to be clarified earlier considering the dataset "
                   "used for testing should no be accessed by the tool before it is \nused for testing the machine "
                   "learning models.\n\n"
                   "After the features extraction process finished, you should see a text box widget displaying the "
                   "training time for features extraction and the top 10 features \nextracted from the dataset. "
                   "These features extracted will also be displayed in a 2D visualize vector space which located "
                   "beside the text box widget.\n\n"
                   "Finally, you may go to the third tab to get the result of testing. This may take a while as the "
                   "tremendous size of dataset will involved in a number of \ncalculation to get the 4 result of evaluation "
                   "metric used, which is the Accuracy score, Precision score, Recall score and F1 score. "
                   "As an additional \ninformation, the score of each model will be compared in 4 bar charts. One pie chart "
                   "that compared the prediction made by each machine learning models \nwill also be shown at the bottom "
                   "of the tab.\n\n"
                   "After training and testing the dataset involved, user can try to make a sentiment prediction on "
                   "a sentence using the machine learning model trained \npreviously. This can be done in the 'Prediction' tab "
                   "and the result of classification will be shown at the bottom of the tab.\n\n\n"
                   "Machine learning models involved: Support Vector Machine (SVM), K-Nearest Neighbors (KNN) and Logistic "
                   "Regression (LR)")
    tb5Label = Label(theRealTab5, text=GuideOrHelp, font=5, justify='left')
    tb5Label.grid(row=1, column=0, padx=5, pady=30)
    #endregion

    #region All Frames
    #datasets preview frame
    datasetsOriFrame = Frame(theRealTab1)
    datasetsOriFrame.grid(column=0,row=3,columnspan=2,rowspan=4,padx=1,pady=5,sticky="ew")
    theRealTab1.grid_columnconfigure(0)

    #datasets results frame
    datasetsResults = Frame(theRealTab1)
    datasetsResults.grid(column=0,columnspan=2,row=8,padx=1,pady=5)
    
    datasetsFEResults = Frame(theRealTab2)
    datasetsFEResults.grid(row=1, column=0, columnspan=3, padx=1, pady=5)
    #endregion
    
    #region Cleaning library:
    nltk.download("stopwords") #stopwords
    stopwords = nltk.corpus.stopwords.words("english") #stopwords in english
    wn = nltk.WordNetLemmatizer() #wn = wordnet
    nltk.download('omw-1.4') #lemmatize
    #endregion
    
    def buttonCreation(self):
        global theRealTab1, theRealTab2, theRealTab3, theRealTab4, buttonTokens
        theRealTab1 = self.theRealTab1
        theRealTab2 = self.theRealTab2
        theRealTab3 = self.theRealTab3
        theRealTab4 = self.theRealTab4
        imgDir = ImageTk.PhotoImage(Image.open(r"C:\Users\USER\OneDrive\sem6\FYP2\fypVirtual\fypProject1\upload.png").resize((30,30)),
                                 master=theRealTab1)
        self.imgDir = imgDir
        buttonDir = Button(theRealTab1,text="Select",compound=LEFT,image=imgDir,command=self.openExcel,font=8)#import datasets
        buttonDir.grid(column=1,row=2)
        imgClean = ImageTk.PhotoImage(Image.open(r"C:\Users\USER\OneDrive\sem6\FYP2\fypVirtual\fypProject1\broom.png").resize((30,30)),
                                 master=theRealTab1)
        self.imgClean = imgClean
        buttonTokens = Button(theRealTab1, text="We need to clean the data first: ", compound=LEFT,
                              image=imgClean, command=self.TriggerCleaning, font=8)#Data Preprocessing
        global buttonFE, buttonResult, buttonPrecision, buttonRecall, buttonF1, buttonPie
        buttonFE = Button(theRealTab2, text="Features Extraction", width=16, command=self.FeaturesExtraction, font=8)#Feature Extraction
        buttonResult = Button(theRealTab3, text='Accuracy', font=8)
        buttonPrecision = Button(theRealTab3, text='Precision', font=8)
        buttonRecall = Button(theRealTab3, text='Recall', font=8)
        buttonF1 = Button(theRealTab3, text='F1', font=8)
        buttonPie = Button(theRealTab3, text='Pie Chart Comparison', font=8)
        global buttonSVM, buttonKNN, buttonLR, tb4InputLabel
        tb4InputLabel = Text(theRealTab4, width=70, font=8)
        buttonSVM = Button(theRealTab4, text='SVM', width=16, command=self.SVMpredict, font=8)
        buttonKNN = Button(theRealTab4, text='KNN', width=16, command=self.KNNpredict, font=8)
        buttonLR = Button(theRealTab4, text='Logistic Regression', width=16, command=self.LRpredict, font=8)
        
    
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
        global dfOri #To manipulate
        global dfOriReference #To read only
        global xTrainLabel, xInput, yTrainLabel, yInput, tb1emptyLabel
        xTrainLabel = self.xTrainLabel
        yTrainLabel = self.yTrainLabel
        xInput = self.xInput
        yInput = self.yInput
        tb1emptyLabel = self.tb1emptyLabel
        #Setup Treeview for Read the Dataset
        self.style.configure('rowStyle.Treeview',rowheight = 40)
        tvPreview = ttk.Treeview(self.datasetsOriFrame, style = 'rowStyle.Treeview')
        errorLabel = Label(self.window, text='')
        filePath = filedialog.askopenfilename(initialdir='C:/Users/USER/OneDrive/sem6/FYP2',
                                              title = 'Choose a csv file as dataset',
                                              filetypes = [('CSV Files','.csv')])
        #File Validation / File Changing
        if filePath:
            try:
                filePath = r'{}'.format(filePath)
                dfOriReference = pd.read_csv(filePath,squeeze=True) #only reference, cannot be altered
                dfOri = pd.read_csv(filePath,squeeze=True)
            except ValueError:
                errorLabel.config(text='Invalid format for the dataset chosen,' +
                                ' try another file and see.', font=8)
                errorLabel.grid(column=0, row=3)
            except FileNotFoundError:
                errorLabel.config(text='File not found. Please choose the correct file'+
                                  'for the tool', font=8)
                errorLabel.grid(column=0, row=3)
        #delete the old dataset        
        tvPreview.delete(*tvPreview.get_children())
        tvPreview['column'] = list(dfOri.columns)
        tvPreview['show'] = 'headings'
        
        #setting the heading
        for column in tvPreview['column']:
            tvPreview.heading(column, text=column)
        
        #setting the row in tree view
        theTVList = dfOri.to_numpy().tolist()
        for row in theTVList:
            tvPreview.insert('','end',values=row)
        tvPreview.grid(column=0, row=0)
        self.CleanedLabel.grid(column=0,row=7)
        buttonTokens.grid(column=1,row=7,pady=5)
        tb1emptyLabel.grid(column=2,row=3,rowspan=4,padx=20)
        xTrainLabel.grid(column=3,row=3)
        xInput.grid(column=3,row=4)
        yTrainLabel.grid(column=3,row=5)
        yInput.grid(column=3,row=6)
        
    def TriggerCleaning(self):
        global titleLabeltab2, full_text, Sentiment, xInputSV, yInputSV, tb2Label1, tb2Input
        global tb2LabelImage
        tb2LabelImage = self.tb2LabelImage
        titleLabeltab2 = self.titleLabeltab2
        xInputSV = self.xInputSV
        yInputSV = self.yInputSV
        full_text = xInputSV.get()
        Sentiment = yInputSV.get()
        tb2Label1 = self.tb2Label1
        tb2Input = self.tb2Input
        dfOri[full_text] = dfOri[full_text].apply(self.Cleaning)
        dfOri[full_text] = dfOri[full_text].apply(nltk.word_tokenize)
        dfOri[full_text] = dfOri[full_text].apply(self.lemmatization)
        dfOri[full_text] = dfOri[full_text].apply(lambda x: [item for item in x if item not in self.stopwords])
        #Setup Treeview for cleaned data
        tvPreview2 = ttk.Treeview(self.datasetsResults)
        tvPreview2['column'] = list(dfOri.columns)
        tvPreview2['show'] = 'headings'
        
        #Setting the heading
        for column in tvPreview2['column']:
            tvPreview2.heading(column, text=column)
        
        #Reading the row
        theTVlist = dfOri.to_numpy().tolist()
        for row in theTVlist:
            tvPreview2.insert('','end',values=row)
        tvPreview2.grid(column=0, row=0)
        titleLabeltab2.grid(column=0,row=0)
        tb2LabelImage.grid(row = 0, column = 2, sticky=E)
        buttonFE.grid(column=0,row=1,pady=5)
        tb2Label1.grid(column=1,row=0,padx=20)
        tb2Input.grid(column=1,row=1,padx=20,pady=5)
        
    def Cleaning(self,text):
        text = text.replace('RT', '') #To remove the RT
        text = text.lower() #To lowercase
        text = re.sub(r'[^\w\s]', '' , text) #To remove punctuation
        text = re.sub(r'[0-9]', '' , text) #To remove number
        text = re.sub(r'_', '', text) #To remove underscore
        return text

    def lemmatization(self,text):
        return [self.wn.lemmatize(word) for word in text]   
        
    def tsne_plot(self,model):
        words = []
        vectors = []

        for word in model.wv.key_to_index:
            vectors.append(model.wv[word])
            words.append(word)
        
        tsne_model = TSNE(perplexity=40, n_components=2, init='pca', n_iter=2500, random_state=23)
        new_values = tsne_model.fit_transform(vectors)

        x = []
        y = []
        for value in new_values:
            x.append(value[0])
            y.append(value[1])
            
        w2vFigure = plt.figure(figsize=(7,7),dpi=100)
        w2v = w2vFigure.add_subplot()
        for i in range(len(x)):
            w2v.scatter(x[i],y[i])
            w2v.annotate(words[i],
                        xy=(x[i], y[i]),
                        xytext=(5, 2),
                        textcoords='offset points',
                        ha='right',
                        va='bottom')
        w2vFC = FigureCanvasTkAgg(w2vFigure, theRealTab2)
        return w2vFC
    def FeaturesExtraction(self):
        global textWidget1
        global titleLabeltab3, tb3Label, tb2LabelImage, tb2InputSV, tb2Label2
        global word2vec, finalw2v
        tb2InputSV = self.tb2InputSV
        tb2Label2 = self.tb2Label2
        testSize = float(tb2InputSV.get())/100
        titleLabeltab3 = self.titleLabeltab3
        tb3Label = self.tb3Label
        tb3LabelImage = self.tb3LabelImage
        
        textWidget1 = Text(theRealTab2, width=60, font=8)
        textWidget1.grid(column=0,row=2,rowspan=2,pady=5)
        logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
        
        #The new word2vec model path
        outputFolder = 'C:/Users/USER/OneDrive/sem6/FYP2/'
        word2vecFolder = outputFolder + 'word2vec300' + '.model'
        
        #splitting the data
        dfOri[Sentiment] =dfOri[Sentiment].apply(self.targetTransform)#change the sentiment column first
        X_train, X_test, y_train, y_test = train_test_split(dfOri[[full_text]],dfOri[Sentiment],test_size=testSize,random_state=1,stratify=dfOri[Sentiment])#8:2
        
        #Word2Vec Training
        starttime = time.time()
        word2vec = Word2Vec(vector_size=300,window=5,min_count=5,workers=8,sg=1)
        word2vec.build_vocab(dfOri[full_text])
        word2vec.train(dfOri[full_text], total_examples=word2vec.corpus_count, epochs=20)#Extract feature (vector) here
        traintime = time.time() - starttime
        word2vec.save(word2vecFolder)
        finalw2v = Word2Vec.load(word2vecFolder)#If using the same file, just load this
        
        #Average the vector for each tweet
        textWidget1.insert(INSERT, "The training time: {0}s".format(traintime))
        word2vecCSV = outputFolder + 'train_review_word2vec.csv'
        with open(word2vecCSV, 'w+') as word2vecFile:
            for index, row in X_train.iterrows():
                modelVector = (np.mean([finalw2v.wv[token] for token in row[full_text] if token in finalw2v.wv.key_to_index], axis=0)).tolist()
                if index == 0:
                    header = ",".join(str(ele) for ele in range(300))
                    word2vecFile.write(header)
                    word2vecFile.write("\n")
                # Check if the line exists, else it is vector of zeros
                if type(modelVector) is list:  
                    line1 = ",".join( [str(vectorElement) for vectorElement in modelVector] )
                else:
                    line1 = ",".join([str(0) for i in range(300)])
                word2vecFile.write(line1)
                word2vecFile.write('\n')
        #Top 10 feature display
        textWidget1.insert(END, "\nThe top 10 features from word2vec:\n")
        index = 0
        for feature in finalw2v.wv.key_to_index:
            textWidget1.insert(END, "\n{}. {}\n".format(index+1, feature))
            index+=1
            if index == 10:
                break
        word2vecCSV = r'{}'.format(word2vecCSV)
        word2vecDF = pd.read_csv(word2vecCSV)
        
        #Average for the X_test
        testFeatures = []
        for index, row in X_test.iterrows():
            modelVector = (np.mean([finalw2v.wv[token] for token in row[full_text] if token in finalw2v.wv.key_to_index], axis=0)).tolist()
            if type(modelVector) is list:
                testFeatures.append(modelVector)
            else:
                testFeatures.append(np.array([0 for i in range(300)]))
        
        #TSNE
        w2vFC = self.tsne_plot(finalw2v)
        
        titleLabeltab3.grid(column=0,row=0)
        tb3Label.grid(column=0,row=1,padx=250)
        tb3LabelImage.grid(row = 1, column = 1, sticky = E)
        buttonResult.grid(column=0, row=2, pady=5)
        tb2Label2.grid(column=1, row=2, padx=20, pady=5)
        w2vFC.get_tk_widget().grid(column=1, row=3, padx=20, pady=5)
        buttonResult.config(command = lambda: self.hyperparameterTuningAndEvaluating(word2vecDF, testFeatures, y_train, y_test))
        
    def hyperparameterTuningAndEvaluating(self, X_train, X_test, y_train, y_test):
        global titleLabeltab4, tb4Label, tb4Label2, tb4Label3, tb4EmptyLabel, SVMmodel, KNNmodel, LRmodel
        tb4Label = self.tb4Label
        titleLabeltab4 = self.titleLabeltab4
        tb4Label2 = self.tb4Label2
        tb4Label3 = self.tb4Label3
        tb4EmptyLabel = self.tb4EmptyLabel
        SVMpositiveCount = SVMnegativeCount = SVMneutralCount = KNNpositiveCount = KNNnegativeCount = KNNneutralCount = LRpositiveCount = LRnegativeCount = LRneutralCount = 0
        #-----------------------------------------------------------------------------------------------#
        #Hyperparameter Tuning for SVM
        print('\nResult: ')    
        SVMparam_grid = {'C':[0.001,0.01,0.1,1,10,100]}
        SVMgrid_search = GridSearchCV(SVC(kernel='linear'),param_grid=SVMparam_grid,n_jobs=-1,verbose=2)
        SVMgrid_search.fit(X_train, y_train)
        print("Best parameters: {}".format(SVMgrid_search.best_params_))
        print("Best score on train set: {:.2f}".format(SVMgrid_search.best_score_))
        #Prediction for SVM
        SVMmodel = SVC(kernel='linear')
        SVMmodel.set_params(**SVMgrid_search.best_params_)
        SVMmodel.fit(X_train, y_train)
        SVMpredict = SVMmodel.predict(X_test)
        #Confusion Matrix for SVM
        SVMcm = confusion_matrix(y_test, SVMpredict, labels=['positive', 'negative', 'neutral'])
        for index, item1D in enumerate(SVMcm):
            for index2, item2D in enumerate(item1D):
                if index2 == 0:
                    SVMpositiveCount += item2D
                elif index2 == 1:
                    SVMnegativeCount += item2D
                else:
                    SVMneutralCount += item2D
        #Result of SVM
        SVMaccuracy = metrics.accuracy_score(y_test,SVMpredict)
        SVMprecision = metrics.precision_score(y_test,SVMpredict,average='macro')
        SVMrecall = metrics.recall_score(y_test,SVMpredict,average='macro')
        SVMf1 = metrics.f1_score(y_test,SVMpredict,average='macro')
        print("Accuracy of SVM: {:.2f}".format(SVMaccuracy))
        print("Precision of SVM: {:.2f}".format(SVMprecision))
        print("Recall of SVM: {:.2f}".format(SVMrecall))
        print("F1 of SVM: {:.2f}".format(SVMf1))
        print()
        #-----------------------------------------------------------------------------------------------#
        #Hyperparameter Tuning for KNN
        KNNparam_grid = {'n_neighbors':[4,7,10,13,15],
                         'weights':['uniform','distance'],
                         'metric':['minkowski','euclidean','manhattan','cosine']}
        KNNgrid_search = GridSearchCV(KNN(), KNNparam_grid, cv=5, n_jobs=-1)
        KNNgrid_search.fit(X_train, y_train)
        print("Best parameters: {}".format(KNNgrid_search.best_params_))
        print("Best score on train set: {:.2f}".format(KNNgrid_search.best_score_))
        #Prediction for KNN
        KNNmodel = KNN(n_jobs=-1)
        KNNmodel.set_params(**KNNgrid_search.best_params_)
        KNNmodel.fit(X_train, y_train)
        KNNpredict = KNNmodel.predict(X_test)
        #Confusion Matrix for KNN
        KNNcm = confusion_matrix(y_test, KNNpredict, labels=['positive', 'negative', 'neutral'])
        for index, item1D in enumerate(KNNcm):
            for index2, item2D in enumerate(item1D):
                if index2 == 0:
                    KNNpositiveCount += item2D
                elif index2 == 1:
                    KNNnegativeCount += item2D
                else:
                    KNNneutralCount += item2D
        #Result of KNN
        KNNaccuracy = metrics.accuracy_score(y_test,KNNpredict)
        KNNprecision = metrics.precision_score(y_test,KNNpredict,average='macro')
        KNNrecall = metrics.recall_score(y_test,KNNpredict,average='macro')
        KNNf1 = metrics.f1_score(y_test,KNNpredict,average='macro')
        print("Accuracy of KNN: {:.2f}".format(KNNaccuracy))
        print("Precision of KNN: {:.2f}".format(KNNprecision))
        print("Recall of KNN: {:.2f}".format(KNNrecall))
        print("F1 of KNN: {:.2f}".format(KNNf1))
        print()
        #-----------------------------------------------------------------------------------------------#
        #Hyperparameter Tuning for LR
        LRparam_grid = {'penalty':['l1','l2','elasticnet','none'],
                        'C':[0.001, 0.01, 0.1, 1, 10, 100],
                        'solver':['newton-cg','lbfgs','liblinear','sag','saga']}
        LRgrid_search = GridSearchCV(LR(), LRparam_grid, cv=5, n_jobs=-1)
        LRgrid_search.fit(X_train, y_train)
        print("Best parameters: {}".format(LRgrid_search.best_params_))
        print("Best score on train set: {:.2f}".format(LRgrid_search.best_score_))
        #Prediction for LR
        LRmodel = LR(n_jobs=-1)
        LRmodel.set_params(**LRgrid_search.best_params_)
        LRmodel.fit(X_train, y_train)
        LRpredict = LRmodel.predict(X_test)
        #Confusion Matrix for LR
        LRcm = confusion_matrix(y_test, LRpredict, labels=['positive', 'negative', 'neutral'])
        for index, item1D in enumerate(LRcm):
            for index2, item2D in enumerate(item1D):
                if index2 == 0:
                    LRpositiveCount += item2D
                elif index2 == 1:
                    LRnegativeCount += item2D
                else:
                    LRneutralCount += item2D
        #Result of LR
        LRaccuracy = metrics.accuracy_score(y_test,LRpredict)
        LRprecision = metrics.precision_score(y_test,LRpredict,average='macro')
        LRrecall = metrics.recall_score(y_test,LRpredict,average='macro')
        LRf1 = metrics.f1_score(y_test,LRpredict,average='macro')
        print("Accuracy of LR: {:.2f}".format(LRaccuracy))
        print("Precision of LR: {:.2f}".format(LRprecision))
        print("Recall of LR: {:.2f}".format(LRrecall))
        print("F1 of LR: {:.2f}".format(LRf1))
        print()
        #-----------------------------------------------------------------------------------------------#
        #Graph plotting here
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
        pieFigure = Figure(figsize=(7,7),dpi=100)
        
        #fc = figure canvas
        accuracyFC = FigureCanvasTkAgg(accuracyFigure, theRealTab3)
        precisionFC = FigureCanvasTkAgg(precisionFigure, theRealTab3)
        recallFC = FigureCanvasTkAgg(recallFigure, theRealTab3)
        f1FC = FigureCanvasTkAgg(f1Figure, theRealTab3)
        pieFC = FigureCanvasTkAgg(pieFigure, theRealTab3)
        
        accuracyAxes = accuracyFigure.add_subplot()
        precisionAxes = precisionFigure.add_subplot()
        recallAxes = recallFigure.add_subplot()
        f1Axes = f1Figure.add_subplot()
        
        piechart = pieFigure.add_subplot(2,3,1)
        piechart.pie([SVMpositiveCount,SVMnegativeCount,SVMneutralCount],
                     labels=['positive','negative','neutral'], startangle=90,
                     autopct='%.2f%%')
        piechart.set_title('SVM Prediction', bbox={'facecolor':'0.8', 'pad':5})
        piechart = pieFigure.add_subplot(2,3,2)
        piechart.pie([KNNpositiveCount,KNNnegativeCount,KNNneutralCount],
                     labels=['positive','negative','neutral'], startangle=90,
                     autopct='%.2f%%')
        piechart.set_title('KNN Prediction', bbox={'facecolor':'0.8', 'pad':5})
        piechart = pieFigure.add_subplot(2,3,3)
        piechart.pie([LRpositiveCount,LRnegativeCount,LRneutralCount],
                     labels=['positive','negative','neutral'], startangle=90,
                     autopct='%.2f%%')
        piechart.set_title('LR Prediction', bbox={'facecolor':'0.8', 'pad':5})
        piechart = pieFigure.add_subplot(2,3,5)
        positiveCount = negativeCount = neutralCount = 0
        for rows in y_test:
            if rows == 'positive':
                positiveCount += 1
            elif rows == 'negative':
                negativeCount += 1
            else:
                neutralCount += 1
        piechart.pie([positiveCount,negativeCount,neutralCount],
                     labels=['positive','negative','neutral'], startangle=90,
                     autopct='%.2f%%')
        piechart.set_title('Original Sentiment Prediction', bbox={'facecolor':'0.8', 'pad':5})
        
        accuracyAxes.bar(accuracyGraph_X, accuracyGraph_Y)
        accuracyAxes.set_title("Accuracy Score")
        accuracyAxes.set_ylabel("Accuracy")
        accuracyAxes.set_xlabel("Text Classifier Models")
        for index, y in enumerate(accuracyGraph_Y):
            accuracyAxes.text(index-0.15, y+0.005, str(round(y,3)), color='blue', fontstyle='oblique',
                              fontweight='bold')
        
        precisionAxes.bar(PrecisionGraph_X, PrecisionGraph_Y)
        precisionAxes.set_title("Precision Score")
        precisionAxes.set_ylabel("Precision")
        precisionAxes.set_xlabel("Text Classifier Models")
        for index, y in enumerate(PrecisionGraph_Y):
            precisionAxes.text(index-0.15, y+0.005, str(round(y,3)), color='blue', fontstyle='oblique',
                              fontweight='bold')
        
        recallAxes.bar(RecallGraph_X, RecallGraph_Y)
        recallAxes.set_title("Recall Score")
        recallAxes.set_ylabel("Recall")
        recallAxes.set_xlabel("Text Classifier Models")
        for index, y in enumerate(RecallGraph_Y):
            recallAxes.text(index-0.15, y+0.005, str(round(y,3)), color='blue', fontstyle='oblique',
                            fontweight='bold')
        
        f1Axes.bar(F1Graph_X, F1Graph_Y)
        f1Axes.set_title("F1 Score")
        f1Axes.set_ylabel("F1")
        f1Axes.set_xlabel("Text Classifier Models")
        for index, y in enumerate(F1Graph_Y):
            f1Axes.text(index-0.15, y+0.005, str(round(y,3)), color='blue', fontstyle='oblique',
                        fontweight='bold')
        
        buttonResult.destroy()
        accuracyFC.get_tk_widget().grid(column=0, row=2)
        
        buttonPrecision.grid(column=0, row=3)
        buttonPrecision.config(command=lambda: [buttonPrecision.destroy(),precisionFC.get_tk_widget().grid(column=0, row=3)
                                                ,buttonRecall.grid(column=0, row=4)])
        buttonRecall.config(command=lambda: [buttonRecall.destroy(),recallFC.get_tk_widget().grid(column=0, row=4)
                                             ,buttonF1.grid(column=0, row=5)])
        buttonF1.config(command=lambda: [buttonF1.destroy(),f1FC.get_tk_widget().grid(column=0, row=5)
                                         ,buttonPie.grid(column=0, row=6)])
        buttonPie.config(command=lambda: [buttonPie.destroy(),pieFC.get_tk_widget().grid(column=0, row=6)])
        titleLabeltab4.grid(row=0, column=1, pady=5, columnspan=3)
        tb4Label.grid(row=1, column=1, pady=5, columnspan=3)
        tb4InputLabel.grid(row=2, column=1, pady=5, columnspan=3)
        buttonSVM.grid(row=3, column=1, pady=5, padx=10)
        buttonKNN.grid(row=3, column=2, pady=5, padx=10)
        buttonLR.grid(row=3, column=3, pady=5, padx=10)
        tb4Label2.grid(row=4, column=2)
        tb4Label3.grid(row=4, column=1)
        tb4EmptyLabel.grid(row=0, column=0, rowspan=4, padx=200)
        
    def SVMpredict(self):
        text = tb4InputLabel.get("1.0", "end-1c")
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
        text = tb4InputLabel.get("1.0", "end-1c")
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
        text = tb4InputLabel.get("1.0", "end-1c")
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