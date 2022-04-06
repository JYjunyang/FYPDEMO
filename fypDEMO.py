from tkinter import *
from tkinter import ttk
from tkinter import filedialog
from pandastable import Table
from textblob import TextBlob
from PIL import Image, ImageTk
from gensim.models import Word2Vec
from collections import defaultdict
import time
import numpy as np
import tkinter
import pandas as pd
import nltk

#some basic things/imports
window = Tk() #The main window
nltk.download("stopwords")
stopwords = nltk.corpus.stopwords.words("english")
wn = nltk.WordNetLemmatizer() #wn = wordnet
nltk.download('omw-1.4')

#word2vec
MAX_EXP = 6#maximum exponential
EXP_TABLE_SIZE = 1000#table for the exponential size
EXP_TABLE = [] #list for the exponential
for i in range(0, EXP_TABLE_SIZE+1):
    EXP_TABLE.append(np.exp(((i-EXP_TABLE_SIZE/2)/(EXP_TABLE_SIZE/2)) * MAX_EXP))
    
class Word2Vec(object):#word2vec class
    
    def __init__(self, sentences, wv_size=100, window=5, min_count=5, sample=1e-4, negative=15, alpha=0.36, min_alpha=0.0001, sg=1):
        np.random.seed(1)
        self.sentences = sentences
        self.wv_size = wv_size
        self.window = window
        self.min_count = min_count
        self.sample = sample
        self.negative = negative
        self.alpha = alpha
        self.min_alpha = min_alpha
        self.sg = sg
        self.vocab = self.vocab() #take the cleaned data(LIST)
        self.input_embedding = np.random.uniform(low=-0.5/(wv_size**(3/4)), high=0.5/(wv_size**(3/4)), size=(len(self.vocab), wv_size))
        self.output_weights = np.zeros([len(self.vocab), wv_size])
        self.word_oh_v = np.zeros(len(self.vocab))
        self.G0 = np.zeros_like(self.input_embedding)
        self.G1 = np.zeros_like(self.output_weights)
        self.fudge_factor = 1e-6
    
    def vocab(self):
        # sentences: list of sentence token lists
        # [[['here', 'are', 'two', 'reasons', 'companies', 'fail', 'they', 'only', 'do', 'more', 'of', 'the', 'same', 'or', 'they', 'only', 'do', 'what', 's', 'new']], [[]], ...]
        sentences = self.sentences
        vocab = defaultdict(dict)
        vocab_words = ['int']
        vocab['int']['word_count'] = 0 
        vocab_size = 0
        for sent_tokens in sentences:
            for word1 in sent_tokens:
                vocab_size += len(word1)
                for word in word1:
                    if not word.isdigit() and word not in vocab:
                        vocab[word]['word_count'] = 1
                        vocab_words.append(word)
                    else:
                        if word.isdigit():
                            vocab['int']['word_count'] += 1 
                        else:
                            vocab[word]['word_count'] += 1
        low_freq_words = []
        for word in vocab:
            if vocab[word]['word_count'] < self.min_count:
                low_freq_words.append(word)
        for word in low_freq_words:
            vocab_size -= vocab[word]['word_count']
            del vocab[word]
            vocab_words.remove(word)
        sorted_vocab = []
        for word in vocab:
            sorted_vocab.append((word, vocab[word]['word_count']))
        sorted_vocab.sort(key=lambda tup: tup[1], reverse=True)
        for idx, word in enumerate(sorted_vocab):
            vocab[word[0]]['word_freq'] = vocab[word[0]]['word_count'] / vocab_size
            vocab[word[0]]['word_index'] = idx
        return vocab
       
    # Forward Propagation
    def train_batch_sg(self, ):
        sentences = self.sentences
        vocab = self.vocab
        train_step = 0
        neg_word_list = self.neg_sampling(vocab)
        for sent_tokens in sentences:
            for word1 in sent_tokens:
                clean_sent = []
                for word in word1:
                    if word.isdigit():
                        word = 'int'
                    if word not in vocab:
                        continue
                    # Subsampling of High-Freq Word
                    keep_prob = min((np.sqrt(vocab[word]['word_freq'] / self.sample) + 1) * (self.sample / vocab[word]['word_freq']), 1)
                    keep_list = [1] * int(keep_prob * 1000) + [0] * (1000 - int(keep_prob * 1000))
                    if keep_list[np.random.randint(1000)]:
                        clean_sent.append(word)
                for pos, center_word in enumerate(clean_sent):
                    b = np.random.randint(0, self.window)
                    for pos_c, context in enumerate(clean_sent[max(0, pos - (self.window - b)) : pos + (self.window - b) + 1], max(0, pos - (self.window - b))):
                        if pos_c != pos:
                            train_step += 1
                            # Adaptive 
                            context_idx = self.vocab[context]['word_index']
                            if np.min(self.G0) != 0 and self.alpha/np.min(self.G0) < self.min_alpha:
                                print(train_step)
                            self.train_pair_sg(center_word, context, neg_word_list=neg_word_list, neg=self.negative)

        # Save the final embedding vector matrix
        fname1 = './parameter_data/word_embedding_vector_matrix_test_%sf1.txt' % str(train_step)
        np.savetxt(fname1, self.input_embedding)
        fname2 = './parameter_data/word_embedding_vector_matrix_test_%sf2.txt' % str(train_step)
        np.savetxt(fname2, (self.input_embedding + self.output_weights) / 2)
                
    # Back Propagation
    def train_pair_sg(self, center_w, context_w, neg_word_list, neg=0):
        if neg > 0:
            context_idx = self.vocab[context_w]['word_index']
            center_idx = self.vocab[center_w]['word_index']
            neg_sample = [(center_w, 1)]
            wv_h = self.input_embedding[context_idx]
            # wv_j = self.input_embedding[self.vocab[context_w]['word_index']]
            for i in range(0, neg):
                neg_word = neg_word_list[np.random.randint(0, len(neg_word_list))] 
                if (neg_word, 0) not in neg_sample and neg_word != center_w:
                    neg_sample.append((neg_word, 0))
            # log(P(Wo|Wi)) = log(sigmoid(np.dot(Vt, Vi))) + np.sum(sigmoid(-np.dot(Vn, Vi))  for neg_w in neg_sample[1:]) / (len(neg_sample) - 1)
            # Adagrad
            dh = np.zeros(self.wv_size)
            for neg_w in neg_sample:
                target, label = neg_w[0], neg_w[1]
                neg_w_idx = self.vocab[target]['word_index']
                wv_j = self.output_weights[neg_w_idx]
                dwjh = self.sigmoid(np.dot(wv_h, wv_j)) - label
                dwj = dwjh * wv_h
                self.G1[neg_w_idx] += np.power(dwj, 2)
                dwj /= np.sqrt(self.G1[neg_w_idx]) + self.fudge_factor
                assert dwj.shape == wv_j.shape
                dh += dwjh * wv_j
                # Update the output weight matrix
                self.output_weights[neg_w_idx] -= self.alpha * dwj
            # Update the input embedding matrix
            self.G0[context_idx] += np.power(dh, 2)
            dh /= np.sqrt(self.G0[context_idx]) + self.fudge_factor
            assert dh.shape == wv_h.shape
            self.input_embedding[context_idx] -= self.alpha * dh
    
    # Negative Sampling
    def neg_sampling(self, vocab):
        NEG_SIZE = 1e6
        neg_word_list = []
        sorted_vocab = []
        freq_sum = np.sum(vocab[word]['word_freq']**0.75 for word in vocab)
        for word in vocab:
            sorted_vocab.append((word, vocab[word]['word_freq']))
        sorted_vocab.sort(key=lambda tup: tup[1], reverse=True)
        for word in sorted_vocab:
            neg_word_list.extend([word[0]] * int((word[1]**0.75 / freq_sum) * NEG_SIZE))
        return neg_word_list
    
    def sigmoid(self, x):
        if x > MAX_EXP:
            x = MAX_EXP
        if x < -MAX_EXP:
            x = -MAX_EXP
        exp_x = EXP_TABLE[int((-x / MAX_EXP) * 500 + 500)]
        return 1 / (1 + exp_x)
                        
    def cosine_distance(self, vec1, vec2):
        assert vec1.shape == vec2.shape
        return np.dot(vec1, vec2) / (np.sqrt(np.dot(vec1, vec1)) * np.sqrt(np.dot(vec2, vec2)))
    
    def most_similar(self, word, topn):
        most_similar_word = []
        word_idx = self.vocab[word]['word_index']
        for w in self.vocab:
            w_idx = self.vocab[w]['word_index']
            if w_idx != word_idx:
                cos_dis = self.cosine_distance(self.input_embedding[word_idx], self.input_embedding[w_idx])
                most_similar_word.append((w, cos_dis))
        most_similar_word.sort(key=lambda tup : tup[1], reverse=True)
        return most_similar_word[0:topn]
    
def FeaturesExtraction():
    word2vec = Word2Vec(sentences)
    vocabulary = word2vec.vocab()
    b1 = time.time()
    word2vec.train_batch_sg()
    train_time = time.time() - b1

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
    LemmaColumn = SWcolumn[['text']]
    LemmaColumn['text'] = LemmaColumn['text'].map(lambda x: lemmatization(x))
    global sentences 
    sentences = LemmaColumn.values.tolist()#the input list
    tableAfterLemma = Table(datasetsResults3, dataframe=LemmaColumn, width=300, editable=True)
    LemmaLabel.grid(column=7,row=6)
    tableAfterLemma.show()
    
def lemmatization(stopWordsOnly):
    LemmaColumn = [wn.lemmatize(word) for word in stopWordsOnly]
    return LemmaColumn
    
def TriggerStopWords():
    global SWcolumn
    SWcolumn = theColumn[['text']]
    SWcolumn['text'] = SWcolumn['text'].map(lambda x: stopWordsRemoval(x))
    tableAfterSW = Table(datasetsResults2, dataframe=SWcolumn,width=300, editable=True)
    SWLabel.grid(column=4,row=6)
    tableAfterSW.show()
    
def stopWordsRemoval(tokenizedTextOnly):
    SWcolumn = [word for word in tokenizedTextOnly if word not in stopwords]
    return SWcolumn

def TriggerTokenization():
    global theColumn
    theColumn = dfOri[['text']]
    theColumn['text'] = theColumn['text'].map(lambda x: Tokenization(str(x)))
    tableAfterToken = Table(datasetsResults, dataframe=theColumn, width=300, editable=True)
    TokenizedLabel.grid(column=0,row=6)
    tableAfterToken.show()

def Tokenization(text):
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