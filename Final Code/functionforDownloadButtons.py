import streamlit as st
import pickle
import pandas as pd
import json
import base64
import uuid
import re
import nltk
from sklearn.svm import SVC
nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords
import re
import string
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()
import importlib.util
import ftfy
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline
from langdetect import detect
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
cv = CountVectorizer()
tfidf = TfidfVectorizer(max_features=3000)
import spacy
from spacy import displacy
from sklearn import metrics

def null_val_rows(df):
  Rows = df.shape[0] 
  Columns = df.shape[1]


  # Find the Number of Rows that has Nan Value in it

  Null_Data = df.isnull().sum()
  print(Null_Data)

  #valuecons - check for better approach
  # List for storing the Null Column Names

  Null_Columns = []

  #check pandas dataframe for better approach - action tbd

  for i in range(len(Null_Data)):


    # If the number of Null Values in the Row is equal to the total number of Records, then it means that the whole column contains NUll value in it. 

    if Null_Data[i] == Rows - 1 or Null_Data[i] == Rows:
      
      Null_Columns.append(Column_Names[i])


  # Print all Columns which has only NULL values

  return Null_Columns

def del_col_null(df,Null_Columns):
  # Delete all NULL Columns which has only NULL values

  for i in Null_Columns:

    del df[i]

def row_null(df):
  # Display the Rows which has one or more NULL values in it
  df.dropna(inplace=True)
  df.reset_index(drop=True)

def stnd_prepro(text):
        text = str(text)

        text = text.lower()
        text = nltk.word_tokenize(text)
        
        
        removeword = ""
        y = []
        for i in text:
            if i.isalnum():
                y.append(i)
        
        text = y[:]
        y.clear()
        
        for i in text:
            if i not in stopwords.words('english') and i not in string.punctuation and i != removeword:
                y.append(i)
                
        text = y[:]
        y.clear()
        
        for i in text:
            y.append(ps.stem(i))
            
        return " ".join(y)

def removedigits(text):
        text = str(text)
        pattern = r'[0-9]'
        text = re.sub(pattern, '', text)
        return text

def removealphabets(text):
        text = str(text)
        pattern = r'[a-z | A-Z]'
        text = re.sub(pattern, '', text)
        return text
    
def removealphanum(text):
        text = str(text)
        pattern = r'[0-9 | a-z | A-Z]'
        text = re.sub(pattern, '', text)
        return text
    
def onlyalphanum(text):
        text = str(text)
        pattern = r'[\W_]+'
        text = re.sub(pattern, '', text)
        return text

def remove_special(text):
        text = str(text)
        text = re.sub(r"[^a-zA-Z0-9 ]", "", text)
        return text

def lang_detect(df):
    for i,r  in enumerate(df["text"]):
      try:
        #print(r)
        df.loc[i,"language"] = detect(r)
        #print(detect(r))
      except:
        df.loc[i,"language"] = "NA"

def import_from_file(module_name: str, filepath: str):
    """
    Imports a module from file.
    Args:
        module_name (str): Assigned to the module's __name__ parameter (does not
            influence how the module is named outside of this function)
        filepath (str): Path to the .py file
    Returns:
        The module
    """
    spec = importlib.util.spec_from_file_location(module_name, filepath)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def notebook_header(text):
    """
    Insert section header into a jinja file, formatted as notebook cell.
    Leave 2 blank lines before the header.
    """
    return f"""# # {text}
"""


def code_header(text):
    """
    Insert section header into a jinja file, formatted as Python comment.
    Leave 2 blank lines before the header.
    """
    seperator_len = (75 - len(text)) / 2
    seperator_len_left = math.floor(seperator_len)
    seperator_len_right = math.ceil(seperator_len)
    return f"# {'-' * seperator_len_left} {text} {'-' * seperator_len_right}"

def knn(new_df):
    X = tfidf.fit_transform(new_df["Pre-Processed"].values.astype('U')).toarray()
    Y = new_df['spam'].values
    X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.20)
    k = 2
    neigh = KNeighborsClassifier(n_neighbors=k)
    neigh.fit(X_train, Y_train)
    Y_pred = neigh.predict(X_test)
    st.write(confusion_matrix(Y_test, Y_pred))
    st.write("KNN Accuracy : ", (metrics.accuracy_score(Y_test, Y_pred) * 100.0))
    st.download_button("Download KNN Model",data=pickle.dumps(neigh),file_name="model.pkl")
    filename = 'finalized_model.sav'
    pickle.dump(neigh, open(filename, 'wb'))
    #self.algo_compute(neigh)

def logistic(new_df):
    X = tfidf.fit_transform(new_df["Pre-Processed"].values.astype('U')).toarray()
    Y = new_df['spam'].values
    X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.20)
    lr = LogisticRegression()
    lr.fit(X_train, Y_train)
    Y_pred = lr.predict(X_test)
    st.write(confusion_matrix(Y_test, Y_pred))
    st.write("Logistic Regression Accuracy: ",(metrics.accuracy_score(Y_test, Y_pred) * 100.0))
    st.download_button("Download Logistic Regression Model",data=pickle.dumps(lr),file_name="model.pkl")
    #self.algo_compute(lr)

def svmobj(new_df):
    X = tfidf.fit_transform(new_df["Pre-Processed"].values.astype('U')).toarray()
    Y = new_df['spam'].values
    X_train, X_test, Y_train, Y_test = train_test_split( X, Y, test_size = 0.2, random_state = 2 )
    clf = svm.SVC(kernel = 'linear')
    clf.fit(X_train, Y_train)
    Y_pred = clf.predict(X_test)
    st.write(confusion_matrix(Y_test, Y_pred))
    st.write("SVM Accuracy: ",(metrics.accuracy_score(Y_test, Y_pred) * 100.0))
    st.download_button("Download SVM Model",data=pickle.dumps(clf),file_name="model.pkl")
    #self.algo_compute(clf)

def naivebayes(new_df):
    X = tfidf.fit_transform(new_df["Pre-Processed"].values.astype('U')).toarray()
    Y = new_df['spam'].values
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.5, random_state=0)
    gnb = GaussianNB()
    y_pred = gnb.fit(X_train, Y_train).predict(X_test)
    st.write(confusion_matrix(Y_test, y_pred))
    st.write("Naive Bayes Accuracy: ",(metrics.accuracy_score(Y_test, y_pred) * 100.0))

    st.download_button("Download Naive Bayes Model",data=pickle.dumps(gnb),file_name="model.pkl")

    #self.algo_compute(gnb)
  
def randomforest(new_df):
    X = tfidf.fit_transform(new_df['Pre-Processed'].values.astype('U')).toarray()
    Y = new_df['spam'].values
    X_train, X_test, Y_train, Y_test = train_test_split( X, Y, test_size = 0.2, random_state = 2 )
    clf=RandomForestClassifier(n_estimators=100)
    clf.fit(X_train,Y_train)
    Y_pred=clf.predict(X_test)
    st.write(confusion_matrix(Y_test, Y_pred))
    st.write("Random Forest Accuracy: ",(metrics.accuracy_score(Y_test, Y_pred) * 100.0))
    st.download_button("Download Random Forest Model",data=pickle.dumps(clf),file_name="model.pkl")



def to_notebook(code):
    """Converts Python code to Jupyter notebook format."""
    notebook = jupytext.reads(code, fmt="py")
    return jupytext.writes(notebook, fmt="ipynb")


def open_link(url, new_tab=True):
    """Dirty hack to open a new web page with a streamlit button."""
    # From: https://discuss.streamlit.io/t/how-to-link-a-button-to-a-webpage/1661/3
    if new_tab:
        js = f"window.open('{url}')"  # New tab or window
    else:
        js = f"window.location.href = '{url}'"  # Current tab
    html = '<img src onerror="{}">'.format(js)
    div = Div(text=html)
    st.bokeh_chart(div)


def download_button(object_to_download, pre_p, option, download_filename, button_text):
    """
    Generates a link to download the given object_to_download.
    From: https://discuss.streamlit.io/t/a-download-button-with-custom-css/4220
    Params:
    ------
    object_to_download:  The object to be downloaded.
    download_filename (str): filename and extension of file. e.g. mydata.csv,
    some_txt_output.txt download_link_text (str): Text to display for download
    link.
    button_text (str): Text to display on download button (e.g. 'click here to download file')
    pickle_it (bool): If True, pickle file.
    Returns:
    -------
    (str): the anchor tag to download object_to_download
    Examples:
    --------
    download_link(your_df, 'YOUR_DF.csv', 'Click to download data!')
    download_link(your_str, 'YOUR_STRING.txt', 'Click to download text!')
    """
    # if pickle_it:
    #    try:
    #        object_to_download = pickle.dumps(object_to_download)
    #    except pickle.PicklingError as e:
    #        st.write(e)
    #        return None

    # if:
    if isinstance(object_to_download, bytes):

        pass

    elif isinstance(object_to_download, pd.DataFrame):
        abc = null_val_rows(object_to_download)
        del_col_null(object_to_download,abc)
        row_null(object_to_download)
        object_to_download["text"] = object_to_download["text"].apply(ftfy.fix_text)
        for i in range(0,len(pre_p)):
          if(pre_p[i] == "Standard Pre-Processing"):
            object_to_download["Pre-Processed"] = object_to_download["text"].apply(stnd_prepro)
          elif(pre_p[i] == "Remove Digits"):
            object_to_download["Pre-Processed"] = object_to_download["text"].apply(removedigits)
          elif(pre_p[i] == 'Remove Alphabets'):
            object_to_download["Pre-Processed"] = object_to_download["text"].apply(removealphabets)
          elif(pre_p[i] == "Remove Alphanum"):
            object_to_download["Pre-Processed"] = object_to_download["text"].apply(removealphanum)
          elif(pre_p[i] == "Retain only Alphanum"):
            object_to_download["Pre-Processed"] = object_to_download["text"].apply(onlyalphanum)
          elif(pre_p[i] == "Remove Special Characters"):
            object_to_download["Pre-Processed"] = object_to_download["text"].apply(remove_special)


        if(option == "KNN"):
            knn(object_to_download)
        elif(option == "Logistic Regression"):
            logistic(object_to_download)
        elif(option == "SVM"):
            svmobj(object_to_download)
        elif(option == "Naive Bayes"):
            naivebayes(object_to_download)
        elif(option == "Random Forest"):
            randomforest(object_to_download)

        lang_detect(object_to_download)
        object_to_download = object_to_download.to_csv(header = False,index = False)
    # Try JSON encode for everything else
    else:
        object_to_download = json.dumps(object_to_download)

    try:
        # some strings <-> bytes conversions necessary here
        b64 = base64.b64encode(object_to_download.encode()).decode()
    except AttributeError as e:
        b64 = base64.b64encode(object_to_download).decode()

    button_uuid = str(uuid.uuid4()).replace("-", "")
    button_id = re.sub("\d+", "", button_uuid)

    custom_css = f"""
        <style>
            #{button_id} {{
                display: inline-flex;
                align-items: center;
                justify-content: center;
                background-color: rgb(255, 255, 255);
                color: rgb(38, 39, 48);
                padding: .25rem .75rem;
                position: relative;
                text-decoration: none;
                border-radius: 4px;
                border-width: 1px;
                border-style: solid;
                border-color: rgb(230, 234, 241);
                border-image: initial;
            }}
            #{button_id}:hover {{
                border-color: rgb(246, 51, 102);
                color: rgb(246, 51, 102);
            }}
            #{button_id}:active {{
                box-shadow: none;
                background-color: rgb(246, 51, 102);
                color: white;
                }}
        </style> """

    dl_link = (
        custom_css
        + f'<a download="{download_filename}" id="{button_id}" href="data:file/txt;base64,{b64}">{button_text}</a><br><br>'
    )
    # dl_link = f'<a download="{download_filename}" id="{button_id}" href="data:file/txt;base64,{b64}"><input type="button" kind="primary" value="{button_text}"></a><br></br>'

    st.markdown(dl_link, unsafe_allow_html=True)

def ner_function(raw_text):
  NER = spacy.load("en_core_web_sm")
  text1= NER(raw_text)
  for word in text1.ents:
    st.write(word.text,word.label_)
  


# def download_link(
#     content, label="Download", filename="file.txt", mimetype="text/plain"
# ):
#     """Create a HTML link to download a string as a file."""
#     # From: https://discuss.streamlit.io/t/how-to-download-file-in-streamlit/1806/9
#     b64 = base64.b64encode(
#         content.encode()
#     ).decode()  # some strings <-> bytes conversions necessary here
#     href = (
#         f'<a href="data:{mimetype};base64,{b64}" download="{filename}">{label}</a>'
#     )
#     return href
