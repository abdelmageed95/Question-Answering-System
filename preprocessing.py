from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from pathlib import Path
from sklearn.utils import shuffle
import modeling as md
import nltk
nltk.download('wordnet')
nltk.download('stopwords')

# get the data
def read_process_data(path):
  doc_list = []
  kings_name = []
  for file in Path(path).iterdir():
      kings_name.append(file.stem)
      ls = [ ]
      with open(file, "r", encoding="utf8") as file1:
          i = 0;
          for line in file1:
              ls.append(line)
              i = i + 1;
          doc_list.append(ls)
          
  document_names = [name for name in  kings_name]   
  data = pd.DataFrame()
  for i in range(len(doc_list)):
    one_docu_df = pd.DataFrame()
    for j in range(len(doc_list[i])):
      dic = {}
      dic['paragraph'] = doc_list[i][j]
      dic['doc_name'] = document_names[i]
      df = pd.DataFrame(dic , index = [i])
      one_docu_df = one_docu_df.append(df)
    data = data.append(one_docu_df)

  data= shuffle(data)
  data.reset_index(inplace = True)

  dh = data[data['paragraph'].map(len) == 1]
  data.drop(dh.index, axis=0, inplace=True)
  data.reset_index(inplace=True)

  list_lemma = []
  for i in range(len(data['paragraph'])):
    text= re.findall(r"[a-zA-Z]{3,}", data['paragraph'][i])
    lemmatizer = WordNetLemmatizer()
    lst = []
    for j in text:
      W = j.lower()
      w = lemmatizer.lemmatize(W)
      if w not in set(stopwords.words('english')):
        lst.append(str(w))
        l = " ".join(lst)
    list_lemma.append(l)

  data['lemma'] = list_lemma
  return data
  
def read_docs(path):
	my_dir_path = path
	doc_text = []
	kings_name = []
	for file in Path(my_dir_path).iterdir():
		kings_name.append(file.stem)
		with open(file, "r", encoding="utf8") as file1:
			text = file1.read()
			doc_text.append(text)
	
	document_names = [name for name in  kings_name]
	doc_data = pd.DataFrame()
	doc_data['document']= document_names
	doc_data['doc_content'] = doc_text
	doc_lemma = []
	for i in range(len(doc_data['doc_content'])):
		text= re.findall(r"[a-zA-Z]{3,}", doc_data['doc_content'][i])
		lemmatizer = WordNetLemmatizer()
		lst = []
		for j in text:
			W = j.lower()
			w = lemmatizer.lemmatize(W)
			if w not in set(stopwords.words('english')):
				lst.append(str(w))
				l = " ".join(lst)
		doc_lemma.append(l)
	doc_data['lemma_doc'] = doc_lemma
	return doc_data

def question_preprocessing(question):
	text= re.findall(r"[a-zA-Z]{3,}", question)
	lemmatizer = WordNetLemmatizer()
	lst_word = []
	for word in text:
		W = word.lower()
		w = lemmatizer.lemmatize(W)
		if w not in set(stopwords.words('english')):
			lst_word.append(str(w))
	question_words =[" ".join(lst_word)]
	question_words = pd.Series(question_words)
	vectorizer = TfidfVectorizer()
	tfidf_doc = vectorizer.fit_transform(read_docs('history_data')['lemma_doc'])
	ques = vectorizer.transform(question_words)
	tfidf_df = pd.DataFrame(ques.toarray(), columns=vectorizer.get_feature_names())
	#quess = ques.toarray()
	return tfidf_df
		


def convert_tfidf(data):
	vectorizer = TfidfVectorizer()
	tfidf = vectorizer.fit_transform(data['lemma_doc'])
	X_tfidf = tfidf.toarray()
	return X_tfidf
	