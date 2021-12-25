from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from rank_bm25 import BM25Okapi
from transformers import BertForQuestionAnswering
import torch
from transformers import BertTokenizer
import pickle
import preprocessing as pr


def build_kMeans(n_cluster, tfidf_data):
  tsna = TSNE(n_components= 2, random_state= 0)
  data_tsna = tsna.fit_transform(tfidf_data) 
  kmeans_model= KMeans(n_clusters= n_cluster, init='k-means++', random_state=0, n_init=10)
  y_pred = kmeans_model.fit_predict(data_tsna)
  new_df = pr.read_docs('history_data')
  new_df['cluster_label'] = y_pred
  return new_df


def first_classifier(question):
	model = pickle.load(open('first_classifier.pkl', 'rb'))
	return model.predict(pr.question_preprocessing(question))[0]

def find_doc(first_clf_out, clustered_df, question):
	docu_4_cls2= clustered_df[clustered_df["cluster_label"] == first_clf_out]
	#docu_4_cls2 = docu_4_cls2.drop('cluster_label' , axis= 1 )
	candidate_docs = list(docu_4_cls2['doc_content'])
	tokenized_corpus = [doc.split(" ") for doc in candidate_docs]
	bm25 = BM25Okapi(tokenized_corpus)
	tokenized_query = question.split(" ")
	doc_scores = bm25.get_scores(tokenized_query)
	top_rank_para =  bm25.get_top_n(tokenized_query, candidate_docs, n=3)[1]
	candidate_document = docu_4_cls2.document[docu_4_cls2['doc_content']==top_rank_para ]
	candidate_document = list(candidate_document)[0]
	return candidate_document
	
def find_paragrapgh(document_name, question):
	matched_doc_path = "history_data/{}.txt".format(document_name)
	f = open(matched_doc_path, "r", encoding="utf8")
	paragraphs_lst = []
	for x in f:
		paragraphs_lst.append(x)
	tokenized_corpus = [doc.split(" ") for doc in paragraphs_lst]
	bm25 = BM25Okapi(tokenized_corpus)
	top_rank_para =  bm25.get_top_n(question.split(" "), paragraphs_lst, n=3)[2]
	return top_rank_para
	
def find_answer(question, top_rank_para):
	model = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
	tokenizer = BertTokenizer.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
	input_ids = tokenizer.encode(question, top_rank_para)
	tokens = tokenizer.convert_ids_to_tokens(input_ids)
	# Search the input_ids for the first instance of the `[SEP]` token.
	sep_index = input_ids.index(tokenizer.sep_token_id)
	# The number of segment A tokens includes the [SEP] token istelf.
	num_seg_a = sep_index + 1
	# The remainder are segment B.
	num_seg_b = len(input_ids) - num_seg_a
	# Construct the list of 0s and 1s.
	segment_ids = [0]*num_seg_a + [1]*num_seg_b
	# There should be a segment_id for every input token.
	assert len(segment_ids) == len(input_ids)
	outputs = model(torch.tensor([input_ids]), # The tokens representing our input text.
                             token_type_ids=torch.tensor([segment_ids]), # The segment IDs to differentiate question from answer_text
                             return_dict=True) 
	start_scores = outputs.start_logits
	end_scores = outputs.end_logits
	# Find the tokens with the highest `start` and `end` scores.
	answer_start = torch.argmax(start_scores)
	answer_end = torch.argmax(end_scores)
	# Combine the tokens in the answer and print it out.
	answer = ' '.join(tokens[answer_start:answer_end+1])
	# Start with the first token.
	answer = tokens[answer_start]
	# Select the remaining answer tokens and join them with whitespace.
	for i in range(answer_start + 1, answer_end + 1):
		# If it's a subword token, then recombine it with the previous token.
		if tokens[i][0:2] == '##':
			answer += tokens[i][2:]
		# Otherwise, add a space then the token.
		else:
			answer += ' ' + tokens[i]
	return answer
