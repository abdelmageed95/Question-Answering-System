import json
from flask import Flask, request
import modeling as md
import preprocessing as pr

app = Flask(__name__)
	
@app.route("/predict",methods=['GET'])
def predict():

	question = request.args.get('question')
	lable = md.first_classifier(question)
	data = pr.read_docs('history_data')
	tfidf_data = pr.convert_tfidf(data)
	new_df = md.build_kMeans(3, tfidf_data)
	doc_name = md.find_doc(lable, new_df, question)
	top_ranked = md.find_paragrapgh(doc_name, question)
	result = md.find_answer(question, top_ranked)
	return str(result)
	
if __name__ == '__main__':
    from waitress import serve
    serve(app, port=5000)	