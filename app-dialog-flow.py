import numpy as np
from flask import Flask, request, make_response
#import json
#import modeling as md
import preprocessing as pr


app = Flask(__name__)
@app.route('/')
def hello():
    return 'Hello !'

# geting and sending response to dialogflow
@app.route('/webhook', methods=['POST'])
def webhook():

    req = request.get_json(silent=True, force=True)
    res = processRequest(req)
    res = json.dumps(res)

    r = make_response(res)
    r.headers['Content-Type'] = 'application/json'
    return r  #Final Response sent to DialogFlow

def processRequest(req):    # This method processes the incoming request 

    result = req.get("queryResult")
    parameters = result.get("parameters")
    question = parameters.get("question")
    
    intent = result.get("intent").get('displayName')
    
    if (intent=='DataYes'):
        prediction = md.first_classifier(question)
        output = prediction       
       
        fulfillmentText= "The class of the question is:  {} !".format(output)

        return {
            "fulfillmentText": fulfillmentText
        }

if __name__ == '__main__':
    from waitress import serve
    serve(app, port=5000)