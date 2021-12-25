
![Logo](https://firebasestorage.googleapis.com/v0/b/plantsexpertsystem-f6812.appspot.com/o/Picture1.jpg?alt=media&token=d2540f8c-47bb-4d01-86c1-1d5324955d23)


# Question Answering System (NMEC)

National Museum of Egyptian Civilization has open 
months ago, they announced that they need a system
that helps any visitor to ask any question about 
Ancient Egypt, the visitor is interested in a concise,
comprehensible, and correct answer, which may refer 
to a word, sentence, or a paragraph. Information 
based question answering systems (NMEC) is an
appropriate solution to this case.


## Authors

- [@Mohamed Sayed](https://github.com/Aboalarbe)
- [@Abdelmageed Ahmed](https://github.com/abdelmageed95)
- [@Sara Hossam](https://github.com/HossamSarahh)
- [@Ahmed Shehata](https://github.com/ShehaTaa)


## Documentation

Question Answering System (QAS) is one of the most
promising research areas in NLP (Natural Language
Processing) which consisting of multiple processing
steps such as Corpus Preparations, Information Retrieval
(IR), Information Extraction (IE), Linguistic and
Artificial Intelligence (AI).
The Question Answering system has a lot of applications
such as extracting information from documents, Online
examination system, document management, Language learning, human.
Question Answering (QA) systems have emerged as powerful
platforms for automatically answering questions asked by
humans in natural language using either a pre-structured
database or a collection of natural language documents,
QA systems make it possible asking questions and retrieve
the answers using natural language queries and may be
considered as an advanced form of Information Retrieval
(IR), Question Answering systems in information retrieval
are tasks that automatically answer the questions asked
by humans in natural language using either
a pre-structured database or a collection of naturallanguage document

A lot of studies have been presented in this area.
Most of them defined an architecture of Question
Answering systems in three macro modules as shown
in figure1 [1], Question Processing, Document
Processing and Answer Processing as showed in the next figure.

![fig1](https://firebasestorage.googleapis.com/v0/b/plantsexpertsystem-f6812.appspot.com/o/image2.png?alt=media&token=11c515dd-77de-48f6-98d5-45c4d819c644)
Besides the main architecture, QA systems can be defined by the paradigm each one implements:

1. Information Retrieval QA: Usage of search engines to retrieve answers and then apply filters and ranking on the recovered passage.
2. Natural Language Processing QA: Usage of linguistic intuitions and machine learning methods to extract answers from retrieved snippet.
3. Knowledge Base QA: Find answers from a structured data source (a knowledge base) instead of unstructured text. Standard database queries are used in replacement of word-based searches.

On sending a question to the system as shown in figure. The NMEC system will work to process both documents and questions to extract information and retrieve the answer through several steps as follows:
![fig2](https://firebasestorage.googleapis.com/v0/b/plantsexpertsystem-f6812.appspot.com/o/WhatsApp%20Image%202021-12-11%20at%203.49.47%20AM.jpeg?alt=media&token=16091385-f764-42aa-8c1a-d2acaa3f68c5)

[Full Documentation](https://drive.google.com/file/d/1ZbL_a5QpfzdETIzHRzkYKwYjOl41_jJW/view?usp=sharing)



## Installation

you should install Anaconda Environment and install 
the following packages:

```bash
pip install waitress
```
```bash
pip install rank_bm25
```
```bash
pip install transformers
```
```bash
pip install torch
```
    
## Deployment

To deploy this project run

```bash
  open project dierctory
```
```bash
  run app.py
```
```bash
  open "index.html" file in your broswer
```
```bash
  type your question and hit predict
```
note that first runtime it may take some while
because 'Bert' pre-trained model will be downloaded locally.

## Screenshots

![App Screenshot1](https://firebasestorage.googleapis.com/v0/b/plantsexpertsystem-f6812.appspot.com/o/WhatsApp%20Image%202021-12-11%20at%208.24.15%20PM.jpeg?alt=media&token=034c372a-1014-4362-904f-3c6854ed3c26)
![App Screenshot2](https://firebasestorage.googleapis.com/v0/b/plantsexpertsystem-f6812.appspot.com/o/WhatsApp%20Image%202021-12-11%20at%208.24.15%20PM2.jpeg?alt=media&token=0b257a43-4bac-4bac-b5fc-12afdfcc0ef4)


## Optimizations

now we are working to optimize our piplines to get response in less than 5 seconds !!

## Feedback

If you have any feedback, please reach out to us at mhuss073@uottawa.ca and ahass202@uottawa.ca


## License

[MIT](https://choosealicense.com/licenses/mit/)

