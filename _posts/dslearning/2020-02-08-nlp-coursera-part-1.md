---
title: "Exploring Natural Language Processing with Coursera - Part 1"
date: 2020-02-08
tags: [data science learning, natural language processing, nlp, coursera]
category: [data-science-learning]
author_profile: false
excerpt: "Step by step exploration of NLP on Coursera - Week 1"
toc: True
toc_label: "Contents"
toc_sticky: true
---

Recently, i decided to explore the world of Natural Language Processing. After some research online, I found the course provided by Higher School of Economics on coursera to be one of the highly rated courses and decided to follow it. The course is simply called [Natural Language Processing](https://www.coursera.org/learn/language-processing){:target="_blank"} and covers wide range of topics in NLP from basic to advanced. I decided to document my learning by enriching the course material provided during the course with additional exploration on various subjects that might not have specifically been discussed or presented on the course page.

Ultimately these series of posts will cover my notes during my learning journey. It will also serve as a good reference point to quickly remind myself, and anyone who wishes for that fact, major topics in the course. Please feel free to leave a comment or suggestions on how to improve moving forward my note taking.

Let's get started with Week 1.

# Main approaches in NLP

NLP approaches could be summarized into following:
1. Rule-based methods
    * Regular expressions
    * Context-free grammars
    * ...
2. Probabilistic modeling and machine learning
    * Likelihood maximization
    * Linear classifiers
    * ...
3. Deep Learning
    * Recurrent Neural Networks
    * Convolutional Neural Networks
    * ...

## Semantic Slot Filling

Semantic slot filling is a problem in Natural Language Processing which describes the process of tagging words or tokens that carry meaning in the sentences in order to make sense of text. Various examples approaches in the area of semantic slot filling.

* **Context-free grammar (CFG)** - This method represents a set a of rules that tries to replace non-terminal symbols (symbols or words in a sentence that can be replaced) with terminal ones (symbols or words that can't be replaced). For example in "Show me flights from Boston[origin] to San Francisco[destination] on 12 December[date]". Since its rule based, it needs to be done manually potentially with the involvement of a linguist. Usually this approach has high precision but lower recall.
* **Conditional Random Field (CRF)** - To continue with the example of "Show me flights from Boston to San Francisco". We could also build a machine learning model by training the corpus of tokenized data containing features which we could prepare (feature engineering). Example of new features could be
	- Is the word capitalized?
	- Is the word in a list of city names?
	- What is the previous word?
	- What is the previous slot?
    
    Then we could use a model from a class of discriminative models called Conditional Random Fields (CRF). They are nicely explained by Aditya Prasad in his [post](https://towardsdatascience.com/conditional-random-fields-explained-e5b8256da776){:target="_blank"}. They might be a good candidate in NLP tasks because contextual information or state of the neighbors affect the current prediction of the model. In this scenario we would essentially maximize the probability of the word or word structure given the text. The high level formulas are below

<figure>
    <a href="/images/coursera-nlp/coursera-nlp-w1-1.png"><img src="/images/coursera-nlp/coursera-nlp-w1-1.png"></a>
    <figcaption>NLP Coursera - Week 1 - Semantic Slot Filling CRF</figcaption>
</figure>

* **Long Short Term Memory (LSTM) networks** - are a type of deep learning approach. In this scenario we would simply feed the sequence of words (vectorized by potentially one-hot encoding) to a neural network with a certain architecture/topology and numerous parameters.

### Deep Learning vs Traditional NLP

While Deep Learning performs quite well in many NLP tasks its important not to forget about traditional methods. Some reasons are:
- Traditional methods perform quite nice for tasks such as sequence labeling, which is basically a pattern recognition method where a categorical label is assigned to a member of a sequence of observed values (wikipedia).
- word2vec method which is actually not even deep learning but it is inspired by some neural networks, has really similar ideas as some distributional semantic methods have. Despite not being a DLP method, word2vec acts like a two-layer "neural net" and vectorizes words. Detailed introduction to word2vec [here](https://pathmind.com/wiki/word2vec){:target="_blank"}.
- We can also use knowledge in traditional NLP in improving the deep learning methods and approaches.

Nevertheless, Deep Learning currently seems to be the future for doing NLP and will become more common in the years to come.

Next section, "Brief overview of the next weeks" is skipped since it simply summarizes what's to come.

# Linguistic Knowledge in NLP

Natural Language Processing and understanding is not only about mathematics but also linguistics. Thus its important to cover the following NLP Pyramid.

To understand the above pyramid, let's say we have a sentence as an example. There are multiple stages of analysis of that sentence
* Morphology - Morphology is the study of the structure and formation of words. Basically everything that concerns individual words in the sentence. At this stage we care about 
    - forms of words (sawed, sawn, sawing) [further reading](https://wac.colostate.edu/docs/books/sound/chapter5.pdf){:target="_blank"}
	- part of speech text (noun, pronouns, verbs, etc.)
	- different cases (nominative, accusative, etc.) [further reading](https://en.wikipedia.org/wiki/Case_role){:target="_blank"}
	- genders (masculine, feminine) [further reading](https://findwords.info/term/grammatical%20gender){:target="_blank"}
	- tenses (present, past, future, etc.)
* **Syntax** - is about the different relationships of words within the sentence. It represents the set of rules, principles and processes that govern the structure of sentences in a given language, usually including word order. Interesting video summary [here](https://study.com/academy/lesson/what-is-syntax-in-linguistics-definition-overview.html){:target="_blank"}. Simple sentences follow a basic structure: subject - verb - object. For example - "The girl[subject] bought[verb] a book[object]"
* **Semantics** - as the next step in the pyramid, it represents the meaning and interpretation of words, signs and sentence structure. Some resources for further reading.  [Linguistics 001](https://www.ling.upenn.edu/courses/Fall_2019/ling001/semantics.html){:target="_blank"} and [What does semantics study](https://all-about-linguistics.group.shef.ac.uk/branches-of-linguistics/semantics/what-does-semantics-study/){:target="_blank"}
* **Pragmatics** - is a next level in the abstraction presented in the pyramid and studies the practical meaning of words within various interactional contexts. Fore more explanation of pragmatics read [here](https://all-about-linguistics.group.shef.ac.uk/branches-of-linguistics/pragmatics/what-is-pragmatics/){:target="_blank"}

Some libraries and tools to use to explore more of each of the stages in the pyramid.
1. [NLTK Library](https://www.nltk.org/){:target="_blank"}- contains many small and useful datasets with markup as well as various preprocessing tools (tokenization, normalization, etc.). It also contains some pre-trained models for POS-tagging and parsing. A great book for further exploring NLTK and in general Python in Natural Language Processing (NLP) is [Natural Language Processing with Python](https://amzn.to/2SizenP){:target="_blank"}.
2. [Stanford Parser](https://nlp.stanford.edu/software/lex-parser.shtml){:target="_blank"}  - used for syntactic analysis a natural language parser is a program that works out the grammatical structure of sentences, for instance, which groups of words go together (as "phrases") and which words are the subject or object of a verb.
3. [spaCy](https://spacy.io/){:target="_blank"} - python library for text analysis, e.g. for word embeddings and topic modeling. It is designed specifically for production use and helps you build applications that process and “understand” large volumes of text. It can be used to build information extraction or natural language understanding systems, or to pre-process text for deep learning.
4. [Gensim](https://en.wikipedia.org/wiki/Gensim){:target="_blank"} - is a python library for text analysis, unsupervised topic modeling and nlp. More information on wikipedia and some core concepts explored [here](https://radimrehurek.com/gensim/auto_examples/core/run_core_concepts.html){:target="_blank"}
5. [MALLET](http://mallet.cs.umass.edu/){:target="_blank"}- similar to gensim but written in Java.

To explore further the relationships between words such as synonyms (two different words with similar meaning) or homonyms (two or more words with same spelling but different meanings and origins), libraries such as [WordNet](https://wordnet.princeton.edu/){:target="_blank"} - a lexical database for English and [BabelNet](https://babelnet.org/){:target="_blank"} - the multilingual alternative to wordnet, would be useful to explore.

### Linguistic knowledge + Deep Learning

In NLP, one of the tasks is reasoning. Let's say there is a story. Mary got the football, she went to the kitchen, she left the ball there. Here we have some story, and now we have a question after this story, where is the football now? LSTM networks, a particular type of recurrent neural networks, could be used in this scenario. 

<figure>
    <a href="/images/coursera-nlp/coursera-nlp-w1-2.png"><img src="/images/coursera-nlp/coursera-nlp-w1-2.png"></a>
    <figcaption>NLP Coursera - Week 1 - Linguistic Knowledge + Deep Learning</figcaption>
</figure>

In the picture above we can see that the red lines (or edges) stand for co-reference, which means that mary and she represent the same entity - that is Mary. Meanwhile, green line represents hypernyms [(see Wikipedia)](https://en.wikipedia.org/wiki/Hyponymy_and_hypernymy). In our case football is a type of a ball.

In summary, the knowledge of linguistics allows to identify a method for tackling an NLP challenge, in our case by identifying these relationships we could be adding more edges to our DAG-LTSM approach. DAG-LTSM stands or Dynamic Acyclic Graph - Long Short Term Memory Recurrent Neural Network (a mouthful).

### Syntax: dependency & constituency trees

Another example of linguistic knowledge used in applied NLP is representing syntax via dependency or constituency trees. The images below show examples. In case of dependency trees, (left image) the sentence would be parsed based on various dependencies present in the sentence (e.g subject, object, modifier and etc.). In case of constituency trees a parser would parse the sentence from bottom to top to get a hierarchial structure. Each of the nodes represent a syntactial element (e.g. Noun, noun phrase, verb, verb phrase). By parsing it in a hierarchical structure it allows to find named entities (NE) such as New York City, since NEs are most likely to be noun phrases. 

<figure class="third">
    <a href="/images/coursera-nlp/coursera-nlp-w1-2.png"><img src="/images/coursera-nlp/coursera-nlp-w1-3.png"></a>
    <a href="/images/coursera-nlp/coursera-nlp-w1-2.png"><img src="/images/coursera-nlp/coursera-nlp-w1-4.png"></a>
    <a href="/images/coursera-nlp/coursera-nlp-w1-5.png"><img src="/images/coursera-nlp/coursera-nlp-w1-5.png"></a>
    <figcaption>NLP Coursera - Week 1 - Linguistic Knowledge + Deep Learning</figcaption>
</figure>

It is also useful in sentiment analysis. In sentiment analysis, we try to predict the sentiment given the text. In the image above (3) we can see that a parser similar to the previously mentioned one would have a tree structure with nodes being individual words with a certain sentiment. By looking at nodes the overall sentiment could be calculated. This is an example of using recurrent neural network or DAG networks to predict sentiments, but in practice simple classification at times is sufficient enough.

# Text Preprocessing

Text can be thought of as sequence of:
* Characters
* Words
* Phrases and named entities
* Sentences
* Paragraphs

Let's start with words. Word - a meaningful sequence of characters. Usually we can find boundaries of words, with the exception of some words in German and all words in Japanese. To split a text into words we would need to tokenize the sentence. 

## Tokenization

Tokenization is a process that splits an input sequence into so-called tokens. Token is a useful unit for semantic processing. It could be a word, sentence, paragraph, etc.. A simple whitespace tokenizer is available in nltk, as an example:

```python
from nltk.tokenize import WhitespaceTokenizer
s = "This is Andrew's text, isn't it?"
WhitespaceTokenizer().tokenize(s)
```
    ['This', 'is', "Andrew's", 'text,', "isn't", 'it?']

The problem of tokenizing in this way is that `it?` is the same as `it` and `text,` is same as `text`.

We can try to split by punctuation.

```python
from nltk.tokenize import WordPunctTokenizer
s = "This is Andrew's text, isn't it?"
WordPunctTokenizer().tokenize(s)
```
    ['This', 'is', 'Andrew', "'", 's', 'text', ',', 'isn', "'", 't', 'it', '?']