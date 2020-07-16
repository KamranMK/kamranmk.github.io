---
title: "Exploring NLP with Coursera - Part 3"
date: 2020-05-17
tags: [data science learning, natural language processing, nlp, coursera]
category: [data-science-learning]
classes: full-wide
author_profile: false
excerpt: "Week 1: Part 3 - Simple deep learning for text classification"
toc: True
toc_label: "Contents"
toc_sticky: true
---

# Two approaches to representing text

In this section we think of text as a sequence of words. Let's look at two approaches of representing that sequence.

## Bag of Words (BOW) way (sparse)

BOW is simply a vector where rows represent the sequence at hand (e.g. I saw a very good movie) and all the columns / features are a collection/dictionary of all words in a corpus. This dictionary can be huge (in the video ~100K words) thus making the list of our features enormous. Each sequence of words (sentence) would be a sum of vectors representing each of the words within that sentence, as illustrated in the image below. This representation is usually sparse since each sentence only has a limited number of words from the complete dictionary.

<figure>
    <img src="https://raw.githubusercontent.com/KamranMK/kamranmk.github.io/master/images/coursera-nlp/coursera-nlp-w1-21.png">
    <figcaption>NLP Coursera - Week 1 - BOW representation (sparse)</figcaption>
</figure>

## Neural network way (dense)

Contrary to BOW approach, a neural network approach would generate a dense matrix where each word would be represented by a vector which is shorter. An example of such representation is word2vec embeddings, a vector representation of words prepared in an unsupervised manner. Word2vec has a nice property = words that have a similar context tend to have collinear vectors (vectors that point to the same direction in vector space). For example, man and woman or king and queen would have close/similar values since they are similar in a certain manner.
To come up with a feature descriptor or in other words to represent text we could also sum up all the vectors representing individual words. This could be a good baseline descriptor for text.

<figure>
    <img src="https://raw.githubusercontent.com/KamranMK/kamranmk.github.io/master/images/coursera-nlp/coursera-nlp-w1-22.png">
    <figcaption>NLP Coursera - Week 1 - Neural Way (dense)</figcaption>
</figure>


## A better way: 1D convolutions

Basically, if we want to a vector representation of n-grams (in our case 2-grams), we can simply apply a confolutional filter on top of each token pair vectors (as illustrated) in order to generate a value which will be a vector representation of those two nearby tokens. Thus if we have two token pairs (1. cat sitting and 2. dog resting) which have similar meaning the result of using convolutional filter will be close. This is because word embeddings use cosine distance to calculate similarity of words, cosine distance is a dot product much like application of a convoluational filter. Thus at the end we get a high activation (0.9 and 0.84) which indicate these two 2-grams have a similar meaning ("animal sitting"). Of course, quality of word embeddings will also impact the quality of the n-grams. 

Overall, this allows us to extract meaning from n-grams which can enable us to extract higher/abstract meaning from text.

<figure>
    <img src="https://raw.githubusercontent.com/KamranMK/kamranmk.github.io/master/images/coursera-nlp/coursera-nlp-w1-23.png">
    <figcaption>NLP Coursera - Week 1 - Neural Way (dense)</figcaption>
</figure>

### 1D convolutions

The benefit here is that this approach can be extended to many grams and the feature representation won't expand since its fixed, thanks to word embeddings.

We use the convolutional filter (length 3 since we learn 2-grams) to slide the window through the sentence ("cat sitting there or here"). We would need to use many convolutional filters each of which would represents some sort of meaning. These filters are called 1D since we slide them in one direction. This order represents time. The results on the right are a vector representation of that convolutional filter. 

<figure>
    <img src="https://raw.githubusercontent.com/KamranMK/kamranmk.github.io/master/images/coursera-nlp/coursera-nlp-w1-24.png">
    <figcaption>NLP Coursera - Week 1 - 1D convolutions
    </figcaption>
</figure>

We assume that we can lose the order of words which allows us to simply care about a certain meaning/combination ocurring in a sequence or not (e.g. animal sitting, since the filter would generate a high activation). Thus we choose the maximum value (0.7), which is called maximum pooling over time.

<figure>
    <img src="https://raw.githubusercontent.com/KamranMK/kamranmk.github.io/master/images/coursera-nlp/coursera-nlp-w1-25.png">
    <figcaption>NLP Coursera - Week 1 - 1D convolutions
    </figcaption>
</figure>


## Final Architecture

Following the approach we can finally

* Use windows of size 3, 4, 5 to capture 3, 4, 5 grams
* Each window would have 100 convolutional filters
* This would generate 300 outputs / features
* Last, we train models on top of these features

There are two papers in here which have compared this approach of using 1D convolutions with maximum pooling to Naive Bayes on top of 1, 2 grams, with respective accuracies (slide below). As it seems this approach seems to be more effective in terms of accuracy of prediction.

[Paper 1 - Natural Language Processing (Almost from Scratch)](http://jmlr.org/papers/volume12/collobert11a/collobert11a.pdf){:target="_blank"}

[Convolutional Neural Networks for Sentence Classification](https://arxiv.org/pdf/1408.5882.pdf){:target="_blank"}

<figure>
    <img src="https://raw.githubusercontent.com/KamranMK/kamranmk.github.io/master/images/coursera-nlp/coursera-nlp-w1-26.png">
    <figcaption>NLP Coursera - Week 1 - Final Architecture
    </figcaption>
</figure>



# Neural networks for characters

