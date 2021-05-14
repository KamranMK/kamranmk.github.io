---
title: "Exploring NLP with Coursera - Part 4"
date: 2020-07-31
tags: [data science learning, natural language processing, nlp, coursera]
category: [data-science-learning]
classes: full-wide
author_profile: false
excerpt: "Week 2: Part 1 - Language Modeling: It's all about counting!"
toc: True
toc_label: "Contents"
toc_sticky: true
---

During Week 1, I realised that taking notes that mimic the content doesn't make sense as much, thus for Week 2, the notes cover only highlights and important outcomes.

# Basics of Language Modeling

Let's say we have the text "This is the..." and 4 possible choices

* house
* rat
* did
* malt

For humans, this might not be a trivial question and we could say that for example did doesn't really fit here in any context. For machines, its a bit more challenging.

We can help machines to "understand" better using probability theory. For example, what is the probability of seeing the word 'house' given 'this is the' or 

$$p(house|this\ is\ the)$$

In terms of language modeling we can also ask the probability of the whole sequence. Given our training data how likely are we to see a specific sequence. For example

$$p(this\ is\ the\ house)$$

Mathematically, we want to predict the probability of a sequence of words *This is the house*, which we can annotate as

$$w=(w_1w_2w_3...w_k)$$

Thus the probability of it would be

$$p(w)=p(w_1w_2w_3...w_k)$$

Using the **Chain Rule** we can rewrite the equation as such

$$p(w)\ =\ p(w_1)p(w_2|w_1)...p(w_k|w_1...w_{k-1})$$

Now the challenge here is that the list of probabilities will explode exponentially making it almost impossible to estimate probabilities. Nevertheless, we can use the Markov Assumption to only take the last *n-1* terms. So if we want to look at *trigrams* then we would condition our probabities on last 2 items in the sequence.

Using Markov Assumption we get

$$p(w_i|w_1...w_{i-1})=p(w_i|w_{i-n+1}...w_{i-1})$$

Now, if we use the following probability calculation to our individual probabilities won't add to one and for that we need to add additional start and end tokens to answer the question what is the probability of seeing specific word given its the start and end of the sequence.

$$p(w_1|start)$$

This helps us normalize the probabilities for each sequence.

Thus at the for a n-gram Language Model we have the following representation

$$p(w)=\prod_{i=1}^{k+1}p(w_i|w_{i-n+1}^{i-1})$$

<figure>
    <img src="https://raw.githubusercontent.com/KamranMK/kamranmk.github.io/master/images/coursera-nlp/coursera-nlp-w1-32.png">
    <figcaption>NLP Coursera - Week 2 - N-Gram Language Model
    </figcaption>
</figure>

and the model estimate is simply the countings for those word representations

<figure>
    <img src="https://raw.githubusercontent.com/KamranMK/kamranmk.github.io/master/images/coursera-nlp/coursera-nlp-w1-33.png">
    <figcaption>NLP Coursera - Week 2 - N-Gram Language Model Estimates
    </figcaption>
</figure>

To train the N-gram Language Model, we simply maximize the log likelihood

<figure>
    <img src="https://raw.githubusercontent.com/KamranMK/kamranmk.github.io/master/images/coursera-nlp/coursera-nlp-w1-34.png">
    <figcaption>NLP Coursera - Week 2 - Log-likelihood maximization
    </figcaption>
</figure>

The large N represents the length of the train corpus (all words concatenated)

# Smoothing: Dealing with un-seen n-grams

Now there will be situations where in the test data we observe text (n-grams) that is not present in the training data, essentially leading to zero probabilities. To deal with these situations, various smoothing techniques could be used. 

Essentially smoothing tries to pull some probability from frequent n-grams to infrequent ones.

Let's review various smoothing techniques. This section is pretty straight forward since it simply summarizes various approaches.

## Laplacian Smoothing

<figure>
    <img src="https://raw.githubusercontent.com/KamranMK/kamranmk.github.io/master/images/coursera-nlp/coursera-nlp-w1-35.png">
    <figcaption>NLP Coursera - Week 2 - Laplacian Smoothing
    </figcaption>
</figure>


## Katz Backoff

<figure>
    <img src="https://raw.githubusercontent.com/KamranMK/kamranmk.github.io/master/images/coursera-nlp/coursera-nlp-w1-36.png">
    <figcaption>NLP Coursera - Week 2 - Katz Backoff
    </figcaption>
</figure>

Here and *p* and *alpha* are chosen to ensure normalization.

## Interpolation Smoothing

<figure>
    <img src="https://raw.githubusercontent.com/KamranMK/kamranmk.github.io/master/images/coursera-nlp/coursera-nlp-w1-37.png">
    <figcaption>NLP Coursera - Week 2 - Interpolation Smoothing
    </figcaption>
</figure>

## Absolute Discounting

The approach was developed as a result of the experiment made by Church & Gale in 1991. Experiment [available here](https://web.stanford.edu/~jurafsky/slp3/4.pdf){:target="_blank"}.

<figure>
    <img src="https://raw.githubusercontent.com/KamranMK/kamranmk.github.io/master/images/coursera-nlp/coursera-nlp-w1-38.png">
    <figcaption>NLP Coursera - Week 2 - Absolute Discounting
    </figcaption>
</figure>

<figure>
    <img src="https://raw.githubusercontent.com/KamranMK/kamranmk.github.io/master/images/coursera-nlp/coursera-nlp-w1-39.png">
    <figcaption>NLP Coursera - Week 2 - Absolute Discounting Formula
    </figcaption>
</figure>

## Kneser-Ney Smoothing

This smoothing technique is the most popular smoothing technique mostly because it captures the diversity of contexts for the word. Example here is that the word Kong is less likely to occur with *This is the...* because it only goes with Hong Kong but the word malt can be seen in various context, thus more likely to be preceded by *This is the*.

<figure>
    <img src="https://raw.githubusercontent.com/KamranMK/kamranmk.github.io/master/images/coursera-nlp/coursera-nlp-w1-39.png">
    <figcaption>NLP Coursera - Week 2 - Kneser Ney Smoothing
    </figcaption>
</figure>

## Perplexity Computation

Perplexity is a popular quality measure of language models. We can calculate it using the formula:

<figure>
    <img src="https://raw.githubusercontent.com/KamranMK/kamranmk.github.io/master/images/coursera-nlp/coursera-nlp-w1-40.png">
    <figcaption>NLP Coursera - Week 2 - Perplexity Formula
    </figcaption>
</figure>

There is a nice short article on what Perplexity [here](https://towardsdatascience.com/perplexity-intuition-and-derivation-105dd481c8f3){:target="_blank"}.