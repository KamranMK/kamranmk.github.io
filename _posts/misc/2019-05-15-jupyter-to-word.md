---
title: "Convert Jupyter Notebook to Word"
date: 2019-05-15
tags: [jupyter, word document, convert]
category: [misc]
author_profile: true
classes: wide
excerpt: "Short guide on how to convert .ipynb file to .docx."
comments: True
---

During my Data Mining course at the KU Leuven university, we were asked to present our data analysis project in the form of a word document. The justification behind this was that this was the most convenient form for the teacher to provide feedback to our analysis. As we all usually do, I googled 'How to convert jupyter notebook file (ipynb) to word'. To my surprise there were not many posts describing this.

The best suggestion I found was this one [Using Jupyter Notebooks For Assessment – Export as Word (.docx) Extension](https://blog.ouseful.info/2017/06/13/using-jupyter-notebooks-for-assessment-export-as-word-docx-extension/){:target="_blank"}. The only challenge with this post is that its a bit confusing and cumbersome since it requires you to deal with an extension. Despite that, in the code provided in the post one only needs two commands to run in order to convert the jupyter notebook file (ipynb) to word (docx).

Before that, make sure you have [jupyter nbconvert](https://nbconvert.readthedocs.io/en/latest/usage.html){:target="_blank"} and [pandoc](https://pandoc.org/getting-started.html#){:target="_blank"} set up.

To convert the jupyter notebook to word, first we need to convert the notebook to `html`. You can perform these commands from `cmd`, `powershell` or terminal on mac.

{% highlight terminal %}
jupyter nbconvert --to html my_notebook.ipynb
{% endhighlight %}

Then we need to convert `html` file generated to word (docx)

{% highlight terminal %}
pandoc -s my_notebook.html -o my_notebook.docx
{% endhighlight %}

As a result, we should have a nicely formatted word document ready for consumption.