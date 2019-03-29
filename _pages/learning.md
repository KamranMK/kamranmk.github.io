---
layout: category
permalink: /data-science-learning/
title: "Data Science Learning"
author_profile: true
header:
    image: "/images/ds_learning.jpg"
---

{% include group-by-array category=site.posts field="tags" %}

{% for post in site.categories.data-science-learning %}
  {% include archive-single.html %}
{% endfor %}

{% for tag in group_names %}
  {% assign posts = group_items[forloop.index0] %}
  <h2 id="{{ tag | slugify }}" class="archive__subtitle">{{ tag }}</h2>
  {% for post in posts %}
    {% include archive-single.html %}
  {% endfor %}
{% endfor %}