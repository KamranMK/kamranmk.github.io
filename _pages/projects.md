---
layout: archive
permalink: /projects/
title: "Data Science Projects"
excerpt: "Share and learn together"
author_profile: true
---


{% include group-by-array category=site.posts field="tags" %}

{% for post in site.categories.projects %}
  {% include archive-single.html %}
{% endfor %}

{% for tag in group_names %}
  {% assign posts = group_items[forloop.index0] %}
  <h2 id="{{ tag | slugify }}" class="archive__subtitle">{{ tag }}</h2>
  {% for post in posts %}
    {% include archive-single.html %}
  {% endfor %}
{% endfor %}