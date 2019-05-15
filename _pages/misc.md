---
layout: archive
permalink: /misc/
title: "Miscellaneous"
excerpt: "Tips, tricks, quick fixes"
author_profile: true
header:
    overlay_image: "/images/miscellaneous.jpg"
    overlay_filter: 0.5
    caption: "Photo credit: [**Unsplash**](https://unsplash.com)"
---

{% include group-by-array category=site.posts field="tags" %}

{% for post in site.categories.misc %}
  {% include archive-single.html %}
{% endfor %}

{% for tag in group_names %}
  {% assign posts = group_items[forloop.index0] %}
  <h2 id="{{ tag | slugify }}" class="archive__subtitle">{{ tag }}</h2>
  {% for post in posts %}
    {% include archive-single.html %}
  {% endfor %}
{% endfor %}