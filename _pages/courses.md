---
layout: page
title: courses
permalink: /courses/
description: Structured courses and learning series.
nav: true
nav_order: 2
---

{% assign series_posts_all = site.posts | where_exp: "post", "post.series" %}
{% assign all_series = series_posts_all | map: "series" | uniq | sort %}

{% if all_series.size == 0 %}
<p>No courses yet. Check back soon!</p>
{% else %}
{% for series_name in all_series %}
{% assign series_posts = site.posts | where: "series", series_name | sort: "series_part" %}
{% assign first_post = series_posts | first %}
<div class="course-block mb-5">
<h2>{{ first_post.series_title | default: series_name }}</h2>
{% if first_post.series_description %}
<p class="text-muted">{{ first_post.series_description }}</p>
{% endif %}
<ol>
{% for post in series_posts %}
<li style="margin-bottom: 0.5rem;"><a href="{{ post.url | relative_url }}">{{ post.title }}</a>{% if post.description %} <span class="text-muted">— {{ post.description }}</span>{% endif %}</li>
{% endfor %}
</ol>
</div>
{% unless forloop.last %}<hr>{% endunless %}
{% endfor %}
{% endif %}
