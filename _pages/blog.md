---
layout: default
permalink: /blog/
title: writings
nav: true
nav_order: 1
pagination:
  enabled: true
  collection: posts
  permalink: /page/:num/
  per_page: 5
  sort_field: date
  sort_reverse: true
  trail:
    before: 1 # The number of links before the current page
    after: 3 # The number of links after the current page
---

<div class="post">

{% assign course_posts = site.posts | where_exp: "post", "post.series" %}
{% assign course_series = course_posts | map: "series" | uniq | sort %}
{% if course_series.size > 0 and paginator.page == 1 %}
  <section id="writing-courses" class="writing-courses mb-5" aria-labelledby="writing-courses-title">
    <div class="d-flex align-items-baseline justify-content-between flex-wrap">
      <h2 id="writing-courses-title">Courses</h2>
      <a href="{{ '/courses/' | relative_url }}">View course index&nbsp; →</a>
    </div>
    <p class="text-muted">Structured learning series, separate from the one-off writings below.</p>
    {% for series_name in course_series %}
      {% assign series_posts = course_posts | where: "series", series_name | sort: "series_part" %}
      {% assign first_post = series_posts | first %}
      <article class="course-block mb-4">
        <h3>{{ first_post.series_title | default: series_name }}</h3>
        {% if first_post.series_description %}
          <p>{{ first_post.series_description }}</p>
        {% endif %}
        <ol>
          {% for post in series_posts %}
            <li><a href="{{ post.url | relative_url }}">{{ post.title }}</a>{% if post.description %} <span class="text-muted">— {{ post.description }}</span>{% endif %}</li>
          {% endfor %}
        </ol>
      </article>
    {% endfor %}
  </section>
{% endif %}

<p class="writing-external-links">You can find more of my writing on <a href="https://harshwardhanfartale.substack.com/" target="_blank" rel="noopener">Substack</a> and <a href="https://medium.com/@emharsha1812" target="_blank" rel="noopener">Medium</a>.</p>

<form id="blog-search" class="blog-search" role="search">
  <label for="blog-search-input" class="sr-only">Search writings</label>
  <div class="blog-search-field">
    <i class="fa-solid fa-magnifying-glass" aria-hidden="true"></i>
    <input id="blog-search-input" type="search" placeholder="Search writings..." autocomplete="off" spellcheck="false">
  </div>
  <p id="blog-search-status" class="text-muted" role="status" aria-live="polite"></p>
</form>

<style>
.blog-search { margin: 1.25rem 0 2rem; }
.blog-search-field { display: flex; align-items: center; gap: .75rem; padding: .7rem 1rem; border: 1px solid var(--global-divider-color); border-radius: 8px; background: var(--global-card-bg, var(--global-bg-color)); }
.blog-search-field i { color: var(--global-text-color-light, #888); }
.blog-search-field input { flex: 1; min-width: 0; border: 0; outline: 0; background: transparent; color: var(--global-text-color); font-size: 1rem; }
.blog-search-status { margin: .6rem 0 0; min-height: 1.25rem; }
</style>

  <ul id="blog-post-list" class="post-list">

    {% if page.pagination.enabled %}
      {% assign postlist = paginator.posts | where_exp: "post", "post.series == nil" %}
    {% else %}
      {% assign postlist = site.posts | where_exp: "post", "post.series == nil" %}
    {% endif %}

    {% for post in postlist %}

    {% if post.external_source == blank %}
      {% assign read_time = post.content | number_of_words | divided_by: 180 | plus: 1 %}
    {% else %}
      {% assign read_time = post.feed_content | strip_html | number_of_words | divided_by: 180 | plus: 1 %}
    {% endif %}
    {% assign year = post.date | date: "%Y" %}
    {% assign tags = post.tags | join: "" %}
    {% assign categories = post.categories | join: "" %}

    <li>

{% if post.thumbnail %}

<div class="row">
          <div class="col-sm-9">
{% endif %}
        <h3>
        {% if post.redirect == blank %}
          <a class="post-title" href="{{ post.url | relative_url }}">{{ post.title }}</a>
        {% elsif post.redirect contains '://' %}
          <a class="post-title" href="{{ post.redirect }}" target="_blank">{{ post.title }}</a>
          <svg width="2rem" height="2rem" viewBox="0 0 40 40" xmlns="http://www.w3.org/2000/svg">
            <path d="M17 13.5v6H5v-12h6m3-3h6v6m0-6-9 9" class="icon_svg-stroke" stroke="#999" stroke-width="1.5" fill="none" fill-rule="evenodd" stroke-linecap="round" stroke-linejoin="round"></path>
          </svg>
        {% else %}
          <a class="post-title" href="{{ post.redirect | relative_url }}">{{ post.title }}</a>
        {% endif %}
      </h3>
      <p>{{ post.description }}</p>
      <p class="post-meta">
        {{ read_time }} min read &nbsp; &middot; &nbsp;
        {{ post.date | date: '%B %d, %Y' }}
        {% if post.external_source %}
        &nbsp; &middot; &nbsp; {{ post.external_source }}
        {% endif %}
      </p>
      <p class="post-tags">
        <a href="{{ year | prepend: '/blog/' | prepend: site.baseurl}}">
          <i class="fa-solid fa-calendar fa-sm"></i> {{ year }} </a>

          {% if tags != "" %}
          &nbsp; &middot; &nbsp;
            {% for tag in post.tags %}
            <a href="{{ tag | slugify | prepend: '/blog/tag/' | prepend: site.baseurl}}">
              <i class="fa-solid fa-hashtag fa-sm"></i> {{ tag }}</a>
              {% unless forloop.last %}
                &nbsp;
              {% endunless %}
              {% endfor %}
          {% endif %}

          {% if categories != "" %}
          &nbsp; &middot; &nbsp;
            {% for category in post.categories %}
            <a href="{{ category | slugify | prepend: '/blog/category/' | prepend: site.baseurl}}">
              <i class="fa-solid fa-tag fa-sm"></i> {{ category }}</a>
              {% unless forloop.last %}
                &nbsp;
              {% endunless %}
              {% endfor %}
          {% endif %}
    </p>

{% if post.thumbnail %}

</div>

  <div class="col-sm-3">
    <img class="card-img" src="{{post.thumbnail | relative_url}}" style="object-fit: cover; height: 90%" alt="image">
  </div>
</div>
{% endif %}
    </li>

    {% endfor %}

  </ul>

{% assign writing_posts = site.posts | where_exp: "post", "post.series == nil" %}
<ul id="blog-search-results" class="post-list" hidden></ul>

<script type="application/json" id="blog-search-index">
[
  {% for post in writing_posts %}
    {
      "title": {{ post.title | jsonify }},
      "description": {{ post.description | default: "" | jsonify }},
      "content": {{ post.content | strip_html | strip_newlines | jsonify }},
      "url": {{ post.url | relative_url | jsonify }},
      "date": {{ post.date | date: "%B %d, %Y" | jsonify }}
    }{% unless forloop.last %},{% endunless %}
  {% endfor %}
]
</script>
<script src="{{ '/assets/js/blog-search.js' | relative_url | bust_file_cache }}"></script>

{% if page.pagination.enabled %}
<div id="blog-pagination">
  {% include pagination.liquid %}
</div>
{% endif %}

</div>
