---
layout: page
title: Library
permalink: /papershelf/
description: Curated reading list with quick filters and links to notes.
nav: true
nav_order: 7
---

<div class="papershelf-header">
  <p>A living bookshelf of papers, kernels, and system guides that influence my research. Use the search box to filter by title, author, venue, or tag.</p>
  <input id="paper-search" class="papershelf-search" type="search" placeholder="Search papers" aria-label="Search papers" />
</div>

{% assign all_tags = "" | split: "" %}
{% for paper in site.data.papershelf %}
  {% if paper.tags %}
    {% assign all_tags = all_tags | concat: paper.tags %}
  {% endif %}
{% endfor %}
{% assign unique_tags = all_tags | uniq | sort %}

<div class="papershelf-filters" role="toolbar" aria-label="Filter papers by tag">
  <button type="button" class="filter-btn active" data-tag="all">All</button>
  {% for tag in unique_tags %}
  <button type="button" class="filter-btn" data-tag="{{ tag | downcase }}">{{ tag }}</button>
  {% endfor %}
</div>

<div id="papershelf" class="papershelf-grid">
  {% for paper in site.data.papershelf %}
  <article class="papershelf-card" data-title="{{ paper.title | downcase }}" data-authors="{{ paper.authors | downcase }}" data-tags="{{ paper.tags | join: ' ' | downcase }}">
    <header>
      <p class="paper-meta">{{ paper.venue }} · {{ paper.year }}</p>
      <h3>{{ paper.title }}</h3>
      <p class="paper-authors">{{ paper.authors }}</p>
    </header>
    {% if paper.summary %}
    <p class="paper-summary">{{ paper.summary }}</p>
    {% endif %}
    {% if paper.tags %}
    <p class="paper-tags">{% for tag in paper.tags %}<span class="paper-tag">{{ tag }}</span>{% endfor %}</p>
    {% endif %}
    <p class="paper-links">
      {% if paper.pdf %}<a href="{{ paper.pdf }}" target="_blank" rel="noopener">PDF</a>{% endif %}
      {% if paper.url and paper.url != paper.pdf %}<span>·</span><a href="{{ paper.url }}" target="_blank" rel="noopener">Page</a>{% endif %}
    </p>
  </article>
  {% endfor %}
</div>

{% if site.data.bookshelf %}
<hr class="shelf-divider" />

<section class="bookshelf">
  <div class="bookshelf-header">
    <h2>Bookshelf</h2>
    <p>Reference handbooks and favorite reads that complement the papers above. Covers are optional—add a thumbnail path in <code>_data/bookshelf.yml</code> to display one.</p>
  </div>
  <div class="bookshelf-grid">
    {% for book in site.data.bookshelf %}
    <article class="bookshelf-card">
      {% if book.thumbnail %}
      <img class="book-thumb" src="{{ book.thumbnail | relative_url }}" alt="Cover of {{ book.title }}" loading="lazy" />
      {% endif %}
      <div class="book-body">
        <p class="book-meta">{{ book.publisher }} · {{ book.year }}</p>
        <h3>{{ book.title }}</h3>
        <p class="book-author">{{ book.author }}</p>
        {% if book.url %}<a class="book-link" href="{{ book.url }}" target="_blank" rel="noopener">View</a>{% endif %}
      </div>
    </article>
    {% endfor %}
  </div>
</section>
{% endif %}

<style>
.papershelf-header {margin-bottom: 1rem;}
.papershelf-search {width: 100%; padding: 0.65rem 0.8rem; border: 1px solid var(--global-divider-color); border-radius: 6px; font-size: 1rem;}
.papershelf-filters {display: flex; flex-wrap: wrap; gap: .5rem; margin-bottom: 1.25rem;}
.filter-btn {border: 1px solid var(--global-divider-color); background: var(--global-bg, #f7f7fb); color: var(--global-text, #444); padding: .35rem .9rem; border-radius: 999px; font-size: .9rem; cursor: pointer; transition: all .15s ease;}
.filter-btn:hover {border-color: var(--global-link-color); color: var(--global-link-color);}
.filter-btn.active {background: var(--global-link-color); color: #fff; border-color: var(--global-link-color);}
.papershelf-grid {display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 1.25rem;}
.papershelf-card {padding: 1.25rem; border: 1px solid var(--global-divider-color); border-radius: 10px; background: var(--global-card-bg, var(--global-bg-color)); box-shadow: var(--global-shadow, 0 8px 30px rgba(0,0,0,.05)); display: flex; flex-direction: column; gap: .75rem; color: var(--global-text-color);} 
.papershelf-card h3 {margin: 0; font-size: 1.1rem;}
p.paper-meta {margin: 0; font-size: .9rem; color: var(--global-text-color-light, #999);}
.paper-authors {margin: 0; font-size: .9rem; color: var(--global-text-color, #ccc);}
.paper-summary {margin: 0; font-size: .95rem; color: var(--global-text-color);} 
.paper-tags {margin: 0; display: flex; flex-wrap: wrap; gap: .3rem;}
.paper-tag {background: var(--global-highlight, rgba(79, 70, 229, 0.15)); color: var(--global-text-color); padding: .15rem .45rem; border-radius: 999px; font-size: .8rem; text-transform: lowercase;}
.paper-links {margin: 0; font-size: .9rem;}
.paper-links a {font-weight: 600; color: var(--global-link-color, #4c7ef3);}
.shelf-divider {margin: 2.5rem 0; border: 0; border-top: 1px solid var(--global-divider-color);}
.bookshelf-header {margin-bottom: 1.25rem;}
.bookshelf-grid {display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 1.1rem;}
.bookshelf-card {display: flex; gap: .85rem; padding: 1rem; border: 1px solid var(--global-divider-color); border-radius: 10px; background: var(--global-card-bg, var(--global-bg-color)); box-shadow: var(--global-shadow, 0 8px 30px rgba(0,0,0,.05)); color: var(--global-text-color);} 
.book-thumb {width: 80px; height: 110px; object-fit: cover; border-radius: 6px; box-shadow: inset 0 0 1px rgba(0,0,0,.2);}
.book-body {flex: 1; display: flex; flex-direction: column; gap: .4rem;}
.book-meta {margin: 0; font-size: .85rem; color: var(--global-text-color-light, #aaa);}
.book-author {margin: 0; font-size: .9rem; color: var(--global-text-color);}
.book-link {font-weight: 600;}
.bookshelf-card h3 {margin: 0; font-size: 1rem;}
</style>

<script src="{{ '/assets/js/papershelf.js' | relative_url }}" defer></script>
