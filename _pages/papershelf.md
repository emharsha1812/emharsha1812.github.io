---
layout: page
title: Library
permalink: /papershelf/
description: Books that have shaped my thinking — technical and otherwise.
nav: true
nav_order: 7
---

{% assign non_technical = site.data.bookshelf | where: "category", "non-technical" %}
{% assign technical = site.data.bookshelf | where_exp: "item", "item.category != 'non-technical'" %}

<section class="bookshelf">
  <div class="bookshelf-header">
    <h2>Non-Technical Books</h2>
    <p>Whatever I am, I am because of them</p>
  </div>
  <div class="bookshelf-grid">
    {% for book in non_technical %}
    <article class="bookshelf-card">
      {% if book.thumbnail %}<img class="book-thumb" src="{{ book.thumbnail }}" alt="Cover of {{ book.title }}" loading="lazy" />{% endif %}
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

<hr class="shelf-divider" />

<section class="bookshelf">
  <div class="bookshelf-header">
    <h2>Technical Books</h2>
    <p>ML, systems, and engineering books I've read or am reading</p>
  </div>
  <div class="bookshelf-grid">
    {% for book in technical %}
    <article class="bookshelf-card">
      {% if book.thumbnail %}<img class="book-thumb" src="{{ book.thumbnail }}" alt="Cover of {{ book.title }}" loading="lazy" />{% endif %}
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

<style>
.shelf-divider {margin: 2.5rem 0; border: 0; border-top: 1px solid var(--global-divider-color);}
.bookshelf-header {margin-bottom: 1.25rem;}
.bookshelf-grid {display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 1.1rem; margin-bottom: 1.5rem;}
.bookshelf-card {display: flex; gap: .85rem; padding: 1rem; border: 1px solid var(--global-divider-color); border-radius: 10px; background: var(--global-card-bg, var(--global-bg-color)); box-shadow: var(--global-shadow, 0 8px 30px rgba(0,0,0,.05)); color: var(--global-text-color);}
.book-thumb {width: 80px; height: 110px; object-fit: cover; border-radius: 6px; box-shadow: inset 0 0 1px rgba(0,0,0,.2);}
.book-body {flex: 1; display: flex; flex-direction: column; gap: .4rem;}
.book-meta {margin: 0; font-size: .85rem; color: var(--global-text-color-light, #aaa);}
.book-author {margin: 0; font-size: .9rem; color: var(--global-text-color);}
.book-link {font-weight: 600;}
.bookshelf-card h3 {margin: 0; font-size: 1rem;}
</style>
