---
layout: page
permalink: /frompapertocode/
title: Paper2code
description: A collection of paper-to-code implementations. This page serves as a testament to my skills in deciphering research papers and translating theoretical concepts into functional code.
nav: true
nav_order: 6
---

<div class="paper2code">
{% if site.enable_project_categories and page.display_categories %}
  <!-- Display categorized papers2code -->
  {% for category in page.display_categories %}
  <a id="{{ category }}" href=".#{{ category }}">
    <h2 class="category">{{ category }}</h2>
  </a>
  {% assign categorized_papers = site.papers2code | where: "category", category %}
  {% assign sorted_papers = categorized_papers | sort: "importance" %}
  <!-- Generate cards for each paper2code -->
  {% if page.horizontal %}
  <div class="container">
    <div class="row row-cols-1 row-cols-md-2">
    {% for paper in sorted_papers %}
      {% include papers2code_horizontal.liquid %}
    {% endfor %}
    </div>
  </div>
  {% else %}
  <div class="row row-cols-1 row-cols-md-3">
    {% for paper in sorted_papers %}
      {% include papers2code.liquid %}
    {% endfor %}
  </div>
  {% endif %}
  {% endfor %}

{% else %}
  <!-- Display papers2code without categories -->
  {% assign sorted_papers = site.papers2code | sort: "importance" %}
  
  {% if page.horizontal %}
  <div class="container">
    <div class="row row-cols-1 row-cols-md-2">
    {% for paper in sorted_papers %}
      {% include papers2code_horizontal.liquid %}
    {% endfor %}
    </div>
  </div>
  {% else %}
  <div class="row row-cols-1 row-cols-md-3">
    {% for paper in sorted_papers %}
      {% include papers2code.liquid %}
    {% endfor %}
  </div>
  {% endif %}
{% endif %}
</div>
