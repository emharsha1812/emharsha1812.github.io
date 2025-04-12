---
layout: page
permalink: /stats/
title: stats
description:
nav: true
nav_order: 4
---

Visit my [Github](https://github.com/emharsha1812)

{% if site.data.repositories.github_users %}

### Leetcode stats
![Leetcode Stats](https://leetcard.jacoblin.cool/harshwardhanfartale_nith)

### GeeksForGeeks stats
[![GeeksForGeeks stats](https://gfgstatscard.vercel.app/harshwardhan8ljg)](https://www.geeksforgeeks.org/user/harshwardhan8ljg)


### GitHub users

<div class="repositories d-flex flex-wrap flex-md-row flex-column justify-content-between align-items-center">
  {% for user in site.data.repositories.github_users %}
    {% include repository/repo_user.liquid username=user %}
  {% endfor %}
</div>

---

{% if site.repo_trophies.enabled %}
{% for user in site.data.repositories.github_users %}
{% if site.data.repositories.github_users.size > 1 %}

  <h4>{{ user }}</h4>
  {% endif %}
  <div class="repositories d-flex flex-wrap flex-md-row flex-column justify-content-between align-items-center">
  {% include repository/repo_trophies.liquid username=user %}
  </div>

---

{% endfor %}
{% endif %}
{% endif %}

{% if site.data.repositories.github_repos %}

### GitHub Repositories

<div class="repositories d-flex flex-wrap flex-md-row flex-column justify-content-between align-items-center">
  {% for repo in site.data.repositories.github_repos %}
    {% include repository/repo.liquid repository=repo %}
  {% endfor %}
</div>
{% endif %}
