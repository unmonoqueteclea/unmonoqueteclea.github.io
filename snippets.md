---
layout: page
title: "Notes"
permalink: "/snippets/"
---

This page contains contains all the snippets published.


<ul>
  {% for snippet in site.snippets reversed%}
    <li>
      <a href="{{ snippet.url }}">
      {{snippet.date | date: "%-d %B %Y" }}
       -
      {{ snippet.title }}
      </a>
      <div>
      {% for tag in snippet.tags %}
      <span class="tag-label">{{tag}}</span>
      {% endfor %}
      </div>
      <div>
          {{ snippet.content }}
      </div>
    </li>
  {% endfor %}
</ul>
