---
layout: page
title: "Notes"
permalink: "/notes/"
---

This **microblogging** page contains my daily thoughts and findings as
well as interesting news I find everyday. This is something that I
could have in [**Twitter**](https://twitter.com/unmonoqueteclea) or
[**Mastodon**](https://mastodon.social/@unmonoqueteclea) (actually, I
will try to automatically publish the same content there), but I want
to have *the control of my own content*. My URL, my rules 📣. Check
more [here](../2020/hello-world-again/#microblogging)

In this way, you can subscribe to my [**microblogging
 RSS**](../journal_feed.xml) 📮 and read everything in beautiful ❤
 plain text ❤. You could even read it from
 [**emacs**](https://www.gnu.org/software/emacs/) with
 [**elfeed**](https://github.com/skeeto/elfeed)!


<ul>
  {% for note in site.notes reversed%}
    <li>
      <a href="{{ note.url }}">
      {{note.date | date: "%-d %B %Y" }}
       -
      {{ note.title }}
      </a>
      <div>
      {% for tag in note.tags %}
      <span class="tag-label">{{tag}}</span>
      {% endfor %}
      </div>
      <div>
          {{ note.content }}
      </div>
    </li>
  {% endfor %}
</ul>
