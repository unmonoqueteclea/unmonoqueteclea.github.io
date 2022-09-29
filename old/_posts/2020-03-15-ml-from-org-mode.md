---
title:  "Machine Learning from org mode"
date: 2020-03-15
tags: [emacs]
---

As a Machine Learning engineer at [BigML](https://bigml.com/), I use
[Jupyter Notebooks](https://jupyter.org/) a lot. Although there are
some people creating great Python libraries fully written in notebooks
(see the [great work done by Jeremy
Howard](https://www.fast.ai/2019/12/02/nbdev)) I only use notebooks
for early stage experimenting (actually, I spend many hours every week
experimenting with data). For me, notebooks are a great tool to show
my work, but they were not created as a way to implement and organize
software (see the slides from this [controversial and very interesting
talk at
JupyterCon](https://docs.google.com/presentation/d/1n2RlMdmv1p25Xy5thJUhkKGvjtV-dkAIsUXP-AL4ffI))


![I don't like notebooks](/assets/gifs/notebooks.gif)


I have been exploring, for some months, different ways of doing
things. Emacs is becoming my Operative System and I am starting to
live in a **plain text world**. I use **org mode** to take notes about
absolutely everything,
[org-journal](https://github.com/bastibe/org-journal) to mantain my
personal dialy journal, [elfeed](https://github.com/skeeto/elfeed) to
read RSS feeds, [emacs-slack](https://github.com/yuya373/emacs-slack)
as Slack client, and many more examples that I will write about in a
future blog post.

Sadly, Jupyter Notebooks don't fit well in this new life philosophy. I
realized that people had been doing [literate
programming](https://invidio.us/watch?v=bTkXg2LZIMQ&local=1&nojs=0&player_style=youtube)
long time before it *became cool*, and Jupyter Notebooks are not the only
way to do it. I knew
**[org-babel](https://orgmode.org/worg/org-contrib/babel/)**, that lets
you execute source code within org mode documents, but I wasn't sure
if it would be the right alternative to notebooks. And then I found
these two posts ([literate
programming](https://jao.io/blog/2020-02-26-literate-programming.html)
and [literate programming against rest
apis](https://justinbarclay.me/posts/literate_programming_against_rest_apis/))
about **literate programming in Emacs** and they really opened my eyes.

In this post, I will do **Machine Learning from an org file using
BigML's API**. [BigML](https://bigml.com/) is a Machine Learning As A
Service platform that removes the complexities of Machine Learning so
you can focus on what matters most, enhancing and automating decision
making (I already talked about BigML
[here](https://unmonoqueteclea.github.io/2018/bigml/)) I will use its
**REST API** to create predictive models.

You can download the original **[org
document](/assets/org/ml-from-org-mode.org)** and play with it!

To make HTTP requests from org mode, I will use
**[restclient](https://github.com/pashky/restclient.el)** and an
extension to it that provides org-babel support,
**[ob-restclient](https://github.com/alf/ob-restclient.el)**.

In the file header I will configure the **url and
credentials** to use [BigML's API](https://bigml.com/api) (you can
access the latest full-featured version of BigML with a [free
account](https://bigml.com/accounts/register/))


    #+TITLE:     Machine Learning from org-mode
    #+AUTHOR:    Pablo González Carrizo
    #+PROPERTY:  header-args:restclient :var url="https://bigml.io"
                 :var auth ="?username=YOUR_USERNAME&api_key=YOUR_API_KEY"

Let's explore the data. I will work with the all-known [iris
dataset](https://archive.ics.uci.edu/ml/datasets/Iris). I will use a
**bash** source block to show the 10 first lines of the csv file:

```sh
#+BEGIN_SRC bash
curl "https://static.bigml.com/csv/iris.csv" |  head -10
#+END_SRC
```

This is what it returns:

```sh
    #+RESULTS:
    | sepal length | sepal width | petal length | petal width | species     |
    |          5.1 |         3.5 |          1.4 |         0.2 | Iris-setosa |
    |          4.9 |         3.0 |          1.4 |         0.2 | Iris-setosa |
    |          4.7 |         3.2 |          1.3 |         0.2 | Iris-setosa |
    |          4.6 |         3.1 |          1.5 |         0.2 | Iris-setosa |
    |          5.0 |         3.6 |          1.4 |         0.2 | Iris-setosa |
    |          5.4 |         3.9 |          1.7 |         0.4 | Iris-setosa |
    |          4.6 |         3.4 |          1.4 |         0.3 | Iris-setosa |
    |          5.0 |         3.4 |          1.5 |         0.2 | Iris-setosa |
    |          4.4 |         2.9 |          1.4 |         0.2 | Iris-setosa |

```

The response to the previous command is shown as an **org table**, so I
can use all the [available
commands](https://orgmode.org/manual/Tables.html) for tables in org
mode, and even use
[org mode spreadsheet](https://orgmode.org/worg/org-tutorials/org-spreadsheet-intro.html) capabilities!

With simple bash utilities, I can print the number of rows of the csv
file:

```sh
 #+BEGIN_SRC bash
curl "https://static.bigml.com/csv/iris.csv" |  wc -l
#+END_SRC
```

```sh
#+RESULTS:
: 151
```

Now, let's upload the data to BigML. These are the steps we should
follow:

- Upload the file as a **source** to BigML.
- Create a **dataset** from the **source** (BigML will analyze all the
  content from the original CSV file and prepare it for modelling
  tasks).
- Create a **predictive model** from our dataset.
- Upload new data to obtain **predictions** from the model.


Let's create the **source** with a simple **HTTP POST request**. The
response will be stored in a file called **source.json**

```sh
    #+NAME: source
    #+BEGIN_SRC restclient :file source.json
    POST :url/source/:auth
    Content-Type: application/json
    {"remote": "https://static.bigml.com/csv/iris.csv"}
    #+END_SRC
```


Before creating the **dataset**, let's define a little helper block
(inspired by [this blog
post](https://justinbarclay.me/posts/literate_programming_against_rest_apis/))
that will clean the responses of **restclient** requests, parse the
**BigML's API JSON responses** and extract the id of the resource. All
the resources generated by **BigML's API** contain a *resource* field
with the id of the resource. I will use
[elisp](https://www.gnu.org/software/emacs/manual/html_node/elisp/) to
define this function, but we could have used whatever language we feel
comfortable with.


    #+NAME: resource
    #+BEGIN_SRC emacs-lisp :var path=""

```elisp
  (require 'json)
  (let* ((json-string (string-trim
                       (replace-regexp-in-string
                        "^#\\+BEGIN_SRC js\\|^#\\+END_SRC\\|^//[[:print:]]+"
                        ""
                        (with-temp-buffer
                          (insert-file-contents path)
                          (buffer-string)))))
         (json-object-type 'hash-table)
         (json-array-type 'list)
         (json-key-type 'string)
         (json (json-read-from-string json-string)))
    (gethash "resource" json))
```

    #+END_SRC

From the **source**, we will create a **dataset**. We are using the
**elisp** utility defined previously to extract the **source id** from
the source created previously. See how easy this is:

```sh
    #+BEGIN_SRC restclient :var id=resource("source.json") :file dataset.json
    POST :url/dataset/:auth
    Content-Type: application/json
    {"source": ":id"}
    #+END_SRC
```

Let's stop for a moment to review what we have just done.

- We explored the csv file using a **bash source block**.
- We used a **DSL** (Domain-Specific Langugage) to perform a **HTTP
  POST request** to the API.
- We used **elisp** to extract the **resource id** from the response
  of previous request.
- We used again the **DSL** to perform another **HTTP POST request**,
  using the previously extracted **resource id** as an argument.

And **everything from the same org file**!

**Emacs win**!


I will create now the **predictive model** (an ensemble of decision
trees) from the previously created **dataset**:

```sh
    #+BEGIN_SRC restclient :var id=resource("dataset.json") :file ensemble.json
    POST :url/ensemble/:auth
    Content-Type: application/json
    {"dataset": ":id"}
    #+END_SRC
```

And finally, let's make **predictions** and show the results in this
document:

```sh
    #+BEGIN_SRC restclient :var id=resource("ensemble.json")
    POST :url/prediction/:auth
    Content-Type: application/json
    {"ensemble": ":id",
     "input_data": {"sepal length": 6 , "sepal width": 3,
                     "petal length": 3}}
    #+END_SRC
```

In the **gif** below, you can see how I executed the last block from
and org file and explored the **JSON response** in the same document,
with the help of [yafolding](https://github.com/zenozeng/yafolding.el)
and [Ivy swiper](https://github.com/abo-abo/swiper).


![RestClient example](/assets/gifs/restclient.gif)

Remember that you can download the original [org
file](/assets/org/ml-from-org-mode.org) and play with it! With some
little changes on it, it can be used to explore all your favourite
APIs.
