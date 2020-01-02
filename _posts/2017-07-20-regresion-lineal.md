---
title:  "Introducción a la regresión lineal (I)"
date: 2017-07-20
tags: [machine-learning]
---
Para entender qué es y cómo funciona la `regresión lineal` debemos entender primero en qué consisten los **problemas de regresión**. El análisis de regresión tiene como objetivo la estimación de las relaciones de dependencia entre una variable dependiente y una o más variables independientes. Esta dependencia puede ser funcional o estadística:
-  **Dependencia funcional**: La relación se define mediante una función matemática.
- **Dependencia estadística**: La relación no se puede definir mediante una función matemática (aunque sí, aproximar) porque cuenta con algún componente aleatorio que no es posible modelizar. El análisis del grado de dependencia existente entre las variables se conoce como **problema de correlación**

 En el campo del aprendizaje automático, este análisis es muy usado con la intención de ser capaces de predecir el valor de la variable dependiente para unos valores concretos de las variables independientes. Del análisis de regresión obtendremos lo que conocemos como **función de regresión**, que es una función de las variables independientes a partir de la cual obtenemos el valor de la variable dependiente. Además, también es común la caracterización, mediante una ditribución de probabilidad, de la variación de la variable dependiente con respecto a la función de regresión.

## Clasificación vs Regresión
Los dos tipos principales de problemas de aprendizaje supervisado son los problemas de **clasificación** y los problemas de **regresión**. Por la propia definición de aprendizaje supervisado, los dos tipos de problemas dependen de un conjunto de datos de entrenamiento que contienen las entradas del sistema y las salidas deseadas (u observadas). A partir de ellos, se entrenarán los algoritmos para proporcionar la salida correcta para nuevos datos que no pertenecieran a nuestro conjunto de dato de entrenamiento. La principal diferencia entre los dos tipo de problema es que los problemas de clasificación tratan de obtener una clase o categoría (un **valor discreto**) mientras que, los problemas de regresión proporcionan a la salida un **valor continuo**. Un ejemplo de problema de clasificación es la detección de objetos en imágenes. La salida de este problema es una variable discreta, puesto que el conjunto de posibles objetos que podemos detectar es limitado. Sin embargo, un problema de regresión sería la estimación del precio de una casa en función de ciertas variables como su tamaño, su localización o su antigüedad.

<style>
@media (max-width: 1005px) {
  /* VERSION MOVIL*/
  #classreg{
    width:98vw;
    height:60vw;
  }
}
@media (min-width: 1005px) {
  /* VERSION WEB */
  #classreg{
    width:45vw;
    height:25vw;
  }
}
</style>
<iframe id="classreg" src="../../assets/d3js/classification_regression/classification_regression.html" scrolling="no"  marginwidth="0"  marginheight="0" align="top" frameborder="0" hspace="0" vspace="0"></iframe>


## Relación vs Causalidad
Hago un pequeño aparte para comentar algo que muchas veces nos cuesta asimilar. El análisis de regresión, como hemos comentado, nos permite obtener las relaciones de dependencia entre variables, pero esto no significa que estemos obligatoriamente infiriendo relaciones causales entre variables dependientes e independientes. No olvidemos:

> Relación no implica causalidad

En [Gaussianos](http://gaussianos.com/hay-que-decirlo-mas-correlacion-implica-causalidad/) profundizaban hace unos años en este tema, en un artículo muy interesante que nos hace ver lo común que es este error incluso en el ámbito científico.

## Regresión lineal
Volviendo de nuevo al tema tratado, el tipo de regresión más común es la **regresión lineal**. La regresión lineal se caracteriza porque la función de regresión es una función lineal. Este modelo está definido por la siguiente ecuación:

$$
{ Y=\sum \beta _{k}X_{k}+\varepsilon }
$$

En ella, $$Y$$ representa la variable dependiente, $$X_{k}$$ representa cada una de las $$K$$ variables independientes, y $$\beta _{k}$$ modela el [hiperplano](https://es.wikipedia.org/wiki/Hiperplano) que permite aproximar $$Y$$ en función de las variables independientes. Además, tenemos un **término aleatorio**: $$\varepsilon$$, que representa el posible error obtenido en la medición de las variables independientes $$X_{k}$$.

## Regresión lineal simple
El caso más limple de regresión lineal, es la regresión lineal simple, que tiene una sola variable independiente, $$X$$. En este caso, la función de regresión pasa a ser simplemente una **recta**, en vez de un hiperplano:

$$
{ Y= a + bX}
$$

En la ecuación anterior, $$a$$ es la ordenada en el origen y $$b$$ la pendiente de la recta. Concretamente, en este tipo de problemas, llamamos a $$a$$ **intercepto** y a $$b$$ **coeficiente de regresión** En la siguiente gráfica, vemos un ejemplo de regresión lineal simple. Si pasamos el puntero sobre la líneas que unen los puntos correpondientes a las diversas observaciones con la recta de la regresión veremos el valor del **error** o **residuo**.
<style>
@media (max-width: 1005px) {
  /* VERSION MOVIL*/
  #regresion{
    width:95vw;
    height:70vw;
  }
}
@media (min-width: 1005px) {
  /* VERSION WEB */
  #regresion{
    width:40vw;
    height:25vw;
  }
}
</style>
<iframe id="regresion" src="../../assets/d3js/regresion_lineal/example.html"
        scrolling="no"  marginwidth="0"  marginheight="0" align="top"
        frameborder="0" hspace="0" vspace="0">
</iframe>
<br/>

### Coeficiente de determinación
El coeficiente de determinación, $$R^{2}$$ nos permite medir la *calidad* de la regresión realizada. Determina la proporción de variación de los resultados que puede explicarse por el modelo. En el caso de la regresión lineal simple, este coeficiente es el cuadrado del **coeficiente de correlación**.​ El coeficiente de correlación (también llamado coeficiente de correlación de Pearson) varía entre -1 y 1 mide el grado de asociación entre dos  variables aleatorias. Un valor de -1 representa una correlación perfecta negativa, mientras que un valor de 1 representa una correlación perfecta positiva. Un valor de 0 en el coeficiente de correlación (y por lo tanto, también en el coeficiente de determinación) supondrá la ausencia total de correlación entre las variables. Como vemos, el coeficiente de correlación nos aporta más información que el de determinación, pues nos permite saber si la correlación es positiva o negativa.

El coeficiente de correlación se define como la covarianza muestral entre $$X$$ e $$Y$$ dividida por el producto de las desviaciones típicas de cada variable:

$$
{  r= \frac{S_{xy}}{S_{x}S_{y}} }
$$



$$
{ r = \frac{ \sum _{i=1}^{n}{(x_{i}-{\overline {x}})(y_{i}-{\overline {y}})}}{\sqrt{ {\sum _{i=1}^{n}(                x_{i}-{\overline {x}}  )^{2}} \cdot {\sum _{i=1}^{n}(                y_{i}-{\overline {y}}  )^{2}} }}  }
$$


En la siguiente ecuación, $$\overline {x}$$ y $$\overline {y}$$ representan, respectivamente, los valores medios de $$x$$ e $$y$$:

$$
{\overline {x}=\frac{1}{n}\sum _{i=1}^{n}(x_{i})}
$$


$$
{\overline {y}=\frac{1}{n}\sum _{i=1}^{n}(y_{i})}
$$

En las siguientes figuras vemos dos ejemplos: uno con un valor bajo de correlación y otro con un valor mucho mayor, cercano a 1. Como vemos, cuanto mayor es el coeficiente de determinación, mayor es la calidad del ajuste.
<style>
@media (max-width: 1005px) {
  /* VERSION MOVIL*/
  #corre{
    width:98vw;
    height:75vw;
  }
}
@media (min-width: 1005px) {
  /* VERSION WEB */
  #corre{
    width:40vw;
    height:25vw;
  }
}
</style>
<iframe id="corre"
        src="../../assets/d3js/correlation_regression/regression_correlation.html"
        scrolling="no"  marginwidth="0"  marginheight="0" align="top"
        frameborder="0" hspace="0" vspace="0">
</iframe>
<br/>

Terminamos con un poco de humor, con esta viñeta de [xkcd](https://xkcd.com/1725/)


![linnear-regression](/assets/images/linear_regression.png){:class="img-responsive center-image"}

En el [siguiente artículo](https://unmonoqueteclea.github.io/machine-learning/regresion-lineal-ii/) veremos cómo obtener los parámetros de la función de regresión a partir de nuestro conjunto de datos.

<style>
.center-image{
    margin: 0 auto;
    display: block;
}
</style>
