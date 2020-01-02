---
title:  "Introducción a la regresión logística"
date: 2017-11-12
tags: [machine-learning]
---

En los dos posts anteriores: [Introducción a la regresión lineal
(I)](https://unmonoqueteclea.github.io/machine-learning/regresion-lineal/)
y [Introducción a la regresión lineal
(II)](https://unmonoqueteclea.github.io/machine-learning/regresion-lineal-ii/)
veíamos como funcionaba la regresión lineal. Comentábamos que el
análisis de regresión tiene como objetivo la estimación de las
relaciones de dependencia entre una variable dependiente y una o más
variables independientes. La ``regresión logística``, aunque por su
nombre pueda confundir, nos permite abordar problemas de
**clasificación**, no de regresión. Mientras que en los problemas de
regresión tratamos de predecir una variable continua, en los de
clasificación **tratamos de predecir una variable discreta**. En el
[primer artículo sobre regresión
lineal](https://unmonoqueteclea.github.io/machine-learning/regresion-lineal/)
se profundizaba más en la diferencia entre estos dos tipos de
problemas.

**¡OJO¡** En
[github](https://github.com/unmonoqueteclea/Jupyter-ML/blob/master/Logistic%20Regression.ipynb)
he subido una libreta de Jupyter donde se puede ver una implementación
de todo lo que hablo en el artículo

A partir de los algoritmos y el método desarrollado para la regresión
lineal, la regresión logística intenta resolver problemas de
clasificación. Concretamente nos centraremos en problemas de
**clasificación binaria**, es decir, sólo hay que decidir entre dos
posibles clases. Por lo tanto **y** sólo puede ser 0 o 1. Sin embargo,
como veremos al final, esta clasificación se puede modificar para ser
capaz de distinguir entre una mayor cantidad de clases.

Un ejemplo de problema de este tipo puede ser un programa que detecte
si un email es spam. Sólo hay dos posibles casos, que no sea spam o
que sí lo sea.

## Hipótesis en la regresión logística
Una primera aproximación a la resolución de los problemas de
clasificación a partir de lo aprendido en la regresión lineal,
consiste en ignorar el hecho de que se produce un valor de **y**
continuo, y discretizarlo estableciendo un umbral. Por lo tanto, para
realizar la clasificación, realizaríamos una regresión lineal y
tomaríamos todas las predicciones menores que 0.5 como 0, y todas las
predicciones mayores que 0.5 como 1. Esta aproximación es muy pobre, y
además $$ h_{\theta}(x) $$ puede dar lugar a valores menores que 0 o
mayores que 1 (cuando sabemos que en los problemas de clasificación
binaria, **y**, sólo puede valer 0 o 1). Es por ello por lo que
realizamos una modificación en la hipótesis. Expresada en forma
vectorial, la hipótesis de la función de regresión lineal era:

$$
{
  h_\theta (x) =  \theta^T x
}
$$

Sin embargo, la nueva hipótesis será:

$$
{
 h_\theta (x) = g ( \theta^T x )
}
$$

donde,

$$
{
  g(z) = \dfrac{1}{1 + e^{-z}}
}
$$

Llamamos a g(z), **función sigmoide** o **función logística**, y tiene
la siguiente forma: ![Sigmoid
function](/assets/images/sigmoid.png){:class="img-responsive
center-image"} Como vemos, esta función es capaz de mapear cualquier
numero al intervalo (0,1).

A partir de ahora, la interpretación que le daremos a $$h_\theta(x) $$
es la **probabilidad** de que el valor a la salida sea 1. Por ejemplo,
un valor $$h_\theta(x) = 0.8 $$ supone un 80% de probabilidad de que
el valor a la salida sea 1. Formalmente, definimos esto como:

$$
{
h_\theta(x) = P(y=1 | x ; \theta) = 1 - P(y=0 | x ; \theta)
}
$$

$$
{
  P(y = 0 | x;\theta) + P(y = 1 | x ; \theta) = 1
}
$$

## Frontera de decisión (Decission Boundary)
A partir de la interpretación de la hipótesis como probabilidad, vista
anteriormente, podemos discretizar la salida de la siguiente forma:

$$
{
   h_\theta(x) \geq 0.5 \rightarrow y = 1
}

$$

$$
{
   h_\theta(x) < 0.5 \rightarrow y = 0
}

$$

Como podemos ver en la gráfica anterior, $$ g(z) $$ es mayor o igual
que 0.5 cuando z es mayor o igual que 0. Por lo tanto podemos concluir
que:

$$
{
\theta^T x \geq 0 \Rightarrow y = 1
}
$$

$$
{
   \theta^T x < 0 \Rightarrow y = 0
}
$$

Llamamos **frontera de decisión** a la línea que separa las áreas para
las cuáles **y=0** y las áreas para las cuáles **y=1**.

En la [libreta de Jupyter asociada a este
artículo](https://github.com/unmonoqueteclea/Jupyter-ML/blob/master/Logistic%20Regression.ipynb)
encontramos un ejemplo del cálculo de la frontera de decisión de una
regresión logística. Vemos esta frontera representada en verde:

![Boundary](/assets/images/boundary.png){:class="img-responsive center-image"}
## Función de coste
Si intentáramos usar para la regresión logística la misma función de
coste que para la regresión lineal, llegaríamos a una función con
varios mínimos locales, lo que dificultaría en gran medida la tarea
del algoritmo de descenso de gradiente. Aprovechando que estamos
realizando una clasificación binaria, y por lo tanto sólo puede tomar
los valores 0 o 1, definiremos **una nueva función de coste**:

$$
{
 J(\theta) = \dfrac{1}{m} \sum_{i=1}^m \mathrm{Cost}(h_\theta(x^{(i)}),y^{(i)})
}
$$

donde:

$$
{
\mathrm{Cost}(h_\theta(x),y) = -\log(h_\theta(x)) \qquad  \quad \text{if y = 1}
}
$$

$$
{
\mathrm{Cost}(h_\theta(x),y) = -\log(1 - h_\theta(x)) \qquad \text{if y = 0}
}
$$


A continuación, representaremos esta función, para ver con más
claridad cómo funciona.

![Linear Regression example with bad learning ratio](/assets/images/logisticcost.png)

Como vemos, cuando el valor real es 1 ($$y=1$$) y nuestra función de
hipótesis $$h(x)$$ es cercana a 0, el coste es muy alto (tiende a
infinito). Sin embargo, el coste toma valores más bajos según nos
acercamos a 1. Como era de esperar, justo lo contrario pasa cuando
**y** es 0.

### Simplificación de la función de coste
La función anterior se puede simplificar, para tener evitar tener 2
fórmulas distintas en función del valor de y:

$$
{
\mathrm{Cost}(h_\theta(x),y) = - y \; \log(h_\theta(x)) - (1 - y) \log(1 - h_\theta(x))
}
$$


Cuando **y** sea 1, $$(1-y)$$ será igual a 0, y por lo tanto sólo nos
quedaremos con el primer término. De la misma forma, cuando **y** sea
igual a 0, se anulará el primer término, quedando únicamente el
segundo. Por lo tanto nuestra, función final de coste quedará de la
siguiente forma:

$$
{
J(\theta) = - \frac{1}{m}  \sum_{i=1}^m [y^{(i)}\log (h_\theta (x^{(i)})) + (1 - y^{(i)})\log (1 - h_\theta(x^{(i)}))]
}
$$

Y de forma vectorizada:

$$
{
J(\theta) =  \frac{1}{m}  (-y^{(T)}\log (h) - (1 - y)^{(T)}\log (1 - h))
}
$$

Esta función tiene varias propiedades interesantes. Una de las más
importantes es que es una **función convexa, sin mínimos locales,** lo
que nos asegura que seremos capaces de obtener el mínimo
absoluto. Además, tiene ciertas propiedades (cuya explicación nos
apartaría demasiado del tema central de este artículo), que hacen que
sea fácil encontrar el mínimo absoluto mediante el método *MLE*
(maximum-likelihood estimation).

### Descenso de gradiente
El **descenso de gradiente** realizado para la regresión logística es
exactamente igual que el realizado para la regresión lineal que vimos
en [artículos
anteriores](https://unmonoqueteclea.github.io/machine-learning/regresion-lineal-ii/).
Como sabemos, el descenso de gradiente se basa en la realización de la
derivada de la función de coste:

$$
{ \theta_{j} :=  \theta_{j} - \alpha \frac{\partial}{\partial \theta_{j}} J(\theta_{0}, \theta_{1})  }
$$



Por lo tanto, estas serán las ecuaciones utilizadas:

$$
{
  \theta_{0} :=  \theta_{0} - \alpha \frac{1}{n} \sum_{i=1}^{n}(h(x^{i})-y^{i})
 }
$$

$$
{
  \theta_{1} :=  \theta_{1} - \alpha \frac{1}{n} \sum_{i=1}^{n}(h(x^{i})-y^{i}) \cdot x^{i}
 }
$$

No olvidemos que, aunque aparentemente la fórmula usada para el
descenso de gradiente sea igual a la usada en la regresión lineal, en
este caso:


 $$
 {
   h_{\theta}(x) =\frac{1}{1+e^{-\theta^{T}x}}
 }
 $$

De forma vectorizada quedará:


$$
{
  \theta:=  \theta - \alpha \frac{1}{n}X^{T} (g(X\theta)-\vec{y})
 }
$$




## Clasificación multiclase (One vs All)
Hasta ahora hemos analizado la clasificación binaria, con dos únicas
posibles clases. ¿Cómo lo haríamos para realizar una **clasificación
multiclase**? La respuesta está en el algoritmo conocido como **One vs
All**. Volviendo al ejemplo anterior del clasificador de correo, es
posible que no sólo queramos distinguir entre Spam/No Spam sino que
sería de utilidad separar **correos de trabajo**, **correos
personales** y **correos de promociones**. Al igual que pasaba en la
clasificación binaria, cada una de estas posibles clases tendrá
asignada un valor de la variable de salida. Por ejemplo:

- **y = 0**: Correos de trabajo
- **y = 1**: Correos personales
- **y = 2**: Correos de promociones

Siguiendo con este ejemplo, para resolverlo mediante esta técnica de
clasificación multiclase, tendremos que crear **3 clasificadores
binarios** distintos. Para el primero de ellos, modificaremos el
dataset y asignaremos $$y=1$$ a todos los correos de trabajo e $$y=0$$
al resto (independientemente de si son personales o de
promociones). De este clasifícador obtendremos $$ h_{\theta}^{(1)}(x)
$$ A continuación realizaremos lo mismo para el caso de los correos
personales. Crearemos un clasificador binario y tendremos $$y=1$$ para
los correos personales e $$y=0$$ para el resto. De este clasifícador
obtendremos $$ h_{\theta}^{(2)}(x) $$. Haremos lo mismo para los
correos promocionales obteniendo $$ h_{\theta}^{(3)}(x) $$. A la hora
de realizar una predicción simplemente tendremos que escoger el
clasificador que mayor probabilidad devuelva:

$$
{

\max_{i} h_{\theta}^{(i)}(x)

}
$$

A continuación, vemos un ejemplo de este tipo de clasificación tomado
de la [libreta de
Jupyter](https://github.com/unmonoqueteclea/Jupyter-ML/blob/master/Logistic%20Regression.ipynb)

![linnear-regression](/assets/images/multiclase1.png){:class="img-responsive center-image"}
![linnear-regression](/assets/images/multiclase2.png){:class="img-responsive center-image"}
![linnear-regression](/assets/images/multiclase3.png){:class="img-responsive center-image"}
![linnear-regression](/assets/images/multiclase4.png){:class="img-responsive center-image"}

Y después de este denso artículo, vamos a terminar otra vez con un
poco de humor de [xkcd](https://xkcd.com/833/)

![linnear-regression](/assets/images/convincing.png){:class="img-responsive center-image"}
<style>
.center-image{
    margin: 0 auto;
    display: block;
}
</style>
