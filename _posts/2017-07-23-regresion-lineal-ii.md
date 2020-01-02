---
title:  "Introducción a la regresión lineal (II)"
date: 2017-07-23
tags: [machine-learning]
---

En el [artículo
anterior](https://unmonoqueteclea.github.io/machine-learning/regresion-lineal/)
analizábamos en qué consiste la regresión lineal. Sin embargo, no se
hacía mención a cómo podemos obtener los parámetros de la función de
regresión. En este artículo veremos diferentes métodos para la
obtención de los mismos.

## Método de los mínimos cuadrados
El método de los mínimos cuadrados es uno de los más utilizados para
obtener los parámetros de la función de regresión. Su propósito es
encontrar los parámetros que minimicen la suma de los cuadrados de las
distancias entre los valores observados y los valores estimados por el
modelo lineal. Es decir, queremos minimizar la suma de los cuadrados
de los **residuos**. La razón por la que elevamos al cuadrado los
valores de los residuos es para evitar tener valores negativos. Si no
lo hiciéramos, valores positivos y negativos podrían compensarse al
realizar la suma de todos ellos.

Este método será utilizado para **regresiones lineales
simples**. Suponiendo la siguiente función de regresión:

$$
Y = a + bX
$$

los parámetros $$a$$ y $$b$$ se calculan de la siguiente forma:

$$
{b = \frac{S_{XY}}{S_{X}^{2}} = \frac{ \sum _{i=1}^{n}{(x_{i}-{\overline {x}})(y_{i}-{\overline {y}})}}{ {\sum _{i=1}^{n}(    x_{i}-{\overline {x}}  )^{2}} }  }

$$

$$
{a = \overline{y} - b\overline{x}}
$$

$$
{\overline {x}=\frac{1}{n}\sum _{i=1}^{n}(x_{i})}
$$

$$
{\overline {y}=\frac{1}{n}\sum _{i=1}^{n}(y_{i})}
$$

donde $$ S_{XY} $$ representa la covarianza muestral entre X e Y, y
$$S_{X}^{2}$$ representa la varianza de X (o lo que es lo mismo, el
cuadrado de la desviación típica.)  Por facilitar la lectura, no se
demostrará de donde vienen las ecuaciones para obtener $$a$$ y
$$b$$. Para el lector que quiera una demostración más detallada de
estas ecuaciones recomiendo [este
documento](http://www.ugr.es/~jsalinas/apuntes/C5.pdf) de la
Universidad de Granada.

## Método del descenso de gradiente
En el apartado anterior decíamos que nuestro objetivo es minimizar la
suma de los cuadrados de los residuos. El objetivo del método del
descenso de gradiente es exactamente el mismo. Definiremos la función
de coste, que será la que tendremos que minimizar:

$$
J(\theta_{0},\theta_{1}) = \frac{1}{2n}\sum_{i=1}^{n}(h(x^{(i)})-y^{(i)})^{2}
$$

Hemos realizado un pequeño cambio de notación con respecto al apartado
anterior. En este caso $$h(x)$$ representa la función de regresión y
$$\theta_{0}$$ y $$\theta_{1}$$ los parámetros de dicha función:

$$
{ h(x) = \theta_{0} + \theta_{1} X }
$$

En la función de coste que hemos definido, además de la suma de los
cuadrados de los residuos, tenemos el factor $$\frac{1}{2n}$$
multiplicando, por motivos de simplificación de cálculo.

La idea de este método es, partiendo de unos valores arbitrarios para
$$\theta_{0}$$ y $$\theta_{1}$$, modificarlos de forma **iterativa**
hasta llegar a un mínimo de la función de coste.

He preparado una [libreta interactiva de
Jupyter](https://github.com/unmonoqueteclea/Jupyter-ML/blob/master/Linear%20Regression.ipynb)
donde se puede ver en acción este método. Podemos descargarla desde el
enlace anterior y ejecutarla en nuestro ordenador. En el siguiente gif
(obtenido a partir de la libreta interactiva comentada anteriormente)
vemos cómo se produce el ajuste mediante el descenso de gradiente. Se
puede observar cómo, con cada iteración, disminuye el error hasta que
se llega al valor mínimo, que se mantendrá estable en iteraciones
posteriores. Si se ejecuta la libreta interactiva se pueden cambiar
los distintos parámetros del ajuste o incluso los puntos para los que
se realiza.


 ![Linear Regression example graphs](/assets/gifs/1.lineal_regression.gif)


En la misma [libreta interactiva de Jupyter](https://github.com/unmonoqueteclea/Jupyter-ML/blob/master/Linear%20Regression.ipynb) podemos encontrar una representación de la función de coste en función de $$ \theta_{0} $$ y $$\theta_{1}  $$

 ![Linear Regression error](/assets/gifs/3derror.gif)

 Para comprobar mejor cómo varía el coste en función de cada uno de los parámetros, hemos realizado las siguientes representaciones fijando uno de los parámetros y haciendo un barrido en el otro.

  ![Linear Regression error](/assets/images/2dcost1.png)

  ![Linear Regression error](/assets/images/2dcost2.png)


Veamos ahora qué calculos se realizan en cada iteración. Utilizaremos la siguiente fórmula, que repetiremos hasta encontrar el mínimo:

$$
{ \theta_{j} :=  \theta_{j} - \alpha \frac{\partial}{\partial \theta_{j}} J(\theta_{0}, \theta_{1})  }
$$

Concretando para $$\theta_{0}$$ y $$\theta_{1}$$ y sustituyendo la expresión de $$J$$:

$$
{
  \theta_{0} :=  \theta_{0} - \alpha \frac{\partial}{\partial \theta_{0}} J(\theta_{0}, \theta_{1})
  = \theta_{0} - \alpha \frac{\partial}{\partial \theta_{0}} \frac{1}{2n}\sum_{i=1}^{n}(\theta_{0}+\theta_{1}x^{i}-y^{(i)})^{2}
 }
$$

$$
{
  \theta_{1} :=  \theta_{1} - \alpha \frac{\partial}{\partial \theta_{1}} J(\theta_{0}, \theta_{1})
  = \theta_{1} - \alpha \frac{\partial}{\partial \theta_{1}} \frac{1}{2n}\sum_{i=1}^{n}(\theta_{0}+\theta_{1}x^{i}-y^{(i)})^{2}
 }
$$

Por último, sustituiremos las derivadas por sus correspondientes expresiones:


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

Al parámetro $$\alpha$$, lo llamamos **ratio de aprendizaje**. Veremos a continuación su utilidad.
### Ratio de aprendizaje
El ratio de aprendizaje,**$$\alpha$$** controla la **velocidad** a la que se produce el descenso de gradiente. Cuanto más grande sea, mayores serán los pasos que daremos en cada iteración.

 Es muy importante elegir bien el valor del ratio de aprendizaje. Un valor demasiado pequeño supondrá que el algoritmo necesite muchas más iteraciones para llegar a converger. Por el contrario, un valor demasiado alto puede hacer que el sistema nunca llegue a converger.

 A continuación veremos un claro ejemplo de este problema. En la [libreta comentada anteriormente](https://github.com/unmonoqueteclea/Jupyter-ML/blob/master/Linear%20Regression.ipynb), hemos incrementado ligeramente el valor del ratio de aprendizaje (hemos dado un valor de 0.0008) y este ha sido el resultado obtenido:

  ![Linear Regression example with bad learning ratio](/assets/gifs/2.lineal_regression.gif)

Para ver el efecto más claramente hemos dado un valor a $$ \theta_{0} $$ de 50, para que la linea inicialmente estuviera en el centro de la gráfica. Al tener un ratio de aprendizaje más alto del adecuado, los pasos serán tan grandes que nos saltaremos el valor del error mínimo y en vez de disminuir el error lo aumentaremos. En la siguiente imagen vemos qué es lo que está pasando en realidad. En rojo vemos este caso, y en verde cómo sería con un ratio de aprendizaje adecuado.

  ![Linear Regression example with bad learning ratio](/assets/images/learningrate.png)


### Feature scaling
Cuando realizamos una regresión lineal con varias variables, tenemos que asegurarnos de que **todas estas variables tienen un orden de magnitud similar**. De esta forma, nos aseguremos que el descenso de gradiente tarde lo mínimo posible en encontrar el mínimo local.

La forma más rápida de conseguir el mismo orden de magnitud en todas las variables, es dividir cada una de las variables de entrada del sistema entre su valor máximo para asegurarnos de que los valores vayan de 0 a 1.

 Sin embargo, es más común realizar lo que conocemos como **normalización**, mediante la cual conseguimos que todas ellas tengan un igual oden de magnitud y un valor medio de 0. Para ello, restamos a cada variable su valor medio, y dividimos todo entre la desviación estándar.

$$
{
x \leftarrow \frac{x -\mu}{\sigma}
}
$$

## Resolución mediante ecuación normal
A la hora de encontrar los valores óptimos para los parámetros $$ \theta_0,\theta_1,\theta_2,...,\theta_n$$, una posible alternativa al descenso de gradiente es el uso de la ecuación normal, que permite obtener estos valores de forma analítica.

Supongamos que tenemos una serie de características que influyen en las notas obtenidas por un estudiante en la asignatura de Física. Podemos verlos en la siguiente tabla:

| **Nº Horas de estudio** | **Nº Temas totales** | **Nota en Matemáticas** | **Nota en Física** |
|:----------:|:-------------:|:------:|:------:|
| 15 | 10 |8 | 8|
| 12 | 10 |4 | 5|
| 18 | 12 |9 | 7|
| 5  | 15 |4 | 3|

A partir de estos datos podemos definir las matrices **X** e **Y**.

 **X** será la matriz que contendrá todas las características (le hemos añadido una fila de unos al principio para poder realizar los cálculos). **Y** será la matriz con las notas obtenidas por cada alumno.

$$
{
 X = \begin{bmatrix}
1 &15&10  &8 \\
1 &12&10  &4  \\
1 &18&12  &9 \\
1 &5 &15  &4
\end{bmatrix}

Y = \begin{bmatrix}
8\\
5\\
7\\
3

\end{bmatrix}
}
$$

Mediante la siguiente ecuación podemos obtener directamente los valores de $$ \theta $$:

$$
{
  \theta = (X^{T}X)^{-1}X^{T}Y
}
$$

Este método tiene las ventajas de no necesitar hacer varias iteraciones, ni necesitar establecer un ratio de aprendizaje. Sin embargo, en el caso de tener una gran cantidad de atributos puede llegar a ser mucho más lento de calcular que el descenso de gradiente.

## Regresión polinomial
Hasta el momento, hemos utilizado funciones de regresión del tipo
$$
Y = \theta_{0} + \theta_{1}X_{1}
$$

Sin embargo, podemos definir funciones (hipótesis) no lineales. Un ejemplo de este tipo de hipótesis serían las **regresiones polinomiales** con la siguiente hipótesis:

$$
{
Y = \theta_{0} + \theta_{1}X_{1} + \theta_{2}X_{2}^{2} + ...  + \theta_{n}X_{n}^{n}
}
$$

donde,

$$
X_{2}=X_{1}^{2}\\

X_{3}=X_{1}^{3} \\

... \\

X_{n}=X_{1}^{n}
$$


Vemos que el valor de Y no depende directamente del valor de la características $$ X_{1} $$ sino de las distintas potencias de esta. Por lo tanto ya no tenemos una recta de regresión sino que tendremos distintas formas dependiendo del la función de regresión utilizada.


Terminamos, de nuevo, con un poco de humor, con esta viñeta de [xkcd](https://xkcd.com/605/)

![linnear-regression](/assets/images/extrapolating.png){:class="img-responsive center-image"}
<style>
.center-image{
    margin: 0 auto;
    display: block;
}
</style>
