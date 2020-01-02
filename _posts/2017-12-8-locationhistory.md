---
title:  "Visualizando el historial de ubicaciones de Google"
date: 2017-12-08
tags: [data-visualization]
---


Es sabido por todos que **Google sabe en todo momento nuestra
ubicación** a partir de nuestro smartphone.  De hecho, según conocimos
hace poco, ésta se obtiene [incluso cuando se ha desactivado los
servicios de
localización](https://www.theverge.com/2017/11/21/16684818/google-location-tracking-cell-tower-data-android-os-firebase-privacy).

![Google
location](../../assets/images/googlelocation.png){:class="img-responsive
center-image"}

**¡OJO¡** En
[github](https://github.com/unmonoqueteclea/Jupyter-ML/blob/master/LocationHistory.ipynb)
he subido una libreta de Jupyter donde se puede ver una implementación
de todo lo que hablo en el artículo.

Actualmente, según he podido comprobar, mi smartphone Android está
enviando la localización a los servidores de Google aproximadamente
**una vez por minuto**.

Sin embargo, Google no esconde este hecho. Independientemente de cómo
comercialice estos datos (que lo hace), se ha dado cuenta de que tener
esta información le permite mejorar de forma sustancial la
**experiencia de usuario**. Entre otras muchas funcionalidades ha
creado [Google Timeline](https://www.google.com/maps/timeline), que
nos muestra una visualización con todos estos datos de nuestras
ubicaciones que Google tiene guardados.

![Google Timeline](../../assets/images/google_timeline.png)

Lo que no todo el mundo sabe es que desde [**Google
Takeout**](https://takeout.google.com/), Google (*posiblemente
obligado por las Leyes de Protección de Datos*) permite al usuario
descargar toda la información que tiene guardada sobre él. Por
supuesto, esto es el paraíso de los desarrolladores y de los que
aficionados a la visualización de datos. Uno de los elementos más
importantes que podemos obtener de **Google Takeout** es nuestro
**historial de ubicaciones**. Google nos proporciona un fichero de
tipo *JSON* o *KML* que contiene todas nuestras ubicaciones de los
últimos años. Además, también contiene información sobre cómo hemos
realizado los desplazamientos.


## Obteniendo historial de ubicaciones de Google
Lo primero que necesitaremos, por lo tanto, es descargar nuestro
propio **historial de ubicaciones**. Para ello tendremos que ir a
[Google Takeout](https://takeout.google.com/) y marcar únicamente la
casilla de **Location History**. Habrá que asegurarse que el formato
de fichero elegido sea *JSON*.

![Google Takeout](../../assets/images/takeout_location.png){:class="img-responsive center-image"}


Después, simplemente habrá que hacer click en **Next** y en **Create
file**. Google tardará unos minutos en procesar la información, y nos
enviará un email cuando el fichero esté disponible para descargar. Lo
descargaremos, y dentro del fichero comprimido buscaremos el archivo
con la extensión *.json*. Lo renombraremos dejándolo como
**locations.json** y lo copiaremos a nuestro directorio de trabajo.

## Librerías utilizadas
Además de las librerías imprescindibles en este tipo de proyectos
(**Numpy**, **Pandas**, **Matplotlib**, **Scikit-Learn**) y de otras
centradas principalmente en el tratamiento de datos geográficos
(**Geopy**, **Shapely**), destacaré el uso de dos librerías en
concreto:

- [**Folium**](https://github.com/python-visualization/folium):
  Permite realizar los conocidos mapas de la librería de *Javascript*
  [leaflet](http://leafletjs.com/) desde Python, de forma muy simple.
- [**ipywidgets**](https://github.com/jupyter-widgets/ipywidgets): Nos
  permite realizar visualizaciones con elementos interactivos en
  **Jupyter notebooks**.

Para evitar posibles problemas por las versiones de algunas de estas
librerías, al principio de la `libreta de Jupyter` citada al principio
de este artículo se muestran las instrucciones para crear un nuevo
entorno de `conda` con todas las librerías necesarias.


## Visualizando mapa de calor animado
Cargaremos el contenido del fichero **locations.json** en un
`DataFrame` de **Pandas**. Realizaremos algunas transformaciones
necesarias en las coordenadas y en los campos de fecha para adaptarlos
al formato que usaremos y, además, realizaremos una primera **limpieza
de nuestros datos**. Para ello, eliminaremos todas las localizaciones
cuyo valor de `accuracy` sea menor de 1Km.


Separaremos el dataset en meses, y pasaremos todos ellos al plugin
`HeatMapWithTime` de **Folium**. Este plugin permite tener un mapa de
calor con una dimensión adicional, el tiempo, permitiendo así animar
todas las localizaciones a lo largo de un tiempo determinado.


Folium generará un fichero con el nombre *LocationsHeatMap.html*, muy
pesado (33 Megabytes en mi caso), en el que podremos ver la siguiente
animación:

![Heatmap](../../assets/gifs/heatmap_notreduced.gif)


## Reduciendo nuestro dataset: DBScan
Como vemos, en nuestra animación anterior hay una gran densidad de
puntos cuando en realidad no necesitamos tantos puntos para realizar
una representación de los principales sitios en los que el usuario ha
estado cada mes. Además, si queremos utilizar herramientas de
**Reverse Geocoding** para obtener el nombre de las ciudades a partir
de sus coordenadas, si tenemos casi 1 millón de pares de coordenadas,
será un proceso muy costoso, y supondrá un problema sabiendo que la
mayoría de servicios de reverse geocoding tienen limitado el número de
llamadas que se puede realizar a su API. Por ello, decidimos aplicar
un **algoritmo de clustering**, basándonos en [este
artículo](http://geoffboeing.com/2014/08/clustering-to-reduce-spatial-data-set-size/)

Cuando se habla de **algoritmo de clustering** es inevitable pensar en
el
[K-Means](https://en.wikipedia.org/wiki/K-means_clustering). K-Means
es un muy potente algoritmo de clustering, sin embargo no es el más
indicado para conjuntos de datos geográficos. Como se explica en el
artículo anteriormente citado, esto se debe a que el K-Means minimiza
la **varianza**, no la **distancia geodésica**, lo que provocará
distorsión (sobretodo en latitudes alejadas del ecuador).

Un algoritmo mucho más adecuado para esta tarea es el
[**DBSCAN**](https://en.wikipedia.org/wiki/DBSCAN), que también está
implementado en la librería
[Scikit-learn](http://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html). Este
algoritmo tiene un parámetro **eps** que contiene la máxima distancia
entre dos localizaciones para ser consideradas del mismo cluster. Por
*prueba y error* hemos decidido que el valor adecuado para este
parámetros es de **2.5 kilómetros**. Realizaremos el clustering y
**nos quedaremos con la localización de nuestro dataset más cercana al
centroide de cada cluster**.


![DBScan](../../assets/images/dbscan.png){:class="img-responsive center-image"}


En un principio se había optado por realizar un clustering de todo el
dataset a la vez. Sin embargo, no queremos perder la **dimensión
temporal** de nuestros datos. Además, esto hacía que el algoritmo
tardara varias horas en procesar la información. La solución propuesta
pasa por aplicar el clustering a los datos de cada mes. De esta forma,
además de mantener nuestra dimensión temporal, el tiempo de
procesamiento se reduce a unos pocos minutos. Este es el resultado
final del mapa de calor de nuestro dataset reducido:

![Heatmap](../../assets/gifs/heatmap.gif)

Como vemos, apenas se observan diferencias con el anterior. Sin
embargo, **este archivo (*ReducedLocationsHeatMap.html*) pesa
únicamente 134 Kilobytes**. De hecho, se puede descargar desde [este
enlace](../../assets/html/ReducedLocationsHeatMap.html).

## Estadísticas por meses
A continuación se ha pasado a realizar un **gráfico interactivo** en
el que se muestren los principales lugares en los que el usuario ha
estado cada mes. Para ello, ha sido muy importante realizar un proceso
de **reverse geocoding**. Esto consiste en la obtención de una
dirección a partir de sus coordenadas. Para ello, se ha utilizado la
librería [**geopy**](https://github.com/geopy/geopy), que hemos
utilizado como *wrapper* del servicio [Nominatim de Open Street
Map](http://wiki.openstreetmap.org/wiki/Nominatim). Para evitar
sobrecargar el servicio, no hemos trabajado directamente con todas las
localizaciones del dataset sino con el dataset reducido.

Este ha sido el resultado obtenido:


![locations bargraph](../../assets/gifs/locgraph.gif)




## Estadísticas  totales
Agregando los datos de todos los meses, podemos obtener una gráfica
con cuáles son los principales lugares en los que he estado estos
últimos años.


![locations bargraph](../../assets/images/mainlocs.png)



Por último, recordad, no creáis todo lo que veis en los mapas. Mirad,
si no, esta viñeta de [xkcd](https://xkcd.com/1845/)

![maps-xkcd](../../assets/images/state_word_map.png){:class="img-responsive center-image"}
<style>
.center-image{
    margin: 0 auto;
    display: block;
}
</style>
