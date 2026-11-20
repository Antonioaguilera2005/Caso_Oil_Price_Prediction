# Fuentes y referencias

Este documento reúne las referencias metodológicas que antes estaban repartidas
en informes de revisión. Los informes auxiliares se han retirado de la entrega;
se mantienen las atribuciones y la procedencia de los datos y de las ideas consultadas.

## Material docente y datos

- [Caso y materiales de la asignatura](https://github.com/UAX-VCMT-26-27/aa-case-02-oil-price-prediction):
  enunciado, scripts iniciales y configuración de partida. El PDF está en
  [assignment/ENUNCIADO.pdf](assignment/ENUNCIADO.pdf).
- [FRED/EIA · Brent](https://fred.stlouisfed.org/series/DCOILBRENTEU): precio spot.
- [ALFRED](https://alfred.stlouisfed.org/series?seid=DCOILBRENTEU) y
  [formato de versiones históricas](https://alfred.stlouisfed.org/help/downloaddata):
  fechas de disponibilidad y revisiones de las series.
- [Yahoo Finance · Brent futures](https://finance.yahoo.com/quote/BZ=F/history/):
  futuros y variables de mercado. Las convenciones temporales están en el
  [diccionario de datos](REPRODUCIBILIDAD.md#datos-y-disponibilidad).

Las instantáneas incluidas conservan los metadatos y hashes de su descarga.
Cada proveedor mantiene sus condiciones de uso.

## Referencias de desarrollo consultadas

| Referencia | Aspectos consultados |
|---|---|
| [Alejandro · oil-rnn](https://github.com/alejandrobarreche/oil-rnn) | Arquitecturas recurrentes, cuantiles, presupuesto de búsqueda y selección temporal interna |
| [Pablo · Caso-Oil-Price](https://github.com/PabloJimenezGirardeau/Caso-Oil-Price) | Retornos, regularización, parada temprana, semillas y referencias ingenuas |
| [Álvaro · oil-price-prediction](https://github.com/AlvaroSantamariaAnton/oil-price-prediction) | Disponibilidad temporal, etiquetas por horizonte y comparación con futuros |
| [Rubén · PREDICCION_PETROLEO_RNN](https://github.com/rubenelices/PREDICCION_PETROLEO_RNN) | Señales, modelos por horizonte y diagnóstico con remuestreo por bloques |
| [Marta · breastdcedl](https://github.com/martanavarroguil5/breastdcedl) | Referencia de otro problema para distinguir coste de entrenamiento y unidad de independencia |

El diagnóstico de remuestreo por bloques de `study/diagnostics.py` toma como
referencia la idea metodológica consultada en el proyecto de Rubén, con
implementación local. Los pesos de entrega se entrenaron con los datos locales;
no se incorporan pesos ni resultados numéricos de otros proyectos. Sus métricas
no son directamente comparables cuando cambian calendarios, pérdidas o referencias.

El [README principal](../README.md) y la [guía de reproducción](REPRODUCIBILIDAD.md)
describen el procedimiento y los resultados de la versión entregada.
