---
language:
- es
library_name: pytorch
tags:
- time-series
- forecasting
- rnn
- brent
- educational
model-index:
- name: RNN Brent con señales
  results: []
---

# RNN Brent con señales

Modelo académico para estimar el precio Brent spot en USD/barril a 5, 63 y 252 días de lunes a viernes, incluidos festivos. Es el checkpoint original de la RNN con señales, semilla 42, seleccionado al cerrar un proceso con múltiples arquitecturas, configuraciones y experimentos. Publicar este paquete no entrena ni modifica sus pesos.

## Arquitectura y partición

- RNN tanh de una capa con 32 unidades; último estado y salida lineal de 3 componentes.
- 16 señales causales, ventana de 60 observaciones, 1.699 parámetros.
- Aprendizaje 0,001; época seleccionada 17; sin dropout ni penalización L2 en este checkpoint.
- Entrenamiento y ajuste del escalado hasta el 29 de diciembre de 2017.
- Validación de enero de 2018 a diciembre de 2022 para seleccionar época.
- Comparación histórica guardada de 2023–2024, con los pesos y escalas anteriores intactos.

Las entradas multivariables completas empiezan en 2012. El bloque 2023–2024 se había consultado previamente durante el proyecto; sus resultados no se presentan como una primera prueba ciega. `metrics.json` conserva las métricas del checkpoint elegido.

## Uso en Python

Tras descargar esta carpeta e instalar `requirements.txt`:

```python
import pandas as pd
from inference import predict_frame

inputs = pd.read_csv("informacion_disponible.csv")
result = predict_frame(inputs, origin="2024-12-27")
print(result)
```

También se puede ejecutar `python inference.py informacion_disponible.csv --origin 2024-12-27`.

El CSV requiere `Date` y las columnas `brent_conocido`, `wti_conocido`, `vix_conocido`, `oro_conocido`, `cobre_conocido`, `indice_dolar_conocido`, `brent_fut_Close_conocido`, ordenadas por fecha. Debe representar la información ya disponible en cada origen, con una fila por día de lunes a viernes. No se debe rellenar hacia atrás ni introducir precios que se conocieron después. Se necesitan al menos 252 filas para el calentamiento de las señales y una ventana posterior de 60 señales completas. WTI admite valores negativos; las demás entradas deben ser positivas.

`model.json` contiene los pesos exactos y el escalado, sin pickle. `inference.py` reconstruye explícitamente la red; este modelo personalizado no utiliza `AutoModel` ni tiene un proveedor de inferencia alojado configurado. `engine.js` ofrece la misma preparación e inferencia en el navegador y `package.json` habilita su importación como módulo ES.

## Preprocesamiento

Cambios logarítmicos de Brent a 1, 5, 21 y 63 días; volatilidades muestrales de 21 y 63 días; distancias a medias de 63 y 252 días; drawdown de 252 días; cambio de `asinh(WTI/20)`; cambios logarítmicos de oro, cobre y dólar; logaritmo del VIX y base futuro/spot. Todas las ventanas terminan en el origen.

Se resta la mediana y se divide por la escala guardada de cada señal. La RNN devuelve cambios normalizados; se multiplican por las escalas de objetivo guardadas y se añade el último Brent spot conocido. `preprocessing.json` documenta el contrato y `scaler.json` preserva el escalador original.

## Resultados guardados

| Horizonte | N | MAE | MSE | RMSE | R² | RMSE/ingenuo spot | RMSE/futuro conocido |
|---|---:|---:|---:|---:|---:|---:|---:|
| 5 | 499 | 3.1266 | 15.1453 | 3.8917 | 0.5584 | 0.8856 | 1.0798 |
| 63 | 441 | 7.5916 | 83.4733 | 9.1364 | -1.2741 | 0.9996 | 1.0734 |
| 252 | 258 | 7.7594 | 89.6678 | 9.4693 | -2.1297 | 1.0046 | 1.0807 |

Los cocientes inferiores a 1 indican menor RMSE que la referencia correspondiente, no rentabilidad. El modelo mejora al ingenuo spot a 5 días; a 63 y 252 días prácticamente empata. No supera al futuro conocido en ninguno de esos horizontes. Los R² de los dos horizontes largos son negativos y sus errores están solapados.

Las bandas simétricas del 80 % usan un cuantil empírico de los errores absolutos de validación, conjunto que también seleccionó la época. Son rangos descriptivos y no tienen una garantía formal de cobertura futura. La incertidumbre es especialmente amplia a 252 días.

## Procedencia y alcance

Pesos procedentes de `experiments/fixed_split_2017_2022/RNN/signals/seed_42/model.pt`.

SHA-256 del checkpoint original: `396a44bd61bab1e2928614ea4f68f04559c3fb27ba5c85cd64f01cf88c59c4c4`.

El paquete contiene un modelo, su documentación y métricas; no redistribuye las tablas completas de mercado. Los datos del proyecto proceden de FRED/ALFRED y Yahoo Finance, con disponibilidad histórica documentada en el repositorio académico. Es un ejercicio educativo, sin estrategia de negociación ni evaluación de comisiones, liquidez o rentabilidad.

No se ha asignado una licencia de redistribución a este paquete. Antes de elegirla, debe revisarse la titularidad del código y las condiciones de las fuentes. La integridad de todos los archivos figura en `manifest.json`.
