# Reproducibilidad de la entrega

Ejecuta los comandos desde la raíz del repositorio. El [README](../README.md)
explica cómo instalar el entorno. Aquí `python` significa Python 3.12 del entorno
del proyecto; en Windows puede sustituirse por `.\.venv\Scripts\python.exe`, y
con Poetry por `poetry run python`.

## 1. Comprobar las cifras existentes sin entrenar

No requiere dependencias externas ni conexión a Internet:

```powershell
python -S -m scripts.build_app
python -S -m scripts.verify_release --require-build
```

La primera orden genera `app_site/dist/`; la segunda solo lee archivos. Debe
terminar con código 0 y `"ok": true`. Comprueba los manifiestos SHA-256 del
experimento, el checkpoint y escalas elegidos, las métricas recalculadas desde
los CSV, los datos de evaluación de la web, el paquete portable y el HTML construido.
No confirma una predicción futura: comprueba integridad y coherencia de resultados guardados.

La ejecución local comprobó **31 233 verificaciones, 89 archivos con hash y 90
grupos de métricas**. Los tamaños de muestra del modelo elegido son 499/441/258
a 5/63/252 días. Los seis objetivos largos de 2023 están incluidos.

`.gitattributes` preserva los bytes de los datos, experimentos y fuentes con
hash. Cambiar finales de línea, reexportar CSV o modificar un archivo congelado
puede invalidar la verificación aunque sus valores visibles parezcan iguales.

## 2. Repetir el entrenamiento seleccionado

Requiere el entorno de análisis completo y los datos locales incluidos:

```powershell
python -m scripts.reproduce_delivery --plan-only
python -m scripts.reproduce_delivery
```

El modo de planificación muestra las ventanas y etiquetas sin escribir salidas.
La segunda orden crea una carpeta nueva `runs/reproduction/<fecha-UTC>/`:

| Salida | Uso |
|---|---|
| `model.pt`, `settings.json`, `scaler.json` | Nuevos pesos, configuración y escalas |
| `curve.csv` | Pérdidas y puntuación por época |
| `metrics.csv`, `predictions.csv` | Entrenamiento y validación del nuevo ajuste |
| `diagnostics.json` | Evolución de la curva, épocas, tiempos e indicios de sobreajuste |
| `manifest.json` | Versiones, entradas, hashes y comparación con el checkpoint original |

Para elegir un nombre: `python -m scripts.reproduce_delivery --output runs/reproduction/prueba_01`.
El script rechaza carpetas ya pobladas y rutas protegidas. No sustituye pesos de
entrega, no descarga datos y no evalúa 2023–2024. La validación 2018–2022 vuelve
a usarse para elegir la época, como en el experimento original.

La prueba realizada en este portátil volvió a elegir la **época 17 de 27**, con
pesos idénticos y diferencia máxima de predicción **0 USD**. Conservó los hashes
de 203 archivos originales. El tiempo de entrenamiento depende del equipo y la
carga; no se promete identidad numérica entre todas las versiones o dispositivos.

## 3. Regenerar informe y exportación

Estas órdenes usan el modelo y la evaluación existentes; escriben sus derivados:

```powershell
python -m scripts.report_recovered_test
python -m scripts.export_app
python -m scripts.build_app
python -m scripts.package_huggingface
python -S -m scripts.verify_release --require-build
```

El informe se genera en `experiments/recovered_fixed_test_20261120/report/` y
actualiza el notebook 05. La exportación crea los JSON en `app_site/data/`.
El empaquetado prepara `model_hub/rnn-brent-signals/`; **no lo sube** a Hugging Face.
Los scripts de evaluación con manifiesto existente verifican sus salidas;
no deben confundirse con un nuevo test independiente.

## 4. Ejecutar las pruebas

Con el entorno de análisis:

```powershell
python -m unittest discover -s tests -q
```

Con Node.js disponible, sin instalar paquetes npm:

```powershell
node --test app_site/engine.test.mjs
python -m scripts.check_app_inference
python -m scripts.build_app
```

`check_app_inference` contrasta señales y pronósticos del motor JavaScript con PyTorch
en cinco fechas y actualiza `app_site/data/inference-verification.json`.
La diferencia máxima comprobada fue inferior a **0,0000013 USD/barril**.
El flujo de GitHub Actions está configurado para ejecutar construcción,
verificación de archivos y pruebas del motor en Windows y Linux cuando se suban
los cambios. La validación realizada hasta ahora es local, en Windows; las pruebas
Python completas se ejecutan con el entorno de análisis.

## 5. Regenerar la presentación

Con el grupo `analysis` instalado:

```powershell
python -m scripts.build_final_presentation_figures
python -m scripts.build_final_presentation
```

Se utilizan los resultados guardados, la captura `docs/presentation/assets/app_demo.png`
y la plantilla incluida `docs/presentation/plantilla_UAX_original.pptx`.
`--template` permite indicar otra plantilla compatible. No se necesita una ruta
personal en Downloads. Las órdenes escriben figuras y PPTX dentro de `docs/presentation/`.
El generador elimina las notas de la plantilla y no crea notas del orador.

La exportación a PDF y las previsualizaciones necesitan **Windows y PowerPoint
de escritorio**. Consulta `Get-Help .\scripts\render_presentation.ps1 -Full` o
el inicio del script para sus parámetros. El PDF ya exportado está incluido;
no hace falta PowerPoint para leer los resultados.

## 6. Datos y experimentos anteriores

Para reproducir la entrega se usan las instantáneas incluidas. Los comandos de
descarga y preparación pueden actualizar datos y cambiar sus hashes; no forman
parte del arranque ni de la verificación. Para incorporar nuevas observaciones
sin reentrenar, usa la importación CSV de la aplicación.

La [guía de scripts](REPRODUCIBILIDAD.md#comandos-y-componentes), el [índice de notebooks](../README.md#notebooks)
y el [índice de resultados](REPRODUCIBILIDAD.md#resultados-conservados) señalan las entradas de la
entrega. Se conservan los pesos de la partición fija y la comprobación walk-forward;
las búsquedas anteriores mantienen resúmenes, configuraciones y métricas, sin
todos sus pesos intermedios. Para una búsqueda Optuna nueva utiliza otra salida
con `--output runs/optuna/nueva_busqueda`; no reanudes una instantánea resumida.

## Datos y disponibilidad

| Ruta | Contenido |
|---|---|
| `data/raw/` | Descargas FRED/Yahoo y archivos de versiones ALFRED, con sus metadatos |
| `data/tables/` | Observaciones, futuros, vigencia de versiones y cobertura |
| `data/processed/` | Observaciones sin imputar, información disponible, máscaras y procedencia temporal |
| `data/development_2022/` | Niveles, señales y etiquetas anteriores a 2023, con manifiesto de integridad |
| `data/pilot/` | Instantánea utilizada para la búsqueda y las pruebas de separación temporal |

La fecha observada y la fecha de publicación son distintas. ALFRED permite
reconstruir versiones dentro de su cobertura; no garantiza conocer la publicación
original de cada observación antigua. En Yahoo se adopta disponibilidad desde el
siguiente día de lunes a viernes. Las entradas conservan el último valor conocido;
las etiquetas futuras ausentes no se rellenan. El WTI negativo se conserva.
Los futuros y el spot son instrumentos distintos.

## Configuraciones

`config/delivery_model.json` selecciona el checkpoint; `fixed_split_2017_2022.json`
fija entrenamiento/validación y `recovered_fixed_test_20261120.json` la evaluación.
`validation_2021_2022.json` es el contrato con hash del cargador de
`development_2022`: conserva su nombre histórico, aunque la validación de entrega
es 2018–2022. `protocol.json` y `validation_v2.json` sostienen la búsqueda y sus
pruebas; `optuna_cpu.json` y `optuna_confirmation.json` fijan presupuesto y semillas.
`regularization_20261118.json` define los controles y cortes walk-forward;
`app_events.json` contiene el contexto histórico de la web.

## Comandos y componentes

Los comandos se ejecutan como `python -m scripts.nombre` desde la raíz.

| Etapa | Módulos |
|---|---|
| Datos | `download_data`, `download_alfred`, `build_data_tables`, `prepare_extended`, `prepare_pilot` |
| Entrenamiento seleccionado | `reproduce_delivery`; protocolo original en `run_fixed_split` |
| Búsqueda y controles | `run_optuna`, `confirm_optuna`, `report_optuna`, utilidades de `run_regularized` |
| Informes | `report_fixed_split`, `report_recovered_test` |
| Evaluación y comprobación | `evaluate_recovered_test`, `verify_release` |
| Aplicación y paquete | `export_app`, `build_app`, `check_app_inference`, `package_huggingface` |
| Presentación | `build_final_presentation_figures`, `build_final_presentation`, `render_presentation.ps1` |

En `study/`, `data.py` y `extended.py` preparan ventanas y señales;
`models.py` define RNN/LSTM/GRU; `validation.py` y `fixed.py` realizan la selección
temporal. `regularized.py`, `search.py` y `final_model.py` contienen utilidades
de ajuste y controles. `metrics.py` calcula errores y `diagnostics.py` sus diagnósticos.

## Resultados conservados

| Carpeta de `experiments/` | Contenido |
|---|---|
| `fixed_split_2017_2022/` | Pesos, escalas, curvas y predicciones de entrenamiento/validación |
| `recovered_fixed_test_20261120/` | Evaluación de los seis checkpoints originales, con los 499/441/258 objetivos completos |
| `optuna_20261117_cpu/` | Estudios, tablas y resúmenes de los 112 intentos; configuraciones que alimentaron la partición fija |
| `regularization_20261118/` | Resúmenes de regularización y checkpoints de la comprobación walk-forward |
| `final_model_20261119/` | Registro de la evaluación histórica anterior que explica el contexto del bloque 2023–2024 |

Consulta los [resultados de validación](../experiments/fixed_split_2017_2022/report/INFORME.md),
el [informe de Optuna](../experiments/optuna_20261117_cpu/report/INFORME.md), la
[comprobación temporal](../experiments/regularization_20261118/report/INFORME.md)
y el [informe de evaluación](../experiments/recovered_fixed_test_20261120/report/INFORME.md).
Las búsquedas anteriores mantienen resúmenes, sin todos sus pesos intermedios;
para una nueva búsqueda usa una salida nueva, por ejemplo
`python -m scripts.run_optuna --architecture RNN --output runs/optuna/nueva_busqueda`.

## Problemas frecuentes

| Mensaje o situación | Solución |
|---|---|
| `No module named scripts` | Abre la terminal en la raíz, donde está `pyproject.toml` |
| Falta `standalone.html` | Ejecuta `python -m scripts.build_app` antes de abrir Streamlit |
| Falta PyTorch, Matplotlib o `pptx` | Instala con `setup.ps1` o `poetry install --with analysis`; selecciona ese intérprete |
| VS Code no muestra figuras | Abre el notebook con `.venv`, ejecuta sus celdas o usa el informe HTML incluido |
| No se encuentra Node | Solo es necesario para pruebas del motor y paridad; la app HTML no lo necesita |
| Un hash no coincide | Revisa el archivo indicado frente al checkout original; no edites el manifiesto para forzar su aceptación |
| El entrenamiento da otro tiempo o pequeñas diferencias | Revisa versiones, semilla, número de hilos y dispositivo en `manifest.json` |
