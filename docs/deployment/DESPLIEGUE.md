# Publicar la app en Render y el modelo en Hugging Face

La app HTML ejecuta la RNN en el navegador con los pesos exportados. Render aloja los archivos y Hugging Face conserva una copia documentada del modelo. Los visitantes no necesitan una cuenta de ChatGPT, Render o Hugging Face. El propietario sí necesita entrar en sus cuentas para publicar.

Los controles de edición que muestre el alojamiento de Sites son ajenos al HTML
del proyecto. Una comprobación con Chrome sin sesión no mostró «Edit site»,
«Remix» ni controles de ChatGPT. El archivo `app_site/dist/standalone.html` y
la entrega estática de Render contienen únicamente la app y sus datos.

Esta preparación es local: todavía no se ha creado ningún recurso en tus cuentas de Render o Hugging Face.

## 1. Preparar el repositorio

Desde la raíz del proyecto:

```powershell
.\.venv\Scripts\python.exe -m scripts.build_app
.\.venv\Scripts\python.exe -m scripts.package_huggingface
```

El primer comando prepara `app_site/dist`. El segundo genera `model_hub/rnn-brent-signals`, comprueba los hashes de los artefactos evaluados y verifica que los pesos JSON coinciden exactamente con los del checkpoint seleccionado. Ninguno entrena el modelo. El paquete de Hugging Face incluye pesos JSON sin pickle, escalador, configuración, inferencia Python/JavaScript, métricas y ficha del modelo.

Sube a tu repositorio personal de GitHub los archivos de la app, `app_site/data`, `scripts/build_app.py`, `render.yaml`, el paquete `model_hub` y esta documentación. `app_site/dist` se genera durante el despliegue y puede permanecer ignorado por Git. No subas `.venv`, `.tools`, `.cache`, tokens ni credenciales. Revisa los cambios en la pestaña Control de código fuente de VS Code antes de confirmarlos y sincronizarlos.

## 2. Publicar en Render

Utilizamos un **Static Site**: Render sirve los archivos mediante CDN y no mantiene un proceso Python que pueda quedarse dormido por inactividad. [Documentación de sitios estáticos de Render](https://render.com/docs/static-sites).

1. Entra en tu cuenta de Render y elige **New → Blueprint**.
2. Conecta tu repositorio personal de GitHub y selecciona su rama principal.
3. Render detectará `render.yaml`. Revisa que el servicio sea `brent-observatorio`, de tipo estático, y confirma la creación.
4. Espera al estado **Live**. Copia la URL `https://…onrender.com` que Render asigne; esa será la dirección para compartir con el profesor.
5. Ábrela en una ventana privada del navegador para comprobar que un visitante entra directamente sin iniciar sesión.

Si prefieres **New → Static Site**, configura manualmente:

| Campo | Valor |
|---|---|
| Root Directory | Vacío, raíz del repositorio |
| Build Command | `python3 -m scripts.build_app` |
| Publish Directory | `app_site/dist` |
| Variable de entorno | `SKIP_INSTALL_DEPS=true` |

El Blueprint usa `runtime: static` y publica solamente `app_site/dist`. La construcción usa la biblioteca estándar de Python: no instala PyTorch, no descarga tablas y no entrena. Los pesos y datos ya exportados deben estar en GitHub. La entrada `app.py` con Streamlit permanece en el repositorio como versión Python del trabajo; el servicio estático entrega la interfaz HTML. [Referencia oficial del Blueprint](https://render.com/docs/blueprint-spec).

Tras cada actualización de la rama vinculada, comprueba en Render que el despliegue termina correctamente. La disponibilidad depende del servicio y de sus cuotas; no se promete funcionamiento ininterrumpido ilimitado. La fecha de datos mostrada en la app sigue siendo la del fichero publicado: tener la web disponible no actualiza las cotizaciones automáticamente.

## 3. Publicar el modelo en Hugging Face

La carpeta que se debe subir es **`model_hub/rnn-brent-signals`**, no el repositorio completo. No contiene tablas completas de mercado, cuentas, tokens ni el checkpoint serializado con pickle. Se reconstruye la red a partir de los pesos JSON verificados.

En tu cuenta de Hugging Face, crea un repositorio de tipo **Model** con el nombre que prefieras, por ejemplo `rnn-brent-signals`. En **Files and versions → Add file → Upload files**, añade los archivos de esa carpeta y confirma la subida. Conserva sus nombres: `README.md` es la ficha que mostrará Hugging Face. Elige la visibilidad que necesites; la app de Render seguirá funcionando aunque el repositorio del modelo sea privado, porque lleva su propia copia de los pesos.

Alternativa por terminal, sustituyendo `TU_USUARIO` por tu usuario real:

```powershell
.\.venv\Scripts\python.exe -m pip install --upgrade huggingface_hub
.\.venv\Scripts\hf.exe auth login
.\.venv\Scripts\hf.exe repos create TU_USUARIO/rnn-brent-signals --repo-type model
.\.venv\Scripts\hf.exe upload TU_USUARIO/rnn-brent-signals ./model_hub/rnn-brent-signals . --repo-type model
```

`auth login` pide la autenticación del propietario. No pegues el token en el código, el README ni el chat. Si ya creaste el repositorio desde la web, omite el comando `repos create`. [Guía oficial de autenticación y subida con `hf`](https://huggingface.co/docs/huggingface_hub/guides/cli).

Este modelo tiene una arquitectura personalizada y carga explícita mediante `inference.py`: subirlo al Hub no crea por sí solo una API de predicción ni un Space. Tampoco es necesario activar un servidor de inferencia para esta app, ya que calcula las predicciones localmente en el navegador.

## 4. Comprobación de la entrega

- La URL de Render abre en una ventana privada sin cuentas.
- Se ven los tres horizontes y la fecha real de la copia de datos.
- La evaluación muestra las métricas guardadas y ambas referencias ingenuas.
- Cargar un CSV recalcula entradas y predicciones sin entrenar.
- La ficha de Hugging Face identifica la RNN, los periodos, las métricas y sus límites.
- El README principal contiene las dos direcciones definitivas una vez creadas.

Para abrir la app local sin servicios externos, utiliza `app_site/dist/standalone.html` o ejecuta:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```
