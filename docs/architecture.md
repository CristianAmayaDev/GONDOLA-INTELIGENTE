# Arquitectura

## El flujo, de principio a fin

```
   video.mp4
       |
       v
   [ detect ]  YOLO encuentra personas          -> video_001.detect.jsonl
       |
       v
   [ track ]   enlaza siluetas entre frames     -> video_001.track.jsonl
       |
       v
   [ zones ]   ubica en góndolas, mide tiempo   -> video_001.zones.jsonl
       |
       v
   [ interact ] detecta PICK_UP / PUT_BACK      -> video_001.interact.jsonl
       |
       v
   [ metrics ] agrega los totales               -> video_001.metrics.json
       |
       v
   [ Backend ] lee los archivos  ------------->  PostgreSQL
                                                     |
                                                     v
                                                [ Dashboard ]
```

Cada etapa lee el archivo de la anterior, **rellena sus propios campos** del
mismo evento y escribe el siguiente archivo. Nadie inventa formatos: todos
hablan el `Event` de [`contract.py`](../ai-service/gondola/contract.py).

---

## La decisión más importante: el AI Service nunca toca la base de datos

**El pipeline escribe archivos. Solo el backend escribe en PostgreSQL.**

```
   AI Service  ->  .jsonl  ->  Backend  ->  PostgreSQL
   (6 personas)                (1 persona)
```

Tres razones, por orden de importancia:

**1. Seis personas trabajan sin instalar una base de datos.** Las Personas 2 a
6 escriben etapas del pipeline. Si el pipeline escribiera en PostgreSQL,
cada una tendría que instalarlo, configurarlo, tener las credenciales y
mantener su copia sincronizada. Con archivos, `pip install` y a trabajar. El
día que alguien tiene problemas con la base de datos, es una persona parada,
no seis.

**2. Un fallo a medias se arregla borrando un archivo.** El procesamiento tarda
media hora. Si se corta en el minuto 20 habiendo escrito ya en PostgreSQL, hay
que averiguar qué entró, borrarlo sin tocar lo demás y volver a empezar. Con
archivos: `rm video_001.detect.jsonl` y otra vez. Sin transacciones a medias,
sin limpiezas manuales, sin tablas en estado raro.

**3. Se puede inspeccionar con la vista.** Un `.jsonl` se abre con cualquier
editor. Cuando la Persona 4 dice "mis zonas salen mal", se mira el archivo. Con
la base de datos habría que conectarse y escribir consultas para lo mismo.

**El precio, que sí existe:** los datos no están consultables hasta que el
backend los importa, y ocupan espacio en disco duplicado (archivos + tablas).
Para un proyecto que procesa videos por lotes, no en tiempo real, es un precio
barato. Si algún día hiciera falta procesar en vivo, esta decisión habría que
reconsiderarla.

---

## Las tres capas

| Capa | Carpeta | Qué hace | Quién |
|---|---|---|---|
| **AI Service** | `ai-service/` | Del video a los `.jsonl`. Python, YOLO, OpenCV. | Personas 1–6 |
| **Backend** | `backend/` | De los `.jsonl` a PostgreSQL. Expone la API. | Persona 7 |
| **Frontend** | `frontend/` | Dashboard y recomendaciones. | Persona 8 |

Las capas solo se comunican por **archivos** (AI → Backend) y por **API**
(Backend → Frontend). Nadie importa código de otra capa.

---

## Piezas del AI Service

| Módulo | Responsabilidad |
|---|---|
| `contract.py` | **El contrato.** El único formato de datos del proyecto. |
| `config.py` | Toda la configuración, leída del `.env`. Nadie más llama a `os.getenv`. |
| `pipeline.py` | El registro de etapas. **El código decide los nombres de archivo.** |
| `jsonl.py` | Leer y escribir eventos en streaming. |
| `cli.py` | Punto de entrada único: `python -m gondola`. |
| `video/` | Lo único que sabe de OpenCV: leer video y renderizar. |
| `stages/` | Una etapa por archivo. |
| `verify/` | Comprueba que una salida cumple el contrato y la privacidad. |
| `evaluate/` | Mide la exactitud contra anotaciones humanas. |

Dos reglas transversales:

- **Los nombres de archivo salen de `pipeline.stage_paths()`**, nunca escritos a
  mano. Ocho personas escribiendo nombres de archivo a mano es una integración
  rota garantizada.
- **Las librerías pesadas se importan dentro de las funciones.** `ultralytics`
  arrastra 3 GB de PyTorch. Si estuviera arriba de un archivo, nadie podría
  correr los tests sin instalarlo.

---

## Por qué JSONL y no un JSON grande

Un video de 10 minutos produce decenas de miles de eventos.

- **Un JSON gigante** hay que cargarlo entero en memoria para leer una línea, y
  si se corta a la mitad no sirve nada de él.
- **JSONL** (un objeto JSON por línea) se lee de a poco, se puede procesar
  mientras se escribe, y una línea corrupta solo daña esa línea.

Por eso `read_events()` es un generador y `write_events()` acepta generadores:
una etapa completa se ejecuta sin acumular nada en memoria.

---

## Privacidad en la arquitectura

Que el sistema no identifique personas no es una promesa en un documento: está
en tres capas del diseño.

1. **El contrato usa `extra="forbid"`.** Añadir un campo de edad o rostro rompe
   la validación al instante, en la máquina de quien lo intentó.
2. **`verify` relee los archivos ya escritos** buscando campos prohibidos, por
   si los produjo otra herramienta.
3. **El render por defecto es `privacy`**: dibuja sobre fondo neutro y no
   recibe siquiera el fotograma original.

Detalles en [privacy.md](privacy.md).
