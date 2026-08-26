# Instrucciones del proyecto: Gondola Inteligente

## Que es

Reto de la empresa Scapder: a partir del video de una tienda, cuantificar la
dinamica de los clientes alrededor de las gondolas (flujo, permanencia,
interaccion fisica con productos) para recomendar mejoras de *space management*
y planogramas.

Proyecto universitario de 8 personas. La Persona 1 lidera la arquitectura.

## Regla numero uno: el proyecto debe ser PEQUENO

Quien lo escribe tiene que poder defenderlo ante un jurado. Codigo que no se
entiende es codigo que no sirve, aunque funcione.

- Ninguna abstraccion "por si acaso". **Si no se usa hoy, no existe.**
- Nada de patrones de diseno elaborados, capas extra ni configuracion dinamica.
- Ante dos soluciones, la mas simple, y explicando por que.
- Docstrings en espanol, cortos y utiles.
- Si algo hace falta pero no toca en esta fase: se anota, no se construye.

## Privacidad por diseno (no negociable)

El sistema **no** identifica personas, **no** reconoce rostros, **no** infiere
emociones y **no** crea perfiles biometricos.

Prohibido en todo el codigo: edad, genero, rostro, `embeddings` faciales,
identidad, emocion, biometria, o cualquier caracteristica fisica derivada de la
bounding box. Los modelos usan `extra="forbid"` justamente para que esto falle
solo.

`bbox.width` y `bbox.height` son **pixeles de la caja**, nunca la estatura ni la
contextura de nadie.

## El contrato de datos manda

`ai-service/gondola/contract.py` es la pieza central. Documentado en
`docs/data-contract.md`.

Las etapas **enriquecen** el mismo evento; nunca inventan formatos:

| Campo | Responsable |
|---|---|
| `detection` | Persona 2 (YOLO) |
| `track_id` | Persona 3 |
| `zone`, `metrics.dwell_time` | Persona 4 |
| `interaction` | Persona 5 |

Cambiar la forma del evento implica subir `CONTRACT_VERSION`, actualizar
`docs/data-contract.md` y avisar al equipo. Nunca por cuenta propia.

## Ejecutar el proyecto

Todo pasa por la CLI. Se ejecuta desde `ai-service/`, que es donde vive el
paquete:

```
cd ai-service
python -m gondola doctor     # diagnostico; empieza SIEMPRE por aqui
python -m gondola run        # la cadena completa
```

Codigos de salida: 0 exito, 1 error de ejecucion, 2 falta un requisito (el
video o el archivo de la etapa anterior).

## Convenciones

- **Nombres de archivo: nunca a mano.** Se piden con
  `pipeline.stage_paths(nombre, cfg)`. La tabla `STAGES` de
  `gondola/pipeline.py` es la unica fuente de verdad de la cadena.
- **Leer y escribir .jsonl:** `jsonl.read_events()` y `jsonl.write_events()`,
  que van en streaming. Nunca `open()` a pelo ni cargar todo en una lista.
- Configuracion: solo en `gondola/config.py`. Nadie mas llama a `os.getenv`.
  Toda variable nueva se documenta en `.env.example`.
- Errores: usar la jerarquia de `gondola/errors.py`. El mensaje dice **que
  hacer**, no solo que fallo.
- Logging: `setup_logging()` se llama una sola vez al arrancar. Las etapas solo
  hacen `logging.getLogger(__name__)`.
- OpenCV vive unicamente en `gondola/video/`.
- Serializar siempre con `Event.to_jsonl()`; leer con `Event.from_jsonl()`.

## Correr los tests

```
pip install -r requirements-dev.txt
pytest
```

`requirements-dev.txt` es ligero a proposito: los 7 companeros deben poder
correr los tests sin descargar 3 GB de PyTorch. Lo pesado va en
`requirements.txt` y solo lo necesita quien ejecuta el pipeline completo.

## Estado por fases

- **Fase 1 (hecha):** estructura, contrato, configuracion, errores, logging,
  documentacion del contrato, tests unitarios.
- **Fase 2 (hecha):** registro de etapas, CLI (`doctor`, `run`, `purge` y las
  cinco etapas como placeholders), lectura y escritura de .jsonl en streaming.
- **Fase 3:** deteccion YOLO (Persona 2), lectura de video y render.
- **Fase 4:** verificador de privacidad y de contrato.
- **Fase 5:** evaluacion contra `groundtruth`, esquema SQL, frontend, README
  completo.

No construir cosas de fases futuras.
