"""Etapa 2: seguimiento entre frames. Responsable: Persona 3.

QUE HACE
--------
Lee las detecciones sueltas de la Persona 2 (una caja por persona y por frame,
sin relacion entre si) y les asigna un `track_id`: el numero que dice "esta
caja y aquella son la misma silueta". Ningun otro campo se toca.

COMO DECIDE QUE DOS CAJAS SON LA MISMA PERSONA (acordado en la Fase 1 de diseno)
---------------------------------------------------------------------------
Sin apariencia (prohibida por privacidad, ver docs/data-contract.md) lo unico
que queda es la geometria: donde estaba, hacia donde se movia, y que tanto se
solapa con lo que se ve ahora.

Cada track guarda su ultima caja, el instante en que se vio (`timestamp`) y su
velocidad (px/s, estimada con las dos ultimas posiciones). En cada frame nuevo:

    1. Se PREDICE donde deberia estar cada track activo, extrapolando su
       velocidad hasta el timestamp del frame actual. Emparejar contra la
       posicion predicha (no contra la ultima posicion cruda) es lo que
       distingue a dos personas que se cruzan: aunque sus cajas de este frame
       se solapen casi del todo, sus predicciones -que vienen de direcciones
       distintas- no.
    2. Se EMPAREJA cada prediccion con las cajas del frame por IoU, de mayor a
       menor (voraz). Si el mejor IoU de un par no llega a
       IOU_MINIMO_PARA_EMPAREJAR, no se fuerza: es preferible abrir un track
       nuevo a intercambiar dos identidades en silencio (un ID switch no da
       ningun error, solo datos mal atribuidos).
    3. Un track que no se empareja NO muere de inmediato: sigue "flotando"
       sobre su prediccion hasta MAX_AGE_S segundos sin ver nada. Esto es lo
       que le permite sobrevivir a una persona que YOLO dejo de ver un par de
       frames (confianza inestable) sin creerla otra persona nueva.

POR QUE MAX_AGE Y LA VELOCIDAD SE MIDEN EN SEGUNDOS Y NO EN FRAMES DEL ARCHIVO
-------------------------------------------------------------------------
El pipeline puede correr con FRAME_STRIDE > 1: diez lineas seguidas del
.jsonl pueden ser diez frames reales o cincuenta, segun con que stride corrio
`detect`. Contar EVENTOS o FRAMES para decidir "hace cuanto se vio" haria que
la misma tolerancia real cambiara de tamano sin que nadie lo note. Por eso
todo este modulo trabaja con `evento.timestamp` (segundos reales del video,
ya calculados por la Persona 2) y nunca con `evento.frame` para nada que
tenga que ver con tiempo transcurrido. `frame` solo se usa para saber que
detecciones pertenecen al mismo instante (agruparlas), no para medir tiempo.

SIN APARIENCIA, A PROPOSITO
----------------------------
`track_id` sale unicamente de esta geometria: posicion, velocidad y solape.
Nunca se deriva del contenido de la caja (color, textura, forma de la
silueta). De hecho esta etapa ni siquiera RECIBE pixeles: lee un .jsonl con
cajas y numeros, no el video. Eso no es una promesa de buen comportamiento,
es que la reidentificacion por apariencia es estructuralmente imposible con
esta entrada.

POR QUE NO HAY IMPORTS PESADOS AQUI
------------------------------------
A diferencia de `detect.py` (que necesita ultralytics y OpenCV), esta etapa es
aritmetica pura sobre numeros que ya estan en el .jsonl: no hace falta ninguna
libreria nueva, ni siquiera numpy. Por eso no hay ninguna seccion de imports
diferidos: todo lo que se usa aqui ya esta en requirements-dev.txt.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from itertools import groupby
from pathlib import Path
from typing import Iterator

from gondola import pipeline
from gondola.config import Config
from gondola.contract import CONTRACT_VERSION, BBox, Event
from gondola.jsonl import read_events, write_events

IOU_MINIMO_PARA_EMPAREJAR = 0.3
"""Por debajo de este solape, dos cajas no se consideran la misma persona.
Mas vale abrir un track nuevo que forzar un emparejamiento dudoso: un track de
mas se ve y se puede filtrar despues, un ID switch no avisa a nadie."""

MAX_AGE_S = 0.6
"""Segundos que un track puede pasar sin ninguna deteccion emparejada antes de
darse por perdido. EN SEGUNDOS, no en frames ni en eventos: ver el docstring
del modulo sobre por que (FRAME_STRIDE cambia cuantos eventos representan el
mismo tiempo real)."""

EVENTOS_ENTRE_AVISOS = 2000
"""Cada cuantos eventos se informa del progreso. Esta etapa es mucho mas
rapida que YOLO, pero un video largo igual conviene que avise que sigue viva."""


@dataclass
class Track:
    """Lo que sabemos de una silueta seguida: donde estaba y hacia donde iba.

    Vive solo en memoria mientras corre `run()`; no es lo que se escribe en el
    .jsonl (eso es el `track_id`, un entero, dentro de cada `Event`).
    """

    bbox: BBox
    timestamp: float
    velocity: tuple[float, float] = (0.0, 0.0)  # (vx, vy) en pixeles/segundo


@dataclass
class Resumen:
    """Lo que se va contando durante la corrida y acaba en el JSON de resumen."""

    eventos_procesados: int = 0
    tracks_creados: int = 0
    emparejamientos: int = 0
    tracks_activos_al_cierre: int = 0


# --------------------------------------------------------------------------
# Logica pura: se prueba sin archivos ni video
# --------------------------------------------------------------------------

def iou(a: BBox, b: BBox) -> float:
    """Interseccion sobre union de dos cajas. 0.0 si no se tocan."""
    ix1, iy1 = max(a.x, b.x), max(a.y, b.y)
    ix2 = min(a.x + a.width, b.x + b.width)
    iy2 = min(a.y + a.height, b.y + b.height)
    ancho_i, alto_i = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    interseccion = ancho_i * alto_i
    if interseccion <= 0.0:
        return 0.0
    union = a.width * a.height + b.width * b.height - interseccion
    return interseccion / union if union > 0 else 0.0


def centro(bbox: BBox) -> tuple[float, float]:
    """Centro geometrico de la caja, para medir movimiento.

    No es `bbox.support_point` (los pies): ese punto es para ubicar a la
    persona en el plano del piso (trabajo de la Persona 4), y usarlo aqui
    mezclaria dos ideas distintas. Para seguimiento, el centro de la caja
    basta.
    """
    return (bbox.x + bbox.width / 2, bbox.y + bbox.height / 2)


def predecir_bbox(track: Track, timestamp: float) -> BBox:
    """Donde deberia estar la caja del track AHORA, si siguio a la misma velocidad.

    Con velocidad (0, 0) -si el track solo tiene una deteccion todavia- la
    prediccion es simplemente su ultima caja: no se asume movimiento sin
    evidencia de al menos dos posiciones.

    `x` e `y` se recortan en 0 porque `BBox` (contract.py) exige coordenadas
    no negativas: una prediccion hacia el borde de la imagen no debe romper
    la validacion.
    """
    dt = timestamp - track.timestamp
    vx, vy = track.velocity
    return BBox(
        x=max(0.0, track.bbox.x + vx * dt),
        y=max(0.0, track.bbox.y + vy * dt),
        width=track.bbox.width,
        height=track.bbox.height,
    )


def actualizar_track(track: Track, evento: Event) -> None:
    """Registra una nueva deteccion emparejada: mueve el track y reestima su velocidad."""
    dt = evento.timestamp - track.timestamp
    if dt > 0:
        cx0, cy0 = centro(track.bbox)
        cx1, cy1 = centro(evento.detection.bbox)
        track.velocity = ((cx1 - cx0) / dt, (cy1 - cy0) / dt)
    track.bbox = evento.detection.bbox
    track.timestamp = evento.timestamp


def emparejar(
    tracks: dict[int, Track], eventos: list[Event]
) -> tuple[dict[int, Event], list[Event]]:
    """Empareja los tracks activos con las detecciones de UN mismo frame.

    Voraz: se ordenan todos los pares (track, deteccion) por IoU descendente y
    se van tomando mientras ninguno de los dos ya este usado. No es el optimo
    matematico (eso seria el algoritmo Hungaro), pero con la poca gente que
    hay a la vez frente a una gondola la diferencia es despreciable, y evita
    sumar una dependencia (scipy) solo para esto.

    Devuelve (emparejados: {track_id: evento}, sin_emparejar: detecciones que
    no encontraron track y necesitan uno nuevo).
    """
    timestamp = eventos[0].timestamp
    candidatos: list[tuple[float, int, int]] = []
    for track_id, track in tracks.items():
        prediccion = predecir_bbox(track, timestamp)
        for indice, evento in enumerate(eventos):
            valor = iou(prediccion, evento.detection.bbox)
            if valor >= IOU_MINIMO_PARA_EMPAREJAR:
                candidatos.append((valor, track_id, indice))
    candidatos.sort(key=lambda c: c[0], reverse=True)

    usados_tracks: set[int] = set()
    usados_eventos: set[int] = set()
    emparejados: dict[int, Event] = {}
    for _valor, track_id, indice in candidatos:
        if track_id in usados_tracks or indice in usados_eventos:
            continue
        emparejados[track_id] = eventos[indice]
        usados_tracks.add(track_id)
        usados_eventos.add(indice)

    sin_emparejar = [e for i, e in enumerate(eventos) if i not in usados_eventos]
    return emparejados, sin_emparejar


def asignar_track_ids(eventos: Iterator[Event], resumen: Resumen) -> Iterator[Event]:
    """Recorre los eventos en orden y les va asignando track_id, de a un frame.

    Es un generador: solo mantiene en memoria los tracks activos y las
    detecciones del frame que esta procesando, nunca el archivo completo. Los
    eventos llegan agrupados por frame porque `detect` los escribio asi (todas
    las detecciones de un frame seguidas); agruparlos aqui no exige ordenarlos
    de nuevo.
    """
    tracks: dict[int, Track] = {}
    siguiente_id = 1

    for _frame, grupo_iter in groupby(eventos, key=lambda e: e.frame):
        grupo = list(grupo_iter)
        timestamp = grupo[0].timestamp

        # Los tracks que llevan demasiado sin verse se dan por perdidos ANTES
        # de emparejar, para que no le "roben" el track_id a una persona
        # distinta que aparece despues en el mismo sitio.
        vencidos = [tid for tid, tr in tracks.items() if timestamp - tr.timestamp > MAX_AGE_S]
        for tid in vencidos:
            del tracks[tid]

        emparejados, sin_emparejar = emparejar(tracks, grupo)

        for track_id, evento in emparejados.items():
            actualizar_track(tracks[track_id], evento)
            evento.track_id = track_id
            resumen.emparejamientos += 1

        for evento in sin_emparejar:
            tracks[siguiente_id] = Track(bbox=evento.detection.bbox, timestamp=evento.timestamp)
            evento.track_id = siguiente_id
            resumen.tracks_creados += 1
            siguiente_id += 1

        resumen.eventos_procesados += len(grupo)
        resumen.tracks_activos_al_cierre = len(tracks)
        if resumen.eventos_procesados % EVENTOS_ENTRE_AVISOS < len(grupo):
            _avisar_progreso(resumen)

        yield from grupo


def _avisar_progreso(resumen: Resumen) -> None:
    print(
        f"  eventos {resumen.eventos_procesados:>7}   "
        f"tracks_creados {resumen.tracks_creados:>5}   "
        f"emparejamientos {resumen.emparejamientos:>7}"
    )


# --------------------------------------------------------------------------
# Punto de entrada de la etapa
# --------------------------------------------------------------------------

def run(cfg: Config) -> int:
    """Ejecuta el seguimiento completo. Devuelve el codigo de salida."""
    rutas = pipeline.stage_paths("track", cfg)
    pipeline.require_input("track", cfg)

    print(f"[track] Entrada: {rutas.input_path}")
    print(f"[track] iou_minimo={IOU_MINIMO_PARA_EMPAREJAR}  max_age={MAX_AGE_S}s")
    print()

    resumen = Resumen()
    inicio = time.perf_counter()
    escritos = write_events(
        rutas.output_path,
        asignar_track_ids(read_events(rutas.input_path), resumen),
    )
    transcurrido = time.perf_counter() - inicio

    ruta_resumen = pipeline.summary_path("track", cfg)
    _escribir_resumen(ruta_resumen, cfg, resumen, transcurrido)
    _imprimir_resultado(resumen, escritos, transcurrido, rutas.output_path, ruta_resumen)

    return 0


def _escribir_resumen(destino: Path, cfg: Config, resumen: Resumen, transcurrido: float) -> None:
    """Guarda las metricas de la corrida. Sin esto no se puede comparar nada."""
    datos = {
        "contract_version": CONTRACT_VERSION,
        "stage": "track",
        "video_id": cfg.video_id,
        "params": {
            "iou_minimo_para_emparejar": IOU_MINIMO_PARA_EMPAREJAR,
            "max_age_s": MAX_AGE_S,
        },
        "results": {
            "eventos_procesados": resumen.eventos_procesados,
            "tracks_creados": resumen.tracks_creados,
            "emparejamientos": resumen.emparejamientos,
            "tracks_activos_al_cierre": resumen.tracks_activos_al_cierre,
        },
        "performance": {
            "segundos": round(transcurrido, 2),
            "eventos_por_segundo": round(
                resumen.eventos_procesados / transcurrido, 2
            ) if transcurrido > 0 else 0.0,
        },
    }
    destino.parent.mkdir(parents=True, exist_ok=True)
    destino.write_text(json.dumps(datos, indent=2, ensure_ascii=False), encoding="utf-8")


def _imprimir_resultado(
    resumen: Resumen, escritos: int, transcurrido: float, jsonl: Path, ruta_resumen: Path
) -> None:
    print()
    print("-" * 66)
    print(f"  Eventos procesados     {resumen.eventos_procesados}")
    print(f"  Tracks creados         {resumen.tracks_creados}")
    print(f"  Emparejamientos        {resumen.emparejamientos}")
    print(f"  Tracks activos al cierre  {resumen.tracks_activos_al_cierre}  "
          f"(quedaron 'vivos' al terminar el video)")
    print(f"  Tiempo                 {transcurrido:.2f} s")
    print("-" * 66)
    print(f"  Eventos   {jsonl}  ({escritos} lineas)")
    print(f"  Resumen   {ruta_resumen}")
    print()
    print("  Siguiente etapa:  python -m gondola zones   (Persona 4)")
