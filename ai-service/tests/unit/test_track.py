"""Tests del seguimiento (gondola/stages/track.py).

Todo lo que se prueba aqui es aritmetica pura sobre eventos construidos a
mano: no hace falta YOLO, ni OpenCV, ni un video. Corren con solo
`pip install -r requirements-dev.txt`, igual que test_detect.py.
"""

import json

import pytest

from gondola import pipeline
from gondola.config import load_config
from gondola.contract import BBox, Detection, Event
from gondola.errors import MissingInputError
from gondola.jsonl import read_events, write_events
from gondola.stages.track import (
    IOU_MINIMO_PARA_EMPAREJAR,
    MAX_AGE_S,
    Resumen,
    Track,
    actualizar_track,
    asignar_track_ids,
    centro,
    emparejar,
    iou,
    predecir_bbox,
    run,
)


def crear_evento(frame, timestamp, x, y, w=40.0, h=100.0, conf=0.9,
                 video_id="video_001") -> Event:
    return Event(
        video_id=video_id,
        frame=frame,
        timestamp=timestamp,
        detection=Detection(confidence=conf, bbox=BBox(x=x, y=y, width=w, height=h)),
    )


# --------------------------------------------------------------------------
# iou / centro / prediccion de movimiento
# --------------------------------------------------------------------------

def test_iou_de_cajas_identicas_es_uno():
    caja = BBox(x=10, y=10, width=50, height=100)
    assert iou(caja, caja) == 1.0


def test_iou_de_cajas_que_no_se_tocan_es_cero():
    a = BBox(x=0, y=0, width=10, height=10)
    b = BBox(x=100, y=100, width=10, height=10)
    assert iou(a, b) == 0.0


def test_iou_de_solape_parcial():
    a = BBox(x=0, y=0, width=10, height=10)   # area 100
    b = BBox(x=5, y=0, width=10, height=10)   # solapan en un area de 50
    assert iou(a, b) == pytest.approx(50 / 150)


def test_centro_es_el_punto_medio_de_la_caja():
    assert centro(BBox(x=0, y=0, width=10, height=20)) == (5.0, 10.0)


def test_sin_velocidad_la_prediccion_no_mueve_la_caja():
    """Un track recien creado no asume movimiento sin haber visto una segunda posicion."""
    track = Track(bbox=BBox(x=10, y=10, width=50, height=100), timestamp=0.0)
    predicha = predecir_bbox(track, 1.0)
    assert (predicha.x, predicha.y) == (10.0, 10.0)


def test_con_velocidad_la_prediccion_extrapola():
    track = Track(bbox=BBox(x=10, y=10, width=50, height=100), timestamp=0.0,
                 velocity=(20.0, 0.0))
    predicha = predecir_bbox(track, 0.5)  # medio segundo despues, a 20 px/s
    assert predicha.x == pytest.approx(20.0)


def test_la_prediccion_no_cae_en_coordenadas_negativas():
    """BBox exige x, y >= 0 (contract.py); una prediccion hacia el borde no debe reventar."""
    track = Track(bbox=BBox(x=5, y=5, width=50, height=100), timestamp=0.0,
                 velocity=(-1000.0, -1000.0))
    predicha = predecir_bbox(track, 1.0)
    assert (predicha.x, predicha.y) == (0.0, 0.0)


def test_actualizar_track_calcula_la_velocidad_entre_dos_posiciones():
    track = Track(bbox=BBox(x=0, y=0, width=50, height=100), timestamp=0.0)
    actualizar_track(track, crear_evento(1, 1.0, x=20, y=0, w=50))  # mismo ancho: el
    # centro se mueve exactamente lo que se movio x
    assert track.velocity[0] == pytest.approx(20.0)
    assert track.timestamp == 1.0


def test_actualizar_track_con_el_mismo_timestamp_no_revienta():
    """dt=0 no debe dividir por cero: conserva la velocidad que ya tenia."""
    track = Track(bbox=BBox(x=0, y=0, width=50, height=100), timestamp=1.0, velocity=(5.0, 0.0))
    actualizar_track(track, crear_evento(1, 1.0, x=20, y=0))
    assert track.velocity == (5.0, 0.0)


# --------------------------------------------------------------------------
# emparejar: un frame a la vez
# --------------------------------------------------------------------------

def test_emparejar_prefiere_el_mayor_iou():
    tracks = {1: Track(bbox=BBox(x=0, y=0, width=50, height=100), timestamp=0.0)}
    cerca = crear_evento(1, 0.0, x=2, y=0)
    lejos = crear_evento(1, 0.0, x=500, y=500)
    emparejados, sin_emparejar = emparejar(tracks, [lejos, cerca])
    assert emparejados[1] is cerca
    assert sin_emparejar == [lejos]


def test_emparejar_no_fuerza_por_debajo_del_umbral():
    """Mejor un track nuevo que un emparejamiento dudoso (ver docstring del modulo)."""
    tracks = {1: Track(bbox=BBox(x=0, y=0, width=50, height=100), timestamp=0.0)}
    lejos = crear_evento(1, 0.0, x=1000, y=1000)
    emparejados, sin_emparejar = emparejar(tracks, [lejos])
    assert emparejados == {}
    assert sin_emparejar == [lejos]


def test_emparejar_no_repite_ni_track_ni_deteccion():
    """Dos tracks casi en el mismo sitio: cada uno se lleva UNA deteccion, no las dos."""
    tracks = {
        1: Track(bbox=BBox(x=0, y=0, width=50, height=100), timestamp=0.0),
        2: Track(bbox=BBox(x=3, y=0, width=50, height=100), timestamp=0.0),
    }
    e1 = crear_evento(1, 0.0, x=0, y=0)
    e2 = crear_evento(1, 0.0, x=3, y=0)
    emparejados, sin_emparejar = emparejar(tracks, [e1, e2])
    assert set(id(e) for e in emparejados.values()) == {id(e1), id(e2)}
    assert sin_emparejar == []


# --------------------------------------------------------------------------
# asignar_track_ids: los tres problemas acordados en la Fase 1
# --------------------------------------------------------------------------

def test_una_persona_quieta_conserva_su_track_id():
    eventos = [crear_evento(f, f * 0.04, x=100, y=100) for f in range(5)]
    resumen = Resumen()
    ids = [e.track_id for e in asignar_track_ids(iter(eventos), resumen)]
    assert ids == [1, 1, 1, 1, 1]
    assert resumen.tracks_creados == 1


def test_dos_personas_que_se_cruzan_no_intercambian_id():
    """Problema 1 (oclusion). A camina a velocidad constante hacia la derecha,
    B hacia la izquierda; en el frame 3 sus cajas se solapan fuertemente. La
    prediccion de movimiento (no la ultima posicion cruda) es lo que permite
    distinguir cual caja es cual incluso durante el solape."""
    posiciones_a = [0, 15, 30, 45, 60]     # camina a la derecha, velocidad constante
    posiciones_b = [100, 85, 70, 55, 40]   # camina a la izquierda, velocidad constante
    # en el frame 3 las cajas se solapan fuertemente (A=[45,85], B=[55,95])

    eventos = []
    for f, (xa, xb) in enumerate(zip(posiciones_a, posiciones_b)):
        eventos.append(crear_evento(f, f * 0.04, x=xa, y=0))
        eventos.append(crear_evento(f, f * 0.04, x=xb, y=0))

    resumen = Resumen()
    salida = list(asignar_track_ids(iter(eventos), resumen))

    id_a = salida[0].track_id   # el que arranco en x=0
    id_b = salida[1].track_id   # el que arranco en x=100
    assert id_a != id_b
    assert resumen.tracks_creados == 2  # nadie se confundio con un track nuevo

    por_frame_y_x = {(e.frame, round(e.detection.bbox.x)): e.track_id for e in salida}
    assert por_frame_y_x[(4, 60)] == id_a   # A termino a la derecha con SU id
    assert por_frame_y_x[(4, 40)] == id_b   # B termino a la izquierda con SU id


def test_un_hueco_corto_conserva_el_track_id():
    """Problema 2: la deteccion del frame 1 se 'perdio' (confianza baja), pero
    el tiempo real transcurrido sigue dentro de MAX_AGE_S."""
    eventos = [
        crear_evento(0, 0.0, x=100, y=100),
        crear_evento(2, MAX_AGE_S - 0.1, x=100, y=100),
    ]
    ids = [e.track_id for e in asignar_track_ids(iter(eventos), Resumen())]
    assert ids == [1, 1]


def test_un_hueco_mas_largo_que_max_age_crea_un_track_nuevo():
    """Aunque el archivo pase de un frame al 'siguiente', si en tiempo real
    paso mas de MAX_AGE_S, la persona se da por perdida: es lo esperado, no un
    fallo. La deteccion de vuelta abre un track nuevo."""
    eventos = [
        crear_evento(0, 0.0, x=100, y=100),
        crear_evento(1, MAX_AGE_S + 0.1, x=100, y=100),
    ]
    resumen = Resumen()
    ids = [e.track_id for e in asignar_track_ids(iter(eventos), resumen)]
    assert ids == [1, 2]
    assert resumen.tracks_creados == 2


def test_dos_personas_lejanas_nunca_comparten_id():
    eventos = [
        crear_evento(0, 0.00, x=0, y=0),
        crear_evento(0, 0.00, x=1000, y=1000),
        crear_evento(1, 0.04, x=5, y=0),
        crear_evento(1, 0.04, x=1005, y=1000),
    ]
    ids = [e.track_id for e in asignar_track_ids(iter(eventos), Resumen())]
    assert ids[0] == ids[2]
    assert ids[1] == ids[3]
    assert ids[0] != ids[1]


def test_track_no_toca_los_campos_de_otras_personas():
    """Lo mas importante de esta etapa: solo escribe track_id."""
    e = crear_evento(0, 0.0, x=10, y=10)
    e.zone.zone_id = "gondola_A"
    e.interaction.product_zone = "bebidas"
    e.metrics.dwell_time = 3.5

    salida = next(iter(asignar_track_ids(iter([e]), Resumen())))
    assert salida.zone.zone_id == "gondola_A"
    assert salida.interaction.product_zone == "bebidas"
    assert salida.metrics.dwell_time == 3.5
    assert salida.detection.confidence == 0.9
    assert salida.track_id == 1


def test_el_resumen_empieza_en_cero():
    r = Resumen()
    assert (r.eventos_procesados, r.tracks_creados, r.emparejamientos) == (0, 0, 0)


# --------------------------------------------------------------------------
# El resultado no puede depender de FRAME_STRIDE: solo del timestamp real.
# --------------------------------------------------------------------------

def test_el_resultado_no_depende_del_frame_stride():
    """Si MAX_AGE se contara en frames del archivo, la misma pausa real
    cambiaria de tamano segun el stride con el que corrio `detect`. Dos
    secuencias con los MISMOS timestamps pero numeros de frame distintos
    (una 'consecutiva', otra como si viniera de --stride 5) deben dar
    exactamente el mismo resultado."""
    consecutivos = [
        crear_evento(10, 0.40, x=100, y=100),
        crear_evento(11, 0.44, x=105, y=100),
        crear_evento(13, 0.52, x=115, y=100),  # un frame se perdio en el medio
    ]
    salteados = [
        crear_evento(50, 0.40, x=100, y=100),
        crear_evento(55, 0.44, x=105, y=100),
        crear_evento(65, 0.52, x=115, y=100),
    ]

    ids_consecutivos = [e.track_id for e in asignar_track_ids(iter(consecutivos), Resumen())]
    ids_salteados = [e.track_id for e in asignar_track_ids(iter(salteados), Resumen())]

    assert ids_consecutivos == [1, 1, 1]
    assert ids_salteados == [1, 1, 1]
    assert ids_consecutivos == ids_salteados


# --------------------------------------------------------------------------
# run(): lectura y escritura reales, en streaming
# --------------------------------------------------------------------------

def test_run_asigna_track_id_y_preserva_lo_demas(tmp_path):
    cfg = load_config(env={"VIDEO_ID": "video_001", "OUTPUT_DIR": str(tmp_path / "output")})
    rutas = pipeline.stage_paths("track", cfg)
    write_events(rutas.input_path, [
        crear_evento(0, 0.00, x=0, y=0),
        crear_evento(1, 0.04, x=5, y=0),
    ])

    assert run(cfg) == 0

    salida = list(read_events(rutas.output_path))
    assert [e.track_id for e in salida] == [1, 1]
    assert all(e.detection.confidence == 0.9 for e in salida)

    resumen = json.loads(pipeline.summary_path("track", cfg).read_text(encoding="utf-8"))
    assert resumen["stage"] == "track"
    assert resumen["results"]["tracks_creados"] == 1
    assert resumen["results"]["emparejamientos"] == 1


def test_run_sin_su_entrada_dice_que_corras_detect_antes(tmp_path):
    cfg = load_config(env={"VIDEO_ID": "video_001", "OUTPUT_DIR": str(tmp_path / "output")})
    with pytest.raises(MissingInputError) as error:
        run(cfg)
    assert "python -m gondola detect" in str(error.value)
