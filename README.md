# Gondola Inteligente

Percepcion visual soberana para optimizar el espacio en retail. Reto de Scapder.

A partir del video de una tienda, el sistema cuantifica la dinamica de los
clientes alrededor de las gondolas (flujo, permanencia e interaccion con
productos) para recomendar mejoras de planograma. **No identifica personas, no
reconoce rostros, no infiere emociones y no crea perfiles biometricos.**

## Estado: Fase 1 de 5 (arquitectura y contrato de datos)

Todavia no hay pipeline ejecutable. Lo que existe hoy:

- El **contrato de datos** (`ai-service/gondola/contract.py`), documentado en
  [`docs/data-contract.md`](docs/data-contract.md).
- La configuracion centralizada (`ai-service/gondola/config.py`).
- Tests unitarios en verde.

## Empezar

```bash
cp .env.example .env
pip install -r requirements-dev.txt
pytest
```

Para trabajar con el video: dejalo en `data/videos/` (ver el README de esa
carpeta) y ajusta `VIDEO_PATH` en tu `.env`.

## Documentacion

- [Contrato de datos](docs/data-contract.md) — **empieza por aqui**.
- [CLAUDE.md](CLAUDE.md) — reglas y convenciones del proyecto.

> README completo (arquitectura, uso, resultados): Fase 5.
