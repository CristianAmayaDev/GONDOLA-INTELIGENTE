"""Excepciones propias del proyecto.

Sirven para distinguir "el proyecto detecto un problema y sabe explicarlo" de
"algo se rompio en una libreria". Regla de oro: el mensaje debe decir QUE HACER,
no solo que fallo.

Solo existen las que se usan hoy. Cuando la Fase 2 abra videos y la Fase 3
cargue el modelo, se anadiran `VideoError` y `ModelError`.
"""


class GondolaError(Exception):
    """Error base del proyecto. Captura esto para atrapar todo lo nuestro."""


class ConfigError(GondolaError):
    """La configuracion (.env) esta incompleta o fuera de rango."""


class ContractError(GondolaError):
    """Un evento no cumple el contrato de datos o no se pudo leer."""
