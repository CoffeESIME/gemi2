"""
Phone Tracker - Detección de uso de teléfono celular con YOLOv8 Nano.

Este script captura frames de la cámara, ejecuta inferencia con YOLOv8n
y aplica una heurística espacial + filtro temporal (debouncing) para
determinar si una persona está usando activamente un teléfono celular.

Clases COCO relevantes:
    - 0:  person
    - 67: cell phone
"""

import time
import logging
from ultralytics import YOLO
import cv2

# ---------------------------------------------------------------------------
# Configuración de logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# ---------------------------------------------------------------------------
# Constantes del modelo y umbrales
# ---------------------------------------------------------------------------
MODELO_PATH: str = "yolov8n.pt"  # Se descarga automáticamente la primera vez
CLASE_PERSONA: int = 0
CLASE_CELULAR: int = 67
UMBRAL_CONFIANZA: float = 0.40          # Confianza mínima para aceptar una detección
UMBRAL_SOLAPAMIENTO: float = 0.50       # 50 % del área del celular debe estar dentro
DETECCIONES_NECESARIAS: int = 3         # Frames consecutivos para disparar la alerta
INTERVALO_CAPTURA: float = 3.0          # Segundos entre cada captura


# ---------------------------------------------------------------------------
# Funciones auxiliares de geometría
# ---------------------------------------------------------------------------

def calcular_solapamiento(box_celular: list[float], mitad_superior_persona: list[float]) -> float:
    """
    Calcula la fracción del área del bounding box del celular que se
    intersecta con la mitad superior del bounding box de la persona.

    Heurística: si el celular está en la mitad superior del cuerpo
    (cabeza, torso, manos levantadas) es probable que la persona lo
    esté usando activamente.

    Parámetros
    ----------
    box_celular : list[float]
        Coordenadas [x1, y1, x2, y2] del bounding box del celular.
    mitad_superior_persona : list[float]
        Coordenadas [x1, y1, x2, y2] de la mitad superior del
        bounding box de la persona.

    Retorna
    -------
    float
        Fracción de solapamiento en el rango [0.0, 1.0].
        0.0 = sin intersección, 1.0 = el celular está completamente
        dentro de la mitad superior de la persona.

    Geometría de la intersección (AABB)
    ------------------------------------
    Dados dos rectángulos A y B alineados a los ejes:

        inter_x1 = max(A.x1, B.x1)
        inter_y1 = max(A.y1, B.y1)
        inter_x2 = min(A.x2, B.x2)
        inter_y2 = min(A.y2, B.y2)

    Si inter_x1 < inter_x2 AND inter_y1 < inter_y2, existe
    intersección y su área es:

        area_inter = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)

    La fracción de solapamiento se obtiene dividiendo el área de
    intersección entre el área total del celular:

        solapamiento = area_inter / area_celular
    """
    cx1, cy1, cx2, cy2 = box_celular
    px1, py1, px2, py2 = mitad_superior_persona

    # Coordenadas de la intersección (AABB - Axis-Aligned Bounding Box)
    inter_x1 = max(cx1, px1)
    inter_y1 = max(cy1, py1)
    inter_x2 = min(cx2, px2)
    inter_y2 = min(cy2, py2)

    # Si no existe intersección, retornar 0
    if inter_x1 >= inter_x2 or inter_y1 >= inter_y2:
        return 0.0

    area_interseccion = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
    area_celular = (cx2 - cx1) * (cy2 - cy1)

    # Evitar división por cero en caso de bounding box degenerado
    if area_celular <= 0:
        return 0.0

    return area_interseccion / area_celular


def obtener_mitad_superior(box_persona: list[float]) -> list[float]:
    """
    Dada una bounding box [x1, y1, x2, y2] de una persona, retorna
    las coordenadas de su mitad superior.

    Matemática:
        punto_medio_y = y1 + (y2 - y1) / 2
        mitad_superior = [x1, y1, x2, punto_medio_y]

    Esto cubre la región de la cabeza al torso, donde normalmente
    se sostiene un teléfono celular.
    """
    x1, y1, x2, y2 = box_persona
    punto_medio_y = y1 + (y2 - y1) / 2.0
    return [x1, y1, x2, punto_medio_y]


# ---------------------------------------------------------------------------
# Función principal
# ---------------------------------------------------------------------------

def main() -> None:
    # Cargar modelo YOLOv8 Nano (se descarga automáticamente si no existe)
    logging.info("Cargando modelo %s...", MODELO_PATH)
    modelo = YOLO(MODELO_PATH)
    logging.info("Modelo cargado exitosamente.")

    # Abrir dispositivo de video (cámara por defecto)
    captura = cv2.VideoCapture(0)
    if not captura.isOpened():
        logging.error("No se pudo abrir el dispositivo de video /dev/video0.")
        return

    logging.info("Cámara abierta. Iniciando bucle de detección...")

    contador_detecciones: int = 0

    try:
        while True:
            ret, frame = captura.read()
            if not ret:
                logging.warning("No se pudo leer el frame de la cámara. Reintentando...")
                time.sleep(INTERVALO_CAPTURA)
                continue

            logging.info("Frame capturado. Ejecutando inferencia YOLOv8...")

            # ----- Inferencia -----
            resultados = modelo(frame, conf=UMBRAL_CONFIANZA, verbose=False)

            # Extraer bounding boxes por clase
            personas: list[list[float]] = []
            celulares: list[list[float]] = []

            for r in resultados:
                for box in r.boxes:
                    clase_id = int(box.cls[0])
                    coords = box.xyxy[0].tolist()  # [x1, y1, x2, y2]

                    if clase_id == CLASE_PERSONA:
                        personas.append(coords)
                    elif clase_id == CLASE_CELULAR:
                        celulares.append(coords)

            logging.info(
                "Detecciones — Personas: %d | Celulares: %d",
                len(personas),
                len(celulares),
            )

            # ----- Heurística espacial -----
            uso_detectado = False

            for box_celular in celulares:
                for box_persona in personas:
                    mitad_superior = obtener_mitad_superior(box_persona)
                    solapamiento = calcular_solapamiento(box_celular, mitad_superior)

                    logging.info(
                        "Solapamiento celular/persona: %.2f (umbral: %.2f)",
                        solapamiento,
                        UMBRAL_SOLAPAMIENTO,
                    )

                    if solapamiento >= UMBRAL_SOLAPAMIENTO:
                        uso_detectado = True
                        break  # Una coincidencia es suficiente

                if uso_detectado:
                    break

            # ----- Filtro temporal (debouncing) -----
            if uso_detectado:
                contador_detecciones += 1
                logging.info(
                    "Uso de teléfono detectado. Contador: %d/%d",
                    contador_detecciones,
                    DETECCIONES_NECESARIAS,
                )

                if contador_detecciones >= DETECCIONES_NECESARIAS:
                    logging.warning(
                        "¡ACCIÓN DISPARADA! Uso de teléfono confirmado "
                        "(%d detecciones consecutivas).",
                        DETECCIONES_NECESARIAS,
                    )
                    contador_detecciones = 0
            else:
                if contador_detecciones > 0:
                    logging.info(
                        "Uso de teléfono NO detectado. Reseteando contador."
                    )
                contador_detecciones = 0

            # Pausa para evitar sobrecalentamiento del procesador
            time.sleep(INTERVALO_CAPTURA)

    except KeyboardInterrupt:
        logging.info("Interrupción recibida. Deteniendo el detector...")
    finally:
        captura.release()
        logging.info("Recurso de cámara liberado. Fin del programa.")


if __name__ == "__main__":
    main()
