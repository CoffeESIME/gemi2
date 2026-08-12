import time
import logging

# Configurar el logging a nivel INFO
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def main():
    telefono_detectado = True
    contador_detecciones = 0

    logging.info("Iniciando simulador de detección de teléfonos...")

    try:
        while True:
            logging.info("Simulando captura de imagen y análisis de visión artificial...")
            
            if telefono_detectado:
                contador_detecciones += 1
                logging.info(f"Teléfono detectado. Contador de debouncing: {contador_detecciones}/3")
                
                if contador_detecciones >= 3:
                    logging.warning("¡ALERTA! Se ha detectado el uso de un teléfono (3 veces consecutivas). Acción disparada.")
                    contador_detecciones = 0
            else:
                if contador_detecciones > 0:
                    logging.info("Teléfono no detectado. Reseteando el contador.")
                contador_detecciones = 0
            
            # Pausa de exactamente 3 segundos al final de la iteración
            time.sleep(3)
            
    except KeyboardInterrupt:
        logging.info("Se capturó KeyboardInterrupt. Deteniendo el simulador de forma segura...")

if __name__ == "__main__":
    main()
