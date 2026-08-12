FROM python:3.11-slim

# Establecer el directorio de trabajo
WORKDIR /app

# Copiar solo el archivo de requerimientos primero para aprovechar la caché de Docker
COPY requirements.txt .

# Instalar las librerías de Python
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el script principal
COPY main.py .

# Comando por defecto para ejecutar el contenedor
CMD ["python", "main.py"]