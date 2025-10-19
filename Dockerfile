# Usa uma imagem Python 3.12 (pois 3.14 ainda é novo)
FROM python:3.12-slim

# Define o diretório de trabalho
WORKDIR /app

# --- [NOVO] Instala dependências do sistema para OCR ---
# Atualiza os pacotes do Linux (Debian) e instala Tesseract (com idioma Português) e Poppler
RUN apt-get update && \
    apt-get install -y \
    tesseract-ocr \
    tesseract-ocr-por \
    poppler-utils \
    --no-install-recommends && \
    rm -rf /var/lib/apt/lists/*
# --- [FIM NOVO] ---

# Copia o arquivo de requisitos PRIMEIRO
COPY requirements.txt .

# Instala as dependências do Python (usando a lista atualizada)
RUN pip install --no-cache-dir -r requirements.txt

# Copia o resto do código do aplicativo (main.py, etc.)
COPY . .

# Expõe a porta (O Cloud Run usa $PORT, mas 8080 é um padrão comum)
EXPOSE 8080

# Comando para iniciar o servidor Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]