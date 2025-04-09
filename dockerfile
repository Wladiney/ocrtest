FROM python:3.9-slim

# Instalar dependências do sistema, incluindo o Tesseract OCR e suporte para português
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    tesseract-ocr-por \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Definir diretório de trabalho
WORKDIR /app

# Copiar os arquivos de requisitos e instalar dependências Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar o código da aplicação
COPY app.py .

# Expor a porta que a aplicação vai usar
EXPOSE 8000

# Comando para iniciar a aplicação
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
