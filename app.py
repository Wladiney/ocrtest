from fastapi import FastAPI, File, UploadFile, HTTPException
from typing import Dict, Optional
import pytesseract
from PIL import Image, ImageEnhance, ImageFilter
import io
import re
import uvicorn
import logging
import numpy as np
import cv2

# Configuração de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="API de Extração de Cupom Fiscal",
              description="API que recebe imagem de cupom fiscal e retorna o valor total")

def detectar_e_cortar_cupom(imagem_np):
    """
    Detecta e corta apenas a região do cupom fiscal na imagem
    """
    # Converter para escala de cinza
    cinza = cv2.cvtColor(imagem_np, cv2.COLOR_BGR2GRAY)
    
    # Aplicar desfoque para reduzir ruído
    desfoque = cv2.GaussianBlur(cinza, (5, 5), 0)
    
    # Detecção de bordas com Canny
    bordas = cv2.Canny(desfoque, 50, 150)
    
    # Dilatação para melhorar a detecção de contornos
    kernel = np.ones((5, 5), np.uint8)
    bordas_dilatadas = cv2.dilate(bordas, kernel, iterations=1)
    
    # Encontrar contornos
    contornos, _ = cv2.findContours(bordas_dilatadas, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Se não encontrou contornos, retornar a imagem original
    if not contornos:
        logger.warning("Não foi possível detectar contornos do cupom fiscal")
        return imagem_np
    
    # Encontrar o maior contorno (assumindo que é o cupom)
    maior_contorno = max(contornos, key=cv2.contourArea)
    
    # Obter o retângulo que envolve o contorno
    x, y, w, h = cv2.boundingRect(maior_contorno)
    
    # Verificar se o retângulo é grande o suficiente (evitar falsos positivos)
    area_imagem = imagem_np.shape[0] * imagem_np.shape[1]
    area_contorno = w * h
    
    # O contorno deve ter pelo menos 10% da área total da imagem
    if area_contorno < (area_imagem * 0.1):
        logger.warning("O contorno detectado é muito pequeno, usando imagem original")
        return imagem_np
    
    # Cortar a imagem original
    imagem_cortada = imagem_np[y:y+h, x:x+w]
    
    logger.info(f"Cupom detectado e cortado com dimensões: {w}x{h}")
    return imagem_cortada

def melhorar_imagem_para_ocr(imagem):
    """
    Aplica técnicas de pré-processamento para melhorar a qualidade da imagem para OCR
    """
    # Converter para array numpy se for uma imagem PIL
    if isinstance(imagem, Image.Image):
        imagem_np = np.array(imagem)
    else:
        imagem_np = imagem
    
    # Converter para escala de cinza se ainda não estiver
    if len(imagem_np.shape) == 3:
        cinza = cv2.cvtColor(imagem_np, cv2.COLOR_BGR2GRAY)
    else:
        cinza = imagem_np
    
    # Aplicar threshold adaptativo para melhorar contraste
    thresh = cv2.adaptiveThreshold(
        cinza, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )
    
    # Remover ruído com filtro mediano
    sem_ruido = cv2.medianBlur(thresh, 3)
    
    # Aplicar operações morfológicas para limpar a imagem
    kernel = np.ones((1, 1), np.uint8)
    morph = cv2.morphologyEx(sem_ruido, cv2.MORPH_CLOSE, kernel)
    
    # Converter de volta para imagem PIL
    return Image.fromarray(morph)

@app.post("/extrair-valor/", response_model=Dict[str, float])
async def extrair_valor_cupom(file: UploadFile = File(...), debug: bool = False):
    """
    Recebe uma imagem de cupom fiscal e retorna o valor total encontrado
    Se debug=True, retorna também o texto extraído pelo OCR
    """
    # Verificar se o arquivo é uma imagem
    content_type = file.content_type
    if not content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="O arquivo enviado não é uma imagem")
    
    try:
        # Ler o conteúdo do arquivo
        contents = await file.read()
        
        # Converter para formato que o OpenCV pode usar
        imagem_np = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)
        
        # Detectar e cortar o cupom
        logger.info("Detectando e cortando a região do cupom fiscal")
        cupom_cortado = detectar_e_cortar_cupom(imagem_np)
        
        # Melhorar a imagem para OCR
        logger.info("Aplicando melhorias na imagem para OCR")
        imagem_melhorada = melhorar_imagem_para_ocr(cupom_cortado)
        
        # Extrair texto da imagem usando OCR
        logger.info("Iniciando extração de texto com OCR")
        texto_extraido = pytesseract.image_to_string(imagem_melhorada, lang="por", config='--psm 6')
        logger.info(f"Texto extraído: {texto_extraido[:100]}...")  # Logging dos primeiros 100 caracteres
        
        # Tentar encontrar o valor total
        valor = extrair_valor_total(texto_extraido)
        
        if valor is not None:
            logger.info(f"Valor total encontrado: {valor}")
            response = {"valor_total": valor}
            if debug:
                response["texto_extraido"] = texto_extraido
            return response
        else:
            logger.warning("Valor total não encontrado no cupom")
            raise HTTPException(status_code=404, detail="Não foi possível encontrar o valor total no cupom")
            
    except Exception as e:
        logger.error(f"Erro ao processar a imagem: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erro ao processar a imagem: {str(e)}")
    finally:
        await file.close()

def extrair_valor_total(texto: str) -> Optional[float]:
    """
    Extrai o valor total do texto do cupom fiscal
    """
    # Padrões comuns para identificar o valor total em cupons fiscais brasileiros
    padroes = [
        r"VALOR TOTAL\s*R?\$?\s*(\d+[.,]\d+)",
        r"TOTAL\s*R?\$?\s*(\d+[.,]\d+)",
        r"TOTAL A PAGAR\s*R?\$?\s*(\d+[.,]\d+)",
        r"VALOR A PAGAR\s*R?\$?\s*(\d+[.,]\d+)",
        r"TOTAL:\s*R?\$?\s*(\d+[.,]\d+)",
        r"VL TOTAL\s*R?\$?\s*(\d+[.,]\d+)"
    ]
    
    # Tentar encontrar o valor usando os padrões
    for padrao in padroes:
        matches = re.search(padrao, texto, re.IGNORECASE)
        if matches:
            # Converter o valor encontrado para float
            valor_str = matches.group(1).replace(",", ".")
            return float(valor_str)
    
    # Se nenhum padrão tradicional encontrar, tentar uma abordagem mais genérica
    # Procurar por valores precedidos por R$ próximos às palavras relacionadas a total
    linhas = texto.splitlines()
    for i, linha in enumerate(linhas):
        if any(palavra in linha.upper() for palavra in ["TOTAL", "VALOR", "PAGAR"]):
            # Procurar por valores monetários nesta linha
            valores = re.findall(r"R?\$?\s*(\d+[.,]\d+)", linha)
            if valores:
                # Pegar o último valor da linha (geralmente é o total)
                valor_str = valores[-1].replace(",", ".")
                return float(valor_str)
            
            # Se não encontrou na linha atual, verificar as próximas 2 linhas
            for j in range(1, 3):
                if i + j < len(linhas):
                    valores = re.findall(r"R?\$?\s*(\d+[.,]\d+)", linhas[i + j])
                    if valores:
                        valor_str = valores[-1].replace(",", ".")
                        return float(valor_str)
    
    # Se não encontrou o valor
    return None

@app.post("/debug-processamento/")
async def debug_processamento(file: UploadFile = File(...)):
    """
    Endpoint para debug que mostra o resultado do pré-processamento da imagem
    Retorna base64 das imagens em cada etapa do processo
    """
    import base64
    from io import BytesIO
    
    # Verificar se o arquivo é uma imagem
    content_type = file.content_type
    if not content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="O arquivo enviado não é uma imagem")
    
    try:
        # Ler o conteúdo do arquivo
        contents = await file.read()
        
        # Converter para formato que o OpenCV pode usar
        imagem_original = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)
        
        # Detectar e cortar o cupom
        imagem_cortada = detectar_e_cortar_cupom(imagem_original)
        
        # Melhorar a imagem para OCR
        imagem_melhorada = melhorar_imagem_para_ocr(imagem_cortada)
        
        # Converter imagens para base64
        # Imagem original
        _, buffer_original = cv2.imencode('.png', imagem_original)
        img_original_base64 = base64.b64encode(buffer_original).decode('utf-8')
        
        # Imagem cortada
        _, buffer_cortada = cv2.imencode('.png', imagem_cortada)
        img_cortada_base64 = base64.b64encode(buffer_cortada).decode('utf-8')
        
        # Imagem melhorada
        buffered = BytesIO()
        imagem_melhorada.save(buffered, format="PNG")
        img_melhorada_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        
        # Texto extraído
        texto_extraido = pytesseract.image_to_string(imagem_melhorada, lang="por", config='--psm 6')
        
        return {
            "imagem_original": img_original_base64,
            "imagem_cortada": img_cortada_base64,
            "imagem_melhorada": img_melhorada_base64,
            "texto_extraido": texto_extraido
        }
        
    except Exception as e:
        logger.error(f"Erro ao processar a imagem para debug: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erro ao processar a imagem: {str(e)}")
    finally:
        await file.close()

if __name__ == "__main__":
    # Iniciar o servidor quando executado diretamente
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
