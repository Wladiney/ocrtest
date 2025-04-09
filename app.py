from fastapi import FastAPI, File, UploadFile, HTTPException
from typing import Dict, Optional
import pytesseract
from PIL import Image, ImageEnhance
import io
import re
import uvicorn
import logging
import numpy as np
import cv2
from functools import lru_cache

# Configuração de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="API de Extração de Cupom Fiscal",
              description="API que recebe imagem de cupom fiscal e retorna o valor total")

# Tamanho máximo para redimensionamento (preserva a proporção)
MAX_SIZE = 1200

def redimensionar_imagem(imagem_np, max_size=MAX_SIZE):
    """
    Redimensiona a imagem mantendo a proporção para melhorar a performance
    """
    altura, largura = imagem_np.shape[:2]
    
    # Se já é menor que o tamanho máximo, não faz nada
    if max(altura, largura) <= max_size:
        return imagem_np
    
    # Determina a escala para redimensionar
    if altura > largura:
        nova_altura = max_size
        escala = nova_altura / altura
        nova_largura = int(largura * escala)
    else:
        nova_largura = max_size
        escala = nova_largura / largura
        nova_altura = int(altura * escala)
    
    # Redimensiona a imagem
    return cv2.resize(imagem_np, (nova_largura, nova_altura), interpolation=cv2.INTER_AREA)

def detectar_e_cortar_cupom(imagem_np):
    """
    Detecta e corta apenas a região do cupom fiscal na imagem, com otimizações de performance
    """
    # Redimensionar imagem para melhorar performance
    imagem_np = redimensionar_imagem(imagem_np)
    
    # Converter para escala de cinza
    cinza = cv2.cvtColor(imagem_np, cv2.COLOR_BGR2GRAY)
    
    # Aplicar desfoque mais leve
    desfoque = cv2.GaussianBlur(cinza, (3, 3), 0)
    
    # Detecção de bordas com Canny - ajustando limiares para melhor performance
    bordas = cv2.Canny(desfoque, 30, 200)
    
    # Dilatação mais leve e rápida
    kernel = np.ones((3, 3), np.uint8)
    bordas_dilatadas = cv2.dilate(bordas, kernel, iterations=1)
    
    # Encontrar contornos usando RETR_EXTERNAL para só pegar contornos externos
    contornos, _ = cv2.findContours(bordas_dilatadas, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Se não encontrou contornos, retornar a imagem original
    if not contornos:
        logger.warning("Não foi possível detectar contornos do cupom fiscal")
        return imagem_np
    
    # Filtrar contornos pequenos antes de procurar o maior
    area_minima = imagem_np.shape[0] * imagem_np.shape[1] * 0.05  # 5% da área total
    contornos_grandes = [c for c in contornos if cv2.contourArea(c) > area_minima]
    
    if not contornos_grandes:
        logger.warning("Nenhum contorno grande o suficiente foi encontrado")
        return imagem_np
    
    # Encontrar o maior contorno (assumindo que é o cupom)
    maior_contorno = max(contornos_grandes, key=cv2.contourArea)
    
    # Obter o retângulo que envolve o contorno
    x, y, w, h = cv2.boundingRect(maior_contorno)
    
    # Cortar a imagem original
    imagem_cortada = imagem_np[y:y+h, x:x+w]
    
    logger.info(f"Cupom detectado e cortado com dimensões: {w}x{h}")
    return imagem_cortada

def melhorar_imagem_para_ocr(imagem):
    """
    Aplica técnicas otimizadas de pré-processamento para OCR
    """
    # Converter para array numpy se for uma imagem PIL
    if isinstance(imagem, Image.Image):
        imagem_np = np.array(imagem)
    else:
        imagem_np = imagem
    
    # Redimensionar para tamanho menor se for muito grande
    imagem_np = redimensionar_imagem(imagem_np, max_size=800)  # Tamanho menor para processamento OCR
    
    # Converter para escala de cinza se ainda não estiver
    if len(imagem_np.shape) == 3:
        cinza = cv2.cvtColor(imagem_np, cv2.COLOR_BGR2GRAY)
    else:
        cinza = imagem_np
    
    # Aplicar binarização de Otsu - mais rápido que threshold adaptativo
    _, thresh = cv2.threshold(cinza, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Converter de volta para imagem PIL
    return Image.fromarray(thresh)

@lru_cache(maxsize=32)
def carregar_padroes_regex():
    """
    Carrega e compila os padrões regex - usando cache para melhorar performance
    """
    padroes_texto = [
        r"VALOR TOTAL\s*R?\$?\s*(\d+[.,]\d+)",
        r"TOTAL\s*R?\$?\s*(\d+[.,]\d+)",
        r"TOTAL A PAGAR\s*R?\$?\s*(\d+[.,]\d+)",
        r"VALOR A PAGAR\s*R?\$?\s*(\d+[.,]\d+)",
        r"TOTAL:\s*R?\$?\s*(\d+[.,]\d+)",
        r"VL TOTAL\s*R?\$?\s*(\d+[.,]\d+)"
    ]
    return [re.compile(p, re.IGNORECASE) for p in padroes_texto]

def extrair_valor_total(texto: str) -> Optional[float]:
    """
    Extrai o valor total do texto do cupom fiscal - versão otimizada
    """
    # Usar padrões compilados do cache
    padroes = carregar_padroes_regex()
    
    # Tentar encontrar o valor usando os padrões
    for padrao in padroes:
        matches = padrao.search(texto)
        if matches:
            # Converter o valor encontrado para float
            valor_str = matches.group(1).replace(",", ".")
            return float(valor_str)
    
    # Se nenhum padrão tradicional encontrar, tentar uma abordagem mais simples e eficiente
    # Busca por palavras específicas e valores próximos em vez de procurar linha por linha
    palavras_chave = ["TOTAL", "VALOR", "PAGAR"]
    for palavra in palavras_chave:
        # Encontrar índice da palavra-chave
        idx = texto.upper().find(palavra)
        if idx >= 0:
            # Buscar por números decimais nos próximos 30 caracteres
            sub_texto = texto[idx:idx+30]
            valores = re.findall(r"R?\$?\s*(\d+[.,]\d+)", sub_texto)
            if valores:
                valor_str = valores[-1].replace(",", ".")
                return float(valor_str)
    
    # Se não encontrou o valor
    return None

@app.post("/extrair-valor/", response_model=Dict[str, float])
async def extrair_valor_cupom(file: UploadFile = File(...), debug: bool = False):
    """
    Recebe uma imagem de cupom fiscal e retorna o valor total encontrado
    Versão otimizada para melhor performance
    """
    # Verificar se o arquivo é uma imagem
    content_type = file.content_type
    if not content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="O arquivo enviado não é uma imagem")
    
    try:
        # Ler o conteúdo do arquivo em chunks para economizar memória
        chunk_size = 1024 * 1024  # 1MB chunks
        chunks = []
        while True:
            chunk = await file.read(chunk_size)
            if not chunk:
                break
            chunks.append(chunk)
        
        # Concatenar os chunks
        contents = b''.join(chunks)
        
        # Converter para formato que o OpenCV pode usar
        imagem_np = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)
        
        # Redimensionar imagem grande para economizar recursos
        imagem_np = redimensionar_imagem(imagem_np)
        
        # Detectar e cortar o cupom
        logger.info("Detectando e cortando a região do cupom fiscal")
        cupom_cortado = detectar_e_cortar_cupom(imagem_np)
        
        # Liberar a memória da imagem original
        del imagem_np
        
        # Melhorar a imagem para OCR
        logger.info("Aplicando melhorias na imagem para OCR")
        imagem_melhorada = melhorar_imagem_para_ocr(cupom_cortado)
        
        # Liberar a memória da imagem cortada
        del cupom_cortado
        
        # Extrair texto da imagem usando OCR com configurações otimizadas
        logger.info("Iniciando extração de texto com OCR")
        # Usar configurações mais rápidas para o Tesseract
        texto_extraido = pytesseract.image_to_string(
            imagem_melhorada, 
            lang="por", 
            config='--psm 6 --oem 1'  # Modo de segmentação de página 6 e engine OCR legacy (mais rápido)
        )
        
        # Liberar a memória da imagem melhorada
        del imagem_melhorada
        
        # Tentar encontrar o valor total
        valor = extrair_valor_total(texto_extraido)
        
        if valor is not None:
            logger.info(f"Valor total encontrado: {valor}")
            response = {"valor_total": valor}
            if debug:
                # Incluir apenas os primeiros 500 caracteres do texto extraído para economizar bytes
                response["texto_extraido"] = texto_extraido[:500]
            return response
        else:
            logger.warning("Valor total não encontrado no cupom")
            raise HTTPException(status_code=404, detail="Não foi possível encontrar o valor total no cupom")
            
    except Exception as e:
        logger.error(f"Erro ao processar a imagem: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erro ao processar a imagem: {str(e)}")
    finally:
        await file.close()

@app.post("/debug-processamento-light/")
async def debug_processamento_light(file: UploadFile = File(...)):
    """
    Versão otimizada do endpoint de debug que consome menos recursos
    Retorna apenas o texto extraído e uma versão em baixa resolução das imagens
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
        
        # Redimensionar significativamente para economizar recursos - versão debug não precisa de alta resolução
        imagem_original = redimensionar_imagem(imagem_original, max_size=600)
        
        # Detectar e cortar o cupom
        imagem_cortada = detectar_e_cortar_cupom(imagem_original)
        
        # Melhorar a imagem para OCR
        imagem_melhorada = melhorar_imagem_para_ocr(imagem_cortada)
        
        # Extrair texto
        texto_extraido = pytesseract.image_to_string(imagem_melhorada, lang="por", config='--psm 6 --oem 1')
        
        # Redimensionar imagens para thumbnails antes de convertê-las para base64
        thumb_size = 300
        
        # Imagem original thumbnail
        altura_orig, largura_orig = imagem_original.shape[:2]
        escala_orig = min(thumb_size/largura_orig, thumb_size/altura_orig)
        thumb_original = cv2.resize(
            imagem_original, 
            (int(largura_orig*escala_orig), int(altura_orig*escala_orig)), 
            interpolation=cv2.INTER_AREA
        )
        
        # Imagem cortada thumbnail
        altura_cort, largura_cort = imagem_cortada.shape[:2]
        escala_cort = min(thumb_size/largura_cort, thumb_size/altura_cort)
        thumb_cortada = cv2.resize(
            imagem_cortada, 
            (int(largura_cort*escala_cort), int(altura_cort*escala_cort)), 
            interpolation=cv2.INTER_AREA
        )
        
        # Converter imagens para base64 com JPEG para reduzir tamanho
        _, buffer_original = cv2.imencode('.jpg', thumb_original, [cv2.IMWRITE_JPEG_QUALITY, 75])
        img_original_base64 = base64.b64encode(buffer_original).decode('utf-8')
        
        _, buffer_cortada = cv2.imencode('.jpg', thumb_cortada, [cv2.IMWRITE_JPEG_QUALITY, 75])
        img_cortada_base64 = base64.b64encode(buffer_cortada).decode('utf-8')
        
        # Imagem melhorada (é um PIL Image)
        imagem_melhorada_np = np.array(imagem_melhorada)
        altura_melh, largura_melh = imagem_melhorada_np.shape[:2]
        escala_melh = min(thumb_size/largura_melh, thumb_size/altura_melh)
        thumb_melhorada = cv2.resize(
            imagem_melhorada_np, 
            (int(largura_melh*escala_melh), int(altura_melh*escala_melh)), 
            interpolation=cv2.INTER_AREA
        )
        
        _, buffer_melhorada = cv2.imencode('.jpg', thumb_melhorada, [cv2.IMWRITE_JPEG_QUALITY, 75])
        img_melhorada_base64 = base64.b64encode(buffer_melhorada).decode('utf-8')
        
        # Limitar o texto extraído a 1000 caracteres para economizar bytes
        texto_limitado = texto_extraido[:1000] if len(texto_extraido) > 1000 else texto_extraido
        
        return {
            "imagem_original": img_original_base64,
            "imagem_cortada": img_cortada_base64,
            "imagem_melhorada": img_melhorada_base64,
            "texto_extraido": texto_limitado
        }
        
    except Exception as e:
        logger.error(f"Erro ao processar a imagem para debug: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erro ao processar a imagem: {str(e)}")
    finally:
        await file.close()

if __name__ == "__main__":
    # Iniciar o servidor quando executado diretamente
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
