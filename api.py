from fastapi import FastAPI, File, UploadFile, HTTPException
from typing import Dict
import pytesseract
from PIL import Image
import io
import re
import uvicorn
import logging

# Configuração de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="API de Extração de Cupom Fiscal",
              description="API que recebe imagem de cupom fiscal e retorna o valor total")

@app.post("/extrair-valor/", response_model=Dict[str, float])
async def extrair_valor_cupom(file: UploadFile = File(...)):
    """
    Recebe uma imagem de cupom fiscal e retorna o valor total encontrado
    """
    # Verificar se o arquivo é uma imagem
    content_type = file.content_type
    if not content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="O arquivo enviado não é uma imagem")
    
    try:
        # Ler o conteúdo do arquivo
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        # Extrair texto da imagem usando OCR
        logger.info("Iniciando extração de texto com OCR")
        texto_extraido = pytesseract.image_to_string(image, lang="por")
        logger.info(f"Texto extraído: {texto_extraido[:100]}...")  # Logging dos primeiros 100 caracteres
        
        # Tentar encontrar o valor total
        valor = extrair_valor_total(texto_extraido)
        
        if valor is not None:
            logger.info(f"Valor total encontrado: {valor}")
            return {"valor_total": valor}
        else:
            logger.warning("Valor total não encontrado no cupom")
            raise HTTPException(status_code=404, detail="Não foi possível encontrar o valor total no cupom")
            
    except Exception as e:
        logger.error(f"Erro ao processar a imagem: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erro ao processar a imagem: {str(e)}")
    finally:
        await file.close()

def extrair_valor_total(texto: str) -> float:
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

if __name__ == "__main__":
    # Iniciar o servidor quando executado diretamente
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
