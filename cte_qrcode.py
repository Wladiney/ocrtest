from fastapi import APIRouter, File, UploadFile, HTTPException
import cv2
import numpy as np
import requests
from bs4 import BeautifulSoup
import re
import io
from typing import Dict, Any
import logging
from pyzbar.pyzbar import decode  # Para decodificar o QR code

# Configuração de logging
logger = logging.getLogger(__name__)

# Criar um router para a nova funcionalidade
router = APIRouter()

@router.post("/extrair-cte-qrcode/", response_model=Dict[str, Any])
async def extrair_cte_qrcode(file: UploadFile = File(...)):
    """
    Recebe uma imagem com QR code de um CT-e, acessa a URL e retorna os dados do CT-e em formato JSON
    """
    # Verificar se o arquivo é uma imagem
    content_type = file.content_type
    if not content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="O arquivo enviado não é uma imagem")
    
    try:
        # Ler o conteúdo do arquivo
        contents = await file.read()
        
        # Converter para formato que o OpenCV pode usar
        image_np = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)
        
        # Pré-processamento da imagem para melhorar a detecção do QR code
        gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
        
        # Tentar diversos métodos de pré-processamento para melhorar a leitura
        # Método 1: Aumentar contraste
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        # Tentar decodificar o QR code na imagem original
        qr_codes = decode(image_np)
        
        # Se não encontrou, tentar na versão em escala de cinza
        if not qr_codes:
            qr_codes = decode(gray)
            
        # Se ainda não encontrou, tentar na versão com contraste aprimorado
        if not qr_codes:
            qr_codes = decode(enhanced)
            
        # Se ainda não encontrou, tentar aplicar um threshold adaptativo
        if not qr_codes:
            thresh = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
            )
            qr_codes = decode(thresh)
        
        if not qr_codes:
            raise HTTPException(status_code=404, detail="Nenhum QR code encontrado na imagem")
        
        # Pegar a URL do primeiro QR code encontrado
        qr_url = qr_codes[0].data.decode('utf-8')
        logger.info(f"URL do QR code encontrada: {qr_url}")
        
        # Verificar se a URL parece ser de um CT-e
        if not ("cte" in qr_url.lower() or "nfe" in qr_url.lower() or "fazenda" in qr_url.lower()):
            logger.warning(f"A URL encontrada não parece ser de um CT-e: {qr_url}")
            # Continuar mesmo assim, pois pode ser um domínio diferente
        
        # Acessar a URL do QR code
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        logger.info("Acessando a URL do QR code")
        response = requests.get(qr_url, headers=headers, timeout=10)
        
        if response.status_code != 200:
            raise HTTPException(
                status_code=500, 
                detail=f"Erro ao acessar a URL do QR code. Status code: {response.status_code}"
            )
        
        # Parsear o HTML da página
        logger.info("Fazendo o parsing do HTML da página")
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Extrair dados do CT-e
        cte_dados = extrair_dados_cte(soup)
        
        if not cte_dados:
            raise HTTPException(
                status_code=404, 
                detail="Não foi possível extrair dados de CT-e da página. Verifique se o QR code é válido."
            )
        
        # Adicionar a URL encontrada aos dados
        cte_dados["url_qrcode"] = qr_url
        
        return cte_dados
        
    except HTTPException:
        # Re-lançar exceções HTTP que já foram criadas
        raise
    except Exception as e:
        logger.error(f"Erro ao processar QR code ou fazer scraping: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail=f"Erro ao processar a imagem ou fazer scraping: {str(e)}"
        )
    finally:
        await file.close()
def extrair_dados_cte(soup: BeautifulSoup) -> Dict[str, Any]:
    """
    Extrai os dados de um Conhecimento de Transporte Eletrônico (CT-e) a partir do HTML.
    
    Args:
        soup: Objeto BeautifulSoup contendo o HTML da página do CT-e
        
    Returns:
        Dicionário contendo os dados estruturados do CT-e
    """
    resultado = {
        "chave_acesso": None,
        "dados_cte": {
            "modelo": None,
            "serie": None,
            "numero": None,
            "data_hora_emissao": None,
            "tipo_emissao": None,
            "modal": None,
            "uf_inicio": None,
            "uf_fim": None,
            "natureza_operacao": None,
            "cfop": None,
            "valor_total": None
        },
        "emitente": {
            "cnpj": None,
            "ie": None,
            "nome_razao_social": None,
            "municipio": None,
            "uf": None
        },
        "tomador": {
            "cnpj": None,
            "ie": None,
            "nome_razao_social": None,
            "municipio": None,
            "uf": None,
            "pais": None
        },
        "eventos": [],
        "digest_value": None
    }
    logger.error(f" {str(soup)}")
    # Extrair chave de acesso
    chave_acesso_div = soup.select_one('div:has(> label:-soup-contains("Chave de acesso")) + div')
    if chave_acesso_div:
        resultado["chave_acesso"] = chave_acesso_div.text.strip()
    
    # Extrair dados do CT-e
    dados_cte_div = soup.select_one('#dadosCTe')
    if dados_cte_div:
        # Modelo
        modelo_div = dados_cte_div.select_one('div.row:has(div:has(label:-soup-contains("Modelo"))) + div div:nth-child(1)')
        if modelo_div:
            resultado["dados_cte"]["modelo"] = modelo_div.text.strip()
        
        # Série
        serie_div = dados_cte_div.select_one('div.row:has(div:has(label:-soup-contains("Série"))) + div div:nth-child(2)')
        if serie_div:
            resultado["dados_cte"]["serie"] = serie_div.text.strip()
        
        # Número
        numero_div = dados_cte_div.select_one('div.row:has(div:has(label:-soup-contains("Número"))) + div div:nth-child(3)')
        if numero_div:
            resultado["dados_cte"]["numero"] = numero_div.text.strip()
        
        # Data/Hora da emissão
        data_hora_div = dados_cte_div.select_one('div.row:has(div:has(label:-soup-contains("Data/Hora da emissão"))) + div div:nth-child(4)')
        if data_hora_div:
            resultado["dados_cte"]["data_hora_emissao"] = data_hora_div.text.strip()
        
        # Tipo de Emissão
        emissao_div = dados_cte_div.select_one('div.row:has(div:has(label:-soup-contains("Emissão"))) + div div:nth-child(1)')
        if emissao_div:
            resultado["dados_cte"]["tipo_emissao"] = emissao_div.text.strip()
        
        # Modal
        modal_div = dados_cte_div.select_one('div.row:has(div:has(label:-soup-contains("Modal"))) + div div:nth-child(2)')
        if modal_div:
            resultado["dados_cte"]["modal"] = modal_div.text.strip()
        
        # UF Início
        uf_inicio_div = dados_cte_div.select_one('div.row:has(div:has(label:-soup-contains("UF início"))) + div div:nth-child(3)')
        if uf_inicio_div:
            resultado["dados_cte"]["uf_inicio"] = uf_inicio_div.text.strip()
        
        # UF Fim
        uf_fim_div = dados_cte_div.select_one('div.row:has(div:has(label:-soup-contains("UF fim"))) + div div:nth-child(4)')
        if uf_fim_div:
            resultado["dados_cte"]["uf_fim"] = uf_fim_div.text.strip()
        
        # Natureza da operação
        natureza_operacao_div = dados_cte_div.select_one('div.row:has(div:has(label:-soup-contains("Natureza da operação"))) + div div:nth-child(1)')
        if natureza_operacao_div:
            resultado["dados_cte"]["natureza_operacao"] = natureza_operacao_div.text.strip()
        
        # CFOP
        cfop_div = dados_cte_div.select_one('div.row:has(div:has(label:-soup-contains("CFOP"))) + div div:nth-child(2)')
        if cfop_div:
            resultado["dados_cte"]["cfop"] = cfop_div.text.strip()
        
        # Valor Total
        valor_total_div = dados_cte_div.select_one('div.row:has(div:has(label:-soup-contains("Valor Total"))) + div div:nth-child(3)')
        if valor_total_div:
            valor_texto = valor_total_div.text.strip()
            # Converter para float se possível
            try:
                valor_texto = valor_texto.replace('.', '').replace(',', '.')
                resultado["dados_cte"]["valor_total"] = float(valor_texto)
            except:
                resultado["dados_cte"]["valor_total"] = valor_texto
    
    # Extrair dados do emitente
    emitente_div = soup.select_one('#emitente')
    if emitente_div:
        # CNPJ Emitente
        cnpj_div = emitente_div.select_one('div.row:has(div:has(label:-soup-contains("CNPJ"))) + div div:nth-child(1)')
        if cnpj_div:
            resultado["emitente"]["cnpj"] = cnpj_div.text.strip()
        
        # IE Emitente
        ie_div = emitente_div.select_one('div.row:has(div:has(label:-soup-contains("IE"))) + div div:nth-child(2)')
        if ie_div:
            resultado["emitente"]["ie"] = ie_div.text.strip()
        
        # Nome/Razão Social Emitente
        razao_social_div = emitente_div.select_one('div.row:has(div:has(label:-soup-contains("Nome/Razão Social"))) + div div:nth-child(3)')
        if razao_social_div:
            resultado["emitente"]["nome_razao_social"] = razao_social_div.text.strip()
        
        # Município Emitente
        municipio_div = emitente_div.select_one('div.row:has(div:has(label:-soup-contains("Município"))) + div div:nth-child(1)')
        if municipio_div:
            resultado["emitente"]["municipio"] = municipio_div.text.strip()
        
        # UF Emitente
        uf_div = emitente_div.select_one('div.row:has(div:has(label:-soup-contains("UF"))) + div div:nth-child(2)')
        if uf_div:
            resultado["emitente"]["uf"] = uf_div.text.strip()
    
    # Extrair dados do tomador
    tomador_div = soup.select_one('#tomador')
    if tomador_div:
        # CNPJ Tomador
        cnpj_div = tomador_div.select_one('div.row:has(div:has(label:-soup-contains("CNPJ"))) + div div:nth-child(1)')
        if cnpj_div:
            resultado["tomador"]["cnpj"] = cnpj_div.text.strip()
        
        # IE Tomador
        ie_div = tomador_div.select_one('div.row:has(div:has(label:-soup-contains("IE"))) + div div:nth-child(2)')
        if ie_div:
            resultado["tomador"]["ie"] = ie_div.text.strip()
        
        # Nome/Razão Social Tomador
        razao_social_div = tomador_div.select_one('div.row:has(div:has(label:-soup-contains("Nome/Razão Social"))) + div div:nth-child(3)')
        if razao_social_div:
            resultado["tomador"]["nome_razao_social"] = razao_social_div.text.strip()
        
        # Município Tomador
        municipio_div = tomador_div.select_one('div.row:has(div:has(label:-soup-contains("Município"))) + div div:nth-child(1)')
        if municipio_div:
            resultado["tomador"]["municipio"] = municipio_div.text.strip()
        
        # UF Tomador
        uf_div = tomador_div.select_one('div.row:has(div:has(label:-soup-contains("UF"))) + div div:nth-child(2)')
        if uf_div:
            resultado["tomador"]["uf"] = uf_div.text.strip()
        
        # País Tomador
        pais_div = tomador_div.select_one('div.row:has(div:has(label:-soup-contains("País"))) + div div:nth-child(3)')
        if pais_div:
            resultado["tomador"]["pais"] = pais_div.text.strip()
    
    # Extrair eventos
    eventos_div = soup.select_one('#eventos')
    if eventos_div:
        tabela_eventos = eventos_div.select('table tbody tr')
        for linha in tabela_eventos:
            cells = linha.select('td')
            if len(cells) >= 3:
                evento = {
                    "tipo": cells[0].text.strip(),
                    "protocolo": cells[1].text.strip(),
                    "data_autorizacao": cells[2].text.strip()
                }
                resultado["eventos"].append(evento)
    
    # Extrair DigestValue
    digest_value_div = soup.select_one('div.row:has(div:has(label:-soup-contains("Digest Value"))) + div div')
    if digest_value_div:
        resultado["digest_value"] = digest_value_div.text.strip()
    logger.error(f" {str(resultado)}")
    return resultado 
def extrair_dados_cte2(soup: BeautifulSoup) -> Dict[str, Any]:
    """
    Extrai os dados de um CT-e da página HTML
    
    Esta função tenta identificar e extrair informações relevantes do CT-e
    usando diferentes estratégias, já que o layout pode variar entre estados.
    """
    dados = {}
    
    # Tentar estratégias diferentes para diferentes layouts de páginas de CT-e
    
    # Estratégia 1: Buscar por tabelas com dados estruturados
    try:
        # Buscar informações básicas do CT-e
        dados["numero_cte"] = extrair_por_label(soup, ["Número", "Número do CT-e", "CT-e", "Nro. do CT-e"])
        dados["chave_acesso"] = extrair_por_label(soup, ["Chave de Acesso", "Chave do CT-e"])
        dados["data_emissao"] = extrair_por_label(soup, ["Data de Emissão", "Emissão"])
        
        # Informações do emitente
        dados["emitente"] = {}
        dados["emitente"]["razao_social"] = extrair_por_label(soup, ["Razão Social do Emitente", "Emitente", "Nome/Razão Social"])
        dados["emitente"]["cnpj"] = extrair_por_label(soup, ["CNPJ do Emitente", "CNPJ"])
        dados["emitente"]["inscricao_estadual"] = extrair_por_label(soup, ["Inscrição Estadual", "IE"])
        
        # Informações do remetente
        dados["remetente"] = {}
        dados["remetente"]["razao_social"] = extrair_por_label(soup, ["Razão Social do Remetente", "Remetente"])
        dados["remetente"]["cnpj"] = extrair_por_label(soup, ["CNPJ do Remetente"])
        
        # Informações do destinatário
        dados["destinatario"] = {}
        dados["destinatario"]["razao_social"] = extrair_por_label(soup, ["Razão Social do Destinatário", "Destinatário"])
        dados["destinatario"]["cnpj"] = extrair_por_label(soup, ["CNPJ do Destinatário"])
        
        # Informações do tomador
        dados["tomador"] = {}
        dados["tomador"]["razao_social"] = extrair_por_label(soup, ["Razão Social do Tomador", "Tomador"])
        dados["tomador"]["cnpj"] = extrair_por_label(soup, ["CNPJ do Tomador"])
        
        # Valores
        dados["valor_total"] = extrair_por_label(soup, ["Valor Total", "Total CT-e"])
        dados["valor_receber"] = extrair_por_label(soup, ["Valor a Receber", "Valor a Pagar"])
        
        # Informações do transporte
        dados["transporte"] = {}
        dados["transporte"]["modalidade"] = extrair_por_label(soup, ["Modalidade", "Modal"])
        dados["transporte"]["tipo_servico"] = extrair_por_label(soup, ["Tipo de Serviço", "Serviço"])
        
        # Informações da carga
        dados["carga"] = {}
        dados["carga"]["natureza"] = extrair_por_label(soup, ["Natureza da Carga", "Natureza da Operação"])
        dados["carga"]["valor"] = extrair_por_label(soup, ["Valor da Carga", "Valor Total da Carga"])
        dados["carga"]["produto_predominante"] = extrair_por_label(soup, ["Produto Predominante"])
        
    except Exception as e:
        logger.warning(f"Erro ao extrair dados com a estratégia 1: {str(e)}")
    
    # Estratégia 2: Tentar encontrar elementos por classes ou ids específicos
    # Esta estratégia depende do layout específico da página
    try:
        # Exemplo: buscar divs ou spans com classes que possam conter informações
        for div in soup.find_all("div", class_=lambda x: x and "cte" in x.lower()):
            texto = div.get_text(strip=True)
            if "chave" in texto.lower() and "acesso" in texto.lower():
                # Extrair a chave de acesso usando regex
                chave_match = re.search(r'\d{44}', texto)
                if chave_match:
                    dados["chave_acesso"] = chave_match.group(0)
    except Exception as e:
        logger.warning(f"Erro ao extrair dados com a estratégia 2: {str(e)}")
    
    # Estratégia 3: Para layouts menos estruturados, extrair todos os dados relevantes baseados em regex
    try:
        texto_completo = soup.get_text()
        
        # Extrair chave de acesso (44 dígitos)
        chave_match = re.search(r'\b\d{44}\b', texto_completo)
        if chave_match and "chave_acesso" not in dados:
            dados["chave_acesso"] = chave_match.group(0)
        
        # Extrair número do CT-e
        cte_num_match = re.search(r'CT-e\s*n[º°]?\s*[:.]?\s*(\d+)', texto_completo, re.IGNORECASE)
        if cte_num_match and "numero_cte" not in dados:
            dados["numero_cte"] = cte_num_match.group(1)
        
        # Extrair valores monetários próximos a palavras-chave
        valores = {}
        
        # Padrões para buscar valores
        valor_patterns = [
            (r'valor\s*total\s*(?:do|da|de)?\s*CT-?e\s*[:.]?\s*R?\$?\s*([\d.,]+)', "valor_total"),
            (r'valor\s*a\s*(?:receber|pagar)\s*[:.]?\s*R?\$?\s*([\d.,]+)', "valor_receber"),
            (r'valor\s*(?:do|da|de)?\s*carga\s*[:.]?\s*R?\$?\s*([\d.,]+)', "valor_carga"),
        ]
        
        for pattern, key in valor_patterns:
            match = re.search(pattern, texto_completo, re.IGNORECASE)
            if match:
                valor = match.group(1).replace(".", "").replace(",", ".")
                valores[key] = float(valor)
        
        if valores and "valores" not in dados:
            dados["valores"] = valores
        
    except Exception as e:
        logger.warning(f"Erro ao extrair dados com a estratégia 3: {str(e)}")
    
    # Verificar se conseguimos dados suficientes
    if len(dados) <= 1:  # Apenas a URL ou vazio
        logger.warning("Não foi possível extrair dados suficientes da página")
        return None
    
    return dados

def extrair_por_label(soup: BeautifulSoup, labels: list) -> str:
    """
    Tenta extrair um valor baseado em possíveis labels/textos que precedem o valor
    """
    for label in labels:
        # Tentar encontrar em elementos que contêm o label
        for elemento in soup.find_all(string=re.compile(label, re.IGNORECASE)):
            # Verificar o pai e os irmãos deste elemento
            parent = elemento.parent
            
            # Verificar se o próximo elemento é o valor
            next_sibling = parent.find_next_sibling()
            if next_sibling:
                texto = next_sibling.get_text(strip=True)
                if texto:
                    return texto
            
            # Verificar se o valor está na mesma tag, após o label
            texto_completo = parent.get_text(strip=True)
            if ":" in texto_completo:
                valor = texto_completo.split(":", 1)[1].strip()
                if valor:
                    return valor
    
    return ""
