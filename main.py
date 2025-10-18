# --- [INÍCIO main.py - Backend FastAPI v2.6 (OCR + Gemini)] ---

import io
import re
import json
import time
import os
from typing import List, Dict, Any, Optional

# --- Importações das bibliotecas ---
try:
    # Bibliotecas Principais
    from fastapi import FastAPI, UploadFile, File, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from google import genai
    from PyPDF2 import PdfReader
    import asyncio # Para endpoint assíncrono

    # Bibliotecas OCR (Fallback)
    import cv2
    import pytesseract
    import numpy as np
    from pdf2image import convert_from_bytes, pdfinfo_from_bytes
    from PIL import Image

except ImportError as e:
    print(f"ERRO CRÍTICO: Biblioteca faltando! {e}")
    print("Execute: pip install fastapi uvicorn google-genai PyPDF2 python-multipart opencv-python-headless pytesseract pdf2image Pillow numpy")
    exit()

# --- 1. CONFIGURAÇÃO DA CHAVE GEMINI (Via Variável de Ambiente) ---
try:
    API_KEY = os.environ.get("GEMINI_API_KEY")
    if not API_KEY:
         # Em produção, isso deve gerar um erro ou log, não imprimir a chave.
         # Por segurança, não colocaremos a chave fixa aqui. Render a injetará.
         raise ValueError("Chave da API Gemini (GEMINI_API_KEY) não encontrada nas variáveis de ambiente.")

    # Inicializa o cliente com a chave encontrada
    CLIENT = genai.Client(api_key=API_KEY)
    print("INFO: Cliente Gemini inicializado com sucesso.")

except Exception as e:
    print(f"\nFATAL: Erro de inicialização do Cliente Gemini: {e}")
    # Se isso falhar, o servidor não conseguirá iniciar corretamente.
    raise

# --- 2. Configuração do FastAPI ---
app = FastAPI(title="LabFlow-AI Backend v2.6 (OCR + Gemini)")
origins = [
    # Adicionaremos as URLs do Render aqui depois
    "http://localhost",
    "http://127.0.0.1",
    "null",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins, allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

# --- 3. Funções de Extração de Texto (OCR e Direta) ---

# Função de pré-processamento de imagem (para OCR)
def preparar_imagem_para_leitura(imagem: np.ndarray) -> np.ndarray:
    try:
        # Garante que a imagem esteja no formato correto (BGR ou Cinza)
        if len(imagem.shape) == 3:
             if imagem.shape[2] == 4: imagem = cv2.cvtColor(imagem, cv2.COLOR_RGBA2BGR)
             cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
        elif len(imagem.shape) == 2: cinza = imagem
        else: raise ValueError("Formato de imagem não suportado")

        _, binaria = cv2.threshold(cinza, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        processada = cv2.medianBlur(binaria, 3)
        return processada
    except Exception as e:
        print(f"WARN: Erro no pré-processamento OpenCV: {e}. Usando imagem original.")
        # Retorna a imagem em escala de cinza como fallback mínimo
        try: return cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY) if len(imagem.shape)==3 else imagem
        except: return imagem

# Função de OCR (Tesseract)
def extrair_texto_com_ocr(conteudo_bytes: bytes) -> str:
    texto_final_ocr = ""
    print("INFO: Iniciando extração via OCR...")
    try:
        # Tenta converter PDF para imagens
        paginas_pdf = convert_from_bytes(conteudo_bytes, dpi=300, poppler_path=os.environ.get("POPPLER_PATH")) # Tenta usar path do ambiente se definido
        print(f"INFO: (OCR) PDF convertido em {len(paginas_pdf)} página(s).")
        for i, pagina in enumerate(paginas_pdf):
            try:
                imagem_cv = cv2.cvtColor(np.array(pagina), cv2.COLOR_RGB2BGR)
                imagem_limpa = preparar_imagem_para_leitura(imagem_cv)
                # Configuração Tesseract: --psm 6 tenta tratar como bloco de texto uniforme
                texto_pagina = pytesseract.image_to_string(imagem_limpa, lang='por', config='--psm 6')
                texto_final_ocr += texto_pagina + f"\n--- OCR P{i+1} ---\n"
            except Exception as e_page_ocr:
                print(f"WARN: Erro no OCR da página {i+1}: {e_page_ocr}")
                continue # Pula para a próxima página
    except Exception as e_pdf_to_img:
        # Se falhar como PDF, tenta como imagem única
        print(f"INFO: (OCR) Falha ao converter PDF ({e_pdf_to_img}), tentando como imagem...")
        try:
            imagem_pil = Image.open(io.BytesIO(conteudo_bytes))
            # Converte formatos comuns para BGR que o OpenCV entende
            if imagem_pil.mode == 'RGBA': imagem_cv = cv2.cvtColor(np.array(imagem_pil), cv2.COLOR_RGBA2BGR)
            elif imagem_pil.mode == 'P': imagem_pil_rgb = imagem_pil.convert('RGB'); imagem_cv = cv2.cvtColor(np.array(imagem_pil_rgb), cv2.COLOR_RGB2BGR)
            elif imagem_pil.mode == 'L': imagem_cv = cv2.cvtColor(np.array(imagem_pil), cv2.COLOR_GRAY2BGR) # Converte Cinza para BGR
            else: imagem_cv = cv2.cvtColor(np.array(imagem_pil), cv2.COLOR_RGB2BGR)

            imagem_limpa = preparar_imagem_para_leitura(imagem_cv)
            texto_final_ocr = pytesseract.image_to_string(imagem_limpa, lang='por', config='--psm 6')
        except Exception as e_img_ocr:
            print(f"ERRO: Falha no OCR como PDF e como Imagem: {e_img_ocr}")
            return "" # Retorna vazio se tudo falhar
    print(f"INFO: OCR concluído. Caracteres extraídos: {len(texto_final_ocr)}")
    return texto_final_ocr

# Função de Extração Direta (PyPDF2)
def extrair_texto_direto_pdf(conteudo_bytes: bytes) -> Optional[str]:
     texto_pdf = ""
     try:
          pdf = PdfReader(io.BytesIO(conteudo_bytes))
          print(f"INFO: (Texto Direto) PDF com {len(pdf.pages)} páginas.")
          for i, page in enumerate(pdf.pages):
               try:
                    texto_pagina = page.extract_text()
                    texto_pdf += (texto_pagina or "") + f"\n--- Txt P{i+1} ---\n"
               except Exception as e_page: print(f"WARN: Erro ao extrair texto da pág {i+1}: {e_page}")
          # Considera sucesso se extraiu um número razoável de caracteres
          success = texto_pdf and len(texto_pdf.strip()) > 100 * len(pdf.pages) # Heurística: > 100 chars por página em média
          print(f"INFO: (Texto Direto) Extração: {len(texto_pdf)} chars. Sucesso: {success}")
          return texto_pdf if success else None
     except Exception as e_pdf: print(f"WARN: Erro PyPDF2: {e_pdf}"); return None

# Função Inteligente (Decide entre Direta e OCR)
def extrair_texto_inteligente(conteudo_bytes: bytes, nome_arquivo: str) -> str:
    texto_final = ""
    content_type = nome_arquivo.split('.')[-1].lower() if '.' in nome_arquivo else ''

    if content_type == 'pdf':
        print("INFO: PDF detectado. Tentando extração direta...")
        texto_direto = extrair_texto_direto_pdf(conteudo_bytes)
        if texto_direto:
            print("INFO: Extração direta bem-sucedida.")
            texto_final = texto_direto
        else:
            print("INFO: Falha na extração direta ou pouco texto encontrado. Usando OCR como fallback...")
            texto_final = extrair_texto_com_ocr(conteudo_bytes)
    elif content_type in ['jpg', 'jpeg', 'png', 'bmp', 'tiff']:
        print("INFO: Arquivo de imagem detectado. Usando OCR...")
        texto_final = extrair_texto_com_ocr(conteudo_bytes)
    else:
        print(f"WARN: Tipo de arquivo não suportado ({content_type}). Tentando OCR como último recurso...")
        texto_final = extrair_texto_com_ocr(conteudo_bytes) # Tenta OCR mesmo assim

    return texto_final


# --- 4. Função Principal: Estruturação com API Gemini (v2.5 - Final) ---
# Esta função recebe o texto já extraído (direto ou OCR)
def organizar_dados_com_api_gemini_final(texto_bruto_extraido: str) -> List[Dict[str, Any]]:

    if not texto_bruto_extraido or len(texto_bruto_extraido.strip()) < 50:
        print("WARN: Texto bruto extraído parece vazio ou muito curto. Pulando chamada Gemini.")
        return []

    # Definição do Esquema (JSON Schema)
    json_schema = {
        "type": "array",
        "description": "Lista de exames laboratoriais extraídos.",
        "items": {
            "type": "object",
            "properties": {
                "NomeCompleto": {"type": "string"}, "ResultadoObtido": {"type": "string"},
                "ValorReferencia": {"type": "string"}, "UnidadeMedida": {"type": "string"},
                "DataColeta": {"type": "string"}, "Sigla": {"type": "string"}
            },
            "required": ["NomeCompleto", "ResultadoObtido", "ValorReferencia", "UnidadeMedida", "DataColeta", "Sigla"]
        }
    }
    
    # --- PROMPT V4.0: Ordem e Formatação Final ---
    prompt = f"""
    Você é um assistente de extração de dados clínicos. Sua ÚNICA função é converter o texto do laudo em JSON, aderindo estritamente ao formato JSON definido no schema.
    REGRAS CRÍTICAS:
    1. EXAUSTIVIDADE: Extraia CADA componente (Hemograma, Lipídios, Bilirrubinas, etc.).
    2. ORDEM: Ordene os exames na lista JSON final seguindo as categorias clínicas do laudo: Hemograma (Eritrócitos, Leucócitos, Plaquetas) primeiro, seguido por Perfil Lipídico, Bioquímica, Enzimas Hepáticas, Hormônios e Vitaminas.
    3. VALOR/UNIDADE: Não remova unidades ou valores de referência. Extraia 'ResultadoObtido' e 'UnidadeMedida' EXATAMENTE como aparecem.
    4. PADRONIZAÇÃO DECIMAL: Em valores numéricos, substitua o ponto decimal (.) por vírgula (,) e mantenha separadores de milhar (.). Ex: 5.16 -> 5,16; 230.000 -> 230.000.
    5. SIGLA: Forneça a abreviação padrão (Ex: Creatinina -> Cr, Hemoglobina -> Hb, TGO -> TGO).
    6. DATACOLETA: Encontre a data de coleta ('Coleta: DD/MM/AAAA') e aplique-a a todos os exames. Se houver múltiplas, use a mais frequente.
    TEXTO DO LAUDO:
    ---
    {texto_bruto_extraido} 
    ---
    """
    
    print("INFO: Enviando texto para API Gemini...")
    start_gemini = time.time()
    try:
        response = CLIENT.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt,
            config={ "response_mime_type": "application/json", "response_schema": json_schema, "temperature": 0.0 }
        )
        print(f"INFO: Resposta Gemini recebida em {time.time() - start_gemini:.2f} seg.")
        dados_extraidos_obj = json.loads(response.text)
        
        # Unifica a saída em uma lista simples
        if isinstance(dados_extraidos_obj, list): return dados_extraidos_obj
        elif 'exames' in dados_extraidos_obj and isinstance(dados_extraidos_obj['exames'], list): return dados_extraidos_obj['exames']
        else: print("WARN: Resposta Gemini não é uma lista esperada."); return []
    except Exception as e:
        print(f"ERRO CRÍTICO NA API GEMINI (Processo): {e}")
        # Retorna lista vazia ou lança exceção para o endpoint tratar
        # raise HTTPException(status_code=503, detail=f"Erro na API Gemini: {e}")
        return [] # Retorna vazio para não quebrar o lote

# --- 5. Endpoint da API FastAPI (Lote) ---
@app.post("/api/processar-laudo", response_model=List[Dict[str, Any]])
async def processar_laudo_endpoint(files: List[UploadFile] = File(...)):

    start_time = time.time()
    resultados_combinados = []
    print(f"INFO: Recebido Lote de {len(files)} arquivos.")

    for file in files:
        print(f"\n--- Processando Arquivo: {file.filename} ---")
        try:
            conteudo_bytes = await file.read()
            print(f"INFO: Lido {len(conteudo_bytes)} bytes.")

            # ETAPA 1: Extrair Texto (Inteligente: Direto ou OCR)
            texto_bruto = extrair_texto_inteligente(conteudo_bytes, file.filename)
            if not texto_bruto:
                print(f"WARN: Falha ao extrair texto de {file.filename}. Pulando.")
                continue # Pula para o próximo arquivo no lote

            # ETAPA 2: Estruturar com Gemini
            dados_do_arquivo = organizar_dados_com_api_gemini_final(texto_bruto)

            # ETAPA 3: Adicionar origem e combinar
            for exame in dados_do_arquivo:
                 exame['OrigemArquivo'] = file.filename
            resultados_combinados.extend(dados_do_arquivo)
            print(f"INFO: SUCESSO - {len(dados_do_arquivo)} exames extraídos de {file.filename}")

        except HTTPException as e: # Re-lança erros HTTP específicos
            raise e
        except Exception as e:
            # Captura outros erros, registra e continua o lote se possível
            print(f"ERRO CRÍTICO no processamento do arquivo {file.filename}: {e}")
            # Decide se deve abortar o lote ou apenas pular o arquivo
            # Por enquanto, vamos pular o arquivo com erro:
            # raise HTTPException(status_code=500, detail=f"Erro ao processar {file.filename}: {e}")
            continue # Pula para o próximo arquivo

    end_time = time.time()
    print(f"\nINFO: Lote Concluído. Total de exames: {len(resultados_combinados)}. Tempo: {end_time - start_time:.2f} seg")
    return resultados_combinados

# --- [FIM main.py - Backend FastAPI v2.6] ---