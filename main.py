# --- [INÍCIO main.py - Backend FastAPI v2.7 (Deploy Simplificado - Sem OCR)] ---

import io
import re
import json
import time
import os
from typing import List, Dict, Any, Optional

# --- Importações das bibliotecas (Apenas as necessárias) ---
try:
    from fastapi import FastAPI, UploadFile, File, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from google import genai
    from PyPDF2 import PdfReader # Única dependência de PDF agora
    import asyncio 
except ImportError as e:
    print(f"ERRO: Biblioteca faltando! {e}")
    print("Execute: pip install fastapi uvicorn google-genai PyPDF2 python-multipart")
    exit()

# --- 1. CONFIGURAÇÃO DA CHAVE GEMINI (Via Variável de Ambiente) ---
try:
    API_KEY = os.environ.get("GEMINI_API_KEY")
    if not API_KEY:
         raise ValueError("Chave da API Gemini (GEMINI_API_KEY) não encontrada nas variáveis de ambiente.")
    
    # Inicializa o cliente com a chave encontrada
    CLIENT = genai.Client(api_key=API_KEY)
    print("INFO: Cliente Gemini inicializado com sucesso.")
except Exception as e:
    print(f"\nFATAL: Erro de inicialização do Cliente Gemini: {e}")
    raise

# --- 2. Configuração do FastAPI ---
app = FastAPI(title="LabFlow-AI Backend v2.7 (Deploy Simplificado)")

# --- Configuração do CORS (Deixe o espaço para a URL do Render) ---
origins = [
    "http://localhost",
    "http://127.0.0.1",
    "null",
    # "https://labflow-ai-backend.onrender.com", # Adicionaremos isso depois
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins, allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

# --- 3. Funções de Extração de Texto (Apenas PyPDF2) ---
def extrair_texto_direto_pdf(conteudo_bytes: bytes) -> str:
     texto_pdf = ""
     try:
          pdf = PdfReader(io.BytesIO(conteudo_bytes))
          print(f"INFO: (Texto Direto) PDF com {len(pdf.pages)} páginas.")
          for page in pdf.pages:
               texto_pagina = page.extract_text()
               if texto_pagina: 
                  texto_pdf += texto_pagina + "\n---\n"
          print(f"INFO: (Texto Direto) Extração: {len(texto_pdf)} chars.")
          return texto_pdf if texto_pdf and len(texto_pdf.strip()) > 50 else ""
     except Exception as e:
          print(f"WARN: Erro PyPDF2: {e}")
          return ""

# --- 4. Função Principal: Motor Gemini (v2.5 - Mantida) ---
def organizar_dados_com_api_gemini_final(texto_bruto_extraido: str) -> List[Dict[str, Any]]:
    
    if not texto_bruto_extraido or len(texto_bruto_extraido.strip()) < 50:
        print("WARN: Texto bruto extraído parece vazio ou muito curto. Pulando chamada Gemini.")
        return []

    # Definição do Esquema (JSON Schema)
    json_schema = {
        "type": "array", "description": "Lista de exames laboratoriais extraídos.",
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
    
    # Prompt v4.0 (Mantido)
    prompt = f"""
    Você é um assistente de extração de dados clínicos e sua ÚNICA função é converter o texto do laudo em JSON, aderindo estritamente ao formato JSON definido no schema.
    REGRAS CRÍTICAS DE EXTRAÇÃO:
    1. EXAUSTIVIDADE: Extraia CADA componente (Hemograma, Lipídios, Bilirrubinas, etc.).
    2. ORDEM: Ordene os exames na lista JSON final seguindo as categorias clínicas do laudo: Hemograma (Eritrócitos, Leucócitos, Plaquetas) primeiro, seguido por Perfil Lipídico, Bioquímica, Enzimas Hepáticas, Hormônios e Vitaminas.
    3. VALOR/UNIDADE: Não remova unidades ou valores de referência. Extraia 'ResultadoObtido' e 'UnidadeMedida' EXATAMENTE como aparecem.
    4. PADRONIZAÇÃO DECIMAL: Em valores numéricos, substitua o ponto decimal (.) por vírgula (,) e mantenha separadores de milhar (.). Ex: 5.16 -> 5,16; 230.000 -> 230.000.
    5. SIGLA: Forneça a abreviação padrão (Ex: Creatinina -> Cr, Hemoglobina -> Hb, TGO -> TGO).
    6. DATACOLETA: Encontre a data de coleta ('Coleta: DD/MM/AAAA') e aplique-a a todos os exames.
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
        
        if isinstance(dados_extraidos_obj, list): return dados_extraidos_obj
        elif 'exames' in dados_extraidos_obj and isinstance(dados_extraidos_obj['exames'], list): return dados_extraidos_obj['exames']
        else: print("WARN: Resposta Gemini não é uma lista esperada."); return []
    except Exception as e:
        print(f"ERRO CRÍTICO NA API GEMINI (Processo): {e}")
        raise # Levanta a exceção para ser capturada pelo endpoint

# --- 5. Endpoint da API FastAPI (Lote - Simplificado) ---
@app.post("/api/processar-laudo", response_model=List[Dict[str, Any]])
async def processar_laudo_endpoint(files: List[UploadFile] = File(...)): 
    
    start_time = time.time()
    resultados_combinados = []
    print(f"INFO: Recebido Lote de {len(files)} arquivos.")

    for file in files:
        print(f"\n--- Processando Arquivo: {file.filename} ---")
        # Aceita apenas PDF agora, pois o OCR foi removido
        if file.content_type != 'application/pdf':
             print(f"WARN: Arquivo {file.filename} ({file.content_type}) ignorado. Apenas PDF é suportado.")
             continue

        try: 
            conteudo_bytes = await file.read()
            
            # ETAPA 1: Extrair Texto (Apenas PyPDF2)
            texto_bruto = extrair_texto_direto_pdf(conteudo_bytes)
            if not texto_bruto:
                print(f"WARN: Falha ao extrair texto de {file.filename} (não selecionável?). Pulando.")
                continue

            # ETAPA 2: Estruturar com Gemini
            dados_do_arquivo = organizar_dados_com_api_gemini_final(texto_bruto)
            
            # ETAPA 3: Adicionar origem e combinar
            for exame in dados_do_arquivo:
                 exame['OrigemArquivo'] = file.filename
            resultados_combinados.extend(dados_do_arquivo)
            print(f"INFO: SUCESSO - {len(dados_do_arquivo)} exames extraídos de {file.filename}")

        except HTTPException as e: raise e
        except Exception as e:
            print(f"FALHA CRÍTICA no arquivo {file.filename}: {e}")
            raise HTTPException(status_code=500, detail=f"Erro ao processar {file.filename}: {e}")

    end_time = time.time()
    print(f"\nINFO: Lote Concluído. Total de exames: {len(resultados_combinados)}. Tempo: {end_time - start_time:.2f} seg")
    return resultados_combinados

# --- [FIM main.py - Backend FastAPI v2.7 (Deploy Simplificado)] ---

