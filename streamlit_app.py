import os
import json
import streamlit as st
import openai
from dotenv import load_dotenv
import re
from openai import OpenAI
import geopandas as gpd
import folium
from streamlit_folium import st_folium
import unicodedata


client = OpenAI(
  api_key=st.secrets['openai_key'],  # Poner Key
)

# Configuración de Streamlit
st.set_page_config(page_title="Chatbot Corrupción 💬", layout="centered")

with st.sidebar:
    st.image(".streamlit/logo.png", use_container_width=True)
    st.title('Chatbot Corrupción')
    st.markdown('''
    ## Sobre este Chatbot

    Bienvenido al **Chatbot Corrupción**, una herramienta interactiva diseñada para facilitar el acceso y comprensión de los informes de auditoría relacionados con casos de corrupción en los gobiernos subnacionales del Perú durante el período **2016-2022**.

    Nuestra base de datos incluye **todos los informes de control** emitidos por la **Contraloría General de la República del Perú** en esos años, proporcionando una cobertura completa y actualizada de las acciones de control realizadas a nivel nacional.

    Utiliza este chatbot para explorar información detallada sobre auditorías, hallazgos y recomendaciones, y para obtener respuestas precisas basadas en los documentos oficiales.

    ---
    ### Cómo utilizar este Chatbot
    - **Realiza preguntas claras y específicas** sobre los informes de auditoría.
    - **Indica localidades o períodos de interés** para obtener información detallada.
    - **Recuerda que las respuestas se basan en documentos oficiales**, y si no se dispone de cierta información, se te proporcionará orientación para obtenerla.

    ---
    ''')
    st.markdown('Desarrollado por **Q-Lab** - Laboratorio de Inteligencia Artificial y Métodos Computacionales en Ciencias Sociales ([qlab.pucp.edu.pe](https://qlab.pucp.edu.pe/))')
    st.markdown('Contacto: ✉️ [qlab_csociales@pucp.edu.pe](mailto:qlab_csociales@pucp.edu.pe)')

    if st.button("🗑️ Limpiar conversación"):
        st.session_state.messages = [{"role": "assistant", "content": "Conversación reiniciada. ¿En qué más puedo ayudarte?"}]
        st.experimental_rerun()

load_dotenv()

# Asegúrate de que esta definición esté antes de su llamada

def load_chunks_from_jsonl(input_file='salida_chunks_final.jsonl'): # Renombrado para claridad
    docs_chunks_list = []
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    # Parsea cada línea como un objeto JSON individual
                    chunk = json.loads(line.strip()) 
                    docs_chunks_list.append(chunk)
                except json.JSONDecodeError as e:
                    print(f"Advertencia: Omitiendo línea malformada en '{input_file}': {line.strip()}")
                    print(f"Error de decodificación: {e}")
                    continue # Salta a la siguiente línea
    except FileNotFoundError:
        print(f"Error: El archivo '{input_file}' no fue encontrado.")
        return [] # Devuelve lista vacía si el archivo no existe
    except Exception as e:
        print(f"Ocurrió un error inesperado al leer '{input_file}': {e}")
        return []
    return docs_chunks_list

# En tu streamlit_app.py, cambia la llamada:
# docs_chunks = load_chunks_from_json('salida_chunks_final.jsonl') # Línea antigua
docs_chunks = load_chunks_from_jsonl('salida_chunks_final.jsonl') # Línea nueva y corregida




system_prompt_v2 = """
Eres un asistente virtual experto en analizar y resumir informes de auditoría de la Contraloría General de la República del Perú, enfocados en la gestión de gobiernos subnacionales durante el período 2016-2022. Tu principal tarea es ayudar a los usuarios a entender la situación de la gestión pública y los hallazgos relevantes, incluyendo aquellos que podrían indicar irregularidades o corrupción.

**Principios Clave para tus Respuestas:**
1.  **Basado en Evidencia:** Responde ÚNICAMENTE con información extraída de los chunks de los informes de auditoría proporcionados en el contexto. No inventes información ni hagas suposiciones más allá de lo escrito.
2.  **Referencia Explícita:** SIEMPRE que utilices información de un informe, comienza tu respuesta o el párrafo relevante mencionando el número de informe. Ejemplo: "Según el informe 'NRO-INFORME-AÑO', se observó que..." o "El informe 'NRO-INFORME-AÑO' detalla lo siguiente:..."
3.  **Precisión y Detalle:** Sé preciso y, cuando se soliciten detalles o resúmenes, incluye la información relevante como entidades auditadas, montos involucrados (si los hay en el chunk), principales hallazgos (observaciones), y recomendaciones clave.
4.  **Neutralidad:** Presenta los hechos tal como están en los informes. Aunque los usuarios puedan preguntar sobre "corrupción", los informes detallan "observaciones" o "irregularidades". Utiliza esa terminología, pero entiende que el usuario se refiere a esos hallazgos.
5.  **Manejo de Información Faltante:**
    *   Si no tienes información para una localidad Y período específico, PERO tienes información para esa localidad en OTROS períodos, o para esa región en el período solicitado, indícalo claramente. Ejemplo: "No tengo informes específicos para [Distrito X] en [Año Y]. Sin embargo, para [Distrito X] en [Año Z] el informe '[NRO-INFORME]' señala... Y para la región de [Región W] en [Año Y], el informe '[NRO-INFORME]' indica..."
    *   Si no tienes absolutamente ninguna información relevante para la consulta, responde: "No dispongo de información sobre [tema de la consulta]. Para más detalles, por favor consulte directamente con la Contraloría General de la República del Perú."

**Instrucciones Específicas para Tipos de Preguntas:**

**A. Para "Formular informes" o "Resumir situación" por año y región/localidad:**
    *   Cuando se te pida un resumen o "informe" para un **año y una región específicos**:
        1.  Identifica todos los chunks relevantes proporcionados en el contexto que coincidan con esos criterios (puedes guiarte por los metadatos del chunk si estuvieran disponibles en el contexto, o por la información textual).
        2.  Sintetiza la información de estos chunks.
        3.  Estructura tu respuesta de la siguiente manera (si es posible y la información lo permite):
            *   "Resumen de hallazgos para [Localidad/Región] en el año [Año]:"
            *   Para cada informe relevante encontrado:
                *   "**Informe [NRO-INFORME-AÑO] (Entidad: [ENTIDAD_AUDITADA]):**"
                *   "   **Objetivo Principal de la Auditoría:** [Si está disponible en el chunk de objetivo]"
                *   "   **Principales Observaciones/Hallazgos:**"
                *   "      - [Resumen de la observación 1 del informe, mencionando montos si son relevantes y están en el chunk]"
                *   "      - [Resumen de la observación 2 del informe, etc.]"
                *   "   **Recomendaciones Clave:**"
                *   "      - [Resumen de la recomendación 1 del informe]"
                *   "      - [Resumen de la recomendación 2 del informe, etc.]"
                *   "   **Posibles Implicancias (si se mencionan en los metadatos o el texto del chunk de observación):** [Ej: Responsabilidad Penal, Administrativa, Perjuicio Económico de S/ XXX]"
            *   Si hay múltiples informes, preséntalos secuencialmente.
            *   Finaliza con un breve resumen general si puedes identificar patrones o temas comunes entre los informes de esa localidad/año.
    *   Si no hay informes para la combinación exacta, sigue la política de manejo de información faltante (Principio Clave 5).

**B. Para responder sobre la "situación de la corrupción" o "hallazgos de corrupción" en años y regiones específicas:**
    *   Aplica la misma lógica que en el punto A, pero enfoca tu resumen en las "Observaciones" y las implicancias de responsabilidad (penal, administrativa, perjuicio económico) que encuentres en los chunks.
    *   Interpreta "corrupción" como las irregularidades, observaciones y hallazgos detallados en los informes.
    *   Sé claro al presentar los hechos: "El informe X identificó las siguientes observaciones que podrían ser de su interés respecto a irregularidades en la gestión..."

**C. Para preguntas sobre un informe específico (por número de informe):**
    *   Si el usuario pregunta por un número de informe específico, y tienes chunks de ese informe en el contexto:
        1.  Presenta el título del informe.
        2.  Menciona la entidad auditada, período auditado y fecha de emisión.
        3.  Resume el objetivo general (si está disponible).
        4.  Detalla TODAS las observaciones proporcionadas en los chunks de ese informe, incluyendo montos y responsabilidades si se especifican.
        5.  Detalla TODAS las recomendaciones proporcionadas en los chunks de ese informe.
        6.  No omitas detalles relevantes que estén en los chunks del contexto para ese informe.

**Consideraciones Adicionales:**
*   **Concisión y Relevancia:** Aunque se pide ser completo, evita la verbosidad innecesaria. Prioriza la información que directamente responde a la pregunta del usuario.
*   **Tono Profesional:** Mantén un tono formal e informativo, como corresponde a un experto en auditoría.
*   **Limitación de Conocimiento:** Reitera que tu conocimiento se basa *exclusivamente* en los documentos que se te proporcionan en el contexto para cada consulta.
"""

def main():
    st.title("Chatbot Corrupción 💬")
    st.markdown("Conversa con los informes de la contraloría sobre corrupción en gobiernos subnacionales en Perú (2016-2022).")
    st.write("---")  # Línea divisoria

    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "Hola, soy el Chatbot Corrupción. ¿En qué puedo ayudarte?"}]

    # Mostrar los mensajes previos
    for message in st.session_state.messages:
        if message["role"] == "assistant":
            with st.chat_message("assistant"):
                st.markdown(message["content"])
        elif message["role"] == "user":
            with st.chat_message("user"):
                st.markdown(message["content"])

    # Capturar la entrada del usuario
    if user_input := st.chat_input("Escribe tu pregunta aquí..."):
        user_message = {"role": "user", "content": user_input}
        st.session_state.messages.append(user_message)

        with st.chat_message("user"):
            st.markdown(user_input)

        with st.spinner("Generando respuesta..."):
            # Prepara el historial de mensajes para la API (excluyendo el mensaje del sistema)
            conversation_history = st.session_state.messages[:-1]  # Excluye la entrada actual
            response_text = send_question_to_openai(user_input, docs_chunks, conversation_history)
            if response_text:
                assistant_message = {"role": "assistant", "content": response_text}
                st.session_state.messages.append(assistant_message)

                with st.chat_message("assistant"):
                    st.markdown(response_text)
            else:
                st.error("No se pudo obtener una respuesta.")


    """
    Encuentra chunks relevantes:
    1. Extrae parámetros (año, localidad) de la pregunta.
    2. Pre-filtra chunks basados en estos parámetros.
    3. Calcula un score de relevancia para los chunks pre-filtrados basado en palabras clave.
    4. Devuelve los chunks más relevantes.
    """
    query_params = extract_query_parameters(question)
    
    # --- 1. Pre-filtrado basado en metadatos ---
    pre_filtered_chunks = []
    if not query_params["years"] and not query_params["regions"] and \
       not query_params["provinces"] and not query_params["districts"]:
        # Si no hay filtros de año/localidad en la pregunta, considerar todos los chunks para scoring
        # O, si se prefiere, devolver una lista vacía y que el LLM diga que necesita más especificidad.
        # Por ahora, pasaremos todos al scoring.
        pre_filtered_chunks = all_docs_chunks
    else:
        for chunk in all_docs_chunks:
            metadata = chunk.get("metadata", {})
            
            # Chequeo de Año del Informe
            # (Podrías querer filtrar por 'periodo_inicio'/'periodo_fin' también o en su lugar)
            year_match = True # Asume que coincide si no se especifica año en la pregunta
            if query_params["years"]:
                year_match = str(metadata.get("year", "")).strip() in query_params["years"]

            # Chequeo de Localidad (Región, Provincia, Distrito)
            # Normalizar también los metadatos para la comparación
            region_meta = normalize_text(metadata.get("region", ""))
            provincia_meta = normalize_text(metadata.get("provincia", ""))
            distrito_meta = normalize_text(metadata.get("distrito", ""))
            entidad_meta = normalize_text(metadata.get("entidad_auditada", ""))

            location_match = True # Asume que coincide si no se especifica localidad
            if query_params["regions"] or query_params["provinces"] or query_params["districts"]:
                location_match = False # Requiere al menos un match si se especifica
                if query_params["regions"] and any(q_reg in region_meta for q_reg in query_params["regions"]):
                    location_match = True
                if not location_match and query_params["provinces"] and any(q_prov in provincia_meta for q_prov in query_params["provinces"]):
                    location_match = True
                if not location_match and query_params["districts"] and any(q_dist in distrito_meta for q_dist in query_params["districts"]):
                    location_match = True
                # Opcional: buscar en entidad auditada si los otros no coinciden
                if not location_match and (query_params["regions"] or query_params["provinces"] or query_params["districts"]) and \
                   any(q_loc in entidad_meta for q_loc in query_params["regions"] + query_params["provinces"] + query_params["districts"]):
                    location_match = True
            
            if year_match and location_match:
                pre_filtered_chunks.append(chunk)
        
        # Si el pre-filtrado estricto no devuelve nada, pero hay palabras clave,
        # podríamos considerar relajar y usar todos los chunks para el scoring de keywords.
        # Pero para "formular informes", el pre-filtrado es importante.
        if not pre_filtered_chunks and query_params["keywords"]:
             # print("Prefiltrado no encontró chunks, usando todos para scoring de keywords.")
             pass # O podrías decidir usar all_docs_chunks aquí si la pregunta es más general

    # --- 2. Scoring de Relevancia por Palabras Clave (sobre los pre-filtrados o todos) ---
    chunks_to_score = pre_filtered_chunks if pre_filtered_chunks else all_docs_chunks # Decide si usar todos si el prefiltrado es vacío
    
    if not chunks_to_score: # Si no hay chunks ni prefiltrados ni en total (improbable)
        return []

    relevance_scores = []
    question_norm_keywords = set(query_params["keywords"])

    for chunk_idx, chunk in enumerate(chunks_to_score):
        metadata = chunk.get("metadata", {})
        chunk_text_norm = normalize_text(chunk.get("chunk_text", ""))
        titulo_norm = normalize_text(metadata.get("titulo_informe", ""))
        entidad_norm = normalize_text(metadata.get("entidad_auditada", ""))
        
        # Texto combinado para scoring
        combined_text_for_scoring = f"{chunk_text_norm} {titulo_norm} {entidad_norm}"
        chunk_keywords = set(re.findall(r'\b[a-z]{3,}\b', combined_text_for_scoring)) # Palabras de al menos 3 letras
        
        common_keywords = question_norm_keywords.intersection(chunk_keywords)
        score = len(common_keywords)

        # Bonificaciones por tipo de chunk si la pregunta lo sugiere
        source_field = chunk.get("source_field", "")
        if source_field == "observacion" and ("corrupcion" in question_norm_keywords or \
                                              "irregularidad" in question_norm_keywords or \
                                              "hallazgo" in question_norm_keywords or \
                                              "perjuicio" in question_norm_keywords):
            score += 5 
        if source_field == "objetivo" and "objetivo" in question_norm_keywords:
            score += 2
        if source_field == "recomendacion" and ("recomienda" in question_norm_keywords or "sugiere" in question_norm_keywords):
            score += 2
        
        # Penalizar si el chunk es muy corto y no tiene muchos keywords comunes? (Opcional)
        # if len(chunk_text_norm.split()) < 10 and score < 2:
        #     score *= 0.5

        relevance_scores.append({"score": score, "chunk": chunk, "original_index": chunk_idx})

    # Ordenar por score, y luego por índice original para mantener estabilidad si los scores son iguales
    relevant_chunks_sorted = sorted(relevance_scores, key=lambda x: (x["score"], -x["original_index"]), reverse=True)
    
    # Devolver los N chunks con mayor puntuación
    final_chunks = [item["chunk"] for item in relevant_chunks_sorted[:max_chunks]]
    
    # Si después de todo no hay chunks relevantes (ej. score 0 para todos), devolver lista vacía.
    if not any(item["score"] > 0 for item in relevant_chunks_sorted[:max_chunks]):
        # Si la pregunta tenía filtros de año/localidad y no se encontraron, es mejor devolver vacío.
        if query_params["years"] or query_params["regions"] or query_params["provinces"] or query_params["districts"]:
            return [] 
        # Si era una pregunta general sin filtros y aun así no hay score, quizás devolver los primeros N de all_docs_chunks.
        # Por ahora, si no hay score, devolvemos vacío.
        return []


    return final_chunks

def normalize_text(text):

    """Normaliza el texto a minúsculas y quita tildes."""
    if pd.isna(text) or not text: # Manejo de NaN y strings vacíos
        return ""
    text = str(text).lower()
    nfkd_form = unicodedata.normalize('NFKD', text)
    return "".join([c for c in nfkd_form if not unicodedata.combining(c)])


    """
    Extrae año(s), regiones y provincias de la pregunta del usuario.
    Omite la extracción de distritos.
    Retorna un diccionario con los parámetros encontrados.
    """
    params = {
        "years": [],
        "regions": [],
        "provinces": [],
        # "districts": [], # Omitido
        "keywords": [],
        "is_specific_enough": False # Flag para indicar si la pregunta tiene filtros suficientes
    }
    
    normalized_question = normalize_text(question)
    if not normalized_question:
        return params

    # 1. Extraer Años
    # \b para asegurar que sean palabras completas
    params["years"] = list(set(re.findall(r'\b(201[6-9]|202[0-2])\b', normalized_question)))

    # 2. Listas Normalizadas de Localidades Conocidas (Poblar exhaustivamente)
    # ESTAS LISTAS DEBEN SER COMPLETAS Y NORMALIZADAS (minúsculas, sin tildes)
    known_regions = ["lima", "loreto", "cajamarca", "cusco", "arequipa", "piura", "la libertad", "ancash", "junin", "puno", "lambayeque", "san martin"] # EJEMPLO: Añadir todas las regiones
    known_provinces = ["alto amazonas", "cajamarca", "chiclayo", "sanchez carrion", "lima", "huancayo", "trujillo", "arequipa"] # EJEMPLO: Añadir todas las provincias

    temp_question_for_keywords = normalized_question
    words = re.findall(r'\b\w+\b', normalized_question)
    found_locations_in_query = [] # Para registrar qué partes de la pregunta se identificaron como localidad

    # 3. Extraer Regiones y Provincias (N-gramas)
    # Buscar primero n-gramas más largos
    for n in range(3, 0, -1): # Considerar frases de hasta 3 palabras para localidades
        ngrams = [" ".join(words[i:i+n]) for i in range(len(words)-n+1)]
        for ngram_candidate in ngrams:
            # Evitar que una palabra ya usada en un n-grama más largo sea reprocesada
            already_covered = False
            for found_loc in found_locations_in_query:
                if ngram_candidate in found_loc and ngram_candidate != found_loc: # es subcadena de algo ya encontrado
                    already_covered = True
                    break
            if already_covered:
                continue

            if ngram_candidate in known_regions and ngram_candidate not in params["regions"]:
                params["regions"].append(ngram_candidate)
                found_locations_in_query.append(ngram_candidate)
                params["is_specific_enough"] = True
            elif ngram_candidate in known_provinces and ngram_candidate not in params["provinces"]:
                # Podría ser que una región también sea una provincia (ej. "lima")
                # Si ya se identificó como región, se podría decidir no añadirla como provincia
                # o permitirlo. Por ahora, se permite si no está ya en la lista de provincias.
                params["provinces"].append(ngram_candidate)
                found_locations_in_query.append(ngram_candidate)
                params["is_specific_enough"] = True
    
    if params["years"]:
        params["is_specific_enough"] = True

    # 4. Eliminar Años y Localidades Encontradas para Obtener Keywords Puras
    for year_found in params["years"]:
        temp_question_for_keywords = re.sub(r'\b' + re.escape(year_found) + r'\b', '', temp_question_for_keywords)
    for loc_found in found_locations_in_query:
        # Escapar para regex y asegurar que sea una palabra completa
        temp_question_for_keywords = re.sub(r'\b' + re.escape(loc_found) + r'\b', '', temp_question_for_keywords)
    
    # 5. Extraer Palabras Clave Restantes
    stopwords = [
        "de", "la", "el", "en", "y", "o", "del", "los", "las", "un", "una", "unos", "unas",
        "sobre", "informe", "reporte", "situacion", "caso", "casos", "corrupcion",
        "auditoria", "contraloria", "gobierno", "municipalidad", "region", "provincia",
        "departamento", "distrito", "general", "republica", "peru", "quiero", "saber",
        "dime", "cual", "cuales", "fue", "fueron", "hay", "existe", "acerca", "con", "sin",
        "mas", "menos", "para", "por", "que", "como", "cuando", "donde"
    ]
    params["keywords"] = [
        kw for kw in re.findall(r'\b[a-z]{3,}\b', temp_question_for_keywords) 
        if kw not in stopwords and kw not in params["years"] # No repetir años como keywords
        # No es necesario verificar contra found_locations_in_query si ya los quitamos de temp_question_for_keywords
    ]
    
    # 6. Ajuste Final de `is_specific_enough`
    if not params["is_specific_enough"] and len(params["keywords"]) < 2:
        params["is_specific_enough"] = False
    elif params["keywords"]: # Si hay keywords relevantes, aunque no haya año/localidad explícitos
        params["is_specific_enough"] = True
    # Si solo hay año pero no localidad ni keywords, podría no ser suficiente
    if params["years"] and not params["regions"] and not params["provinces"] and not params["keywords"]:
        params["is_specific_enough"] = False


    return params

def extract_query_parameters_regions_only(question):
    """
    Extrae año(s), REGIONES y palabras clave de la pregunta del usuario.
    Retorna un diccionario con los parámetros encontrados.
    """
    params = {
        "years": [],
        "regions": [], # Solo regiones ahora
        # "provinces": [], # Eliminado
        # "districts": [], # Eliminado
        "keywords": [],
        "is_specific_enough": False # Flag para indicar si la pregunta tiene filtros útiles
    }
    
    normalized_question = normalize_text(question)
    if not normalized_question:
        return params

    # 1. Extraer Años (2016-2022)
    # Usar list(set(...)) para asegurar años únicos si se mencionan múltiples veces
    params["years"] = list(set(re.findall(r'\b(201[6-9]|202[0-2])\b', normalized_question)))
    if params["years"]:
        params["is_specific_enough"] = True

    # 2. Extraer Regiones Conocidas
    # !!! ESTA LISTA DEBE SER POBLADA EXHAUSTIVAMENTE Y NORMALIZADA !!!
    # (minúsculas, sin tildes)
    known_regions = [
        "amazonas", "ancash", "apurimac", "arequipa", "ayacucho", "cajamarca", 
        "callao", "cusco", "huancavelica", "huanuco", "ica", "junin", 
        "la libertad", "lambayeque", "lima", "loreto", "madre de dios", 
        "moquegua", "pasco", "piura", "puno", "san martin", "tacna", 
        "tumbes", "ucayali"
        # Añade más si es necesario o variantes
    ]

    
    # Podrías cargar esta lista desde un archivo si es muy larga

    temp_question_for_keywords = normalized_question
    found_locations_in_query = []

    # Buscar n-gramas (hasta 3 palabras para nombres de regiones compuestos como "madre de dios")
    words = re.findall(r'\b\w+\b', normalized_question)
    
    for n in range(3, 0, -1): # Buscar n-gramas de 3, 2, y 1 palabra
        ngrams = [" ".join(words[i:i+n]) for i in range(len(words)-n+1)]
        for ngram_candidate in ngrams:
            # Verificar si el ngram ya fue procesado como parte de una entidad más larga
            already_processed = False
            for found_loc in found_locations_in_query:
                if ngram_candidate in found_loc and ngram_candidate != found_loc:
                    already_processed = True
                    break
            if already_processed:
                continue

            if ngram_candidate in known_regions:
                if ngram_candidate not in params["regions"]: # Evitar duplicados
                    params["regions"].append(ngram_candidate)
                if ngram_candidate not in found_locations_in_query: # Añadir a la lista para eliminación de keywords
                     # Priorizar el n-grama más largo si hay superposición
                    is_substring_of_existing = any(ngram_candidate in existing_loc and ngram_candidate != existing_loc for existing_loc in found_locations_in_query)
                    if not is_substring_of_existing:
                        # Remover subcadenas si este n-grama es más largo
                        found_locations_in_query = [loc for loc in found_locations_in_query if loc not in ngram_candidate]
                        found_locations_in_query.append(ngram_candidate)
                
                params["is_specific_enough"] = True
    
    # Asegurar que found_locations_in_query solo tenga las regiones efectivamente añadidas
    found_locations_in_query = params["regions"]


    # 3. Extraer Palabras Clave (Keywords)
    # Eliminar años y regiones encontradas para obtener keywords puras
    for year_found in params["years"]:
        temp_question_for_keywords = re.sub(r'\b' + re.escape(year_found) + r'\b', '', temp_question_for_keywords)
    for region_found in found_locations_in_query: # Usar las regiones efectivamente encontradas
        temp_question_for_keywords = re.sub(r'\b' + re.escape(region_found) + r'\b', '', temp_question_for_keywords)
    
    # Lista de stopwords más extensa y específica del dominio
    stopwords = [
        "de", "la", "el", "en", "y", "o", "del", "los", "las", "un", "una", "unos", "unas",
        "sobre", "acerca", "informe", "reporte", "situacion", "caso", "casos", "corrupcion",
        "auditoria", "contraloria", "gobierno", "municipalidad", "region", "provincia", "distrito",
        "general", "republica", "peru", "quiero", "saber", "dime", "podrias", "informacion",
        "detalles", "cual", "cuales", "como", "cuando", "donde", "que", "quien", "porque",
        "mas", "menos", "todo", "todos", "entidad", "publico", "publica"
    ]
    params["keywords"] = [
        kw for kw in re.findall(r'\b[a-z]{3,}\b', temp_question_for_keywords) 
        if kw not in stopwords and kw not in params["years"] and kw not in found_locations_in_query
    ]
    
    # Ajuste final de is_specific_enough: si solo se basa en keywords, que sean al menos algunas.
    if not params["is_specific_enough"] and len(params["keywords"]) >= 1: # Si hay al menos 1 keyword relevante
        params["is_specific_enough"] = True
    elif not params["is_specific_enough"] and not params["keywords"]: # Ni año/loc ni keywords
        params["is_specific_enough"] = False


    return params


    query_params = extract_query_parameters(question)

    if not query_params["is_specific_enough"] and len(query_params["keywords"]) < 1: # Ajustado a <1 keyword
        return {"needs_more_specificity": True, "chunks": []}

    pre_filtered_chunks = []
    # Aplicar pre-filtrado si hay parámetros de año, región o provincia en la pregunta
    apply_pre_filtering = bool(query_params["years"] or query_params["regions"] or query_params["provinces"])

    if apply_pre_filtering:
        for chunk in all_docs_chunks:
            metadata = chunk.get("metadata", {})
            
            year_match = True 
            if query_params["years"]:
                year_match = str(metadata.get("year", "")).strip() in query_params["years"]

            location_match = True 
            if query_params["regions"] or query_params["provinces"]:
                location_match = False 
                region_meta = normalize_text(metadata.get("region", ""))
                provincia_meta = normalize_text(metadata.get("provincia", ""))
                
                if query_params["regions"] and any(q_reg == region_meta for q_reg in query_params["regions"]):
                    location_match = True
                # Si no hay match por región o si la pregunta no especificaba región, chequear provincia
                if not location_match and query_params["provinces"] and any(q_prov == provincia_meta for q_prov in query_params["provinces"]):
                    location_match = True
            
            if year_match and location_match:
                pre_filtered_chunks.append(chunk)
        
        if not pre_filtered_chunks:
            return {"needs_more_specificity": False, "chunks": [], "no_data_for_filter": True, "params": query_params}
    else:
        pre_filtered_chunks = all_docs_chunks

    if not pre_filtered_chunks:
         return {"needs_more_specificity": False, "chunks": []}

    # --- Scoring de Relevancia por Palabras Clave ---
    relevance_scores = []
    question_norm_keywords = set(query_params["keywords"])

    if not question_norm_keywords and apply_pre_filtering and pre_filtered_chunks:
        return {"needs_more_specificity": False, "chunks": pre_filtered_chunks[:max_chunks]}

    for chunk_idx, chunk in enumerate(pre_filtered_chunks):
        metadata = chunk.get("metadata", {})
        chunk_text_norm = normalize_text(chunk.get("chunk_text", ""))
        titulo_norm = normalize_text(metadata.get("titulo_informe", ""))
        entidad_norm = normalize_text(metadata.get("entidad_auditada", ""))
        
        combined_text_for_scoring = f"{chunk_text_norm} {titulo_norm} {entidad_norm}"
        chunk_keywords = set(re.findall(r'\b[a-z]{3,}\b', combined_text_for_scoring))
        
        common_keywords = question_norm_keywords.intersection(chunk_keywords)
        score = len(common_keywords)

        source_field = chunk.get("source_field", "")
        # (Bonificaciones de score sin cambios)
        if source_field == "observacion" and any(kw in question_norm_keywords for kw in ["corrupcion", "irregularidad", "hallazgo", "perjuicio"]):
            score += 5 
        if source_field == "objetivo" and "objetivo" in question_norm_keywords: score += 2
        if source_field == "recomendacion" and any(kw in question_norm_keywords for kw in ["recomienda", "sugiere"]): score += 2

        relevance_scores.append({"score": score, "chunk": chunk, "original_index": chunk_idx})

    relevant_chunks_sorted = sorted(relevance_scores, key=lambda x: (x["score"], -x["original_index"]), reverse=True)
    
    final_chunks = [item["chunk"] for item in relevant_chunks_sorted if item["score"] > 0][:max_chunks]

    if not final_chunks and apply_pre_filtering:
        return {"needs_more_specificity": False, "chunks": [], "no_data_for_filter_after_score": True, "params": query_params}
    
    return {"needs_more_specificity": False, "chunks": final_chunks}

def find_relevant_chunks(question, all_docs_chunks, max_chunks=10):
    """
    Encuentra chunks relevantes:
    1. Extrae parámetros (año, REGIONES) de la pregunta.
    2. Pre-filtra chunks basados en estos parámetros.
    3. Calcula un score de relevancia para los chunks pre-filtrados basado en palabras clave.
    4. Devuelve un diccionario con el estado y los chunks.
    """
    # Llama a la función que solo extrae regiones para la localidad
    query_params = extract_query_parameters_regions_only(question)

    # Caso 1: La pregunta no es suficientemente específica para una búsqueda dirigida
    if not query_params["is_specific_enough"]:
        # print(f"Debug: Pregunta '{question}' no es suficientemente específica. Params: {query_params}")
        return {"needs_more_specificity": True, "chunks": []}

    pre_filtered_chunks = []
    # Determinar si se debe aplicar pre-filtrado basado en si se extrajeron años o regiones
    apply_pre_filtering = bool(query_params["years"] or query_params["regions"])

    if apply_pre_filtering:
        for chunk in all_docs_chunks:
            metadata = chunk.get("metadata", {})
            
            # Chequeo de Año del Informe
            year_match = True # Asume que coincide si no se especifica año en la pregunta
            if query_params["years"]:
                # El año en metadata debe ser int o string, query_params["years"] son strings
                year_meta_str = str(metadata.get("year", "")).strip()
                year_match = year_meta_str in query_params["years"]

            # Chequeo de Región
            region_match = True # Asume que coincide si no se especifica región en la pregunta
            if query_params["regions"]:
                region_meta_norm = normalize_text(metadata.get("region", ""))
                # query_params["regions"] ya están normalizadas por extract_query_parameters
                region_match = any(q_reg == region_meta_norm for q_reg in query_params["regions"])
            
            if year_match and region_match:
                pre_filtered_chunks.append(chunk)
        
        # Si el pre-filtrado estricto no devuelve nada, pero la pregunta SÍ tenía filtros
        if not pre_filtered_chunks:
            # print(f"Debug: Prefiltrado para '{question}' no encontró chunks. Params: {query_params}")
            return {"needs_more_specificity": False, "chunks": [], "no_data_for_filter": True, "params": query_params}
    else:
        # Si no se aplicó pre-filtrado (porque no había parámetros de año/región en la pregunta,
        # pero 'is_specific_enough' fue True debido a keywords), se usan todos los chunks para scoring.
        pre_filtered_chunks = all_docs_chunks

    if not pre_filtered_chunks: # Si all_docs_chunks estaba vacío o el prefiltrado no dio nada y no se pasó a todos
         # print(f"Debug: No hay chunks para score para '{question}'.")
         return {"needs_more_specificity": False, "chunks": []}


    # --- Scoring de Relevancia por Palabras Clave (sobre los pre_filtered_chunks) ---
    relevance_scores = []
    # query_params["keywords"] ya están normalizadas y sin stopwords
    question_norm_keywords = set(query_params["keywords"])

    # Si no hay keywords en la pregunta PERO SÍ HUBO PREFILTRADO exitoso,
    # devolver todos los chunks prefiltrados hasta max_chunks.
    # Esto es para casos como "informes de Lima 2020" donde el filtro es lo principal.
    if not question_norm_keywords and apply_pre_filtering and pre_filtered_chunks:
        # print(f"Debug: Prefiltrado para '{question}' devolvió {len(pre_filtered_chunks)} chunks, sin keywords para scoring adicional.")
        return {"needs_more_specificity": False, "chunks": pre_filtered_chunks[:max_chunks]}

    # Si no hay keywords y NO hubo prefiltrado (pregunta muy general y sin keywords),
    # ya se manejó con is_specific_enough, pero como doble chequeo:
    if not question_norm_keywords and not apply_pre_filtering:
        # print(f"Debug: Pregunta muy general sin keywords '{question}', pidiendo especificidad.")
        return {"needs_more_specificity": True, "chunks": []}


    for chunk_idx, chunk in enumerate(pre_filtered_chunks):
        metadata = chunk.get("metadata", {})
        chunk_text_norm = normalize_text(chunk.get("chunk_text", ""))
        titulo_norm = normalize_text(metadata.get("titulo_informe", ""))
        entidad_norm = normalize_text(metadata.get("entidad_auditada", ""))
        
        combined_text_for_scoring = f"{chunk_text_norm} {titulo_norm} {entidad_norm}"
        # Usar un regex más simple para keywords del chunk, ya que ya están normalizadas
        chunk_keywords = set(re.findall(r'\b[a-z]{3,}\b', combined_text_for_scoring)) 
        
        common_keywords = question_norm_keywords.intersection(chunk_keywords)
        score = len(common_keywords)

        # Bonificaciones (ya tienes esta lógica)
        source_field = chunk.get("source_field", "")
        if source_field == "observacion" and any(kw in question_norm_keywords for kw in ["corrupcion", "irregularidad", "hallazgo", "perjuicio", "delito"]):
            score += 5 
        if source_field == "objetivo" and "objetivo" in question_norm_keywords: score += 3 # Aumentado ligeramente
        if source_field == "recomendacion" and any(kw in question_norm_keywords for kw in ["recomienda", "sugiere", "recomendacion"]): score += 3 # Aumentado ligeramente
        
        # Bonus si hay una coincidencia de región en el chunk_text en sí, si la pregunta tenía región
        if query_params["regions"]:
            if any(q_reg in chunk_text_norm for q_reg in query_params["regions"]):
                score += 1 # Pequeño bonus por mención en el texto
        
        relevance_scores.append({"score": score, "chunk": chunk, "original_index": chunk_idx})

    relevant_chunks_sorted = sorted(relevance_scores, key=lambda x: (x["score"], -x["original_index"]), reverse=True)
    
    # Devolver solo chunks con score > 0, a menos que no haya keywords en la pregunta y el prefiltrado haya sido el único criterio
    if not question_norm_keywords and apply_pre_filtering and pre_filtered_chunks:
         final_chunks = [item["chunk"] for item in relevant_chunks_sorted][:max_chunks] # Devuelve prefiltrados si no hay keywords
    else:
        final_chunks = [item["chunk"] for item in relevant_chunks_sorted if item["score"] > 0][:max_chunks]


    # Si después del scoring, no hay chunks (y no era el caso de solo prefiltrado sin keywords)
    if not final_chunks and apply_pre_filtering: 
        # print(f"Debug: Scoring no encontró chunks relevantes para '{question}' después del prefiltrado. Params: {query_params}")
        return {"needs_more_specificity": False, "chunks": [], "no_data_for_filter_after_score": True, "params": query_params}
    
    return {"needs_more_specificity": False, "chunks": final_chunks}


def send_question_to_openai(question, all_docs_chunks, conversation_history):
    retrieval_result = find_relevant_chunks(question, all_docs_chunks, max_chunks=15)

    if retrieval_result.get("needs_more_specificity"):
        return "Por favor, proporciona más detalles en tu consulta, como un año específico, región, provincia o distrito para poder ayudarte mejor."

    relevant_chunks = retrieval_result.get("chunks", [])

    if not relevant_chunks:
        if retrieval_result.get("no_data_for_filter") or retrieval_result.get("no_data_for_filter_after_score"):
            # Crear un mensaje más informativo si se usaron filtros
            params = retrieval_result.get("params", {})
            year_str = ", ".join(params.get("years", [])) or "el período consultado"
            loc_parts = params.get("districts", []) + params.get("provinces", []) + params.get("regions", [])
            loc_str = ", ".join(loc_parts) or "la localidad consultada"
            if params.get("years") or loc_parts:
                 return f"No encontré informes que coincidan exactamente con tu consulta para {loc_str} en {year_str}. Intenta con otros parámetros o consulta directamente a la Contraloría General de la República del Perú."
        # Si era una pregunta general y aun así no hay chunks (o todos con score 0)
        return "No dispongo de información específica para tu consulta. Por favor, intenta reformularla o consulta directamente a la Contraloría General de la República del Perú."


    context_text = "\n\n---\n\n".join([
        f"Del Informe: {chunk['metadata'].get('numero_informe', 'N/A')}\n"
        f"Entidad Auditada: {chunk['metadata'].get('entidad_auditada', 'N/A')}\n"
        f"Año del Informe: {chunk['metadata'].get('year', 'N/A')}\n"
        f"Región: {chunk['metadata'].get('region', 'N/A')}\n"
        # f"Provincia: {chunk['metadata'].get('provincia', 'N/A')}\n" # Puedes añadir más metadatos si es útil
        # f"Distrito: {chunk['metadata'].get('distrito', 'N/A')}\n"
        f"Tipo de Información (Chunk): {chunk.get('source_field', 'N/A')}\n"
        f"Texto del Chunk:\n{chunk.get('chunk_text', '')}"
        for chunk in relevant_chunks
    ])

    # ... (resto de la función send_question_to_openai igual que antes)
    MAX_HISTORY_MESSAGES = 10
    trimmed_history = conversation_history[-MAX_HISTORY_MESSAGES:]
    messages = []
    combined_system_prompt = f"{system_prompt_v2}\n\nContexto relevante de los informes:\n{context_text}" # Usar system_prompt_v2
    messages.append({"role": "system", "content": combined_system_prompt})
    messages.extend(trimmed_history)
    messages.append({"role": "user", "content": question})

    response = client.chat.completions.create(
        model="gpt-4.1-nano",
        messages=messages,
        temperature=0,
        max_tokens=1500, # Aumentar si se espera que sintetice "informes"
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    return response.choices[0].message.content

if __name__ == "__main__":
    main()
