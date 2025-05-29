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
def load_chunks_from_json(input_file='salida_chunks_final.jsonl'):
    with open(input_file, 'r', encoding='utf-8') as f:
        docs_chunks = json.load(f)
    return docs_chunks

# Ahora puedes llamar a la función después de su definición
docs_chunks = load_chunks_from_json('salida_chunks_final.jsonl')  # Asegúrate de especificar la ruta correcta al archivo JSON

system_prompt = """
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
    *   Cuando se te pida un resumen o "informe" para un **año y una región/distrito/provincia específicos**:
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

def normalize_text(text):
    """Normaliza el texto a minúsculas y quita tildes."""
    if not text:
        return ""
    text = str(text).lower()
    # Quitar tildes
    nfkd_form = unicodedata.normalize('NFKD', text)
    return "".join([c for c in nfkd_form if not unicodedata.combining(c)])

def extract_query_parameters(question):
    """
    Extrae año(s), regiones, provincias y distritos de la pregunta del usuario.
    Retorna un diccionario con los parámetros encontrados.
    """
    params = {
        "years": [],
        "regions": [],
        "provinces": [],
        "districts": [],
        "keywords": []
    }
    
    normalized_question = normalize_text(question)

    # Extraer años (2016-2022 como mencionaste)
    # \b para asegurar que sean palabras completas (evitar parte de un número más largo)
    params["years"] = re.findall(r'\b(201[6-9]|202[0-2])\b', normalized_question)
    
    # Palabras clave para identificar localidades (esto es una simplificación)
    # Idealmente, tendrías listas de regiones, provincias, distritos válidos para una mejor extracción.
    # O podrías usar NER si la pregunta es muy libre.
    
    # Ejemplo de palabras clave para regiones (deberías tener una lista más completa)
    # Estas deberían ser normalizadas (minúsculas, sin tildes)
    known_regions = ["lima", "loreto", "cajamarca", "cusco", "arequipa", "piura", "la libertad", "ancash", "junin", "puno", "lambayeque", "san martin"] # añadir más
    known_provinces = ["alto amazonas", "cajamarca", "chiclayo", "sanchez carrion"] # añadir más
    known_districts = ["yurimaguas", "jesus", "san bartolo", "miraflores", "san isidro"] # añadir más

    words = re.findall(r'\b\w+\b', normalized_question)
    
    # Una heurística simple: si una palabra de la pregunta está en la lista de regiones/provincias/distritos
    # Esta parte puede mejorarse mucho con listas más completas o técnicas de NLP.
    # También considerar frases de varias palabras.
    # Para mejorar, busca n-gramas.
    
    potential_locations = []
    # Buscar n-gramas (hasta 3 palabras)
    for n in range(3, 0, -1):
        ngrams = [" ".join(words[i:i+n]) for i in range(len(words)-n+1)]
        for ngram in ngrams:
            if ngram in known_regions:
                params["regions"].append(ngram)
            elif ngram in known_provinces:
                params["provinces"].append(ngram)
            elif ngram in known_districts:
                params["districts"].append(ngram)
            # Eliminar las palabras del n-grama encontrado para no re-procesarlas
            # (esto es una simplificación, podría necesitar lógica más robusta)

    # Extraer palabras clave restantes (que no sean años ni localidades ya identificadas)
    non_entity_words = normalized_question
    for year in params["years"]:
        non_entity_words = non_entity_words.replace(year, "")
    for loc_list in [params["regions"], params["provinces"], params["districts"]]:
        for loc in loc_list:
            non_entity_words = non_entity_words.replace(loc, "")
            
    params["keywords"] = [kw for kw in re.findall(r'\b[a-z]{3,}\b', non_entity_words) 
                          if kw not in ["de", "la", "el", "en", "y", "o", "del", "sobre", "informe", "reporte", "situacion", "caso", "casos", "corrupcion"]] # stopwords comunes

    return params

def find_relevant_chunks(question, all_docs_chunks, max_chunks=10):
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

def send_question_to_openai(question, docs_chunks, conversation_history):
    # Encuentra los chunks más relevantes para la pregunta
    relevant_chunks = find_relevant_chunks(question, docs_chunks)

    # Construye el contexto incluyendo el título y el contenido de cada chunk
    context_text = "\n\n".join([
        f"Título del Documento: {chunk['title']}\nContenido:\n{chunk['content']}"
        for chunk in relevant_chunks
    ])

    # Limita el historial a los últimos N mensajes para controlar el número de tokens
    MAX_HISTORY_MESSAGES = 5  # Puedes ajustar este número según tus necesidades
    trimmed_history = conversation_history[-MAX_HISTORY_MESSAGES:]

    # Construye la lista de mensajes para la API
    messages = []

    # Añade el mensaje del sistema combinado con el contexto relevante
    combined_system_prompt = f"{system_prompt}\n\nContexto relevante:\n{context_text}"
    messages.append({"role": "system", "content": combined_system_prompt})

    # Añade el historial de conversación previo
    messages.extend(trimmed_history)

    # Añade la pregunta actual del usuario
    messages.append({"role": "user", "content": question})

    # Llama a la API de OpenAI con los mensajes actualizados
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        temperature=0,
        max_tokens=1024,  # Ajusta según sea necesario
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )

    # Devuelve la respuesta generada
    return response.choices[0].message.content

if __name__ == "__main__":
    main()
