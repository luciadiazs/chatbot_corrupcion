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

def find_relevant_chunks(question, docs_chunks, max_chunks=5):
    question_keywords = set(re.findall(r'\w+', question.lower()))
    relevance_scores = []

    for chunk in docs_chunks:
        # Combina el título y el contenido para la comparación
        combined_text = (chunk["title"] + " " + chunk["content"]).lower()
        chunk_keywords = set(re.findall(r'\w+', combined_text))
        common_keywords = question_keywords.intersection(chunk_keywords)
        relevance_scores.append((len(common_keywords), chunk))

    relevant_chunks = [chunk for _, chunk in sorted(relevance_scores, key=lambda x: x[0], reverse=True)]
    return relevant_chunks[:max_chunks]

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
