import os
import json
import streamlit as st
import openai
from dotenv import load_dotenv
import re
from openai import OpenAI

client = OpenAI(
  api_key=st.secrets['openai_key'],  # Poner Key
)

# Configuración de Streamlit
st.set_page_config(page_title="Chatbot Corrupción 💬", layout="centered")

with st.sidebar:
    st.image(".streamlit/logo.png", use_column_width=True)
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

def load_chunks_from_json(input_file='data/processed/docs_chunks.json'):
    with open(input_file, 'r', encoding='utf-8') as f:
        docs_chunks = json.load(f)
    return docs_chunks

docs_chunks = load_chunks_from_json('data/processed/docs_chunks.json')

system_prompt = """
Eres un experto en informes de auditoría sobre corrupción en los gobiernos subnacionales de Perú. Responde a las preguntas basándote únicamente en los datos de los documentos proporcionados (Informes de Servicios de Control) de la Contraloría General de la República del Perú.

Al elaborar tus respuestas:

- Proporciona información precisa y útil basada en los documentos disponibles.
- Cuando utilices información específica de un documento, menciona al inicio el número de informe de donde proviene. Por ejemplo: "Según el informe '002-2017-2-5510', se encontró que..."
- Si se te pregunta sobre corrupción en una localidad y/o período específico, y solo tienes información de otros períodos, indica primero qué información tienes disponible. Por ejemplo: "No dispongo de información sobre Chiclayo en 2017, pero sí del 2014. Según el informe 'XXX-XXXX-XXXX', se encontró que...". Si tienes información de varios períodos, mencionalos todos.
- Si no conoces la respuesta a una pregunta o no tienes información al respecto, responde: "No dispongo de esa información, por favor consulte https://buscadorinformes.contraloria.gob.pe/..."
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
    MAX_HISTORY_MESSAGES = 10
    trimmed_history = conversation_history[-MAX_HISTORY_MESSAGES:]

    # Construye la lista de mensajes para la API
    messages = []

    # Añade el mensaje del sistema
    messages.append({"role": "system", "content": system_prompt})

    # Añade el historial de conversación previo
    messages.extend(trimmed_history)

    # Añade el contexto (fragmentos relevantes) como mensaje del asistente
    messages.append({"role": "assistant", "content": context_text})

    # Añade la pregunta actual del usuario
    messages.append({"role": "user", "content": question})

    # Llama a la API de OpenAI con los mensajes actualizados
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages,
        temperature=0,
        max_tokens=1024,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )

    # Devuelve la respuesta generada
    return response.choices[0].message.content

if __name__ == "__main__":
    main()
