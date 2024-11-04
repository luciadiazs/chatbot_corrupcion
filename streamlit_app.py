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

def main():
    st.title("Chatbot Corrupción 💬")
    st.markdown("Conversa con los informes de la contraloría sobre corrupción en gobiernos subnacionales en Perú (2016-2022).")
    st.write("---")  # Línea divisoria

    system_prompt = """
    Eres un experto en informes de auditoría sobre corrupción en los gobiernos subnacionales de Perú. Responde a las preguntas basándote en los datos de los documentos proporcionados (Informes de Servicios de Control) que proceden de la Contraloría General de La República del Perú.

    Al elaborar tus respuestas:

    - Proporciona información precisa y útil basada en los documentos.
    - Cuando utilices información específica de un documento, siempre menciona al inicio el número de informe de donde proviene. Por ejemplo: "Según el informe '002-2017-2-5510-informe', se encontró que..."
    - Si se te pregunta sobre corrupción en una localidad específica, menciona la información que tengas de todos informes sobre esa localidad. 
    - Si no conoces la respuesta a una pregunta, simplemente responde: «No dispongo de esa información, por favor consulte https://buscadorinformes.contraloria.gob.pe/BuscadorCGR/Informes/inicio.html?utm_source=gobpee&utm_medium=otsbuscador&utm_campaign=buscador.»
    """

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
            response_text = send_question_to_openai(user_input, docs_chunks)
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

def send_question_to_openai(question, docs_chunks):
    # Encuentra los chunks más relevantes para la pregunta
    relevant_chunks = find_relevant_chunks(question, docs_chunks)
    
    # Construye el contexto incluyendo el título y el contenido de cada chunk
    context_text = "\n\n".join([
        f"Título del Documento: {chunk['title']}\nContenido:\n{chunk['content']}"
        for chunk in relevant_chunks
    ])
    
    # Construye los mensajes a enviar a la API
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "assistant", "content": context_text},
        {"role": "user", "content": question}
    ]
    
    # Llama a la API de OpenAI con los mensajes actualizados
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages,
        temperature=0,
        max_tokens=2048,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    
    # Devuelve la respuesta generada
    return response.choices[0].message.content

if __name__ == "__main__":
    main()
