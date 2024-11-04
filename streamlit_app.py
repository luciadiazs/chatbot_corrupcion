import os
import json
import streamlit as st
import openai
from dotenv import load_dotenv
import re  # Importante para las expresiones regulares en find_relevant_chunks
from openai import OpenAI

client = OpenAI(
  api_key=st.secrets['openai_key'],  # this is also the default, it can be omitted
)

# Configuración de Streamlit
st.set_page_config(page_title="Chatbot Corrupción 💬", layout="centered")

with st.sidebar:
    st.title('Chatbot Corrupción')
    st.markdown('''
    ## Sobre este Chatbot
    Este es un chatbot que te permite conversar con los informes de auditoría sobre corrupción en gobiernos subnacionales en Perú 2016-2022.
                ''')

load_dotenv()

# Asegúrate de que esta definición esté antes de su llamada
def load_chunks_from_json(input_file='data/processed/docs_chunks.json'):
    with open(input_file, 'r', encoding='utf-8') as f:
        docs_chunks = json.load(f)
    return docs_chunks

# Ahora puedes llamar a la función después de su definición
docs_chunks = load_chunks_from_json('data/processed/docs_chunks.json')  # Asegúrate de especificar la ruta correcta al archivo JSON

def main():
    st.header("Conversa con los informes de la contraloría💬")

# Define el system_prompt

system_prompt = """
Eres un experto en informes de auditoría sobre corrupción en los gobiernos subnacionales de Perú. Responde a las preguntas basándote en los datos de los documentos proporcionados (Informes de Servicios de Control) que proceden de la Contraloría General de La República del Perú.

Al elaborar tus respuestas:

- Proporciona información precisa y útil basada en los documentos.
- Cuando utilices información específica de un documento, siempre menciona al inicio el número de informe de donde proviene. Por ejemplo: "Según el informe '002-2017-2-5510-informe', se encontró que..."
- Si se te pregunta sobre corrupción en una localidad específica, menciona la información que tengas de todos informes sobre esa localidad. 
- Si no conoces la respuesta a una pregunta, simplemente responde: «No dispongo de esa información, por favor consulte https://buscadorinformes.contraloria.gob.pe/BuscadorCGR/Informes/inicio.html?utm_source=gobpee&utm_medium=otsbuscador&utm_campaign=buscador.»
"""

if "messages" not in st.session_state:
    st.session_state["messages"] = []

prompt = st.text_input("Tu pregunta:", "")

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

if st.button("Enviar"):
    if prompt:  # Check if the prompt is not empty
        user_message = {"role": "user", "content": prompt}
        st.session_state.messages.append(user_message)

        with st.spinner("Generando respuesta..."):
            response_text = send_question_to_openai(prompt, docs_chunks)
            if response_text:  # Check if the response_text is not None or empty
                assistant_message = {"role": "assistant", "content": response_text}
                st.session_state.messages.append(assistant_message)
            else:
                st.error("Failed to get a response.")  # Display an error if no response was received

# Display the messages
for index, message in enumerate(st.session_state.messages):
    if message["role"] == "user":
        st.text_area("Pregunta", value=message["content"], height=75, disabled=True, key=f"user_{index}")
    elif message["role"] == "assistant":  # Ensure this is an 'elif' to check specifically for "assistant" role
        st.text_area("Respuesta", value=message["content"], height=100, disabled=True, key=f"assistant_{index}")


if __name__ == "__main__":
    main()


