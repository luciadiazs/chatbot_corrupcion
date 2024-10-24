import os
import json
import streamlit as st
import openai
from dotenv import load_dotenv
import re  # Importante para las expresiones regulares en find_relevant_chunks
from openai import OpenAI

client = OpenAI(
  api_key=os.environ['openai_key'],  # this is also the default, it can be omitted
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
system_prompt = "You are an expert in audit reports on corruption in subnational governments in Peru. Answer the questions based on the data in the documents provided."

if "messages" not in st.session_state:
    st.session_state["messages"] = []

prompt = st.text_input("Tu pregunta:", "")


openai.api_key = st.secrets['openai_key']

def find_relevant_chunks(question, docs_chunks, max_chunks=5):
    # Tokeniza la pregunta para extraer palabras clave significativas
    question_keywords = set(re.findall(r'\w+', question.lower()))
    relevance_scores = []

    # Calcula un puntaje de relevancia para cada chunk (puede ser el conteo de palabras clave coincidentes)
    for chunk in docs_chunks:
        chunk_text = chunk["content"].lower()
        chunk_keywords = set(re.findall(r'\w+', chunk_text))
        common_keywords = question_keywords.intersection(chunk_keywords)
        relevance_scores.append((len(common_keywords), chunk))

    # Ordena los chunks por su puntaje de relevancia, de mayor a menor
    relevant_chunks = [chunk for _, chunk in sorted(relevance_scores, key=lambda x: x[0], reverse=True)]

    # Retorna los top N chunks más relevantes
    return relevant_chunks[:max_chunks]

def send_question_to_openai(question, docs_chunks):
    # Encuentra los chunks más relevantes para la pregunta
    relevant_chunks = find_relevant_chunks(question, docs_chunks)
    
    # Construye el prompt completo con el system_prompt y los chunks de texto relevantes
    prompt_text = system_prompt + "\n\n" + "\n\n".join([chunk["content"] for chunk in relevant_chunks]) + "\n\nQuestion: " + question

    # Llama a la API de OpenAI con el prompt para chat
    response = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question}
        ],
        "temperature": 0.0
    )
    
    # Return the message content directly
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


