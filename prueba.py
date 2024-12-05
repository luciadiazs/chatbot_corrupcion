from openai import OpenAI
import os

client = OpenAI(
    # This is the default and can be omitted
    api_key='sk-hvJC0yT1jLP5_VeaYVHwTQuwPX-7jopZambdxVonp5T3BlbkFJfR8Lp2sqOdJkE_MSmrxeQcbNphDzq6GI-FCQ-OIx8A',
)

response = client.chat.completions.create(
  model="gpt-3.5-turbo",
  messages=[
    {
      "role": "system",
      "content": [
        {
          "type": "text",
          "text": "You are an expert in audit reports on corruption in subnational governments in Peru. Answer the questions based on the data in the documents provided (Informes de Servicios de Control), which come from la Contraloría General de La República del Perú. If you don't know the answer to a question, simply respond 'I don't have that information available, please consult https://buscadorinformes.contraloria.gob.pe/BuscadorCGR/Informes/inicio.html?utm_source=gobpee&utm_medium=otsbuscador&utm_campaign=buscador'"
        }
      ]
    },
    {
      "role": "user",
      "content": [
        {
          "type": "text",
          "text": "De donde obtienes tus datos?"
        }
      ]
    }
  ],
  temperature=0,
  max_tokens=2048,
  top_p=1,
  frequency_penalty=0,
  presence_penalty=0,
  response_format={
    "type": "text"
  }
)

print(response.choices[0].message)