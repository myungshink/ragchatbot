import streamlit as st  #conda install streamlit -c conda-forge -y , conda install streamlit-chat -c conda-forge -y
from streamlit_chat import message
import openai
import pandas as pd
import ast
import numpy as np
import os

df = pd.read_csv("./completion/embedding.csv")
df["embedding"] = df["embedding"].apply(ast.literal_eval)

client = openai.OpenAI(api_key='sk-proj-EUPfxVARWBdvhaZFzHeeT3BlbkFJ3izJT6SzdmquaBDH3DFX')

def get_embedding(text):
    response = client.embeddings.create(
        input=text,
        model="text-embedding-ada-002"
    )
    return response.data[0].embedding

def search_similar_document(df,query) :
    
    df["similarity"] = df["embedding"].apply(lambda x : cosine_similarity(get_embedding(query),x))
    top_documents = df.nlargest(3,"similarity")
    top_documents.reset_index(drop=True,inplace=True)
    return top_documents

def cosine_similarity(A, B):
    dot_product = np.dot(A, B)
    norm_A = np.linalg.norm(A)
    norm_B = np.linalg.norm(B)
    similarity = dot_product / (norm_A * norm_B)
    return similarity

def create_prompt(df,query ) :
    tops = search_similar_document(df,query)
    system_prompt = f"""
      당신은 문서 요약의 달인입니다.
      다음 문서의 내용을 이해 하기 쉬운 표현으로 요약해 주십시요.
      문서 1: {str(tops.iloc[0]["text"])}
      문서 2: {str(tops.iloc[1]["text"])}
      문서 3: {str(tops.iloc[2]["text"])}
      요약된 내용은 한글로 작성해 주십시요.
      답변은 반드시 이 문서들에 기반해서 답변되어야 합니다.
    """
    user_prompt=f"{query}"

    return (system_prompt,user_prompt)


def geterate_response(system_prompt,user_prompt) :
    response = client.chat.completions.create(
      model="gpt-3.5-turbo",
      messages= [
    {
      "role": "system",
      "content": [
        {
          "type": "text",
          "text": system_prompt
        }
      ]
    },
    {
      "role": "user",
      "content": [
        {
          "type": "text",
          "text": user_prompt
        }
      ]
    }]
    )
    return response.choices[0].message.content

# 화면에 보여주기 위해 챗봇의 답변을 저장할 공간 할당
if 'generated' not in st.session_state:
    st.session_state['generated'] = []

# 화면에 보여주기 위해 사용자의 질문을 저장할 공간 할당
if 'past' not in st.session_state:
    st.session_state['past'] = []

# 사용자의 입력이 들어오면 user_input에 저장하고 Send 버튼을 클릭하면
# submitted의 값이 True로 변환.
with st.form('form', clear_on_submit=True):
    user_input = st.text_input('정책을 물어보세요!', '', key='input')
    submitted = st.form_submit_button('Send')

# submitted의 값이 True면 챗봇이 답변을 하기 시작
if submitted and user_input:
    # 프롬프트 생성
    prompt = create_prompt(df, user_input)
    
    # 생성한 프롬프트를 기반으로 챗봇 답변을 생성
    chatbot_message = geterate_response(prompt[0],prompt[1])

    # 화면에 보여주기 위해 사용자의 질문과 챗봇의 답변을 각각 저장
    st.session_state['past'].append(user_input)
    st.session_state['generated'].append(chatbot_message)

# 챗봇의 답변이 있으면 사용자의 질문과 챗봇의 답변을 가장 최근의 순서로 화면에 출력
if st.session_state['generated']:
    for i in reversed(range(len(st.session_state['generated']))):
        message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')
        message(st.session_state['generated'][i], key=str(i))

