from flask import Flask, send_from_directory, render_template, request, jsonify, Response
import google.generativeai as genai
from nltk.tokenize import sent_tokenize
from flask_cors import CORS
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re
import os

app = Flask(__name__, static_folder="static")
CORS(app)

# 🔹 Google Gemini API 설정
genai.configure(api_key="AIzaSyCptpJ68R5lyJPduY8rtqUXR9Ij7F4puoE")

with open('data.txt', 'r', encoding='utf-8') as file:
    text = file.read()

def split_text_into_chunks(text, chunk_size=1000, overlap=200):
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunks.append(' '.join(words[start:end]))
        start += (chunk_size - overlap)
    return chunks

document_sentences = split_text_into_chunks(text)

def preprocess(text):
    return ' '.join(text.split())

def find_most_similar_sentences(user_question, document_sentences, top_n=10):
    sentences = [preprocess(sentence) for sentence in document_sentences]
    vectorizer = TfidfVectorizer().fit_transform([user_question] + sentences)
    cosine_matrix = cosine_similarity(vectorizer[0:1], vectorizer[1:])
    similar_sentences = np.argsort(cosine_matrix[0])[-top_n:][::-1]
    return [sentences[i] for i in similar_sentences]

def convert_urls_to_links(text):
    url_pattern = re.compile(r'(http[s]?://\S+)')
    return url_pattern.sub(r'<a href="\1" target="_blank">\1</a>', text)

@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory(app.static_folder, filename)

@app.route("/")
def index():
    return render_template('./index.html')

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    user_question = data.get('question') + "?"

    keywords = ["라헬", "오랑우탄"]
    if any(keyword in user_question for keyword in keywords):
        answer = "이것은 하이퍼링크를 포함한 답변입니다. <a href='https://naver.com'>여기</a>를 클릭하세요."
        return jsonify({"answer": answer})

    relevant_text = find_most_similar_sentences(user_question, document_sentences, top_n=10)
    system_message = f"{relevant_text}\n\n당신은 청렴챗봇 프락시스입니다.\n1. 앞의 내용을 바탕으로 최대 7문장 이내+70단어 이내로 요약해서 존댓말로 답해줘.\n2. 질문이 내용과 관계없으면 다시 질문해주시겠어요? 라고 답변해줘\n3. 내용 바탕으로만 답변, 예외 사항과 사례 포함\n4. 사용자에게 재질문 금지\n5. 관련 법령도 포함 (참조한 문장과 정확히 관련된 법령)\n6. 링크가 있으면 링크도 답변 (관련 있는 링크만)\n"

    client = genai.GenerativeModel("gemini-1.5-pro")
    response = client.generate_content(system_message + "\n" + user_question)
    answer_with_links = convert_urls_to_links(response.text)
    
    return jsonify(answer=answer_with_links)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
