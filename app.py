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

GEMINI_MODELS = [
    "gemini-2.0-flash",                    # 1️⃣ Gemini 2.0 Flash
    "gemini-2.0-flash-lite-preview",        # 2️⃣ Gemini 2.0 Flash-Lite 미리보기
    "gemini-1.5-flash",                     # 3️⃣ Gemini 1.5 Flash
    "gemini-1.5-flash-8b",                  # 4️⃣ Gemini 1.5 Flash-8B
    "gemini-2.0-pro-experimental-02-05",    # 5️⃣ Gemini 2.0 Pro Experimental
    "gemini-1.5-pro"                        # 6️⃣ Gemini 1.5 Pro
]

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    user_question = data.get('question') + "?"

    response = None
    last_exception = None  # 🔹 마지막 오류 저장
    switched_model = None  # 🔹 사용된 모델 저장

    for model in GEMINI_MODELS:
        try:
            print(f"[🔄] 모델 시도: {model}")  # 🔹 로그에는 모델 변경 내역 표시
            client = genai.GenerativeModel(model)
            response = client.generate_content(user_question)
            
            if response and hasattr(response, 'text') and response.text:
                print(f"[✅] 모델 {model} 사용 성공!")
                switched_model = model  # 🔹 모델 변경 감지
                break

        except Exception as e:
            error_message = str(e).lower()
            last_exception = e  # 🔹 마지막 오류 저장
            
            if "quota exceeded" in error_message or "rate limit" in error_message or "429" in error_message:
                print(f"[⚠️] {model} 한도 초과! 다음 모델로 전환 중...")
                continue  # 다음 모델 시도
            
            print(f"[❌] {model} 호출 오류: {e}")
            return jsonify({"answer": f"AI 응답 오류: {str(e)}"})

    if response is None or not hasattr(response, 'text') or not response.text:
        print("[❌] 모든 모델이 한도를 초과했습니다.")
        return jsonify({"answer": f"현재 모든 AI 모델이 사용 불가 상태입니다. (에러: {last_exception})"})

    # 🔹 사용된 모델 정보를 로그에는 남기지만, 실제 응답에는 포함하지 않음
    if switched_model:
        print(f"[ℹ️] 최종 사용된 모델: {switched_model}")

    response_data = {"answer": response.text}  # 🔥 사이트 대화에서는 모델 변경 메시지 제거

    return app.response_class(
        response=json.dumps(response_data, ensure_ascii=False),  # ✨ 한글 깨짐 방지
        status=200,
        mimetype="application/json"
    )

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
