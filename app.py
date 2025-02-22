import google.generativeai as genai
from flask import Flask, send_from_directory, render_template, request, jsonify
from flask_cors import CORS
import json
import numpy as np
import re
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__, static_folder="static")
CORS(app)

# 🔹 Google Gemini API 설정
genai.configure(api_key="AIzaSyCptpJ68R5lyJPduY8rtqUXR9Ij7F4puoE")

# 🔹 모델 순서대로 정렬
GEMINI_MODELS = [
     "gemini-3.0-flash",  
    "gemini-2.0-flash",                    # 1️⃣ Gemini 2.0 Flash
    "gemini-2.0-flash-lite-preview",        # 2️⃣ Gemini 2.0 Flash-Lite 미리보기
    "gemini-1.5-flash",                     # 3️⃣ Gemini 1.5 Flash
    "gemini-1.5-flash-8b",                  # 4️⃣ Gemini 1.5 Flash-8B
    "gemini-2.0-pro-experimental-02-05",    # 5️⃣ Gemini 2.0 Pro Experimental
    "gemini-1.5-pro"                        # 6️⃣ Gemini 1.5 Pro
]

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

    # 🔹 유사한 문장 찾기 (문자열 변환 필요)
    relevant_sentences = find_most_similar_sentences(user_question, document_sentences, top_n=10)
    relevant_text = "\n".join(relevant_sentences)  # 🔥 리스트를 문자열로 변환

    # 🔹 AI에게 전달할 프롬프트 생성 (사용자 질문 포함)
    system_message = f"""{relevant_text}

    당신은 학교행정업무 서포터입니다. 
    1. 앞의 내용을 바탕으로 최대 7문장 이내+70단어 이내로 요약해서 존댓말로 답해줘. 
    2. 질문이 내용과 관계없으면 다시 질문해주시겠어요? 라고 답변해줘. 
    3. 내용을 바탕으로만 답해줘. 또한 예외 사항이나 사례가 있으면 그것도 답해줘. 
    4. 사용자에게 재질문 금지. 
    5. 관련 법령도 같이 답변(답변시 참조한 문장과 정확히 관련된 법령). 
    6. 링크가 있으면 링크도 답변(답변과 관련있는 링크만)."""

    final_prompt = f"{system_message}\n\n사용자 질문: {user_question}"

    response = None
    last_exception = None  # 🔹 마지막 오류 저장
    switched_model = None  # 🔹 사용된 모델 저장

    for model in GEMINI_MODELS:
        try:
            print(f"[🔄] 모델 시도: {model}")  # 🔹 로그에는 모델 변경 내역 표시
            client = genai.GenerativeModel(model)

            # 🔥 AI 모델에게 user_question과 관련 문장을 포함하여 전달
            response = client.generate_content(final_prompt)

            if response and hasattr(response, 'text') and response.text:
                print(f"[✅] 모델 {model} 사용 성공!")
                switched_model = model  # 🔹 모델 변경 감지
                break

        except Exception as e:
            error_message = str(e).lower()
            last_exception = e  # 🔹 마지막 오류 저장

            # 🔹 특정 에러 메시지 감지 후 자동으로 다음 모델로 스위칭
            if any(keyword in error_message for keyword in ["quota exceeded", "rate limit", "429", "not found", "unsupported"]):
                print(f"[⚠️] {model} 사용 불가 ({error_message}). 다음 모델로 전환 중...")
                continue  # 다음 모델 시도
            
            print(f"[❌] {model} 호출 오류: {e}")
            return jsonify({"answer": f"AI 응답 오류: {str(e)}"})

    if response is None or not hasattr(response, 'text') or not response.text:
        print("[❌] 모든 모델이 한도를 초과했거나 지원되지 않습니다.")
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
