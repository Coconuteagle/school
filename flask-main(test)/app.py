from flask import Flask, send_from_directory
from flask import Flask, render_template, request, jsonify, Response
import openai
from nltk.tokenize import sent_tokenize
from flask_cors import CORS
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re
import os
from datetime import datetime

app = Flask(__name__, static_folder="static")
CORS(app)
# 🔹 방문자 데이터 저장 파일
VISITOR_FILE = 'visitors.json'

# 🔹 방문자 수 로드 함수
def load_visitors():
    if os.path.exists(VISITOR_FILE):
        with open(VISITOR_FILE, 'r', encoding='utf-8') as file:
            return json.load(file)
    return {"today": 0, "total": 0, "last_date": "", "ips": []}

# 🔹 방문자 수 저장 함수
def save_visitors(data):
    with open(VISITOR_FILE, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)

# 🔹 방문자 수 API 엔드포인트 (중복 방문 방지 추가)
@app.route('/visitors', methods=['GET'])
def get_visitors():
    visitors = load_visitors()
    today_date = datetime.now().strftime('%Y-%m-%d')

    # 🔹 실제 클라이언트 IP 가져오기
    if request.headers.get("X-Forwarded-For"):
        user_ip = request.headers.get("X-Forwarded-For").split(',')[0]
    else:
        user_ip = request.remote_addr

    print(f"🔹 Detected IP: {user_ip}")  # 디버깅 로그

    # 🔹 날짜가 변경되면 today 방문자 수 초기화
    if visitors["last_date"] != today_date:
        visitors["last_date"] = today_date
        visitors["today"] = 0
        visitors["ips"] = []  # IP 목록 초기화

    # 🔹 중복 방문 방지: 같은 IP에서 새로고침해도 방문자 수 증가 X
    if user_ip not in visitors["ips"]:
        visitors["ips"].append(user_ip)
        visitors["today"] += 1
        visitors["total"] += 1
        save_visitors(visitors)

    return jsonify({"today": visitors["today"], "total": visitors["total"]})

openai.api_key = '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'

with open('data.txt', 'r', encoding='utf-8') as file:
    text = file.read()

def find_relevant_parts(text, user_question):
    keywords = user_question.lower().split()
    text_parts = text.lower().split('content_box')  # Split by ';'

    selected_parts = []

    # For the first word
    if len(keywords) > 0:
        first_word = keywords[0]
        for length in [3, 4, 5]:  # Prefix lengths (3, 4, 5)
            first_word_prefix = first_word[:length]  # Prefix of the first word
            prefix_parts = [part for part in text_parts if any(word.startswith(first_word_prefix) for word in part.split()) and part not in selected_parts]
            selected_parts.extend(prefix_parts[:3])

    # For the second word
    if len(keywords) > 1:
        second_word = keywords[1]
        for length in [2, 3]:  # Prefix lengths (2, 3)
            second_word_prefix = second_word[:length]
            prefix_parts = [part for part in text_parts if any(word.startswith(second_word_prefix) for word in part.split()) and part not in selected_parts]
            selected_parts.extend(prefix_parts[:3])

        # First and Second word combined prefix
        for part in text_parts:
            if first_word[:2] in part and second_word[:2] in part and part not in selected_parts:
                selected_parts.append(part)
            if len(selected_parts) >= 4:  # Max of 7 parts with the current criteria
                break

    # For the third word
    if len(keywords) > 2:
        third_word = keywords[2]
        for length in [2, 3]:  # Prefix lengths (2, 3)
            third_word_prefix = third_word[:length]
            prefix_parts = [part for part in text_parts if any(word.startswith(third_word_prefix) for word in part.split()) and part not in selected_parts]
            selected_parts.extend(prefix_parts[:1])
            
        # Second and Third word combined prefix
        for part in text_parts:
            if second_word[:2] in part and third_word[:2] in part and part not in selected_parts:
                selected_parts.append(part)
            if len(selected_parts) >= 6:  # Max of 10 parts with the current criteria
                break

    # For the fourth word
    if len(keywords) > 3:
        fourth_word = keywords[3]
        # Third and Fourth word combined prefix
        for part in text_parts:
            if third_word[:2] in part and fourth_word[:2] in part and part not in selected_parts:
                selected_parts.append(part)
            if len(selected_parts) >= 8:  # Max of 12 parts with the current criteria
                break

    return selected_parts[:2]

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
    input_sentence = preprocess(user_question)
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
        response = Response(response=json.dumps({"answer": answer}, ensure_ascii=False),
                            status=200,
                            mimetype="application/json")
        return response

    relevant_text = find_most_similar_sentences(user_question, document_sentences, top_n=10)

    system_message = f"{relevant_text}\n\n 당신은 청렴챗봇 프락시스입니다. 1. 앞의 내용을 바탕으로 최대 7문장 이내+70단어 이내로 요약해서 존댓말로 답해줘. 2. 질문이 내용과 관계없으면 다시 질문해주시겠어요? 라고 답변해줘 3.내용을 바탕으로만 답해줘.또한 예외 사항이나 사례가 있으면 그것도 답해줘 4. 사용자에게 재질문 금지 5. 관련 법령도 같이 답변(답변시 참조한 문장과 정확히 관련된 법령) 6. 링크가 있으면 링크도 답변(답변과 관련있는 링크만) 7. 답변과 관련있는 링크란: 질문과 답변 청크는 hide를 기준으로 나뉘어진다. 관련 없는 링크는 포함시키면 안됩니다."

    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_question}
        ]
    )
    answer_with_links = convert_urls_to_links(response['choices'][0]['message']['content'])

    return jsonify(answer=answer_with_links)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
