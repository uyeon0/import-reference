"""Flask 서버 — 로컬: python -m server.app → http://127.0.0.1:5000 · 배포: gunicorn server.app:app"""
import json

from flask import Flask, Response, jsonify, render_template, request
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from werkzeug.middleware.proxy_fix import ProxyFix

from .chain import KeyProblem, RagError, answer_events, resolve_key
from .search import Index

app = Flask(__name__)
# 배포 플랫폼(Render 등)은 앞단 프록시를 거쳐 요청이 온다. X-Forwarded-For 의 첫 프록시 한 단계만 믿어
# request.remote_addr 를 실제 사용자 IP 로 (요청 횟수 제한이 IP 기준이라 필요하다)
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1)
app.config["MAX_CONTENT_LENGTH"] = 64 * 1024   # 요청 본문 64KB 까지 (질문 + 대화 기록이면 충분하다)

# 요청 횟수 제한: IP 마다 /ask 를 분당 10번 · 하루 200번. 기록은 프로세스 메모리에 둔다 (서버 1대 기준)
limiter = Limiter(get_remote_address, app=app, storage_uri="memory://")

MAX_QUESTION = 1000   # 질문 글자 수
MAX_HISTORY = 6       # 대화 기록 메시지 수 (질문 · 답 3쌍)
MAX_TURN = 4000       # 기록 메시지 하나의 글자 수

_index = None   # 인덱스는 서버 프로세스마다 처음 요청 때 한 번만 불러 둔다


def get_index():
    global _index
    if _index is None:
        _index = Index()
    return _index


@app.get("/")
def home():
    return render_template("index.html")


def clean_history(raw):
    """브라우저가 보낸 대화 기록을 믿지 않고 정리한다: 형식이 맞는 것만, 최근 MAX_HISTORY 개, 메시지마다 MAX_TURN 자"""
    if not isinstance(raw, list):
        return []
    out = []
    for h in raw[-MAX_HISTORY:]:
        if isinstance(h, dict) and h.get("role") in ("user", "assistant") and isinstance(h.get("content"), str):
            out.append({"role": h["role"], "content": h["content"][:MAX_TURN]})
    return out


@app.errorhandler(429)
def too_many(e):
    return jsonify(error="요청이 너무 많아요. 1분 뒤에 다시 시도해 주세요."), 429


@app.errorhandler(413)
def too_large(e):
    return jsonify(error="요청이 너무 커요."), 413


@app.post("/ask")
@limiter.limit("10 per minute;200 per day")
def ask():
    body = request.get_json(silent=True)
    if not isinstance(body, dict):
        return jsonify(error="요청 형식이 올바르지 않아요."), 400
    question = body.get("question")
    question = question.strip() if isinstance(question, str) else ""
    if not question:
        return jsonify(error="질문을 입력해 주세요."), 400
    if len(question) > MAX_QUESTION:
        return jsonify(error=f"질문은 {MAX_QUESTION:,}자까지 쓸 수 있어요."), 400
    history = clean_history(body.get("history"))
    # 사용자 키는 요청 헤더로만 받는다. 저장하지도, 로그에 남기지도 않는다
    api_key = request.headers.get("X-OpenAI-Key", "").strip() or None
    try:
        resolve_key(api_key)   # 키가 아예 없으면 스트림을 열기 전에 401
        index = get_index()
    except KeyProblem as e:
        return jsonify(error=str(e), key=True), 401
    except RagError as e:
        return jsonify(error=str(e)), 500

    # NDJSON 스트리밍: 한 줄에 JSON 하나. 진행 단계 → 출처 → 글자 조각들 → 끝
    # 스트림을 연 뒤에는 상태 코드를 바꿀 수 없어서, 틀린 키 같은 에러도 error 이벤트로 보낸다
    def generate():
        try:
            for ev in answer_events(index, question, history, api_key):
                yield json.dumps(ev, ensure_ascii=False) + "\n"
        except KeyProblem as e:
            yield json.dumps({"type": "error", "text": str(e), "key": True}, ensure_ascii=False) + "\n"
        except RagError as e:
            yield json.dumps({"type": "error", "text": str(e)}, ensure_ascii=False) + "\n"
        yield json.dumps({"type": "done"}) + "\n"

    return Response(generate(), mimetype="application/x-ndjson")


if __name__ == "__main__":
    # debug=True 는 로컬 개발용. 배포할 때는 gunicorn 같은 WSGI 서버로 띄운다
    app.run(debug=True, threaded=True)
