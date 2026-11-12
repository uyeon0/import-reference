"""LLM 쪽: 설정 · 키 · 에러 변환 · 검색용 질문 다시 쓰기 · 답 이벤트 스트리밍
흐름: 질문 다시 쓰기(LLM) → 하이브리드 검색(search.py) → 답 스트리밍(LLM)"""
import os
from pathlib import Path

import openai
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

ROOT = Path(__file__).resolve().parent.parent      # 프로젝트 최상위 (.env · data/ 가 있는 곳)
load_dotenv(ROOT / ".env")

CHAT_MODEL = os.getenv("OPENAI_MODEL", "gpt-6-luna")
EMBED_MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")
NOT_FOUND = "문서에서 찾지 못했습니다."


class RagError(Exception):
    """사용자에게 보여 줄 한국어 한 줄 에러. 키 값은 절대 메시지에 넣지 않는다"""


class KeyProblem(RagError):
    """키가 없거나 틀린 경우. 서버는 401 로 돌려주고 화면은 키 입력 칸을 다시 연다"""


def _api_errors(fn):
    """OpenAI 에러를 한국어 RagError 로 바꾼다. 에러 원문에 키 일부가 섞일 수 있어 원문은 버린다"""
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except openai.AuthenticationError:
            raise KeyProblem("OpenAI 키가 틀렸거나 만료됐어요. 키를 확인해 주세요.") from None
        except (openai.PermissionDeniedError, openai.NotFoundError):
            # 403 권한 없음 · 404 모델 없음: 이 키의 계정에서 모델을 쓸 수 없는 경우
            raise KeyProblem(f"이 키로는 {CHAT_MODEL} · {EMBED_MODEL} 모델을 쓸 수 없어요. 다른 키를 넣어 주세요.") from None
        except openai.RateLimitError:
            # 429: 요청이 너무 잦거나 계정 크레딧이 없는 경우
            raise RagError("OpenAI 사용 한도에 걸렸어요. 잠시 뒤 다시 시도하거나 계정의 크레딧을 확인해 주세요.") from None
        except openai.APIConnectionError:
            raise RagError("OpenAI 서버에 연결하지 못했어요.") from None
        except openai.APIError:
            # 그 밖의 OpenAI 에러. 원문에 키 일부가 섞일 수 있어 보여 주지 않는다
            raise RagError("OpenAI 요청이 실패했어요. 잠시 뒤 다시 시도해 주세요.") from None
    return wrapper


def resolve_key(api_key=None):
    """앱은 사용자가 화면에서 넣은 키만 쓴다. 서버 .env 의 키로 대신 답하지 않는다
    (배포 서버에 실수로 키를 넣어도 남이 내 키로 질문할 수 없게).
    .env 의 키는 scripts/build_index.py · scripts/eval_search.py 처럼 관리용 스크립트만 명시적으로 넘겨 쓴다"""
    key = api_key
    if not key:
        raise KeyProblem("OpenAI API 키를 입력해 주세요.")
    return key


def embeddings(api_key=None):
    """임베딩 모델. 저장할 때와 질문할 때 같은 모델을 써야 한다"""
    return OpenAIEmbeddings(model=EMBED_MODEL, api_key=resolve_key(api_key), check_embedding_ctx_length=False)


# ── 검색용 질문 다시 쓰기 (대화 맥락 · 문서 용어) ─────────

REWRITE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "마지막 질문을 문서 검색에 쓸 질문 한 문장으로 다시 써. 답하지 말고 질문만 써.\n"
               "- 앞의 대화가 있으면 '그거', '그럼' 같은 말을 가리키는 대상으로 바꿔.\n"
               "- 기호나 키워드(is, ==, with, *args 등)가 있으면 공식 문서에서 그것을 부르는 이름도 같이 써."),
    ("placeholder", "{history}"),
    ("human", "{question}"),
])


def to_messages(history):
    """브라우저가 보낸 [{role, content}] → LangChain 메시지 튜플. 최근 6개만 (토큰 비용)"""
    return [("human" if h["role"] == "user" else "ai", h["content"]) for h in history[-6:]]


@_api_errors
def standalone_question(llm, question, history):
    """LLM 호출 1번으로 검색용 질문을 다시 쓴다. 두 가지 이유:
    - '그럼 None은?'을 그대로 임베딩하면 무엇에 대한 질문인지 몰라 엉뚱한 조각을 찾는다
    - 'is 와 =='처럼 기호뿐인 질문은 문서 문장('아이덴티티 비교')과 벡터가 멀다"""
    return llm.invoke(REWRITE_PROMPT.format_messages(history=to_messages(history), question=question)).content


# ── 답하기 (스트리밍) ──────────────────────────────────

ANSWER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """너는 파이썬 언어 레퍼런스를 근거로 답하는 도우미야. 아래 규칙을 지켜.

1. [자료] 안에 있는 내용으로만 답해. 원래 알던 지식은 쓰지 마.
2. 자료에 답이 없으면 다른 말 없이 이 문장만 답해: "{not_found}"
3. 숫자 · 조건 · 문법 규칙은 자료에 적힌 그대로 써. 코드 예시는 자료에 있는 것을 우선 써.
4. 근거로 쓴 문장 끝에 [번호] 형식으로 자료 번호를 붙여. 자료에 없는 번호는 쓰지 마.
5. 짧고 분명하게, 합니다체(~입니다, ~합니다)로 써. ~해요체는 쓰지 마.

[자료]
{context}"""),
    ("placeholder", "{history}"),
    ("human", "{question}"),
])


def build_context(hits):
    """조각마다 [번호] 출처 머리줄. 화면에서는 답에 처음 나온 순서로 번호를 다시 매긴다"""
    return "\n\n".join(
        f"[{i}] {d.metadata['source']}\n{d.page_content}"
        for i, (d, _) in enumerate(hits, 1)
    )


def get_llm(api_key=None):
    # gpt-6-luna 는 temperature 기본값(1)만 받는다
    return ChatOpenAI(model=CHAT_MODEL, api_key=resolve_key(api_key))


def answer_events(index, question, history, api_key=None):
    """답 하나를 만드는 과정을 이벤트(dict)로 하나씩 내보낸다. 화면은 이걸 받아 진행 단계를 보여 준다
    stage(rewrite) → stage(rewrite 끝 + 검색용 질문) → stage(search) → meta(출처) → stage(answer) → token …"""
    llm = get_llm(api_key)

    yield {"type": "stage", "stage": "rewrite"}
    query = standalone_question(llm, question, history)   # 검색에만 쓴다 (LLM 1번)
    yield {"type": "stage", "stage": "search", "query": query}

    # 키워드 검색에는 원래 질문 + 다시 쓴 질문을 같이 넣는다 (원래 말과 공식 용어 둘 다 걸리게)
    hits = index.search(query, api_key, keyword_query=f"{question} {query}")
    sources = [
        {"n": i, "label": d.metadata['source'], "url": d.metadata.get("url"),
         "score": round(s, 3), "text": d.page_content}
        for i, (d, s) in enumerate(hits, 1)
    ]
    yield {"type": "meta", "query": query, "sources": sources}
    yield {"type": "stage", "stage": "answer"}

    messages = ANSWER_PROMPT.format_messages(
        not_found=NOT_FOUND, context=build_context(hits),
        history=to_messages(history), question=question,
    )
    # 제너레이터는 호출할 때가 아니라 next() 로 꺼낼 때 실행된다. 그래서 @_api_errors 로 감싸면
    # 에러가 데코레이터 밖(Flask 가 꺼낼 때)에서 터진다 → 한 조각씩 꺼내는 함수를 감싼다
    stream = llm.stream(messages)
    next_chunk = _api_errors(lambda: next(stream, None))
    while (chunk := next_chunk()) is not None:   # 모델이 만든 글자를 오는 대로 넘긴다
        if chunk.content:
            yield {"type": "token", "text": chunk.content}
