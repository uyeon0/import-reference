"""검색 비교 — 임베딩 · 키워드(BM25) · 하이브리드가 답이 들어 있는 조각을 찾아오는지
실행: python -m scripts.eval_search   (질문마다 LLM 1번 + 임베딩 3번. 다시 쓴 질문이 실행마다 달라 결과가 조금씩 달라진다)
출력: 질문별로 각 방식이 정답 조각을 몇 위로 찾았는지 (✗ = 못 찾음), 마지막 줄에 방식별 맞힌 수"""

import os

from server.chain import get_llm, standalone_question
from server.search import Index
# (질문, 정답 조각에 꼭 있는 문구) — 절 이름이 아니라 답이 들어 있는 조각을 찾았는지 본다
Q = [("데코레이터는 어떻게 동작해?", "f1(arg)(f2(func))"), ("is 와 == 는 어떻게 달라?", "아이덴티티"),
     ("for 문에 else 붙이면 언제 실행돼?", "else 절을 실행하지 않고"), ("with 문은 어떻게 동작해?", "__enter__"),
     ("*args 랑 **kwargs 차이", "**identifier"), (":= 이게 뭐야?", ":="),
     ("기본 인자로 리스트를 쓰면 왜 값이 공유돼?", "pre-computed"), ("제너레이터에서 return 을 쓰면 어떻게 돼?", "StopIteration"),
     ("클래스에 __eq__를 정의하면 == 비교가 어떻게 바뀌어?", "__eq__"), ("람다 안에 대입문 쓸 수 있어?", "람다")]
KEY = os.getenv("OPENAI_API_KEY")   # 관리용 스크립트라 .env 의 키를 명시적으로 넘긴다
idx, llm = Index(), get_llm(KEY)
tot = {"vector": 0, "keyword": 0, "hybrid": 0}
for q, want in Q:
    rq = standalone_question(llm, q, [])   # 앱과 같은 방식으로 검색용 질문을 다시 쓴다
    row = []
    for mode in tot:
        hits = idx.search(rq, KEY, mode=mode, keyword_query=f"{q} {rq}")
        pos = next((i for i, (d, _) in enumerate(hits, 1) if want in d.page_content), None)
        tot[mode] += pos is not None
        row.append(f"{mode}:{pos or '✗'}")
    print(f"{q[:22]:<24} " + "  ".join(row))
print({m: f"{c}/{len(Q)}" for m, c in tot.items()})
