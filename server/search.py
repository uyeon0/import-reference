"""검색 쪽: data/python_ref.json 을 불러와 임베딩 검색 + 키워드(BM25) 검색을 섞는다"""
import json
import math
import re
from collections import Counter

import numpy as np
from langchain_core.documents import Document

from .chain import ROOT, RagError, _api_errors, embeddings

INDEX_PATH = ROOT / "data" / "python_ref.json"   # scripts/build_index.py 가 만든다
K = 5   # mode='vector' · 'keyword' 일 때 가져오는 조각 수


def tokenize(text):
    """키워드 검색용 토큰. 영문 · 숫자 · 기호 단어는 그대로, 한글은 두 글자씩 겹쳐 자른다(bigram).
    '데코레이터는' → 데코 · 코레 · 레이 · 이터 · 터는 → '데코레이터'의 bigram 이 모두 들어 있어
    형태소 분석기 없이도 조사(는 · 를 · 에서)가 붙은 말이 걸린다"""
    tokens = []
    for w in re.findall(r"[가-힣]+|[A-Za-z_][A-Za-z0-9_]*|\d+|[^\sA-Za-z0-9가-힣]{1,3}", text.lower()):
        if re.match(r"[가-힣]", w):
            tokens += [w] if len(w) == 1 else [w[i:i + 2] for i in range(len(w) - 1)]
        else:
            tokens.append(w)
    return tokens


class BM25:
    """키워드 검색 점수. 질문 토큰이 조각에 많이 · 드물게(다른 조각엔 잘 없는 말일수록) 나올수록 높다"""

    def __init__(self, texts, k1=1.5, b=0.75):
        self.k1, self.b = k1, b
        self.docs = [Counter(tokenize(t)) for t in texts]
        self.lens = np.array([sum(d.values()) for d in self.docs])
        self.avg = self.lens.mean()
        df = Counter(tok for d in self.docs for tok in d)
        n = len(self.docs)
        self.idf = {tok: math.log(1 + (n - c + 0.5) / (c + 0.5)) for tok, c in df.items()}

    def scores(self, query):
        q = set(tokenize(query))
        out = np.zeros(len(self.docs))
        for i, d in enumerate(self.docs):
            norm = self.k1 * (1 - self.b + self.b * self.lens[i] / self.avg)
            out[i] = sum(self.idf[t] * d[t] * (self.k1 + 1) / (d[t] + norm) for t in q if t in d)
        return out


class Index:
    """scripts/build_index.py 가 저장한 python_ref.json 을 불러와 두 가지 검색을 한다.
    - 임베딩 검색: 뜻이 가까운 조각 (질문 벡터 ↔ 조각 벡터 코사인 유사도)
    - 키워드 검색: 같은 말이 들어 있는 조각 (BM25)
    하이브리드는 양쪽 상위 3개씩을 모으고 RRF 점수로 순서를 정한다. '데코레이터'처럼 용어가 분명한 질문은 키워드가, 풀어 쓴 질문은 임베딩이 잡는다"""

    def __init__(self):
        if not INDEX_PATH.exists():
            raise RagError("data/python_ref.json 이 없어요. 먼저 python -m scripts.build_index 를 실행해 주세요.")
        # 불러오기는 파일만 읽어서 키가 필요 없다. dump 형식: {id: {"vector", "text", "metadata"}}
        items = list(json.loads(INDEX_PATH.read_text(encoding="utf-8")).values())
        self.docs = [Document(page_content=it["text"], metadata=it["metadata"]) for it in items]
        m = np.array([it["vector"] for it in items])
        self.matrix = m / np.linalg.norm(m, axis=1, keepdims=True)
        self.bm25 = BM25([it["text"] for it in items])

    @_api_errors
    def search(self, query, api_key=None, k=K, mode="hybrid", keyword_query=None, pool=30, rrf_k=60):
        """(Document, 코사인 유사도) 목록. mode: 'vector' · 'keyword' (각 k개) · 'hybrid' (양쪽 상위 3개씩, 겹치면 3~6개)"""
        # 질문 벡터는 그 사용자의 키로 만들고(임베딩 API 1번), 비교는 저장된 벡터와 로컬에서 한다
        v = np.array(embeddings(api_key).embed_query(query))
        cos = self.matrix @ (v / np.linalg.norm(v))
        vec_rank = np.argsort(-cos)[:pool]
        kw = self.bm25.scores(keyword_query or query)
        kw_rank = [i for i in np.argsort(-kw)[:pool] if kw[i] > 0]
        if mode == "vector":
            top = list(vec_rank[:k])
        elif mode == "keyword":
            top = kw_rank[:k]
        else:
            # 각 검색의 상위 3개를 모두 가져온다(겹치면 하나로). 한쪽에서만 1~3위인 조각도 빠지지 않게
            # (RRF 점수만으로 고르면 한쪽 검색에서만 상위인 조각이 양쪽에서 중간쯤인 조각들에 밀린다)
            picked = list(dict.fromkeys([int(i) for i in vec_rank[:3]] + [int(i) for i in kw_rank[:3]]))
            # 순서는 RRF 점수 순: 각 검색에서의 순위 r 마다 1/(60+r) 를 더한다 (코사인 · BM25 점수 단위가 달라 순위로 합친다)
            fused = Counter()
            for ranks in (vec_rank, kw_rank):
                for r, i in enumerate(ranks, 1):
                    fused[int(i)] += 1 / (rrf_k + r)
            top = sorted(picked, key=lambda i: -fused[i])
        return [(self.docs[i], float(cos[i])) for i in top]
