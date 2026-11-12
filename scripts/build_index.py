"""파이썬 언어 레퍼런스(한국어) → 절 단위 Document → 임베딩 → data/python_ref.json
실행: python -m scripts.build_index   (문서를 새로 받을 때만. 앱은 저장된 파일을 불러온다)"""
import os
import re
import sys
import time
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup
from langchain_core.documents import Document

from langchain_text_splitters import RecursiveCharacterTextSplitter

from server.chain import RagError, embeddings
from server.search import INDEX_PATH

BASE = "https://docs.python.org/ko/3/reference/"
HEADERS = {"User-Agent": "import-reference (unofficial study project)"}
SKIP = {"grammar.html"}   # 10장 '전체 문법 규격'은 문장 없이 문법 규칙만 있어 검색을 흐린다


def get(url):
    res = requests.get(url, headers=HEADERS, timeout=20)
    res.raise_for_status()
    time.sleep(1)   # 1초에 1번 이하로 요청 (서버에 부담을 주지 않도록)
    res.encoding = "utf-8"   # 응답 헤더에 charset 이 없어 requests 가 latin-1 로 읽는다
    return BeautifulSoup(res.text, "html.parser")


def chapters():
    """목차 페이지에서 장 링크 (1. 개요 ~ 9. 최상위 요소들)"""
    soup = get(BASE + "index.html")
    links = soup.select("li.toctree-l1 > a.reference.internal")
    return [(a.get_text(strip=True), urljoin(BASE, a["href"])) for a in links if a["href"] not in SKIP]


def heading_text(h):
    return h.get_text(" ", strip=True).replace("¶", "").strip()


def section_text(sec):
    """이 절의 글만 (안에 들어 있는 하위 절은 빼고). 코드는 줄바꿈 · 들여쓰기를 살린다"""
    parts = []
    for child in sec.children:
        if not getattr(child, "name", None) or child.name == "section" or re.fullmatch(r"h[1-6]", child.name):
            continue
        # Sphinx 코드 블록: <div class="highlight-python3"><div class="highlight"><pre>…
        is_code = child.name == "pre" or any(c.startswith("highlight") for c in child.get("class", []))
        if is_code:
            parts.append(child.get_text().rstrip())
        else:
            parts.append(re.sub(r"\s+", " ", child.get_text(" ", strip=True)))
    return "\n\n".join(p for p in parts if p)


def page_docs(title, url):
    """한 장(페이지) → 절마다 Document. 출처는 '장 › 절' 이름과 그 절로 바로 가는 주소"""
    soup = get(url)
    docs = []
    for sec in soup.select("div.body section[id]"):
        h = sec.find(re.compile(r"h[1-6]"), recursive=False)
        text = section_text(sec)
        if not h or len(text) < 30:   # 제목만 있는 절, 색인용 빈 절은 건너뛴다
            continue
        name = heading_text(h)
        label = title if name == title else f"{title} › {name}"
        docs.append(Document(
            page_content=f"{label}\n\n{text}",   # 제목도 같이 임베딩해야 '비교 연산자' 같은 질문이 걸린다
            metadata={"source": label, "url": f"{url}#{sec['id']}"},
        ))
    return docs


def split(docs):
    """빈 줄 → 줄 → 문장 끝 → 띄어쓰기 순으로 자른다. 긴 절만 실제로 잘린다.
    조각마다 원래 Document 의 metadata(출처 · 주소)가 따라간다"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=600, chunk_overlap=80, separators=["\n\n", "\n", ". ", " ", ""]
    )
    return splitter.split_documents(docs)


def main():
    from langchain_core.vectorstores import InMemoryVectorStore

    docs = []
    for title, url in chapters():
        page = page_docs(title, url)
        print(f"  {title}: 절 {len(page)}개")
        docs += page
    chunks = split(docs)
    # 긴 절은 여러 조각으로 잘리는데, 제목은 첫 조각에만 있다. 나머지 조각에도 '장 › 절'을 붙여야
    # 'for 문 else' 질문에 for 문의 두 번째 조각(else 설명)이 걸린다
    for c in chunks:
        if not c.page_content.startswith(c.metadata["source"]):
            c.page_content = f"{c.metadata['source']}\n\n{c.page_content}"
    print(f"절 {len(docs)}개 → 조각 {len(chunks)}개 · {sum(len(c.page_content) for c in chunks):,}자 임베딩")
    # 관리용 스크립트라 .env 의 키를 명시적으로 넘긴다 (앱은 .env 키를 쓰지 않는다)
    store = InMemoryVectorStore(embeddings(os.getenv("OPENAI_API_KEY")))
    store.add_documents(chunks)
    store.dump(str(INDEX_PATH))
    print(f"저장: {INDEX_PATH.name}")


if __name__ == "__main__":
    try:
        main()
    except RagError as e:
        print("⚠️", e)
        sys.exit(1)
    except requests.RequestException:
        print("⚠️ docs.python.org 에서 문서를 받지 못했어요. 인터넷 연결을 확인해 주세요.")
        sys.exit(1)
