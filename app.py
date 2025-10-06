import streamlit as st
from docx import Document
import pdfplumber
import re
import nltk
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import requests
import feedparser
import time

# ----------------- NLTK downloads -----------------
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
###### Title ######
st.set_page_config(page_title="Research Navigator", layout="wide")
st.title("Research Navigator -Intelligent Research Paper Analyzer")
st.markdown(
    "Upload a PDF/DOCX  "

)
###  text preprocessing###
@st.cache_data(show_spinner=False)
def clean_text_raw(raw: str) -> str:
    txt = re.sub(r"http\S+|www\S+|https\S+", "", raw)
    txt = re.sub(r"(Volume|Issue|ISSN|©|Page \d+|\d{1,4}-\d{1,4})", "", txt, flags=re.I)
    txt = re.sub(r'\s+', ' ', txt)
    return txt.strip()

@st.cache_data(show_spinner=False)
def extract_text_from_file(uploaded_file) -> str:
    text = ""
    if uploaded_file.type == "application/pdf":
        with pdfplumber.open(uploaded_file) as pdf:
            for page in pdf.pages:
                p = page.extract_text()
                if p:
                    text += p + " "
    elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        doc = Document(uploaded_file)
        text = " ".join([p.text for p in doc.paragraphs])
    else:
        try:
            text = uploaded_file.getvalue().decode('utf-8')
        except Exception:
            raise ValueError("Unsupported file type")
    return clean_text_raw(text)

@st.cache_data(show_spinner=False)
def split_sentences(text: str):
    sents = sent_tokenize(text)
    return [s.strip() for s in sents if len(s.split()) >= 5]

@st.cache_data(show_spinner=False)
def extract_keyphrases_tfidf(text: str, top_n=12):
    v = TfidfVectorizer(stop_words='english', max_features=2000, ngram_range=(1,3))
    X = v.fit_transform([text])
    scores = X.toarray().flatten()
    terms = v.get_feature_names_out()
    idx = np.argsort(scores)[::-1][:top_n]
    return [(terms[i], float(scores[i])) for i in idx]

@st.cache_data(show_spinner=False)
def lda_topics_for_sentences(sentences, n_topics=3):
    if len(sentences) < 2:
        return []
    v = TfidfVectorizer(stop_words='english', max_features=1000)
    X = v.fit_transform(sentences)
    n_topics = min(n_topics, X.shape[0], 6)
    lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
    lda.fit(X)
    feat = v.get_feature_names_out()
    topics = []
    for comp in lda.components_:
        top_words = [feat[i] for i in comp.argsort()[-6:][::-1]]
        topics.append(top_words)
    return topics

#### web scrapping####
SEMANTIC_SCHOLAR_URL = "https://api.semanticscholar.org/graph/v1/paper/search"

@st.cache_data(show_spinner=False)
def fetch_semanticscholar(query: str, limit: int = 3, api_key: str = None):
    headers = {}
    if api_key:
        headers["x-api-key"] = api_key
    params = {"query": query, "limit": limit, "fields": "title,abstract,url,authors,year"}
    try:
        r = requests.get(SEMANTIC_SCHOLAR_URL, params=params, headers=headers, timeout=8)
        if r.status_code == 200:
            data = r.json().get("data", [])
            out = []
            for p in data:
                out.append({
                    "title": p.get("title"),
                    "abstract": p.get("abstract") or "",
                    "url": p.get("url"),
                    "source": "ss",
                    "authors": [a.get("name") for a in p.get("authors", [])] if p.get("authors") else [],
                    "year": p.get("year")
                })
            return out
        else:
            return []
    except Exception:
        return []

@st.cache_data(show_spinner=False)
def fetch_arxiv(query: str, limit: int = 3):
    q = "all:" + query
    url = f"http://export.arxiv.org/api/query?search_query={requests.utils.quote(q)}&start=0&max_results={limit}"
    try:
        feed = feedparser.parse(url)
        out = []
        for e in feed.entries:
            title = getattr(e, 'title', '')
            summary = getattr(e, 'summary', '')
            link = getattr(e, 'link', '')
            out.append({"title": title, "abstract": summary, "url": link, "source": "arxiv"})
        return out
    except Exception:
        return []

#### Question answering part########
def build_corpus_tfidf(local_sents, external_papers):
    corpus_texts = local_sents.copy()
    metadata = [{"source": "local", "index": i} for i in range(len(local_sents))]
    for i,p in enumerate(external_papers):
        text = (p.get("title","") + ". " + p.get("abstract","")).strip()
        if len(text) < 30:
            continue
        corpus_texts.append(text)
        metadata.append({"source": p.get("source","external"), "index": i, "meta": p})
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(corpus_texts)
    return corpus_texts, X, metadata, vectorizer

def retrieve_top_k_tfidf(question, vectorizer, X, corpus_texts, metadata, top_k=5, external_weight=0.9):
    q_vec = vectorizer.transform([question])
    sims = cosine_similarity(q_vec, X).flatten()
    for i,m in enumerate(metadata):
        if m.get("source") != "local":
            sims[i] *= external_weight
    idx = np.argsort(sims)[::-1][:top_k]
    results = [{"score": float(sims[i]), "text": corpus_texts[i], "meta": metadata[i]} for i in idx]
    return results

def synthesize_answer_tfidf(retrieved, max_sentences=3):
    if not retrieved:
        return "No relevant content found locally or on the web."
    local_parts = []
    external_parts = []
    for r in retrieved:
        src = r["meta"].get("source")
        if src == "local":
            local_parts.append((r["score"], r["text"]))
        else:
            meta = r["meta"].get("meta", {})
            title = meta.get("title")
            abstract = meta.get("abstract","")
            external_parts.append((r["score"], f"{title}. {abstract[:300]}... [{meta.get('url','')}]"))
    local_parts = sorted(local_parts, key=lambda x:x[0], reverse=True)[:max_sentences]
    out = [t for _,t in local_parts]
    if len(out) < max_sentences and external_parts:
        external_parts = sorted(external_parts, key=lambda x:x[0], reverse=True)
        out += [t for _,t in external_parts[:max_sentences-len(out)]]
    return " ".join(out)

# Interface ##

st.sidebar.header("Options & Retrieval")
use_web = st.sidebar.checkbox("Enable web retrieval (Semantic Scholar + arXiv)", value=True)
ss_limit = st.sidebar.slider("Semantic Scholar results", 0, 5, 3)
arxiv_limit = st.sidebar.slider("arXiv results", 0, 5, 2)
ss_api_key = st.sidebar.text_input("Semantic Scholar API Key (optional)", type="password")
external_weight = st.sidebar.slider("External doc score weight", 0.1, 1.0, 0.9)

uploaded_file = st.file_uploader("Upload PDF or DOCX", type=["pdf","docx"])
if uploaded_file:
    with st.spinner("Processing document..."):
        raw_text = extract_text_from_file(uploaded_file)
        local_sents = split_sentences(raw_text)
    st.subheader("Document Summary")
    st.write(f"Total sentences: {len(local_sents)}")
    st.write("Top keyphrases (TF-IDF):")
    kp = extract_keyphrases_tfidf(raw_text, top_n=12)
    st.write(", ".join([k for k,_ in kp]))
    st.write("LDA Topics:")
    topics = lda_topics_for_sentences(local_sents, n_topics=3)
    for i,t in enumerate(topics):
        st.write(f"Topic {i+1}: {', '.join(t)}")

    web_papers = []
    if use_web:
        seed_query = " ".join([k for k,_ in kp[:4]]) or raw_text[:200]
        if ss_limit>0:
            ss = fetch_semanticscholar(seed_query, limit=ss_limit, api_key=ss_api_key)
            web_papers.extend(ss)
        time.sleep(0.2)
        if arxiv_limit>0:
            ax = fetch_arxiv(seed_query, limit=arxiv_limit)
            web_papers.extend(ax)

    corpus_texts, X, metadata, vectorizer = build_corpus_tfidf(local_sents, web_papers)

 
    st.subheader("Ask Questions")
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    for item in st.session_state.chat_history:
        st.markdown(f"**Q:** {item['q']}")
        st.markdown(f"**A:** {item['a']}")
        st.markdown("---")

    q = st.text_input("Enter your question:", key="q")
    if st.button("Ask"):
        if q.strip() == "":
            st.warning("Please enter a question.")
        else:
            retrieved = retrieve_top_k_tfidf(q, vectorizer, X, corpus_texts, metadata, top_k=6, external_weight=external_weight)
            ans = synthesize_answer_tfidf(retrieved, max_sentences=3)
            st.session_state.chat_history.append({"q": q, "a": ans})
            st.markdown(f"**A:** {ans}")


    if use_web and web_papers:
        st.subheader("Retrieved External Papers ")
        for p in web_papers:
            st.markdown(f"**{p.get('title','(no title)')}** — source: {p.get('source')}, year: {p.get('year','NA')}")
            if p.get("abstract"):
                st.write(p.get("abstract")[:400] + ("..." if len(p.get("abstract"))>400 else ""))
            if p.get("url"):
                st.write(p.get("url"))
            st.markdown("---")
else:
    st.info("Upload a PDF or DOCX to start analysing.")
