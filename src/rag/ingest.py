from pathlib import Path
import json,re,hashlib
import networkx as nx
import numpy as np
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import faiss

def clean_text(text):
    text=re.sub(r"-\s*\n\s*","",text); text=re.sub(r"\s*\n\s*"," ",text); text=re.sub(r"\s{2,}"," ",text)
    return text.strip()

def extract_pdf(pdf_path):
    reader=PdfReader(str(pdf_path)); pages=[]
    for page_num,page in enumerate(reader.pages,1):
        text=clean_text(page.extract_text() or "")
        if text: pages.append({"page":page_num,"text":text})
    return pages

def chunk_pages(pages,chunk_words=450,overlap=60):
    chunks=[]
    for item in pages:
        words=item["text"].split()
        start=0
        while start<len(words):
            end=min(start+chunk_words,len(words))
            chunks.append({"page":item["page"],"text":" ".join(words[start:end])})
            if end==len(words): break
            start=end-overlap
    return chunks

def build_chunks(raw_dir="data/raw",output="data/knowledge/chunks.jsonl"):
    rows=[]
    for pdf in sorted(Path(raw_dir).glob("*.pdf")):
        print(f"Reading {pdf.name}")
        for i,c in enumerate(chunk_pages(extract_pdf(pdf))):
            digest=hashlib.sha1(f"{pdf.name}|{c['page']}|{i}|{c['text']}".encode()).hexdigest()[:16]
            rows.append({"chunk_id":digest,"document":pdf.name,"page":c["page"],"text":c["text"]})
    Path(output).parent.mkdir(parents=True,exist_ok=True)
    with open(output,"w",encoding="utf-8") as f:
        for row in rows: f.write(json.dumps(row,ensure_ascii=False)+"\n")
    print(f"Saved {len(rows):,} chunks -> {output}")
    return rows

def build_faiss(chunks,output_dir="data/knowledge"):
    model=SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    texts=[x["text"] for x in chunks]
    embeddings=model.encode(texts,batch_size=32,show_progress_bar=True,normalize_embeddings=True).astype("float32")
    index=faiss.IndexFlatIP(embeddings.shape[1]); index.add(embeddings)
    out=Path(output_dir); out.mkdir(parents=True,exist_ok=True)
    faiss.write_index(index,str(out/"faiss.index"))
    (out/"metadata.json").write_text(json.dumps(chunks,ensure_ascii=False,indent=2),encoding="utf-8")

def build_basic_knowledge_graph(chunks,output="data/knowledge/knowledge_graph.json"):
    concepts={"value investing":["intrinsic value","margin of safety","valuation"],"risk":["risk","margin of safety","diversification"],"market":["market fluctuations","market price","market sentiment"],"fundamental analysis":["earnings","balance sheet","valuation"],"investment":["investment","speculation","investor"]}
    graph=nx.MultiDiGraph()
    for concept,related in concepts.items():
        graph.add_node(concept,type="concept")
        for rel in related:
            graph.add_node(rel,type="financial_concept"); graph.add_edge(concept,rel,relation="related_to")
    for chunk in chunks:
        text=chunk["text"].lower(); node=chunk["chunk_id"]
        graph.add_node(node,type="document_chunk",document=chunk["document"],page=chunk["page"])
        for concept in concepts:
            if concept in text: graph.add_edge(node,concept,relation="mentions")
    (Path(output)).parent.mkdir(parents=True,exist_ok=True)
    Path(output).write_text(json.dumps(nx.node_link_data(graph),indent=2),encoding="utf-8")

def build_rag(raw_dir="data/raw"):
    chunks=build_chunks(raw_dir)
    if not chunks: raise RuntimeError("No PDF files found in data/raw.")
    build_faiss(chunks); build_basic_knowledge_graph(chunks)
