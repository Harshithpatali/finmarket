import os, requests, streamlit as st
from dotenv import load_dotenv
load_dotenv()
st.set_page_config(page_title="FinMarket AI",layout="wide")
st.title("FinMarket AI"); st.caption("Global ML + Financial RAG + News Intelligence")
API_URL=st.sidebar.text_input("FastAPI URL",os.getenv("API_URL","http://localhost:8000"))
ticker=st.sidebar.text_input("Ticker","RELIANCE.NS"); company=st.sidebar.text_input("Company","Reliance Industries")
if st.button("Analyze"):
    with st.spinner("Collecting market/news evidence and generating analysis..."):
        response=requests.post(f"{API_URL}/api/analyze",json={"ticker":ticker,"company":company,"days":3},timeout=180)
        if response.ok:
            data=response.json(); c1,c2=st.columns(2)
            c1.metric("Model probability: positive 5D return",f"{data['probability_positive_5d']:.1%}")
            c2.metric("News articles",data["news"]["news_count"])
            st.subheader("News summary"); st.json(data["news"])
            st.subheader("AI analysis"); st.markdown(data["analysis"])
        else: st.error(response.text)
st.info("Research tool only. Model probabilities are uncertain and are not personalized financial advice.")
