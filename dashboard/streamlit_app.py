import streamlit as st
from dashboard.latency_tracker import LatencyTracker
from dashboard.ranking_metrics import RankingMetrics
import random

st.title("Local Multimodal RAG Dashboard")

latency_tracker = LatencyTracker()

# Simulate steps
steps = ["PDF Crawl", "Image Crawl", "Extraction", "BM25 Search", "Vector Search", "Rerank", "Answer Generation"]
for step in steps:
    duration = random.uniform(0.2,2.0)
    latency_tracker.records.append({"step":step, "latency":duration})

st.subheader("Latency (seconds)")
st.bar_chart(latency_tracker.summary())

st.subheader("Sample Ranking Metrics")
precision = random.uniform(0.5,1.0)
recall = random.uniform(0.5,1.0)
st.caption("Demo data (replace with real pipeline metrics)")

st.write(f"Precision@10: {precision:.2f}")
st.write(f"Recall@10: {recall:.2f}")