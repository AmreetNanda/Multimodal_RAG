import time 
from utils.logger import get_logger

logger = get_logger("LatencyTracker")

class LatencyTracker:
    """
    Tracks the latency of each step in the multimodal RAG pipeline
    """

    def __init__(self):
        self.records = []
    
    def track(self, step_name, func, *args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        latency = end - start
        self.records.append({"step":step_name, "latency":latency})
        logger.info(f"{step_name} took {latency:.2f}s")
        return result
    
    def summary(self):
        return {r["step"]: r["latency"] for r in self.records}