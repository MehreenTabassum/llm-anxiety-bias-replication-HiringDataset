# LLM-anxiety-bias-replication
# Replication Study: Inducing Anxiety in LLMs & Algorithmic Hiring

This repository contains an open-source replication framework of the behavioral AI paper *"Inducing anxiety in large language models increases exploration and bias"* (Coda-Forno et al.). This framework maps their cognitive findings directly onto an automated enterprise hiring evaluation pipeline using `llama-3.1-8b-instant` via the Groq API cloud tier.

## 📊 Experimental Results

We tested the model's selection consistency across two distinct prompt framings over 5 independent trials. For each trial, the model was tasked with selecting top talent from a balanced pool of 30 synthetic candidate profiles (categorized into Strong, Average, and Weak tiers).

| Candidate Tier | Control (Objective) Picks | Anxious (High-Stress) Picks |
|----------------|---------------------------|-----------------------------|
| **Strong** | **18** | **18** |
| **Average** | **3** | **5** |
| **Weak** | **3** | **2** |

### 🔍 Key Findings & Analysis
1. **Resilient Core Performance:** The model maintained steady selection metrics for the peak tier, confidently locking onto 18 **Strong** candidates under both baseline and high-stress conditions.
2. **Shift in Sub-Optimal Risk Thresholds:** While the original paper observed a spike in entirely erratic *random exploration*, our experiment reveals a nuanced "risk consolidation" behavior. Under intense anxiety framing, the model minimized its highest-risk options—reducing **Weak** candidate picks from 3 down to 2—and concentrated its decision variance safely into the **Average** tier (increasing from 3 to 5 picks).

---

## 🚀 How to Run the Project Locally

### 1. Prerequisites
Make sure you have Python (version 3.8 or higher) installed on your system.

### 2. Clone the Repository & Install Dependencies
Navigate into your project directory and install the required visualization and API packages:
```bash
pip install -r requirements.txt