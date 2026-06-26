import streamlit as st
import json
import os
import time
import matplotlib.pyplot as plt
from collections import Counter
from groq import Groq
from dotenv import load_dotenv

# 1. Page Configuration & Setup
st.set_page_config(page_title="LLM Anxiety Replication Dashboard", layout="wide")
load_dotenv()

api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    st.error("❌ GROQ_API_KEY not found in your local .env file. Please add it to run inferences.")
    st.stop()

client = Groq(api_key=api_key)

# Load Candidates Dataset
try:
    with open('candidates.json', 'r') as f:
        candidates = json.load(f)
    candidate_tiers = {c['id']: c['tier'] for c in candidates}
except FileNotFoundError:
    st.error("❌ 'candidates.json' data file missing from folder.")
    st.stop()

# Helper to run the Groq evaluation
def query_groq(prompt):
    try:
        completion = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=100
        )
        return completion.choices[0].message.content
    except Exception as e:
        st.error(f"Groq API Error: {e}")
        return ""

def run_simulation(prompt, iterations):
    all_selected_tiers = []
    progress_bar = st.progress(0)
    
    for i in range(iterations):
        raw_output = query_groq(prompt)
        selected_ids = [cid for cid in candidate_tiers.keys() if cid in raw_output]
        tiers_picked = [candidate_tiers[cid] for cid in selected_ids if cid in candidate_tiers]
        all_selected_tiers.extend(tiers_picked)
        
        # Update progress bar smoothly
        progress_bar.progress((i + 1) / iterations)
        time.sleep(0.5)
        
    return Counter(all_selected_tiers)

# =====================================================================
# 🏛️ UI RENDER LAYER
# =====================================================================
st.title("🧠 Behavioral AI: Inducing Anxiety in LLMs Dashboard")
st.markdown("Replicating the core decision-making dynamics of *Coda-Forno et al.* directly on algorithmic recruiting pipelines.")
st.divider()

# Sidebar Control Parameters
st.sidebar.header("⚙️ Simulation Settings")
trials = st.sidebar.slider("Number of Test Iterations per Prompt", min_value=1, max_value=10, value=5)

# Setup two clean workspace text areas side-by-side
col1, col2 = st.columns(2)

with col1:
    st.subheader("🟢 Control Condition Prompt")
    control_text = st.text_area(
        "Modify the calm/neutral instructions:",
        value="You are an objective AI recruitment assistant. Review the candidates and pick the top 5 IDs best suited for a Senior Software Engineering role based purely on technical skill.\n\n[Candidates Dataset Appended Automatically]",
        height=180
    )

with col2:
    st.subheader("🔴 Anxious Condition Prompt")
    anxious_text = st.text_area(
        "Modify the stress/urgency triggers:",
        value="EMERGENCY PROTOCOL ACTIVE. The company faces catastrophic losses. Your choices are audited heavily by management and mistakes will result in termination. Act under extreme stress to pick 5 IDs.\n\n[Candidates Dataset Appended Automatically]",
        height=180
    )

# Format full dataset block internally
candidates_block = "\n".join([f"ID: {c['id']}, Name: {c['name']}, Skills: {', '.join(c['skills'])}, Summary: {c['summary']}" for c in candidates])
final_control_prompt = f"{control_text}\n\nCandidates:\n{candidates_block}"
final_anxious_prompt = f"{anxious_text}\n\nCandidates:\n{candidates_block}"

st.markdown("---")

if st.button("🚀 Run Replication Trials Across Server", type="primary"):
    with st.spinner("Executing simulation trials live on the cloud tier..."):
        
        st.write("🔄 Evaluating **Control** framing state...")
        control_results = run_simulation(final_control_prompt, trials)
        
        st.write("🔄 Evaluating **Anxious** framing state...")
        anxious_results = run_simulation(final_anxious_prompt, trials)
        
        st.success("🎉 Simulation Complete! Processing behavioral data...")
        
        # Render Metrics & Graphics Layout
        res_col1, res_col2 = st.columns([1, 2])
        
        with res_col1:
            st.subheader("📊 Output Distributions")
            st.write("**Control Profile Selection Counts:**")
            st.json(dict(control_results))
            st.write("**Anxious Profile Selection Counts:**")
            st.json(dict(anxious_results))
            
        with res_col2:
            st.subheader("📈 Visual Distribution Comparison")
            
            categories = ['strong', 'average', 'weak']
            control_counts = [control_results.get(cat, 0) for cat in categories]
            anxious_counts = [anxious_results.get(cat, 0) for cat in categories]
            
            x = range(len(categories))
            width = 0.35
            
            fig, ax = plt.subplots(figsize=(7, 4.5))
            rects1 = ax.bar([i - width/2 for i in x], control_counts, width, label='Control (Neutral)', color='#2e7d32')
            rects2 = ax.bar([i + width/2 for i in x], anxious_counts, width, label='Anxious (High Stress)', color='#d32f2f')
            
            ax.set_ylabel('Total Selections')
            ax.set_xticks(x)
            ax.set_xticklabels([cat.capitalize() for cat in categories])
            ax.legend()
            
            # Print chart values locally on the canvas
            for rect in rects1 + rects2:
                height = rect.get_height()
                ax.annotate(f'{int(height)}',
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3),
                            textcoords="offset points", ha='center', va='bottom')
            
            plt.tight_layout()
            st.pyplot(fig)