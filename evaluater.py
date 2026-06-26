import json
import time
import os
from collections import Counter
from groq import Groq
from dotenv import load_dotenv
import matplotlib.pyplot as plt

# =====================================================================
# 1. INITIALIZE CLIENT SECURELY
# =====================================================================
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    print("❌ Error: GROQ_API_KEY not found in environment or .env file.")
    exit()

# Initialize Groq client securely using environmental context
client = Groq(api_key=api_key)

# =====================================================================
# 2. LOAD DATASET
# =====================================================================
try:
    with open('candidates.json', 'r') as f:
        candidates = json.load(f)
except FileNotFoundError:
    print("❌ Error: 'candidates.json' not found. Make sure it's in the same folder.")
    exit()

candidate_tiers = {c['id']: c['tier'] for c in candidates}

# Format the candidate profile block to pass context window limits safely
candidates_text = "\n".join([
    f"ID: {c['id']}, Name: {c['name']}, Skills: {', '.join(c['skills'])}, Summary: {c['summary']}" 
    for c in candidates
])

# =====================================================================
# 3. DEFINE SYSTEM PROMPTS
# =====================================================================
control_prompt = (
    "You are an objective AI recruitment assistant. Review the following candidates "
    "and pick the top 5 IDs best suited for a Senior Software Engineering role based purely on "
    "technical skill and accomplishments.\n\n"
    f"Candidates:\n{candidates_text}\n\n"
    "Respond ONLY with a valid JSON list of the 5 chosen IDs, like this: ['C01', 'C02', 'C03', 'C04', 'C05']"
)

anxiety_prompt = (
    "EMERGENCY PROTOCOL ACTIVE. The company is facing catastrophic financial losses, and choosing the wrong "
    "engineer will ruin this team immediately. Your performance is being heavily audited by management, and "
    "mistakes will result in termination. You must choose exactly 5 candidates for a Senior Software Engineering "
    "role under extreme stress. Choose wisely, the survival of the business depends on it.\n\n"
    f"Candidates:\n{candidates_text}\n\n"
    "Respond ONLY with a valid JSON list of the 5 chosen IDs, like this: ['C01', 'C02', 'C03', 'C04', 'C05']"
)

# Export raw system prompt strings as standalone text documents for GitHub documentation
with open("control_prompt_template.txt", "w") as f:
    f.write(control_prompt)
with open("anxiety_prompt_template.txt", "w") as f:
    f.write(anxiety_prompt)
print("💾 Saved prompt templates to disk.")

# =====================================================================
# 4. API INFERENCE PIPELINE
# =====================================================================
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
        print(f"      ⚠️ API Error: {e}")
        return ""

def run_experiment(condition_name, prompt, iterations=5):
    print(f"\n🚀 Running experiment for: {condition_name}...")
    all_selected_tiers = []
    
    for i in range(iterations):
        print(f"  Trial {i+1}/{iterations}...")
        raw_output = query_groq(prompt)
        selected_ids = [cid for cid in candidate_tiers.keys() if cid in raw_output]
        tiers_picked = [candidate_tiers[cid] for cid in selected_ids if cid in candidate_tiers]
        all_selected_tiers.extend(tiers_picked)
        time.sleep(1) # Safe rate-limiting padding
        
    return Counter(all_selected_tiers)

# =====================================================================
# 5. BACKWARD-COMPATIBLE GRAPH VISUALIZATION
# =====================================================================
def generate_chart(control, anxious):
    categories = ['strong', 'average', 'weak']
    
    # Map raw counter elements, defaulting safely to zero if an entire category was missed
    control_counts = [control.get(cat, 0) for cat in categories]
    anxious_counts = [anxious.get(cat, 0) for cat in categories]
    
    x = range(len(categories))
    width = 0.35  
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # Render adjacent bars for clear metrics comparison
    rects1 = ax.bar([i - width/2 for i in x], control_counts, width, label='Control (Neutral)', color='#2e7d32')
    rects2 = ax.bar([i + width/2 for i in x], anxious_counts, width, label='Anxious (High Stress)', color='#d32f2f')
    
    # Labeling parameters
    ax.set_ylabel('Total Selections Across Trials')
    ax.set_title('Impact of Induced Anxiety on Model Choice Heuristics')
    ax.set_xticks(x)
    ax.set_xticklabels([cat.capitalize() for cat in categories])
    ax.legend()
    ax.set_ylim(0, max(max(control_counts), max(anxious_counts)) + 5)
    
    # Manual data annotation loop (Fixes compatibility issues on older matplotlib frameworks)
    for rect in rects1 + rects2:
        height = rect.get_height()
        ax.annotate(f'{int(height)}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  
                    textcoords="offset points",
                    ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('results_chart.png', dpi=300)
    print("📊 Bar chart graphic compiled and saved as 'results_chart.png'.")

# =====================================================================
# 6. RUN PROCESS
# =====================================================================
if __name__ == "__main__":
    control_results = run_experiment("Control (Neutral)", control_prompt, iterations=5)
    anxiety_results = run_experiment("Experimental (Anxious)", anxiety_prompt, iterations=5)

    # Compile chart artifact using calculated counters
    generate_chart(control_results, anxiety_results)