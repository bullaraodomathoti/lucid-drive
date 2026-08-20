"""Generate all figures for LUCID-Drive dataset paper."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from matplotlib.gridspec import GridSpec
import matplotlib.ticker as mticker

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 150,
})

COLORS = ['#2C7BB6', '#D7191C', '#1A9641', '#FDAE61', '#ABD9E9',
          '#F46D43', '#74ADD1', '#A6D96A', '#FEE090', '#D9EF8B']

# ── Figure 1: Scenario Category Distribution ────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

categories = ['Urban\nIntersections', 'Highway\nMerging', 'Pedestrian-\ndense Zones',
              'Long-tail\nEdge Cases', 'Parking /\nLow-speed']
counts = [3847, 2156, 2891, 2318, 1635]
colors_cat = COLORS[:5]

bars = axes[0].bar(categories, counts, color=colors_cat, edgecolor='white', linewidth=0.8, width=0.6)
axes[0].set_ylabel('Number of Scenarios')
axes[0].set_title('(a) Scenario Category Distribution')
axes[0].set_ylim(0, 4500)
for bar, count in zip(bars, counts):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 60,
                 f'{count:,}', ha='center', va='bottom', fontsize=9, fontweight='bold')
axes[0].spines['top'].set_visible(False)
axes[0].spines['right'].set_visible(False)
axes[0].grid(axis='y', alpha=0.3, linestyle='--')

# Annotation type distribution (pie)
ann_types = ['Causal QA\nPairs', 'Counterfactual\nQA Pairs', 'Planning\nInstructions',
             'Scene\nDescriptions', 'Risk\nAssessment']
ann_counts = [312456, 198327, 156832, 89247, 90431]
explode = (0.05, 0.05, 0, 0, 0)
wedges, texts, autotexts = axes[1].pie(ann_counts, labels=ann_types, autopct='%1.1f%%',
                                        explode=explode, colors=colors_cat,
                                        startangle=140, pctdistance=0.75,
                                        textprops={'fontsize': 9})
for at in autotexts:
    at.set_fontsize(8)
    at.set_fontweight('bold')
axes[1].set_title('(b) Annotation Type Distribution\n(Total: 847,293 annotations)')

plt.tight_layout()
plt.savefig('/home/sandbox/dataset_paper/figures/scenario_and_annotation.png',
            dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("Figure 1 saved.")

# ── Figure 2: Dataset Comparison Bar Chart ──────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 7))

datasets = ['LUCID-Drive\n(Ours)', 'DriveLM', 'NuScenes-QA', 'NuPrompt',
            'DRAMA', 'BDD-X', 'Box-QAymo', 'STRIDE-QA', 'Rank2Tell',
            'CoVLA', 'WEDGE', 'IEDD', 'HAD', 'PotentialRiskQA', 'CARScenes']
qa_counts = [847293, 150000, 459941, 35000, 17785, 26000, 90000, 20000,
             10000, 12000, 5000, 8000, 45000, 6000, 14000]

colors_comp = ['#D7191C'] + ['#74ADD1'] * 14
bars = ax.barh(datasets, qa_counts, color=colors_comp, edgecolor='white', linewidth=0.6, height=0.65)
ax.set_xscale('log')
ax.set_xlabel('Number of Language Annotations (log scale)')
ax.set_title('Dataset Scale Comparison: Language Annotations Across LLM4AD Datasets', fontweight='bold')
ax.axvline(x=100000, color='gray', linestyle='--', linewidth=0.8, alpha=0.6)
ax.text(120000, -0.8, '100K', fontsize=8, color='gray', va='bottom')

for bar, count in zip(bars, qa_counts):
    ax.text(bar.get_width() * 1.05, bar.get_y() + bar.get_height()/2,
            f'{count:,}', va='center', ha='left', fontsize=8)

ax.invert_yaxis()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(axis='x', alpha=0.3, linestyle='--')

patch_ours = mpatches.Patch(color='#D7191C', label='LUCID-Drive (proposed)')
patch_other = mpatches.Patch(color='#74ADD1', label='Existing datasets')
ax.legend(handles=[patch_ours, patch_other], loc='lower right', fontsize=10)

plt.tight_layout()
plt.savefig('/home/sandbox/dataset_paper/figures/dataset_comparison.png',
            dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("Figure 2 saved.")

# ── Figure 3: Baseline Results Bar Chart ────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(13, 9))

models = ['GPT-4V', 'DriveLM-\nAgent', 'LLaMA-\n3.2V', 'LLaVA-\n1.6', 'InstructBLIP', 'BLIP-2']
col_models = ['#D7191C', '#F46D43', '#FDAE61', '#74ADD1', '#ABD9E9', '#D9EF8B']

# Task 1: CSU - BLEU-4
bleu4 = [18.4, 16.7, 15.1, 14.2, 12.3, 10.8]
bars1 = axes[0,0].bar(models, bleu4, color=col_models, edgecolor='white', width=0.6)
axes[0,0].set_title('(a) Causal Scene Understanding (BLEU-4)')
axes[0,0].set_ylabel('BLEU-4 Score')
axes[0,0].set_ylim(0, 22)
for b, v in zip(bars1, bleu4):
    axes[0,0].text(b.get_x()+b.get_width()/2, b.get_height()+0.2, f'{v}', ha='center', fontsize=9)
axes[0,0].spines['top'].set_visible(False)
axes[0,0].spines['right'].set_visible(False)
axes[0,0].grid(axis='y', alpha=0.3, linestyle='--')

# Task 2: CFR - Accuracy
acc = [58.3, 52.1, 48.6, 44.7, 41.9, 38.2]
bars2 = axes[0,1].bar(models, acc, color=col_models, edgecolor='white', width=0.6)
axes[0,1].set_title('(b) Counterfactual Reasoning (Accuracy %)')
axes[0,1].set_ylabel('Accuracy (%)')
axes[0,1].set_ylim(0, 68)
for b, v in zip(bars2, acc):
    axes[0,1].text(b.get_x()+b.get_width()/2, b.get_height()+0.5, f'{v}', ha='center', fontsize=9)
axes[0,1].spines['top'].set_visible(False)
axes[0,1].spines['right'].set_visible(False)
axes[0,1].grid(axis='y', alpha=0.3, linestyle='--')

# Task 3: IFN - Success Rate
sr = [42.7, 38.9, 35.7, 32.4, 29.3, 25.8]
bars3 = axes[1,0].bar(models, sr, color=col_models, edgecolor='white', width=0.6)
axes[1,0].set_title('(c) Instruction-following Navigation (Success Rate %)')
axes[1,0].set_ylabel('Success Rate (%)')
axes[1,0].set_ylim(0, 52)
for b, v in zip(bars3, sr):
    axes[1,0].text(b.get_x()+b.get_width()/2, b.get_height()+0.4, f'{v}', ha='center', fontsize=9)
axes[1,0].spines['top'].set_visible(False)
axes[1,0].spines['right'].set_visible(False)
axes[1,0].grid(axis='y', alpha=0.3, linestyle='--')

# Task 4: RAP - Collision Avoidance
ca = [78.3, 72.4, 68.1, 63.7, 58.9, 54.2]
bars4 = axes[1,1].bar(models, ca, color=col_models, edgecolor='white', width=0.6)
axes[1,1].set_title('(d) Risk-aware Planning (Collision Avoidance %)')
axes[1,1].set_ylabel('Collision Avoidance Rate (%)')
axes[1,1].set_ylim(0, 92)
for b, v in zip(bars4, ca):
    axes[1,1].text(b.get_x()+b.get_width()/2, b.get_height()+0.5, f'{v}', ha='center', fontsize=9)
axes[1,1].spines['top'].set_visible(False)
axes[1,1].spines['right'].set_visible(False)
axes[1,1].grid(axis='y', alpha=0.3, linestyle='--')

plt.suptitle('Baseline Model Performance on LUCID-Drive Benchmark Tasks', fontsize=13, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig('/home/sandbox/dataset_paper/figures/baseline_results.png',
            dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("Figure 3 saved.")

# ── Figure 4: Weather and Lighting Distribution ──────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

weather = ['Clear', 'Cloudy', 'Rainy', 'Snowy', 'Foggy', 'Hazy', 'Mixed']
w_counts = [4120, 2847, 2156, 1312, 842, 730, 840]
w_colors = ['#FFD700', '#B0C4DE', '#4682B4', '#87CEEB', '#C0C0C0', '#D2B48C', '#9370DB']
bars_w = axes[0].bar(weather, w_counts, color=w_colors, edgecolor='white', linewidth=0.8, width=0.65)
axes[0].set_ylabel('Number of Scenarios')
axes[0].set_title('(a) Weather Condition Distribution')
for bar, cnt in zip(bars_w, w_counts):
    axes[0].text(bar.get_x()+bar.get_width()/2, bar.get_height()+40,
                 f'{cnt:,}', ha='center', va='bottom', fontsize=9, fontweight='bold')
axes[0].spines['top'].set_visible(False)
axes[0].spines['right'].set_visible(False)
axes[0].grid(axis='y', alpha=0.3, linestyle='--')
axes[0].set_ylim(0, 4700)

lighting = ['Daytime', 'Dusk/Dawn', 'Nighttime', 'Low-light\nAdverse']
l_counts = [6423, 2341, 2847, 1236]
l_colors = ['#FFD700', '#FF8C00', '#191970', '#483D8B']
wedges, texts, autos = axes[1].pie(l_counts, labels=lighting, autopct='%1.1f%%',
                                    colors=l_colors, startangle=90,
                                    textprops={'fontsize': 10}, pctdistance=0.78)
for at in autos:
    at.set_fontsize(9)
    at.set_fontweight('bold')
    at.set_color('white')
axes[1].set_title('(b) Lighting Condition Distribution\n(Total: 12,847 scenarios)')

plt.tight_layout()
plt.savefig('/home/sandbox/dataset_paper/figures/weather_lighting.png',
            dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("Figure 4 saved.")

print("\nAll figures generated successfully.")
