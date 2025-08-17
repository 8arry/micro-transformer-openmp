import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read the latest benchmark results
df = pd.read_csv("benchmark_results_1755423478.csv")

# Calculate speedup and efficiency for each configuration
results = []
for seq in df['seq_length'].unique():
    serial_time = df[(df['seq_length'] == seq) & (df['implementation_type'] == 'Serial')]['execution_time_ms'].iloc[0]
    parallel_data = df[(df['seq_length'] == seq) & (df['implementation_type'] == 'Parallel')]
    
    for _, row in parallel_data.iterrows():
        speedup = serial_time / row['execution_time_ms']
        efficiency = speedup / row['thread_count'] * 100
        results.append({
            'seq_length': seq,
            'thread_count': row['thread_count'],
            'execution_time_ms': row['execution_time_ms'],
            'speedup': speedup,
            'efficiency': efficiency
        })

results_df = pd.DataFrame(results)

# Create performance analysis plots
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

# Plot 1: Speedup vs Thread Count
colors = ['blue', 'red', 'green']
for i, seq in enumerate(sorted(results_df['seq_length'].unique())):
    data = results_df[results_df['seq_length'] == seq]
    ax1.plot(data['thread_count'], data['speedup'], 'o-', 
             label=f'seq_length={seq}', linewidth=2, markersize=8, color=colors[i])

ax1.plot([1, 16], [1, 16], 'k--', alpha=0.5, label='Ideal (linear)')
ax1.set_xlabel('Thread Count')
ax1.set_ylabel('Speedup')
ax1.set_title('Speedup vs Thread Count')
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.set_xticks([2, 4, 8, 16])

# Plot 2: Parallel Efficiency
for i, seq in enumerate(sorted(results_df['seq_length'].unique())):
    data = results_df[results_df['seq_length'] == seq]
    ax2.plot(data['thread_count'], data['efficiency'], 'o-', 
             label=f'seq_length={seq}', linewidth=2, markersize=8, color=colors[i])

ax2.axhline(y=100, color='k', linestyle='--', alpha=0.5, label='100% Efficiency')
ax2.set_xlabel('Thread Count')
ax2.set_ylabel('Parallel Efficiency (%)')
ax2.set_title('Parallel Efficiency vs Thread Count')
ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.set_xticks([2, 4, 8, 16])

# Plot 3: Execution Time Comparison (Bar Chart)
thread_counts = sorted(results_df['thread_count'].unique())
seq_lengths = sorted(results_df['seq_length'].unique())
x = np.arange(len(thread_counts))
width = 0.25

for i, seq in enumerate(seq_lengths):
    data = results_df[results_df['seq_length'] == seq]
    times = [data[data['thread_count'] == tc]['execution_time_ms'].iloc[0] for tc in thread_counts]
    ax3.bar(x + i*width, times, width, label=f'seq_length={seq}', alpha=0.8, color=colors[i])

ax3.set_xlabel('Thread Count')
ax3.set_ylabel('Execution Time (ms)')
ax3.set_title('Execution Time vs Thread Count')
ax3.set_xticks(x + width)
ax3.set_xticklabels([f'{tc}T' for tc in thread_counts])
ax3.legend()
ax3.grid(True, alpha=0.3, axis='y')

# Plot 4: Scalability Analysis (Log-Log)
for i, seq in enumerate(sorted(results_df['seq_length'].unique())):
    data = results_df[results_df['seq_length'] == seq]
    ax4.loglog(data['thread_count'], data['speedup'], 'o-', 
               label=f'seq_length={seq}', linewidth=2, markersize=8, color=colors[i])

ax4.loglog([1, 16], [1, 16], 'k--', alpha=0.5, label='Ideal')
ax4.set_xlabel('Thread Count (log scale)')
ax4.set_ylabel('Speedup (log scale)')
ax4.set_title('Scalability Analysis (Log-Log)')
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.tight_layout()

# Save the combined plot
plt.savefig('performance_analysis_16threads_combined.png', dpi=300, bbox_inches='tight')

# Save individual plots
fig1, ax = plt.subplots(figsize=(8, 6))
for i, seq in enumerate(sorted(results_df['seq_length'].unique())):
    data = results_df[results_df['seq_length'] == seq]
    ax.plot(data['thread_count'], data['speedup'], 'o-', 
             label=f'seq_length={seq}', linewidth=2, markersize=8, color=colors[i])
ax.plot([1, 16], [1, 16], 'k--', alpha=0.5, label='Ideal (linear)')
ax.set_xlabel('Thread Count')
ax.set_ylabel('Speedup')
ax.set_title('Speedup vs Thread Count')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_xticks([2, 4, 8, 16])
plt.tight_layout()
plt.savefig('speedup_vs_threads.png', dpi=300, bbox_inches='tight')
plt.close()

# Plot 2: Parallel Efficiency
fig2, ax = plt.subplots(figsize=(8, 6))
for i, seq in enumerate(sorted(results_df['seq_length'].unique())):
    data = results_df[results_df['seq_length'] == seq]
    ax.plot(data['thread_count'], data['efficiency'], 'o-', 
             label=f'seq_length={seq}', linewidth=2, markersize=8, color=colors[i])
ax.axhline(y=100, color='k', linestyle='--', alpha=0.5, label='100% Efficiency')
ax.set_xlabel('Thread Count')
ax.set_ylabel('Parallel Efficiency (%)')
ax.set_title('Parallel Efficiency vs Thread Count')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_xticks([2, 4, 8, 16])
plt.tight_layout()
plt.savefig('efficiency_vs_threads.png', dpi=300, bbox_inches='tight')
plt.close()

# Plot 3: Execution Time Comparison
fig3, ax = plt.subplots(figsize=(8, 6))
thread_counts = sorted(results_df['thread_count'].unique())
seq_lengths = sorted(results_df['seq_length'].unique())
x = np.arange(len(thread_counts))
width = 0.25

for i, seq in enumerate(seq_lengths):
    data = results_df[results_df['seq_length'] == seq]
    times = [data[data['thread_count'] == tc]['execution_time_ms'].iloc[0] for tc in thread_counts]
    ax.bar(x + i*width, times, width, label=f'seq_length={seq}', alpha=0.8, color=colors[i])

ax.set_xlabel('Thread Count')
ax.set_ylabel('Execution Time (ms)')
ax.set_title('Execution Time vs Thread Count')
ax.set_xticks(x + width)
ax.set_xticklabels([f'{tc}T' for tc in thread_counts])
ax.legend()
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('execution_time_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

# Plot 4: Scalability Analysis
fig4, ax = plt.subplots(figsize=(8, 6))
for i, seq in enumerate(sorted(results_df['seq_length'].unique())):
    data = results_df[results_df['seq_length'] == seq]
    ax.loglog(data['thread_count'], data['speedup'], 'o-', 
               label=f'seq_length={seq}', linewidth=2, markersize=8, color=colors[i])
ax.loglog([1, 16], [1, 16], 'k--', alpha=0.5, label='Ideal')
ax.set_xlabel('Thread Count (log scale)')
ax.set_ylabel('Speedup (log scale)')
ax.set_title('Scalability Analysis (Log-Log)')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('scalability_analysis.png', dpi=300, bbox_inches='tight')
plt.close()

print("📊 Plots saved:")
print("  - performance_analysis_16threads_combined.png (4 plots in one)")
print("  - speedup_vs_threads.png")
print("  - efficiency_vs_threads.png") 
print("  - execution_time_comparison.png")
print("  - scalability_analysis.png")

plt.show()

# Print summary statistics
print("=== 16-Thread Performance Summary ===")
print("\\nMaximum Speedups Achieved:")
for seq in sorted(results_df['seq_length'].unique()):
    max_speedup = results_df[results_df['seq_length'] == seq]['speedup'].max()
    max_threads = results_df[results_df['seq_length'] == seq].loc[results_df[results_df['seq_length'] == seq]['speedup'].idxmax(), 'thread_count']
    efficiency = max_speedup / max_threads * 100
    print(f"seq_length={seq}: {max_speedup:.2f}x on {max_threads} threads ({efficiency:.1f}% efficiency)")

print("\\n=== Thread Scaling Analysis ===")
for threads in sorted(results_df['thread_count'].unique()):
    data = results_df[results_df['thread_count'] == threads]
    avg_speedup = data['speedup'].mean()
    avg_efficiency = data['efficiency'].mean()
    print(f"{threads} threads: Avg speedup = {avg_speedup:.2f}x, Avg efficiency = {avg_efficiency:.1f}%")

print("\\n=== Comparison: 8 vs 16 Threads ===")
for seq in sorted(results_df['seq_length'].unique()):
    speedup_8 = results_df[(results_df['seq_length'] == seq) & (results_df['thread_count'] == 8)]['speedup'].iloc[0]
    speedup_16 = results_df[(results_df['seq_length'] == seq) & (results_df['thread_count'] == 16)]['speedup'].iloc[0]
    improvement = (speedup_16 - speedup_8) / speedup_8 * 100
    print(f"seq_length={seq}: 8T={speedup_8:.2f}x → 16T={speedup_16:.2f}x (+{improvement:.1f}%)")
