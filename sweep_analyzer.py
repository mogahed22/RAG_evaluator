import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from eval_retrieval import run_evaluation # الدالة اللي عدلناها عشان ترجع metrics

def start_sweep():
    # 1. حدد القيم اللي عايز تجربها
    chunk_sizes = [40, 140, 65]
    sweep_results = []

    print(f"🚀 Starting automated sweep for sizes: {chunk_sizes}")

    for size in chunk_sizes:
        print(f"\n" + "="*40)
        print(f"🔎 Testing Chunk Size: {size}")
        print("="*40)

        # 2. تشغيل الـ populate_database بالحجم الحالي
        # ده بينادي الملف اللي عدلناه بـ argparse وبيمسح الـ DB القديمة
        os.system(f"python3 populate_database.py --chunk_size {size} --chunk_overlap {int(size*0.2)}")

        # 3. تشغيل التقييم وأخذ النتائج
        # الدالة دي دلوقتي بتعمل paraphrase وبتحسب الـ MRR
        metrics = run_evaluation(chunk_size_label=size)
        sweep_results.append(metrics)

    # 4. تجميع النتائج ورسم الـ Line Chart للمقارنة
    df_sweep = pd.DataFrame(sweep_results)
    df_sweep.to_csv("results/sweep_comparison_results.csv", index=False)

    print("\n📊 Generating Comparison Chart...")
    sns.set_theme(style="darkgrid")
    plt.figure(figsize=(10, 6))
    
    # رسم الـ MRR كخط أساسي للمقارنة
    sns.lineplot(data=df_sweep, x="Chunk Size", y="MRR Score", marker='o', linewidth=2.5, label="MRR %")
    sns.lineplot(data=df_sweep, x="Chunk Size", y="Hit@1 (%)", marker='s', linestyle='--', label="Hit@1 %")
    sns.lineplot(data=df_sweep, x="Chunk Size", y="Hit@3 (%)", marker='s', label="Hit@3", linewidth=2)
    sns.lineplot(data=df_sweep, x="Chunk Size", y="Hit@5 (%)", marker='D', label="Hit@5", linewidth=2)


    plt.title("Effect of Chunk Size on All Retrieval Metrics", fontsize=14, fontweight='bold')
    plt.xlabel("Chunk Size", fontsize=12)
    plt.ylabel("Percentage (%)", fontsize=12)
    plt.legend(title="Metrics", loc='lower right')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    chart_path = "results/sweep_line_chart.png"
    plt.savefig(chart_path, dpi=300)
    print(f"✅ Sweep Complete! Comparison chart saved to: {chart_path}")

if __name__ == "__main__":
    start_sweep()