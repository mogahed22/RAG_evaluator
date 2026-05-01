import pandas as pd
import os
import matplotlib
matplotlib.use('Agg') # لضمان التشغيل في خلفية Ubuntu
import matplotlib.pyplot as plt
import seaborn as sns
from langchain_chroma import Chroma
from langchain_ollama import OllamaLLM
from get_embedding_function import get_embedding_function
from similarity_pipeline import step_1_similarity_search

# الإعدادات الأساسية
DATA_PATH = "data/golden_test_set_squad.csv"
CHROMA_PATH = "chroma"
RESULTS_DIR = "results"

if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)

# دالة إعادة صياغة السؤال لجعله ديناميكي وواقعي
def paraphrase_query(original_query, llm):
    prompt = f"""You are a technical user. 
    Rewrite the following question in a natural, conversational way while keeping the same technical meaning.
    Output ONLY the rewritten question text.
    Original Question: {original_query}"""
    
    try:
        new_query = llm.invoke(prompt).strip()
        # تنظيف الرد من أي علامات تنصيص زائدة
        new_query = new_query.replace('"', '').replace("'", "")
        return new_query
    except Exception as e:
        print(f"Error paraphrasing: {e}")
        return original_query

def run_evaluation(chunk_size_label):
    """
    نسخة معدلة لاستقبال حجم الـ Chunk كبرامتر وترجيع النتائج لاستخدامها في الـ Sweep.
    """
    # تحميل الداتا والموديلات
    df = pd.read_csv(DATA_PATH)
    if len(df) > 10:
        print(f"🎲 Selecting 10 random questions from {len(df)} for faster evaluation...")
        df = df.sample(n=10, random_state=42).reset_index(drop=True)
    # ملاحظة: يتم إعادة تحميل الـ DB هنا لأن الـ populate_database قام بتغييرها فعلياً على القرص
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=get_embedding_function())
    model = OllamaLLM(model="mistral") 
    
    total_queries = len(df)
    hit_counts = {"Hit@1": 0, "Hit@3": 0, "Hit@5": 0}
    total_reciprocal_rank = 0

    print(f"\n[Evaluating] Chunk Size: {chunk_size_label} | Queries: {total_queries}")
    print("-" * 50)

    for i, row in df.iterrows():
        original_q = row['question']
        expected_src = str(row['expected_source']).strip()
        
        # 1. إعادة صياغة السؤال (Dynamic Query)
        dynamic_q = paraphrase_query(original_q, model)
        
        # 2. البحث (k=5 للحصول على صورة كاملة للترتيب)
        sources, _ = step_1_similarity_search(dynamic_q, db, k=5)
        
        # 3. حساب مقاييس Hit@k
        if any(expected_src in str(src) for src in sources[:1]): hit_counts["Hit@1"] += 1
        if any(expected_src in str(src) for src in sources[:3]): hit_counts["Hit@3"] += 1
        if any(expected_src in str(src) for src in sources[:5]): hit_counts["Hit@5"] += 1

        # 4. حساب مقياس MRR
        rank_score = 0
        for idx, src in enumerate(sources):
            if expected_src in str(src):
                rank_score = 1 / (idx + 1)
                break
        total_reciprocal_rank += rank_score

    # حساب المتوسطات النهائية
    mrr_actual = total_reciprocal_rank / total_queries
    
    # تجهيز قاموس النتائج لإرجاعه لملف الـ Sweep
    metrics = {
        "Chunk Size": chunk_size_label,
        "Hit@1 (%)": (hit_counts["Hit@1"] / total_queries) * 100,
        "Hit@3 (%)": (hit_counts["Hit@3"] / total_queries) * 100,
        "Hit@5 (%)": (hit_counts["Hit@5"] / total_queries) * 100,
        "MRR Score": mrr_actual * 100  # ضربناه في 100 لتوحيد الرسم البياني لاحقاً
    }
    
    print(f"✅ Finished Eval for Chunk {chunk_size_label}: MRR = {mrr_actual:.4f}")
    
    # (اختياري) يمكنك إبقاء كود الرسم هنا إذا أردت صورة منفصلة لكل Chunk 
    # لكن الأفضل تركه لملف sweep_analyzer ليرسم مقارنة مجمعة.
    
    return metrics
if __name__ == "__main__":
    run_evaluation()