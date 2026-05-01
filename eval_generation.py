import pandas as pd
from langchain_ollama import OllamaLLM

def test_generation(context, question, expected_answer, model):
    #  
    prompt = f"Context: {context}\nQuestion: {question}\nAnswer:"
    generated_answer = model.invoke(prompt)

    #  
    judge_prompt = f"""
    Question: {question}
    Expected Answer: {expected_answer}
    Actual Answer: {generated_answer}
    
    Does the following answer correctly address the question? Answer only yes or no.
    """
    
    # 
    judge_response = model.invoke(judge_prompt).strip().lower()
    return "yes" in judge_response

def run_gen_evaluation():
    df = pd.read_csv("data/golden_test_set_squad.csv")
    model = OllamaLLM(model="mistral")
    
    pass_count = 0
    total = len(df)

    print("Starting Generation Evaluation (this might take a while)...")
    for _, row in df.iterrows():
        # بنستخدم الـ expected_answer كـ context لضمان اختبار التوليد فقط
        if test_generation(row['expected_answer'], row['question'], row['expected_answer'], model):
            pass_count += 1

    pass_rate = (pass_count / total) * 100
    print(f"Generation Pass Rate: {pass_rate}%")
    
    # حفظ النتيجة
    with open("results/generation_status.txt", "w") as f:
        f.write(f"Pass Rate: {pass_rate}%")

if __name__ == "__main__":
    run_gen_evaluation()