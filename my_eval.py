from modelscope import AutoModelForCausalLM, AutoTokenizer
import pandas as pd

cot_prompt = """
Respond in the following format:
<reasoning>
...
</reasoning>
<answer>
...
</answer>
"""

cosplay_prompt='''
# Role
I am an elementary school math tutor. Currently, I will give a clear and easy-to-understand explanation for a specific elementary school math word problem.
Note that the response should be in the form of spoken textual expression, and avoid including the formats such as pictures, hyperlinks, etc.

# Precautions
1. Fully consider the cognitive level and knowledge reserve of primary school students.
2. If the problem is complex, break it down into simple steps. Avoid using professional terms and explain it in plain language.
3. Make more use of real-life examples and intuitive teaching aids to assist in teaching.

# Teaching Style
Adopt a step-by-step teaching method, combined with interactive Q&A sessions and classroom exercises; encourage students through positive feedback to strengthen their understanding of mathematical concepts.

# Answer Format
<reasoning>
Break down, analyze, and reflect on the problem step by step to sort out the thinking process of the solution.
</reasoning>
<answer>
Start explaining the problem from the first-person perspective of a math tutor.
</answer>
'''

# cosplay_prompt='''
# # Role
# You are required to answer the question.

# # Answer Format
# <reasoning>
# Break down, analyze, and reflect on the problem step by step to sort out the thinking process of the solution.
# </reasoning>
# <answer>
# right answer to the question.
# </answer>
# '''

def eval_qwen(model_name,prompt,question):
    model_dir = "models/" + model_name
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        torch_dtype="auto",
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_dir)

    messages = [
        {'role':'system','content':prompt}, 
        {'role':'user','content': question}, 
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=1024
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response

if __name__ == "__main__":
    
    # 定义要评估的问题列表
    questions = [
        "Janet’s ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?",
        "A robe takes 2 bolts of blue fiber and half that much white fiber. How many bolts in total does it take?",
        "Kylar went to the store to buy glasses for his new apartment. One glass costs $5, but every second glass costs only 60% of the price. Kylar wants to buy 16 glasses. How much does he need to pay for them?",
        "A new program had 60 downloads in the first month. The number of downloads in the second month was three times as many as the downloads in the first month, but then reduced by 30% in the third month. How many downloads did the program have total over the three months?",
        "Mishka bought 3 pairs of shorts, 3 pairs of pants, and 3 pairs of shoes. One pair of shorts costs $16.50. One pair of pants costs $22.50 and one pair of shoes costs $42. How many dollars did Mishka spend on all the clothing items?"
    ]
    
    # 定义要评估的模型列表
    models_to_evaluate = [
        "Qwen2.5-0.5B-Instruct",	
        "Qwen2.5-0.5B-Instruct-GRPO",		
        "Qwen2.5-0.5B-Instruct-GRPO-SFT",		
        "Qwen2.5-3B-Instruct",		
        "Qwen2.5-3B-Instruct-SFT",		
        "Qwen2.5-3B-Instruct-GRPO_lora_bz_1",		
        "Qwen2.5-3B-Instruct-GRPO_lora_bz_32",	
    ]
    
    # 创建一个以模型名为索引，问题为列的DataFrame
    results_df = pd.DataFrame(index=models_to_evaluate, columns=questions)
    
    # 评估每个模型对每个问题的回答
    for model_name in models_to_evaluate:
        for question in questions:
            print(f"正在评估模型: {model_name}，问题: {question[:50]}...")
            completion=eval_qwen(model_name,cot_prompt,question)
            results_df.loc[model_name, question] = completion
    
    # 保存结果到CSV
    csv_path = "model_evaluation.csv"
    results_df.to_csv(csv_path)
    
    print(f"评估完成，结果已保存至: {csv_path}")
    print(f"共评估 {len(questions)} 个问题，{len(models_to_evaluate)} 个模型。")