# -*- coding: utf-8 -*-
"""
情绪小小侦探 evaluate.py
自动评测 submission.py
"""
# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "5"  # 只使用GPU 0
import json
import time
import torch
import re
from transformers import AutoTokenizer, AutoModelForCausalLM
from submission import (
    build_system_prompt,
    apply_chat_template_dialogue,
    generate,
    dialogue_cosine_similarity,
    dialogue_to_text,
)


MODEL_NAME = "weights/Qwen3-0.6B"
MAX_NEW_TOKENS = 256


# ------------------------------------------------------------
# 清洗输出（移除 <think> 和前后空格）
# ------------------------------------------------------------
def clean_output(text: str):
    import re
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    return text.strip()
def remove_answer_prefix(text: str) -> str:
    """移除文本中的 [Answer]: 前缀"""
    text = text.strip()
    # 匹配 [Answer]: 或 [Answer] : 等变体
    pattern = r"^\s*\[?\s*Answer\s*\]?\s*:?\s*"
    text = re.sub(pattern, "", text, flags=re.IGNORECASE)
    return text.strip()

# ------------------------------------------------------------
# 载入数据
# ------------------------------------------------------------
def load_test_data(path: str):
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                items.append(json.loads(line))
    return items


# ------------------------------------------------------------
# 主流程
# ------------------------------------------------------------
def main():
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        padding_side="left",
        truncation_side="left",
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        device_map="auto",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    system_prompt = build_system_prompt()

    test_items = load_test_data("data/emo_dialogues1.jsonl")

    print("==== 测试1：情感分类是否正确 ====")
    correct = 0
    start = time.time()
    for item in test_items:
        dialogue = item["dialogue"]
        gold = str(item["label"])
        dialogue_text = dialogue_to_text(dialogue)

        rendered_text = apply_chat_template_dialogue(tokenizer, system_prompt, dialogue_text)
        inputs = tokenizer(
            rendered_text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            )
        outputs = generate(model, tokenizer, inputs, MAX_NEW_TOKENS, do_sample=False)

        # 解码
        full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        ans = remove_answer_prefix(clean_output(full_text.split("assistant")[-1]).strip())


        print("Dialogue:", dialogue)
        print("Model:", ans, " | Gold:", gold)
      
        if gold in ans:
            correct += 1

    end = time.time()
    print("正确：", correct, "/", len(test_items))
    print("平均耗时：", (end - start) / len(test_items), "秒")

    # 第一阶段得分计算（20题，60分）
    stage1_score = correct / len(test_items) * 60
    print(f"第一阶段得分：{stage1_score:.1f}/60")

    original_dialogues = [{"dialogue":[{"role": "user", "content": "今天生日收到很多礼物，我特别喜欢那个小熊玩偶！"},{"role": "assistant", "content": "生日快乐！得到心仪的礼物肯定很兴奋吧～"}],"label":"高兴"},
            {"dialogue":[{"role": "user", "content": "我画的图画在班级展示上获得了第一名！"},{"role": "assistant", "content": "真了不起！你的绘画才华真出色！"}],"label":"高兴"},
            {"dialogue":[{"role": "user", "content": "今天体育课跑步比赛我拿了第一名！"},{"role": "assistant", "content": "哇！你的体育素质真好，继续保持！"}],"label":"高兴"},
            {"dialogue":[{"role": "user", "content": "爸爸同意明天带我去游乐场玩！"},{"role": "assistant", "content": "真为你开心，希望你在游乐场玩得高兴！"}],"label":"高兴"},
            {"dialogue":[{"role": "user", "content": "今天老师夸奖我作业完成得认真"},{"role": "assistant", "content": "获得老师的赞赏感觉一定很自豪！"}],"label":"高兴"},
            {"dialogue":[{"role": "user", "content": "我的好伙伴转学了，以后没有人和我玩耍了"},{"role": "assistant", "content": "和伙伴分离确实会让人伤心，但你们还能继续联系"}],"label":"难过"},
            {"dialogue":[{"role": "user", "content": "今天不小心把最珍爱的水杯打碎了"},{"role": "assistant", "content": "真遗憾，失去珍视的物品确实令人难过"}],"label":"难过"},
            {"dialogue":[{"role": "user", "content": "数学测验没考好，心里挺难受"},{"role": "assistant", "content": "一次测验不理想不要紧，以后加油就好"}],"label":"难过"},
            {"dialogue":[{"role": "user", "content": "今天和好伙伴闹矛盾了，现在很懊悔"},{"role": "assistant", "content": "和朋友闹矛盾确实会不开心，尝试和解看看？"}],"label":"难过"},
            {"dialogue":[{"role": "user", "content": "我的宠物小仓鼠昨天去世了"},{"role": "assistant", "content": "失去心爱的宠物确实很悲痛，它肯定很感激你的陪伴"}],"label":"难过"}]
    original_dialogue_texts =[]
    for i in original_dialogues:
        original_dialogue_texts.append(dialogue_to_text(i["dialogue"]))
    answers = [i for i in range(len(original_dialogue_texts))]
    
    print("\n==== 测试2：相似度分类 ====")
    test_items = load_test_data("data/emo_dialogues2.jsonl")
    ll_correct = 0
    num = 0
    for item in test_items:
        dialogue = item["dialogue"]
        gold = item["label"]
        dialogue_text = dialogue_to_text(dialogue)

        scores = []
        for lab in original_dialogue_texts:
            scores.append(dialogue_cosine_similarity(model, tokenizer, dialogue_text, lab))
        pred = scores.index(max(scores))

        print("Dialogue:", dialogue)
        print("LL scores:", scores, "Pred:", pred, "Gold:",answers[num])

        if pred == answers[num]:
            ll_correct += 1
        num += 1
        
    print("文本相似度准确：", ll_correct, "/", len(test_items))
    
    # 第二阶段得分计算（10题，40分）
    stage2_score = ll_correct / len(test_items) * 40
    print(f"第二阶段得分：{stage2_score:.1f}/40")
    
    # 总分计算
    total_score = stage1_score + stage2_score
    print("\n" + "="*50)
    print("最终评分结果：")
    print("="*50)
    print(f"第一阶段（生成测试）：{stage1_score}/60")
    print(f"第二阶段（相似度分类）：{stage2_score}/40")
    print(f"总分：{total_score}/100")
    print("="*50)


if __name__ == "__main__":
    main()