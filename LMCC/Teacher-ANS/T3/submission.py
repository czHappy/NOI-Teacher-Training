# -*- coding: utf-8 -*-
"""
情绪小小侦探 submission.py
你需要实现 5 个核心函数供 evaluate.py 调用：
1. build_system_prompt()
2. apply_chat_template_dialogue()
3. dialogue_to_text()
4. generate()
5. dialogue_cosine_similarity()
"""

from typing import List, Dict
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


# ============================================================
# 1. system prompt 规范
# ============================================================
def build_system_prompt() -> str:
    """
    考生实现：构建系统提示词
    - 定义情绪分类任务的规则和输出格式
    - 明确情绪类别为高兴和难过两类
    - 指定输出格式为 [Answer]: 词 的形式
    - 确保模型只输出指定格式，不产生额外解释内容
    """
    # ============ 考生在此处实现代码 ============
    return (
        "你是一个简洁的情绪分类助手。\n"
        "【任务】给定一段儿童与模型的多轮中文对话，判断整体情绪只属于下面两类之一。\n"
        "【类别】高兴 / 难过\n"
        "【输出格式】仅输出一行：[Answer]: 高兴  或  [Answer]: 难过\n"
        "【限制】不得输出除上述格式外的任何文字、标点、解释、推理、标签、概率或空格。不得输出英文。\n"
        "【提示】 如果遇见不舒服、生病、委屈等词，肯定是难过。 \n"
        "现在请输出结果。"
    )
    # ===========================================


# ============================================================
# 2. 对话模板
# ============================================================
def apply_chat_template_dialogue(
    tokenizer: AutoTokenizer,
    system_prompt: str,
    dialogue_text: str,
) -> str:
    """
    考生实现：构建模型输入模板
    - 将系统提示词和对话文本组合成完整的输入格式
    - 使用tokenizer的apply_chat_template方法格式化对话
    - 返回格式化后的对话
    """
    # ============ 考生在此处实现代码 ============
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": dialogue_text},
    ]
    rendered = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    return rendered
    # ===========================================


def dialogue_to_text(dialogue: List[Dict[str, str]]) -> str:
    """
    考生实现：对话格式转换
    - 将结构化的对话列表转换为自然语言文本格式
    - 为不同角色（user/assistant）添加对应的前缀标识
    - 用换行符分隔各轮对话，形成可读的连续文本
    - 便于后续的情绪分析和相似度计算
        示例：
        输入: [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]
        输出: "小学生：...\n模型：..."
    """
    # ============ 考生在此处实现代码 ============
    lines: List[str] = []
    for turn in dialogue:
        # 结构形式1：{"role": "user", "content": "..."}
        if "role" in turn and "content" in turn:
            if turn["role"] == "user":
                lines.append(f"小学生：{turn['content']}")
            elif turn["role"] == "assistant":
                lines.append(f"模型：{turn['content']}")
        # 结构形式2：{"user": "...", "assistant": "..."}
        else:
            if "user" in turn:
                lines.append(f"小学生：{turn['user']}")
            if "assistant" in turn:
                lines.append(f"模型：{turn['assistant']}")
    return "\n".join(lines)
    
    # ===========================================


# ============================================================
# 3. 推理：生成回复
# ============================================================
def generate(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    model_inputs: Dict[str, torch.Tensor],
    max_new_tokens: int = 128,
    do_sample: bool = False,
) -> torch.Tensor:
    """
    考生实现：单条推理生成
    - 将tokenized的输入数据移动到模型所在设备
    - 调用模型的generate方法进行文本生成
    - 配置生成参数（最大生成长度、采样策略、终止标记等）
    - 返回包含输入和输出的完整token序列
    """
    # ============ 考生在此处实现代码 ============
    for k, v in model_inputs.items():
        model_inputs[k] = v.to(model.device)
    outputs = model.generate(
        input_ids=model_inputs["input_ids"],
        attention_mask=model_inputs.get("attention_mask"),
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
    return outputs
    
    # ===========================================


def dialogue_cosine_similarity(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    text1: str,
    text2: str,
) -> float:
    """
    考生实现：基于余弦相似度的对话情感分析
    - 分别对两个对话文本进行编码处理
    - 提取模型最后一层隐藏状态的最后一个token向量
    - 对提取的向量进行L2归一化处理
    - 计算两个归一化向量的余弦相似度作为情感相似度分数
    - 返回[-1,1]范围内的相似度值，用于情绪分类任务
    """
    tokenizer.padding_side = "left"
    
    #=============step1: 文本处理+模型前向传播=====================
    encoded_text1 = tokenizer(
        text1, return_tensors="pt", padding=True, truncation=True
    ).to(model.device)
    
    encoded_text2 = tokenizer(
        text2, return_tensors="pt", padding=True, truncation=True
    ).to(model.device)
    
    # TODO: 在这里实现,替换掉下面内容对应的None
    with torch.no_grad():
        outputs_text1 = model(**encoded_text1, output_hidden_states=True, use_cache=False)
        outputs_text2 = model(**encoded_text2, output_hidden_states=True, use_cache=False)
        
    #=============step2: 提取最后一个token的隐藏状态=====================
    # TODO: 在这里实现,替换掉下面内容对应的None
    # 获取最后一个隐藏层
    hidden_text1 = outputs_text1.hidden_states[-1]  # (1, seq_len1, hidden_dim)
    hidden_text2 = outputs_text2.hidden_states[-1]  # (1, seq_len2, hidden_dim)

    # 分别取各自序列的最后一个token
    last_token_text1 = hidden_text1[:, -1, :]  # (1, hidden_dim)
    last_token_text2 = hidden_text2[:, -1, :]  # (1, hidden_dim)

    # L2 归一化
    norm_text1 = torch.nn.functional.normalize(last_token_text1, p=2, dim=-1)
    norm_text2 = torch.nn.functional.normalize(last_token_text2, p=2, dim=-1)

    # 计算余弦相似度
    similarity = torch.sum(norm_text1 * norm_text2, dim=-1).item()
    # 考生实现结束
    
    return float(similarity)
