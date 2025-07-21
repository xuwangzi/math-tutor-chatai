# math-tutor-chatai

数学应用题指导ChatAI —— 强化学习

## 项目背景

构建数学应用题智能辅导LLM，通过强化学习与知识蒸馏技术，赋予小参数模型逻辑推理和个性化教学能力，以低成本实现高质量的数学应用题智能辅导。通过实践，掌握LLM微调框架、强化学习算法与模型蒸馏。

## 我的工作

- 设计并优化不同参数量模型训练流程，对比分析其解题能力差异；

- 基于trl库对Qwen-2.5-0.5B进行全量微调，使用GRPO强化学习提升模型的解题能力与CoT思维链推理能力，基于Deepseek-R1蒸馏数据完成SFT知识注入，完成个性化辅导角色模拟。

- 采用unsloth框架结合QLoRA技术对Qwen-2.5-3B进行高效微调，使用GRPO强化学习显著提升模型的解题能力，在GSM8K数据集上准确率从79%提升至85%。

## 项目难点与解决方案

- Scaling Laws约束：针对小模型效果差问题，从0.5B指令微调模型转向3B模型尝试；

- 资源约束：面对GPU内存不足，采用unsloth框架与QLoRA微调技术，将3B模型训练显存占用降至20GB。

## 代码结构

[my_r1_distill.ipynb](https://github.com/xuwangzi/math-tutor-chatai/blob/main/my_r1_distill.ipynb) 数据蒸馏脚本

[my_sft_lora.ipynb](https://github.com/xuwangzi/math-tutor-chatai/blob/main/my_sft_lora.ipynb) SFT 训练脚本（后者使用了 LoRA 高效微调）

[my_grpo.ipynb](https://github.com/xuwangzi/math-tutor-chatai/blob/main/my_grpo.ipynb) / [my_grpo_unsloth.ipynb](https://github.com/xuwangzi/math-tutor-chatai/blob/main/my_grpo_unsloth.ipynb)  GRPO 训练脚本（后者使用了 Unsloth 框架和 QLoRA ）

