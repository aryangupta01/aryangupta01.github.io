+++
title = "Text-to-SQL Model Fine-tuning with LoRA"
description = """
This project leverages `LoRA (Low-Rank Adaptation)` technique to fine-tune `Qwen3-0.6B` language model for natural language to SQL query translation. Built with `Transformers` and TRL frameworks, it implements `Parameter Efficient Fine-Tuning (PEFT)` using the `SQL-Create-Context` dataset with 20K samples. The system features `conversation-style training`, `schema-aware prompting`, and `completion-only optimization` for enhanced SQL generation accuracy. Achieves 400% improvement in exact match scores through `mixed precision training`, `gradient checkpointing`, and custom evaluation metrics including `BLEU scoring` and `SQL keyword accuracy` assessment.
"""
weight = 3

[extra]
remote_image = "/llm_finetunning.jpg"
link_to = "https://github.com/aryangupta01/ML-Projects/tree/main/Text%20to%20Sql%20Finetune%20LLM%20Model"
+++