# story-to-dialgoue
Dialogue Generation from a given story.


Folder Structure:
```
story-to-dialgoue/
├── src/
│   ├── dataset/                # Dataset pre-process modules
│   └── utils/                  # Utility functions
├── notebooks/                  # Jupyter notebooks for experimentation
├── config/                     # Configuration files
├── requirements.txt
```

# Dataset
SODA: Million-scale Dialogue Distillation with Social Commonsense Contextualization, Kim et al., 2022 - [allenai/soda](https://huggingface.co/datasets/allenai/soda)

# Models
- Full fine-tuning - [abirmondalind/story2dialogue-SODA-BERT](https://huggingface.co/abirmondalind/story2dialogue-SODA-BERT)
- LoRA fine-tuning - [abirmondalind/story2dialogue-SODA-BERT-LoRA](https://huggingface.co/abirmondalind/story2dialogue-SODA-BERT-LoRA)

# WANDB Logs
- Full fine-tuning - [story2dialogue-soda-bert](https://wandb.ai/abirmondalind/story2dialogue-soda-bert)
- LoRA fine-tuning - [story2dialogue-soda-bert-lora](https://wandb.ai/abirmondalind/story2dialogue-soda-bert-lora)

# Inferencing
To inference the models please use the notebooks:
- [bart-exp.ipynb](https://github.com/abirmondal/story-to-dialgoue/blob/main/notebooks/bart-exp.ipynb)
- [bart-lora-exp.ipynb](https://github.com/abirmondal/story-to-dialgoue/blob/main/notebooks/bart-lora-exp.ipynb)
