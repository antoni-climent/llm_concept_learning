# Project development notes

## Next steps:
- Augment diversity in true bench -> DONE
- Improve the rhinolume definition so that it has sth unusual -> DONE
- Split train/test data + do finetuning on training -> DONE
- Generate dataset for the definition -> DONE

23 Diciembre 2025
- Generate data with qwen -> DONE
- Test with gemma -> DONE
- Automatizar proceso + que se guarden los logs -> DONE
- Test on test set -> DONE
- Debug prompt -> DONE
- Chain of thought
- True/False/Unknown. If you have no info about the topic, say unknown.
- Generate dataset with llama nano nvidia
- Generate 100 topics to train on and generate questions about them using gemini

## Results 
USING DATASET gen_v0 (done with Gemma 3-4B IT model)

TRAINING SET RESULTS:
Model: google/gemma-3-4b-it + LoRA finetuned v2
Confusion Matrix:
TP: 87  FP: 32
FN: 9  TN: 64
Accuracy: 78.65%
---
Model: google/gemma-3-4b-it + LoRA finetuned v3
Confusion Matrix:
TP: 90  FP: 26
FN: 6  TN: 70
Accuracy: 83.33%
---

TEST SET RESULTS:
Model: google/gemma-3-4b-it base model
Confusion Matrix:
TP: 20  FP: 1
FN: 1  TN: 20
Accuracy: 95.24%

Model: google/gemma-3-4b-it + LoRA finetuned v3
Confusion Matrix:
TP: 21  FP: 0
FN: 0  TN: 21
Accuracy: 100.00%

=============================================

USING DATASET gen_v1 (done with Qwen 2.5B instruct model)

TRAINING SET RESULTS:
Confusion Matrix google/gemma-3-4b-it base model:
TP: 87  FP: 26
FN: 9  TN: 70
Accuracy: 81.77%
---
Model: Qwen/Qwen2.5-3B-Instruct
Confusion Matrix:
TP: 18  FP: 4
FN: 78  TN: 92
Accuracy: 57.29%
--- 
Model: Qwen/Qwen2.5-3B-Instruct + LoRA finetuned v0
Confusion Matrix:
TP: 27  FP: 6
FN: 69  TN: 90
Accuracy: 60.94%


TEST SET RESULTS
Model: Qwen/Qwen2.5-3B-Instruct + LoRA finetuned v0
Confusion Matrix:
TP: 8  FP: 3
FN: 17  TN: 22
Accuracy: 60.00%

Model: Qwen/Qwen2.5-3B-Instruct base model
Confusion Matrix:
TP: 7  FP: 3
FN: 18  TN: 22
Accuracy: 58.00%

Model: google/gemma-3-4b-it + LoRA finetuned v3
Confusion Matrix:
TP: 25  FP: 11
FN: 0  TN: 14
Accuracy: 78.00%

Model: google/gemma-3-4b-it base model
Confusion Matrix:
TP: 22  FP: 7
FN: 3  TN: 18
Accuracy: 80.00%