import pandas as pd
results_df = pd.read_csv("gen_v1/results_bench.csv")

TP = FP = TN = FN = 0

for _, row in results_df.iterrows():
    y_true = row['text_type']
    y_pred = row['answer']
    print(type(y_true), type(y_pred))

    if y_true == True and y_pred == True:
        TP += 1
    elif y_true == False and y_pred == False:
        TN += 1
    elif y_true == False and y_pred == True:
        FP += 1
    elif y_true == True and y_pred == False:
        FN += 1

print("Confusion Matrix:")
print(f"TP: {TP}  FP: {FP}")
print(f"FN: {FN}  TN: {TN}")

accuracy = (TP + TN) / len(results_df) * 100
print(f"Accuracy: {accuracy:.2f}%")