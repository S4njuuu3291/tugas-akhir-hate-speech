import scipy
import numpy as np
import pickle
import itertools
from EXPLAINABLE_CODE.svm_lr_shap import shap_kernel_instance as predict_asli

def predict_proba(x_vec, clf, target_index):
    return clf.predict_proba(x_vec)[:,target_index][0]

def predict_on_subset(subset, x_instance, clf, target_index):
    x_masked = x_instance.copy().toarray().flatten()
    keep = set(subset)
    all_features_indices = list(range(x_instance.shape[1]))

    for j in all_features_indices:
        if j not in keep:
            x_masked[j]=0.0
    return predict_proba(x_masked.reshape(1,-1),clf,target_index)

def kernel_weight(M,k):
    if k==0 or k== M:
        return 1e6
    else:
        return (M-1)/(scipy.special.comb(M,k)*k*(M-k))

def shap_kernel_instance(sentence,vectorizer,clf,features_list,labels,target_label,num_samples = 400):
    target_index = labels.index(target_label)
    x_instance = vectorizer.transform([sentence])
    print(x_instance)
    idx_features = x_instance.nonzero()[1]
    print(idx_features)
    active_features = [features_list[i] for i in idx_features]
    print(active_features)

    M = len(active_features)
    total_subsets = 2**M

    if num_samples >= total_subsets:
        subsets = [list(s) for k in range(M+1) for s in itertools.combinations(idx_features,k)]
    else:
        subsets = []
        subsets.append([])
        subsets.append(list(idx_features))

        while len(subsets) < num_samples:
            k = np.random.randint(0,M+1)
            s = np.random.choice(idx_features,size=k,replace=False)
            subset = list(s)
            if subset not in subsets:
                subsets.append(subset)

    X_mask = []
    y = []
    weight = []
    
    for subset in subsets:
        print(f"{subset} ---->",predict_on_subset(subset,x_instance,clf,target_index))
        pred = predict_on_subset(subset,x_instance,clf,target_index)
        y.append(pred)

        mask = np.zeros(M)
        for s_idx in subset:
            position = idx_features.tolist().index(s_idx)
            mask[position] = 1
        w = kernel_weight(M,len(subset))
        weight.append(w)
        
        # X_mask.append(mask)
        X_mask.append(np.concatenate(([1], mask)))


    X_mask = np.array(X_mask)
    y = np.array(y)
    W = np.diag(weight)

    print(X_mask)
    print(y)
    print(W)

    beta = np.linalg.pinv(X_mask.T @ W @ X_mask) @ (X_mask.T @ W @ y)

    base_value = beta[0]
    shap_values = beta[1:]

    shap_results = list(zip(active_features, shap_values))
    
    sorted_shap_results = sorted(
        shap_results, 
        key=lambda item: abs(item[1]), 
        reverse=True
    )
    
    print("-" * 50)
    print(f"Kalimat: {sentence}")
    print(f"Penjelasan untuk: {target_label}")
    print("-" * 50)

    for f, val in sorted_shap_results:
        print(f"{f}: {val:.4f}")

    original_proba = predict_proba(x_instance, clf, target_index)
    
    print("\nBaseline (E[f(x)]):", base_value)
    print("Baseline + ΣKernelSHAP:", base_value + shap_values.sum())
    print("Prediksi Asli f(x):", original_proba)


with open('MODELS\\baseline\\model_artifact_2.pkl', 'rb') as f:
    model_artifact = pickle.load(f)

# predict_on_subset()
vectorizer = model_artifact['vectorizer']
model = model_artifact['ovr_svm']
features_list = vectorizer.get_feature_names_out()
sentence = "aku benci bapak kamu"

x_instance = vectorizer.transform([sentence])
labels = ["HS_Individual","HS_Group","HS_Religion","HS_Race","HS_Physique","HS_Bla"]
target = "HS_Individual"

print(predict_proba(x_instance, model, target_index=0))

print(predict_on_subset([0, 2], x_instance, model, target_index=0))

shap_kernel_instance(sentence,vectorizer,model,features_list,labels,target)


print("\n\n==================================================================================")
print("==================================== INDOBERT ====================================")
print("==================================================================================\n\n")

import numpy as np
import pandas as pd
import tensorflow as tf
