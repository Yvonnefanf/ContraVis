
import numpy as np

def semantic_diff(r,t,ref_data_provider,tar_data_provider,REF_EPOCH, TAR_EPOCH, k_neibour=15):
    pred_r = ref_data_provider.get_pred(REF_EPOCH,[r])[0]
    pred_t = tar_data_provider.get_pred(TAR_EPOCH,[t])[0]

def difference(vector_a,vector_b):
    # calculate the variance
    difference = vector_a - vector_b
    variance = np.var(difference)
    # print("Variance:", variance)

    # 归一化
    normalized = (difference - np.min(difference)) / (np.max(difference) - np.min(difference))
    print("Normalized:", normalized)
    return normalized

    
