import sys
import pandas as pd
import pickle
import numpy as np
import numpy as np
import os


def softmax(x):
    """
    Compute the softmax of vector x.

    Parameters:
    x (numpy.ndarray): Input array or matrix.

    Returns:
    numpy.ndarray: Softmax of the input.
    """
    # Subtract the max value from each element to prevent overflow
    x_max = np.max(x, axis=-1, keepdims=True)
    e_x = np.exp(x - x_max)
    return e_x / np.sum(e_x, axis=-1, keepdims=True)


def softmax_to_logits(softmax_output):
    """
    Recover logits from softmax output.

    Parameters:
    softmax_output (numpy.ndarray): Probability distribution from softmax output, 2D array.

    Returns:
    numpy.ndarray: Recovered logits, 2D array.
    """
    if softmax_output.ndim != 2:
        raise ValueError("Input must be a 2D array.")

    # Take the logarithm to get log-probabilities
    log_probs = np.log(softmax_output)

    # Subtract the maximum log-probability from each row
    logits = log_probs - np.max(log_probs, axis=1, keepdims=True)

    return logits

if __name__ == '__main__':
    train_gemma_path = sys.argv[1]
    oof_qwen_path = sys.argv[2]
    oof_llama3_path = sys.argv[3]
    save_path = sys.argv[4]
    with open(train_gemma_path,'rb') as f:
        data = pickle.load(f)
    with open(oof_qwen_path,'rb') as f:
        data_qwen = pickle.load(f)
    with open(oof_llama3_path,'rb') as f:
        data_llama3 = pickle.load(f)
    data_llama3['winner_model_b_2'] = data_llama3['winner_model_b']
    data_llama3['winner_model_a_2'] = data_llama3['winner_model_a']
    data_llama3['winner_tie_2'] = data_llama3['winner_tie']
    data = data[['order_index','label','text']]
    print(data)
    data = data.merge(data_qwen,on=['order_index'])
    data = data.merge(data_llama3,on=['order_index'])
    print(data.isnull().sum())
    with open(save_path,'wb') as f:
        pickle.dump(data,f)
    print(data.shape)
