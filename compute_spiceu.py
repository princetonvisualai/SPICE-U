import json
import pickle
import argparse
import numpy as np
from scipy import stats


def get_concept_word(concpet_tuple):
    """Convert a concept tuple to string"""
    return' '.join(concpet_tuple)


def compute_uniqueness_score(test_tuples, ref_tuples, uniqueness_dict):
    test_uniqueness = 0.
    meta_uniqueness_list = []
    
    for test_tuple in test_tuples:
        test_tuple_word = get_concept_word(test_tuple['tuple'])
        tuple_uniqueness = uniqueness_dict.get(test_tuple_word, 1.)
        test_uniqueness += tuple_uniqueness
        meta_uniqueness_list.append((test_tuple_word, tuple_uniqueness))
        
    for ref_tuple in ref_tuples:
        ref_tuple_word = get_concept_word(ref_tuple['tuple'])
        tuple_uniqueness = uniqueness_dict.get(ref_tuple_word, 1.)
        meta_uniqueness_list.append((ref_tuple_word, tuple_uniqueness))
        
    meta_uniqueness_list = list(set(meta_uniqueness_list))
    meta_uniqueness_list.sort(key=lambda x:x[1], reverse=True)
    
    max_uniqueness_score = 0.
    min_uniqueness_score = 0.
    for i in range(len(test_tuples)):
        max_uniqueness_score += meta_uniqueness_list[i][1]
        min_uniqueness_score += meta_uniqueness_list[-i-1][1]
        
    if max_uniqueness_score == min_uniqueness_score:
        uniqueness_score = 1.
    else:
        uniqueness_score = (test_uniqueness - min_uniqueness_score) \
                            / (max_uniqueness_score - min_uniqueness_score)
    return uniqueness_score


def compute_spiceu_score_on_tuplues_pair(test_tuples, ref_tuples, uniqueness_dict):
    tp = 0
    for test_tuple in test_tuples:
        if test_tuple['truth_value']:
            tp += 1
                
    fp = len(test_tuples) - tp
    fn = len(ref_tuples) - tp
    
    if (tp+fp) > 0:
        pr = tp*1. / (tp + fp)
    else:
        pr = 1.
    
    if (tp+fn) > 0:
        re = tp*1. / (tp + fn)
    else:
        re = 0.
    
    uq = compute_uniqueness_score(test_tuples, ref_tuples, uniqueness_dict)
    
    if pr*re*uq > 0:
        spiceu = stats.hmean([stats.hmean([pr, re]), uq])
    else:
        spiceu = 0.
    
    return spiceu


def compute_spcieu_score(parsed_prediction_list, uniqueness_dict):
    spiceu_score_list = []
    for parsed_prediction in parsed_prediction_list:
        spiceu_score_list.append(
            compute_spiceu_score_on_tuplues_pair(
                                                 parsed_prediction['test_tuples'],
                                                 parsed_prediction['ref_tuples'],
                                                 uniqueness_dict
                                                )
            )
        
    return np.mean(spiceu_score_list)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--parsed_input', type=str, required=True)
    parser.add_argument('--uniqueness_dict', type=str, required=True)
    args = parser.parse_args()

    with open(args.parsed_input, 'r') as f:
        parsed_input = json.load(f)

    with open(args.uniqueness_dict, 'rb') as f:
        uniqueness_dict = pickle.load(f)

    spiceu_score = compute_spcieu_score(parsed_input, uniqueness_dict)
    print(f'The SPICE-U socre is {spiceu_score:.5f}')

