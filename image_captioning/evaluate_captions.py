from tqdm import tqdm, tqdm_notebook
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction
import numpy as np
from pycocotools.coco import COCO
import json

## SAMPLE USAGE
"""
true_annotations_file ='./data/annotations/captions_val2014.json'
pred_annotations_file = 'baseline_lstm_captions.json' -> this is just a dictionary

BLEU1, BLEU4 = evaluate_captions( true_annotations_file, pred_annotations_file )
"""

def evaluate_captions( true_captions_path, generated_captions_path ):
    """
    Takes json formatted true and predicted captions, and calculates BLEU1, BLEU4 scores
    :param true_captions_path: path to json file with true COCO captions, used with pycocotools.coco
    :param generated_captions_path: path to json file with predicted COCO captions, used with pycocotools.coco
    
    :return: BLEU1, BLEU2 score tuple
    """
    coco = COCO(true_captions_path)

    with open(generated_captions_path) as f:
        cocoRes = json.load(f)
    score1 = 0
    score4 = 0

    smoother = SmoothingFunction()

    for i in tqdm(cocoRes.keys()):
        candidate = cocoRes[i]
        reference = [entry['caption'] for entry in coco.imgToAnns[int(i)]]

        score1 += sentence_bleu(reference, candidate, weights=(1, 0, 0, 0), smoothing_function=smoother.method1)
        score4 += sentence_bleu(reference, candidate, weights=(0, 0, 0, 1), smoothing_function=smoother.method1)

    bleu1 = 100*score1/len(cocoRes)
    bleu4 = 100*score4/len(cocoRes)
    
    return bleu1, bleu4

