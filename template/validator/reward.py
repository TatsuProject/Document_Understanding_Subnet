# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# TODO(developer): Set your name
# Copyright © 2023 <your name>

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
import numpy as np
from typing import List
import bittensor as bt
from template.protocol import ProfileSynapse
from fuzzywuzzy import fuzz


def hard_match_strings(string1: str, string2: str, minimum_match_percentage: float) -> float:
    # If lengths are different, it's an immediate mismatch
    if len(string1) != len(string2):
        return 0.0

    # Count character matches
    matches = sum(1 for c1, c2 in zip(string1, string2) if c1 == c2)

    # Calculate match percentage
    match_percentage = (matches / len(string1)) * 100 if string1 else 100.0

    # Apply minimum match threshold
    return match_percentage if match_percentage >= minimum_match_percentage else 0.0


def time_score_calculation(time_taken, Tn=2.0):
    """
    Calculate the time score based on the time taken by the miner.
    
    Parameters:
    - time_taken (float): Time taken by the miner (Tt).
    - Tn (float): Normal time, default is 2.2.
    
    Returns:
    - float: The calculated time score.
    """
    if time_taken >= 10 * Tn:
        return 0.0  # Score is zero if Tt >= 10 * Tn
    elif time_taken <= 0.01 * Tn:
        return 1.0  # Score is one if Tt <= 0.01 * Tn
    else:
        # Calculate the score for the range (0.01 * Tn < Tt < 10 * Tn)
        score = 1 - (time_taken - (0.01 * Tn)) / ((10 * Tn) - (0.01 * Tn))
        return score

def calculate_overlap(box1, box2):
    """
    Calculate the overlap area between two bounding boxes.
    
    Parameters:
    - box1 (list): Bounding box of detected text.
    - box2 (list): Bounding box of ground truth checkbox.

    Returns:
    - float: Overlap ratio between the two boxes.
    """
    # Extract coordinates
    x1_min, y1_min = min(box1[0], box1[6]), min(box1[1], box1[7])
    x1_max, y1_max = max(box1[2], box1[4]), max(box1[3], box1[5])
    x2_min, y2_min = min(box2[0], box2[6]), min(box2[1], box2[7])
    x2_max, y2_max = max(box2[2], box2[4]), max(box2[3], box2[5])
    
    # Calculate overlap area
    overlap_x1 = max(x1_min, x2_min)
    overlap_y1 = max(y1_min, y2_min)
    overlap_x2 = min(x1_max, x2_max)
    overlap_y2 = min(y1_max, y2_max)
    
    if overlap_x2 < overlap_x1 or overlap_y2 < overlap_y1:
        return 0.0  # No overlap

    overlap_area = (overlap_x2 - overlap_x1) * (overlap_y2 - overlap_y1)
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)

    if box1_area>=2*box1_area:
        return 0.0
    
    return overlap_area / max(box1_area, box2_area)

def accuracy_score_calculation(detected_checkboxes, ground_truths):
    """
    Calculate the accuracy score based on detected checkboxes and ground truths.

    Parameters:
    - detected_checkboxes (list): List of detected checkbox data.
    - ground_truths (list): List of ground truth checkbox data.

    Returns:
    - float: Overall accuracy score.
    """
    scores = []
    if abs(len(detected_checkboxes) - len(ground_truths))>1:
        return 0.0

    for detected in detected_checkboxes:
        detected_bbox = detected['checkbox_boundingBox']
        detected_text = detected['text']
        
        score_for_detected_pair = 0.0
        for ground_truth in ground_truths:
            ground_truth_bbox = ground_truth['checkbox_boundingBox']
            ground_truth_text = ground_truth['text']
            
            # Calculate CBS (Checkbox Score)
            overlap = calculate_overlap(detected_bbox, ground_truth_bbox)
            if overlap > 0.95:
                cbs = 1.0
            elif overlap > 0.7:
                cbs = 1.0 - (0.95 - overlap) / (0.95 - 0.7) * 0.5  # Decrease score gradually
            else:
                cbs = 0.0
            
            # print("---- checkbox score: ", cbs)
            # Calculate TS (Text Similarity)
            # ts = fuzz.token_sort_ratio(detected_text, ground_truth_text)
            ts = hard_match_strings(detected_text, ground_truth_text, 75.0)
            if ts >= 100:
                ts_score = 1.0
            elif ts >= 30:
                ts_score = (ts - 30) / 70  # Decrease score gradually
            else:
                ts_score = 0.0
            
            # Calculate score for this pair
            score = (cbs + ts_score) / 2
            if score>score_for_detected_pair:
                score_for_detected_pair = score
        scores.append(score_for_detected_pair)
    
    # Calculate overall accuracy score
    if scores:
        accuracy_score = sum(scores) / len(scores)
    else:
        accuracy_score = 0.0
    
    return accuracy_score

def final_score_calculation(time_score, accuracy_score):
    final_score = 0.3*time_score + 0.7*accuracy_score
    return final_score


def reward(ground_truth: list, response: ProfileSynapse, Tt: float) -> float:
    """
    Reward the miner response to the dummy request. This method returns a reward
    value for the miner, which is used to update the miner's score.

    Returns:
    - float: The reward value for the miner.
    """
    checkboxes_detected = response.miner_output

    bt.logging.info(f"*************** Detected Checkbox-Text:")
    bt.logging.info(checkboxes_detected)
    bt.logging.info("************** End")
    bt.logging.info(f"*************** Ground Truth:")
    bt.logging.info(ground_truth)
    bt.logging.info("************** End")
    tim_score = time_score_calculation(Tt)
    acc_score = accuracy_score_calculation(checkboxes_detected, ground_truth)
    # score = final_score_calculation(tim_score, acc_score)
    score = acc_score
    return score


def doc_class_reward(ground_truth: list, response: ProfileSynapse, Tt: float) -> float:
    """
    Reward the miner response to the dummy request. This method returns a reward
    value for the miner, which is used to update the miner's score.

    Returns:
    - float: The reward value for the miner.
    """
    doc_class_detected = str(response.miner_output[0])
    actual_class = ground_truth[0]

    bt.logging.info(f"*************** Detected Document Class:")
    bt.logging.info(doc_class_detected)
    bt.logging.info("************** End")
    bt.logging.info(f"*************** Ground Truth:")
    bt.logging.info(actual_class)
    bt.logging.info("************** End")
    # tim_score = time_score_calculation(Tt)
    acc_score = hard_match_strings(doc_class_detected, actual_class, 75.0)
    # score = final_score_calculation(tim_score, acc_score)
    score = acc_score/100
    return score


def get_rewards(
    self,
    ground_truth: list,
    responses: List[ProfileSynapse],
    Tt: float,
    redis_score: float
) -> np.ndarray:
    """
    Returns an array of rewards for the given query and responses.

    Args:
    - query (int): The query sent to the miner.
    - responses (List[float]): A list of responses from the miner.

    Returns:
    - np.ndarray: An array of rewards for the given query and responses.
    """
    # Get all the reward results by iteratively calling your reward() function.
    scores_array=np.zeros(len(responses))
    for idx, each_resp in enumerate(responses):
        if each_resp[0].task_type=="redis":
            scores_array[idx]=redis_score
        else:
            if each_resp[0].task_sub_type=="checkbox":
                scores_array[idx]=reward(ground_truth.get("checkboxes", []), each_resp[0], Tt)
            elif each_resp[0].task_sub_type=="doc-class":
                scores_array[idx]=doc_class_reward([ground_truth.get("document_class", "")], each_resp[0], Tt)

    return scores_array