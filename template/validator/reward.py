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
    if len(box1)==8 and len(box2)==8:
        x1_min, y1_min = min(box1[0], box1[6]), min(box1[1], box1[7])
        x1_max, y1_max = max(box1[2], box1[4]), max(box1[3], box1[5])
        x2_min, y2_min = min(box2[0], box2[6]), min(box2[1], box2[7])
        x2_max, y2_max = max(box2[2], box2[4]), max(box2[3], box2[5])
    elif len(box1)==4 and len(box2)==4:
        x1_min, y1_min, x1_max, y1_max = box1
        x2_min, y2_min, x2_max, y2_max = box2
    
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

    if box1_area>=2*box2_area:
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
    if isinstance(response.miner_output[0], str):
        doc_class_detected = str(response.miner_output[0])
    elif isinstance(response.miner_output[0], dict) and "document_class" in response.miner_output[0]:
        doc_class_detected = str(response.miner_output[0]["document_class"])

    if isinstance(ground_truth[0], str):
        actual_class = ground_truth[0]
    elif isinstance(ground_truth[0], dict) and "document_class" in ground_truth[0]:
        actual_class = ground_truth[0]["document_class"]

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


def are_keys_same(dict1, dict2):
    return set(dict1.keys()) == set(dict2.keys())

def doc_parse_basic_unit_reward(detected_dict, actual_dict):
    """ Computes reward based on string match and bounding box overlap. """
    try:
        string_score = hard_match_strings(actual_dict.get("text", ""), detected_dict.get("text", ""), 75.0) / 100
        bbox_overlapping = (
            calculate_overlap(detected_dict["bounding_box"], actual_dict["bounding_box"])
            if "bounding_box" in actual_dict and "bounding_box" in detected_dict
            and len(actual_dict["bounding_box"]) in [4, 8] and len(detected_dict["bounding_box"]) in [4, 8]
            else 0.0
        )
        return (string_score + bbox_overlapping) / 2
    except Exception as e:
        import traceback
        bt.logging.error(f"{traceback.format_exc()}")
        bt.logging.error(f"Error in basic unit reward calculation: {e}")
        return 0.0

def compute_section_score(detected_section, actual_section):
    """ Recursively computes the score for different sections. """
    try:
        # Case 1: Both are dicts with "text"
        if isinstance(actual_section, dict) and "text" in actual_section and isinstance(detected_section, dict) and "text" in detected_section:
            return doc_parse_basic_unit_reward(detected_section, actual_section)

        # Case 2: Both are dicts without "text" (nested dictionaries)
        elif isinstance(actual_section, dict) and isinstance(detected_section, dict):
            if not are_keys_same(actual_section, detected_section):
                return 0.0
            scores = [compute_section_score(detected_section[sub_key], actual_section[sub_key]) for sub_key in actual_section]
            return sum(scores) / len(scores) if scores else 0.0

        # Case 3: Both are lists
        elif isinstance(actual_section, list) and isinstance(detected_section, list):
            if len(actual_section) != len(detected_section):
                return 0.0
            scores = []
            for each_detected_component in detected_section:
                highest_score = max(
                    (compute_section_score(each_detected_component, each_actual_component) for each_actual_component in actual_section),
                    default=0.0
                )
                scores.append(highest_score)
            return sum(scores) / len(scores) if scores else 0.0

    except Exception as e:
        import traceback
        bt.logging.error(f"{traceback.format_exc()}")
        bt.logging.error(f"Error in computing section score: {e}")
    return 0.0

def doc_parse_reward(ground_truth: list, response: ProfileSynapse, Tt: float) -> float:
    """
    Reward function for evaluating miner response.

    Parameters:
    - ground_truth: List containing the expected parsed response.
    - response: ProfileSynapse object containing the miner's response.
    - Tt: Threshold value (not used in function, but can be incorporated).

    Returns:
    - float: Reward score.
    """
    try:
        doc_parsing_detected = response.miner_output[0].get("NER", {})
        actual_parsing = ground_truth[0].get("NER", {})

        bt.logging.info(f"*************** Detected Document Parsing:\n{doc_parsing_detected}\n************** End")
        bt.logging.info(f"*************** Ground Truth:\n{actual_parsing}\n************** End")

        if not are_keys_same(doc_parsing_detected, actual_parsing):
            return 0.0

        reward_dict = {
            key: compute_section_score(doc_parsing_detected[key], actual_parsing[key])
            for key in actual_parsing
        }

        score = sum(reward_dict.values()) / len(reward_dict) if reward_dict else 0.0
        return score

    except Exception as e:
        import traceback
        bt.logging.error(f"{traceback.format_exc()}")
        bt.logging.error(f"Error in doc_parse_reward function: {e}")
        return 0.0


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
            elif each_resp[0].task_sub_type=="doc-parse":
                classification_score = doc_class_reward([ground_truth.get("document_class", "")], each_resp[0], Tt)
                parsing_score = doc_parse_reward([ground_truth], each_resp[0], Tt)
                weighted_avg_score = 0.3*classification_score + 0.7*parsing_score
                scores_array[idx]=weighted_avg_score

    return scores_array