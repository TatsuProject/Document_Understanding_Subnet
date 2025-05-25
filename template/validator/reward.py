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
from shapely.geometry import Polygon

def hard_match_strings(string1: str, string2: str, minimum_match_percentage: float) -> float:
    try:
        # If lengths are different, it's an immediate mismatch
        if len(string1) != len(string2):
            return 0.0

        # Count character matches
        matches = sum(1 for c1, c2 in zip(string1, string2) if c1 == c2)

        # Calculate match percentage
        match_percentage = (matches / len(string1)) * 100 if string1 else 100.0

        # Apply minimum match threshold
        return match_percentage if match_percentage >= minimum_match_percentage else 0.0
    except Exception as e:
        bt.logging.error(f"[hard_match_strings] Error {e}")
        return 0.0
    
    
def hard_match_strings_2(string1: str, string2: str, minimum_match_percentage: float) -> float:
    """ Slightly softer logic
    """
    try:
        # If length difference is more than one, it's an immediate mismatch
        if abs(len(string1) - len(string2)) > 1:
            return 0.0

        # Determine shorter and longer strings
        shorter, longer = (string1, string2) if len(string1) <= len(string2) else (string2, string1)
        
        # If strings are the same length, do a direct comparison
        if len(shorter) == len(longer):
            matches = sum(1 for c1, c2 in zip(shorter, longer) if c1 == c2)
        else:
            # Look for the best alignment of the shorter string within the longer string
            # (we'll try both possible positions since the difference is only 1)
            max_matches = 0
            for start_pos in range(2):  # Try positions 0 and 1
                if start_pos + len(shorter) <= len(longer):
                    current_matches = sum(1 for c1, c2 in zip(shorter, longer[start_pos:]) if c1 == c2)
                    max_matches = max(max_matches, current_matches)
            
            matches = max_matches

        # Calculate match percentage based on the longest string
        longest_length = len(longer)
        match_percentage = (matches / longest_length) * 100 if longest_length else 100.0

        # Apply minimum match threshold
        return match_percentage if match_percentage >= minimum_match_percentage else 0.0
    except Exception as e:
        bt.logging.error(f"[hard_match_strings_2] Error {e}")
        return 0.0


def time_score_calculation(time_taken, Tn=2.0):
    """
    Calculate the time score based on the time taken by the miner.
    
    Parameters:
    - time_taken (float): Time taken by the miner (Tt).
    - Tn (float): Normal time, default is 2.2.
    
    Returns:
    - float: The calculated time score.
    """
    try:
        if time_taken >= 10 * Tn:
            return 0.0  # Score is zero if Tt >= 10 * Tn
        elif time_taken <= 0.01 * Tn:
            return 1.0  # Score is one if Tt <= 0.01 * Tn
        else:
            # Calculate the score for the range (0.01 * Tn < Tt < 10 * Tn)
            score = 1 - (time_taken - (0.01 * Tn)) / ((10 * Tn) - (0.01 * Tn))
            return score
    except Exception as e:
        bt.logging.error(f"[time_score_calculation] Error {e}")
        return 0.0

def calculate_overlap(box1, box2):
    """
    Calculate the overlap ratio between two bounding boxes using polygon intersection.
    Maintains strictness by:
    1. Returning 0.0 if detected box is 2x+ larger than ground truth box
    2. Using max(area1, area2) as denominator instead of union area
    3. Ensuring the result is within [0.0, 1.0] range
    
    Parameters:
    - box1 (list): Bounding box coordinates of detected text.
    - box2 (list): Bounding box coordinates of ground truth checkbox.

    Returns:
    - float: Overlap ratio between the two boxes, always within [0.0, 1.0].
    """
    try:
        # Convert to polygon points
        if len(box1) == 4 and len(box2) == 4:
            # Format: [x_min, y_min, x_max, y_max]
            rect1_points = [(box1[0], box1[1]), (box1[2], box1[1]), 
                           (box1[2], box1[3]), (box1[0], box1[3])]
            rect2_points = [(box2[0], box2[1]), (box2[2], box2[1]), 
                           (box2[2], box2[3]), (box2[0], box2[3])]
        elif len(box1) == 8 and len(box2) == 8:
            # Format: [x1, y1, x2, y2, x3, y3, x4, y4]
            rect1_points = [(box1[0], box1[1]), (box1[2], box1[3]), 
                           (box1[4], box1[5]), (box1[6], box1[7])]
            rect2_points = [(box2[0], box2[1]), (box2[2], box2[3]), 
                           (box2[4], box2[5]), (box2[6], box2[7])]
        else:
            return 0.0  # Invalid format
        
        # Create Polygon objects
        poly1 = Polygon(rect1_points)
        poly2 = Polygon(rect2_points)
        
        if not poly1.is_valid or not poly2.is_valid:
            return 0.0
        
        # Get accurate areas using polygon calculation
        box1_area = poly1.area
        box2_area = poly2.area
        
        # Add a small epsilon to prevent division by zero
        epsilon = 1e-10
        
        # Maintain strictness rule: return 0.0 if detected box is 2x+ larger than ground truth
        if box1_area >= 2 * box2_area:
            return 0.0
        
        # Calculate intersection area with polygon geometry
        intersection_area = poly1.intersection(poly2).area
        
        # Use max area as denominator (stricter than union)
        denominator = max(box1_area, box2_area)
        
        if denominator < epsilon:  # Avoid division by very small values
            return 0.0
            
        # Calculate ratio and clamp to [0.0, 1.0] range
        overlap_ratio = intersection_area / denominator
        
        # Clamp to valid range [0.0, 1.0]
        return max(0.0, min(1.0, overlap_ratio))
        
    except Exception as e:
        bt.logging.error(f"[calculate_overlap] Error {e}")
        return 0.0

def accuracy_score_calculation(detected_checkboxes, ground_truths):
    """
    Calculate the accuracy score based on detected checkboxes and ground truths.

    Parameters:
    - detected_checkboxes (list): List of detected checkbox data.
    - ground_truths (list): List of ground truth checkbox data.

    Returns:
    - float: Overall accuracy score.
    """
    try:
        scores = []
        
        #both lists are empty
        if not detected_checkboxes and not ground_truths:
            return 1.0
        
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
    except Exception as e:
        bt.logging.error(f"[accuracy_score_calculation] Error {e}")
        return 0.0
    
    
def highlight_accuracy_score_calculation(detected_highlights, ground_truths):
    """
    Calculate the accuracy score based on detected highlights and ground truths.

    Parameters:
    - detected_highlights (list): List of detected highlight data.
    - ground_truths (list): List of ground truth highlight data.

    Returns:
    - float: Overall accuracy score.
    """
    try:
        scores = []
        
        # Both lists are empty
        if not detected_highlights and not ground_truths:
            return 1.0
        
        # If difference in count is too large, return 0
        if abs(len(detected_highlights) - len(ground_truths)) > 1:
            return 0.0

        for detected in detected_highlights:
            detected_bbox = detected.get('boundingBox', [])
            detected_text = detected.get('text', '')
            
            score_for_detected_pair = 0.0
            for ground_truth in ground_truths:
                ground_truth_bbox = ground_truth.get('boundingBox', [])
                ground_truth_text = ground_truth.get('text', '')
                
                # Calculate HBS (Highlight Bounding Box Score)
                overlap = calculate_overlap(detected_bbox, ground_truth_bbox)
                if overlap > 0.95:
                    hbs = 1.0
                elif overlap > 0.7:
                    hbs = 1.0 - (0.95 - overlap) / (0.95 - 0.7) * 0.5  # Decrease score gradually
                else:
                    hbs = 0.0
                
                # Calculate TS (Text Similarity)
                ts = hard_match_strings_2(detected_text, ground_truth_text, 75.0)
                if ts >= 100:
                    ts_score = 1.0
                elif ts >= 30:
                    ts_score = (ts - 30) / 70  # Decrease score gradually
                else:
                    ts_score = 0.0
                
                # Calculate score for this pair
                score = (hbs + ts_score) / 2
                if score > score_for_detected_pair:
                    score_for_detected_pair = score
            
            scores.append(score_for_detected_pair)
        
        # Calculate overall accuracy score
        if scores:
            accuracy_score = sum(scores) / len(scores)
        else:
            accuracy_score = 0.0
        
        return accuracy_score
    except Exception as e:
        bt.logging.error(f"[highlight_accuracy_score_calculation] Error {e}")
        return 0.0


def encircle_accuracy_score_calculation(detected_encircles, ground_truths):
    """
    Calculate the accuracy score based on detected encircles and ground truths.

    Parameters:
    - detected_encircles (list): List of detected encircle data.
    - ground_truths (list): List of ground truth encircle data.

    Returns:
    - float: Overall accuracy score.
    """
    try:
        scores = []
        
        # Both lists are empty
        if not detected_encircles and not ground_truths:
            return 1.0
        
        # If difference in count is too large, return 0
        if abs(len(detected_encircles) - len(ground_truths)) > 1:
            return 0.0

        for detected in detected_encircles:
            detected_bbox = detected.get('boundingBox', [])
            detected_text = detected.get('text', '')
            
            score_for_detected_pair = 0.0
            for ground_truth in ground_truths:
                ground_truth_bbox = ground_truth.get('boundingBox', [])
                ground_truth_text = ground_truth.get('text', '')
                
                # Calculate EBS (Encircle Bounding Box Score)
                overlap = calculate_overlap(detected_bbox, ground_truth_bbox)
                if overlap > 0.95:
                    ebs = 1.0
                elif overlap > 0.7:
                    ebs = 1.0 - (0.95 - overlap) / (0.95 - 0.7) * 0.5  # Decrease score gradually
                else:
                    ebs = 0.0
                
                # Calculate TS (Text Similarity)
                ts = hard_match_strings_2(detected_text, ground_truth_text, 75.0)
                if ts >= 100:
                    ts_score = 1.0
                elif ts >= 30:
                    ts_score = (ts - 30) / 70  # Decrease score gradually
                else:
                    ts_score = 0.0
                
                # Calculate score for this pair
                score = (ebs + ts_score) / 2
                if score > score_for_detected_pair:
                    score_for_detected_pair = score
            
            scores.append(score_for_detected_pair)
        
        # Calculate overall accuracy score
        if scores:
            accuracy_score = sum(scores) / len(scores)
        else:
            accuracy_score = 0.0
        
        return accuracy_score
    except Exception as e:
        bt.logging.error(f"[encircle_accuracy_score_calculation] Error {e}")
        return 0.0
    

def final_score_calculation(time_score, accuracy_score):
    try:
        final_score = 0.3*time_score + 0.7*accuracy_score
        return final_score
    except Exception as e:
        bt.logging.error(f"[final_score_calculation] Error {e}")
        return 0.0


def reward(ground_truth: list, response: ProfileSynapse, Tt: float) -> float:
    """
    Reward the miner response to the dummy request. This method returns a reward
    value for the miner, which is used to update the miner's score.

    Returns:
    - float: The reward value for the miner.
    """
    try:
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
    except Exception as e:
        bt.logging.error(f"[reward] Error {e}")
        return 0.0
    
def highlight_reward(ground_truth: list, response: ProfileSynapse, Tt: float) -> float:
    """
    Reward the miner response for highlighted text detection. This method returns a reward
    value for the miner, which is used to update the miner's score.

    Returns:
    - float: The reward value for the miner.
    """
    try:
        highlights_detected = response.miner_output

        bt.logging.info(f"*************** Detected Highlights:")
        bt.logging.info(highlights_detected)
        bt.logging.info("************** End")
        bt.logging.info(f"*************** Ground Truth:")
        bt.logging.info(ground_truth)
        bt.logging.info("************** End")
        
        tim_score = time_score_calculation(Tt)
        acc_score = highlight_accuracy_score_calculation(highlights_detected, ground_truth)
        # score = final_score_calculation(tim_score, acc_score)
        score = acc_score
        return score
    except Exception as e:
        bt.logging.error(f"[highlight_reward] Error {e}")
        return 0.0


def encircle_reward(ground_truth: list, response: ProfileSynapse, Tt: float) -> float:
    """
    Reward the miner response for encircled text detection. This method returns a reward
    value for the miner, which is used to update the miner's score.

    Returns:
    - float: The reward value for the miner.
    """
    try:
        encircles_detected = response.miner_output

        bt.logging.info(f"*************** Detected Encircles:")
        bt.logging.info(encircles_detected)
        bt.logging.info("************** End")
        bt.logging.info(f"*************** Ground Truth:")
        bt.logging.info(ground_truth)
        bt.logging.info("************** End")
        
        tim_score = time_score_calculation(Tt)
        acc_score = encircle_accuracy_score_calculation(encircles_detected, ground_truth)
        # score = final_score_calculation(tim_score, acc_score)
        score = acc_score
        return score
    except Exception as e:
        bt.logging.error(f"[encircle_reward] Error {e}")
        return 0.0


def doc_class_reward(ground_truth: list, response: ProfileSynapse, Tt: float) -> float:
    """
    Reward the miner response to the dummy request. This method returns a reward
    value for the miner, which is used to update the miner's score.

    Returns:
    - float: The reward value for the miner.
    """
    try:
        if len(response.miner_output)==0:
            return 0.0
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
    except Exception as e:
        bt.logging.error(f"[doc_class_reward] Error {e}")
        return 0.0


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
            # Handle empty dictionaries
            if not actual_section and not detected_section:
                return 1.0
            if not are_keys_same(actual_section, detected_section):
                return 0.0
            scores = [compute_section_score(detected_section[sub_key], actual_section[sub_key]) for sub_key in actual_section]
            return sum(scores) / len(scores) if scores else 0.0

        # Case 3: Both are lists
        elif isinstance(actual_section, list) and isinstance(detected_section, list):
            # Handle empty lists
            if not actual_section and not detected_section:
                return 1.0
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
        if len(response.miner_output)==0:
            return 0.0
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
    sub_task_type: str
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
    if sub_task_type == "checkbox":
        return np.array(
            [reward(ground_truth, each_resp, Tt) for each_resp in responses]
        )
    
    elif sub_task_type == "highlight":
        return np.array(
            [highlight_reward(ground_truth, each_resp, Tt) for each_resp in responses]
        )
    
    elif sub_task_type == "encircle":
        return np.array(
            [encircle_reward(ground_truth, each_resp, Tt) for each_resp in responses]
        )

    elif sub_task_type == "doc-class":
        return np.array(
            [doc_class_reward(ground_truth, each_resp, Tt) for each_resp in responses]
        )

    elif sub_task_type == "doc-parse":
        scores_array = np.zeros(len(responses))
        for idx, each_resp in enumerate(responses):
            if len(ground_truth) > 0:
                classification_score = doc_class_reward([ground_truth[0].get("document_class", "")], each_resp, Tt)
                parsing_score = doc_parse_reward(ground_truth, each_resp, Tt)
                weighted_avg_score = 0.3*classification_score + 0.7*parsing_score
                scores_array[idx] = weighted_avg_score
            else:
                scores_array[idx] = 0.0

        return scores_array