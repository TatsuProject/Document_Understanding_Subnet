import requests
import time
import pandas as pd
import logging
import math
import threading
import numpy as np
from fuzzywuzzy import fuzz
import json
import base64
import uuid

class YoloCheckboxDetector():

    def __init__(self):
        self.lines_list = None
        self.request_id = ""

    def reduce_image_dimension(self, image):
        try:
            # Resize the image
            maximum_acceptable_size = 9990
            height, width = image.shape[:2]
            scaling_factor = min(maximum_acceptable_size / height, maximum_acceptable_size / width)
            new_height = int(height * scaling_factor)
            new_width = int(width * scaling_factor)
            resized_image = cv2.resize(image, (new_width, new_height))
            
            # Encode the resized image as bytes
            _, encoded_image = cv2.imencode('.png', resized_image)
            
            return encoded_image.tobytes()
        except Exception as e:
            logging.error(f"Error in image size reduction: {e}")
            return None

    def check_image_validity_for_ocr(self, file_content):
        try:
            maximum_acceptable_size = 10000
            
            # Read image from bytes
            image = cv2.imdecode(np.frombuffer(file_content, np.uint8), -1)
            img_height, img_width = image.shape[:2]
            
            if img_height < 50 or img_width < 50:
                logging.error("Image is smaller than required size (50x50)")
                return file_content, True
            
            if img_height > maximum_acceptable_size or img_width > maximum_acceptable_size:
                resized_image_bytes = self.reduce_image_dimension(image)
                if resized_image_bytes is not None:
                    return resized_image_bytes, False
            return file_content, False
        except Exception as e:
            logging.error("Error in image validity check: ", e)
            return file_content, False

    def get_selected_checkboxes(self, checkbox_response):
        selected_checkboxes = []
        for checkbox in checkbox_response:
            if checkbox['state'] == 'selected':
                if 'polygon' in checkbox.keys():
                    checkbox["boundingBox"] = checkbox.pop("polygon")
                    if 'span' in checkbox.keys():
                        checkbox.pop("span")
                selected_checkboxes.append(checkbox)

        return selected_checkboxes

    def strip_string_at_left_of_checkbox(self, text_string, x1_text, x1_checkbox, x2_text):
        """When the text line lies at the left as well as at the right side of checkbox, this function will strip it out
        and discard its left part

            x1_text_____________x1_checkbox___________________x2_text
            |                   |               |                   |
            |                   |               |                   |
            |                   |               |                   |
            |___________________|_______________|___________________|
        """
        # calculating discard ratio using bounding box
        discard_ratio = (x1_checkbox - x1_text) / (x2_text - x1_text)
        # calculating discard length using string length
        discard_length = math.floor(discard_ratio * len(text_string))
        stripped_text_string = ''
        # if letter lies in discarded region, don't pick it
        for idx, letter in enumerate(text_string):
            if idx>=discard_length:
                stripped_text_string+=letter

        return stripped_text_string

    def screen_checkboxes_based_on_confidence(self, checkboxes_with_text, threshold_conf = 0.3):
        """
        This function will select checkboxes along with the text with confidence greater than a threshold
        """
        new_checkboxes_list = []
        for each_checkbox in checkboxes_with_text:
            if each_checkbox["confidence"]>threshold_conf:
                new_checkboxes_list.append(each_checkbox)

        return new_checkboxes_list


    def convert_ocr_to_line_list(self, model_request_data):
        try:
            lines = model_request_data.get('lines', [])
            return lines

        except Exception as e:
            logging.error(f"Error in convert_ocr_to_line_list: {e}")
            return []

    def nearest_text_loop(self, checkbox_bbox, lines_list):
        # Initialize variables to store the nearest text and its distance
        nearest_text = None
        nearest_text_bbox = None
        min_distance = float('inf')  # Initialize with a large value

        # Calculate the center coordinates of the checkbox
        checkbox_center_x = (checkbox_bbox[0] + checkbox_bbox[2]) / 2
        checkbox_center_y = (checkbox_bbox[1] + checkbox_bbox[5]) / 2

        # defining thresholds for top, bottom, left and right
        y_margin_above = 15
        y_margin_below = 10
        x_margin_right = 60
        x_margin_left = 20

        # Iterate through each text bounding box
        for ind, content in enumerate(lines_list):
            strip_string = False
            text_bbox = lines_list[ind]["boundingBox"]
            # Calculate the center coordinates of the text bounding box
            text_center_x = (text_bbox[0] + text_bbox[2]) / 2
            text_center_y = (text_bbox[1] + text_bbox[5]) / 2

            # if text lies vertically within the range of checkbox
            if ((checkbox_bbox[1] - y_margin_above) < text_bbox[1]) and ((checkbox_bbox[7] + y_margin_below) > text_bbox[7]):
                # if text lies horizontally within the range of checkbox
                if ((checkbox_bbox[0] - x_margin_left) < text_bbox[0]) and ((checkbox_bbox[2] + x_margin_right) > text_bbox[0]) or \
                (text_bbox[0] < checkbox_bbox[0] and text_bbox[2] > checkbox_bbox[0]):
                # if text lies horizontally at the left as well as at the right side of checkbox

                    # Check if the text is to the right or below the checkbox
                    if text_center_x >= checkbox_center_x or text_center_y >= checkbox_center_y:
                        # Calculate the Euclidean distance between the checkbox and text center
                        distance = ((checkbox_center_x - text_center_x) ** 2 +
                                    (checkbox_center_y - text_center_y) ** 2) ** 0.5
                        if (text_bbox[0] < checkbox_bbox[0] and text_bbox[2] > checkbox_bbox[0]):
                            strip_string = True
                        # Check if this text is closer than the current nearest text
                        if distance > 0.5 and distance < min_distance:
                            min_distance = distance
                            nearest_text = lines_list[ind]["text"]
                            if strip_string:
                                nearest_text = self.strip_string_at_left_of_checkbox(nearest_text, text_bbox[0], checkbox_bbox[0], text_bbox[2])
                                text_bbox[0], text_bbox[6] = checkbox_bbox[0], checkbox_bbox[6]
                            nearest_text_bbox = text_bbox
        if nearest_text:
            nearest_text = nearest_text.lstrip("X").lstrip("x")
        return nearest_text, nearest_text_bbox


    def nearest_text_loop_at_left(self, checkbox_bbox, lines_list, left_right_clusters):
        # Initialize variables to store the nearest text and its distance
        nearest_text = None
        nearest_text_bbox = None
        min_distance = float('inf')  # Initialize with a large value

        # Calculate the center coordinates of the checkbox
        checkbox_center_x = (checkbox_bbox[0] + checkbox_bbox[2]) / 2
        checkbox_center_y = (checkbox_bbox[1] + checkbox_bbox[5]) / 2

        # defining thresholds for top, bottom, left and right
        y_margin_above = 15
        y_margin_below = 10
        x_margin_right = 5
        x_margin_left = 60
        # Iterate through each text bounding box
        for ind, content in enumerate(lines_list):
            strip_string = False
            text_bbox = lines_list[ind]["boundingBox"]
            # Calculate the center coordinates of the text bounding box
            text_center_x = (text_bbox[0] + text_bbox[2]) / 2
            text_center_y = (text_bbox[1] + text_bbox[5]) / 2

            # if text lies vertically within the range of checkbox
            if ((checkbox_bbox[1] - y_margin_above) < text_bbox[1]) and ((checkbox_bbox[7] + y_margin_below) > text_bbox[7]):
                # if text lies horizontally within the range of checkbox
                if ((checkbox_bbox[0] + x_margin_right) > text_bbox[2]) and ((checkbox_bbox[0] - x_margin_left) < text_bbox[2]):
                # if text lies horizontally at the left as well as at the right side of checkbox

                    # Check if the text is to the right or below the checkbox
                    if text_center_x < checkbox_center_x or text_center_y <= checkbox_center_y:
                        # Calculate the Euclidean distance between the checkbox and text center
                        distance = ((checkbox_center_x - text_center_x) ** 2 +
                                    (checkbox_center_y - text_center_y) ** 2) ** 0.5
                        # Check if this text is closer than the current nearest text
                        if distance > 0.5 and distance < min_distance:
                            min_distance = distance
                            nearest_text = lines_list[ind]["text"]
                            if self.use_spacing_method:
                                nearest_text, nearest_text_bbox = self.check_on_word_level(lines_list, ind, checkbox_bbox, nearest_text, text_bbox)
                            if nearest_text is None or nearest_text=="":
                                nearest_text = lines_list[ind]["text"]
                            nearest_text_bbox = text_bbox

        return nearest_text, nearest_text_bbox

    def get_associated_text(self, checkboxes_list):
        checkboxes_with_text = []
        i = 1
        for checkboxes in checkboxes_list:
            print("Check Boxes:", i)
            nearest_text, nearest_text_bbox = self.nearest_text_loop(checkboxes["boundingBox"], self.lines_list)
            if nearest_text:
                checkboxes["text"] = nearest_text
                checkboxes["checkbox_boundingBox"] = checkboxes["boundingBox"]
                checkboxes["boundingBox"] = nearest_text_bbox
                checkboxes_with_text.append(checkboxes)
            i+=1

        return checkboxes_with_text
    

    def to_xyxy(self, bbox):
        box = [bbox[0], bbox[1], bbox[2], bbox[1], bbox[2], bbox[3], bbox[0], bbox[3]]
        return box
    
    def isOverlapping(self, bbox1, bbox2, method="min"):
        """
        Calculate the overlapping area between two bounding boxes.

        Arguments:
        bbox1 (list): List representing the coordinates of the first bounding box in the format [x1, y1, x2, y1, x2, y2, x1, y2].
        bbox2 (list): List representing the coordinates of the second bounding box in the same format.

        Returns:
        float: Overlapping ratio between the two bounding boxes.
        """
        # Extract coordinates from the bounding box lists
        x1_min, y1_min, x2_max, y2_max = min(bbox1[::2]), min(bbox1[1::2]), max(bbox1[::2]), max(bbox1[1::2])
        x1_min2, y1_min2, x2_max2, y2_max2 = min(bbox2[::2]), min(bbox2[1::2]), max(bbox2[::2]), max(bbox2[1::2])

        # Calculate the overlapping area
        x_overlap = max(0, min(x2_max, x2_max2) - max(x1_min, x1_min2))
        y_overlap = max(0, min(y2_max, y2_max2) - max(y1_min, y1_min2))
        overlapping_area = x_overlap * y_overlap

        # Calculate the area of bbox1 and bbox2
        area_bbox1 = (x2_max - x1_min) * (y2_max - y1_min)
        area_bbox2 = (x2_max2 - x1_min2) * (y2_max2 - y1_min2)

        # Calculate the overlapping ratio
        if area_bbox1 == 0 or area_bbox2 == 0:
            return 0.0  # Return 0 if any of the bounding boxes has zero area
        elif method=="max":
            overlap_ratio = overlapping_area / max(area_bbox1, area_bbox2)
            return overlap_ratio
        else:
            overlap_ratio = overlapping_area / min(area_bbox1, area_bbox2)
            return overlap_ratio
    
    def merge_bbox_xywh(self, bbox1, bbox2):
        """this will merge bboxes in format (x,y,w,h)"""
        return [min(bbox1[0], bbox2[0]), min(bbox1[1], bbox2[1]), 
                max(bbox1[0]+bbox1[2], bbox2[0]+bbox2[2]), max(bbox1[1]+bbox1[3], bbox2[1]+bbox2[3])]

    def merge_bbox(self, bbox1, bbox2):
        """this will merge bboxes in format (x1,y1,x2,y2)"""
        return [min(bbox1[0], bbox2[0]), min(bbox1[1], bbox2[1]), max(bbox1[2], bbox2[2]), max(bbox1[3], bbox2[3])]

    def merge_polygon_bbox(self, bbox1, bbox2):
        """
        Merges two bounding boxes represented as polygons and returns the merged bounding box.
        
        Args:
            bbox1 (list): The first bounding box in the form [x1, y1, x2, y1, x2, y2, x1, y2].
            bbox2 (list): The second bounding box in the same form.
            
        Returns:
            list: The merged bounding box in the form [x1, y1, x2, y1, x2, y2, x1, y2].
        """
        
        # Extract the x and y coordinates from both bounding boxes
        x_coords = [bbox1[i] for i in range(0, len(bbox1), 2)] + [bbox2[i] for i in range(0, len(bbox2), 2)]
        y_coords = [bbox1[i] for i in range(1, len(bbox1), 2)] + [bbox2[i] for i in range(1, len(bbox2), 2)]
        
        # Determine the min and max values for x and y
        x1, x2 = min(x_coords), max(x_coords)
        y1, y2 = min(y_coords), max(y_coords)
        
        # Construct the merged bounding box
        merged_bbox = [x1, y1, x2, y1, x2, y2, x1, y2]
        
        return merged_bbox

    
    def are_strings_similar(self, string1, string2, threshold=90.0):
        total_average_similarity = fuzz.token_sort_ratio(string1.lower(), string2.lower())
        return total_average_similarity>threshold
 

    def get_selected_checkboxes_with_text(self, checkbox_response, ocr_data, request_id=""):
        self.request_id = request_id if request_id else str(uuid.uuid4())
        try:
            self.lines_list = self.convert_ocr_to_line_list(ocr_data)
            if len(self.lines_list) == 0:
                logging.info("lines dataframe is empty in checkbox service")
                return []

            # Since call_form_recognizer takes more time so this function is called in a thread

            # get selected checkboxes
            checkboxes_response = self.get_selected_checkboxes(checkbox_response)
            # get text near selected checkboxes
            checkboxes_with_text = self.get_associated_text(checkboxes_response)
            # filter out checkboxes based on confidence
            final_check_boxes = self.screen_checkboxes_based_on_confidence(checkboxes_with_text)
            return final_check_boxes
        except Exception as e:
            logging.error(e)
            return []


if __name__ == "__main__":

    img_path = ""
    ocr_path = img_path.replace(".png", ".png.ocr.json")

    with open(ocr_path, 'r') as f:
        ocr_json = json.load(f)
    # Convert the loaded OCR JSON object to a string
    ocr_string = json.dumps(ocr_json)
    # Encode the OCR string to base64
    encoded_ocr = base64.b64encode(ocr_string.encode('utf-8')).decode('utf-8')
    # Create the structure to pass to the function
    ocr_data = {
        "data": {
            "instances": [
                {
                    "ocr": encoded_ocr
                }
            ]
        }
    }

    file_data = open(img_path, "rb").read()

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    logging.info(f"checkboxes_with_text")

    api = FormRecognizer()
    result = api.get_selected_checkboxes_with_text(file_data, img_path, ocr_data, "")
    logging.info(f"checkboxes_with_text {result}")
