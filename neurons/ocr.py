from PIL import Image
import pytesseract
import pandas as pd
import numpy as np
import json
import os
import logging
import base64
from io import BytesIO

def get_bounding_box(left, top, width, height):
    """
    Convert left, top, width, height into a bounding box format [x1, y1, x2, y1, x2, y2, x1, y2].
    """
    x1, y1 = left, top
    x2, y2 = left + width, top + height
    # Return the bounding box in the format: top-left, top-right, bottom-right, bottom-left
    return [x1, y1, x2, y1, x2, y2, x1, y2]

def are_words_on_same_line(word1, word2, y_threshold=15, x_threshold=50):
    """
    Determine if two words are on the same line based on the vertical position (y-axis)
    and their proximity in the x-axis.
    """
    # Check vertical proximity (y-axis)
    if abs(word1['top'] - word2['top']) < y_threshold:
        # Check horizontal proximity (x-axis)
        if abs(word1['left'] + word1['width'] - word2['left']) < x_threshold:
            return True
    return False

def group_words_into_lines(ocr_data, y_threshold=15, x_threshold=50):
    """
    Group words into lines based on their bounding box proximity.
    """
    lines = []
    current_line = []

    for i in range(len(ocr_data['text'])):
        word_text = ocr_data['text'][i]
        if word_text.strip():  # Ignore empty strings
            word_data = {
                'text': word_text,
                'left': ocr_data['left'][i],
                'top': ocr_data['top'][i],
                'width': ocr_data['width'][i],
                'height': ocr_data['height'][i],
                'bounding_box': get_bounding_box(ocr_data['left'][i], ocr_data['top'][i],
                                                 ocr_data['width'][i], ocr_data['height'][i])
            }

            if not current_line:
                current_line.append(word_data)
            else:
                last_word = current_line[-1]
                if are_words_on_same_line(last_word, word_data, y_threshold, x_threshold):
                    current_line.append(word_data)
                else:
                    lines.append(current_line)
                    current_line = [word_data]
    
    if current_line:
        lines.append(current_line)

    return lines

def ocr_image_with_custom_line_detection(binary_image, save_ocr=False):
    """
    Perform OCR on an image and return the text organized by lines based on proximity-based grouping.
    Optionally save the result to a .json file if save_ocr is set to True.
    """
    # img = Image.open(BytesIO(binary_image))
    # Step 1:
    image_data = base64.b64decode(binary_image)

    # Step 2: Wrap the bytes in a BytesIO stream
    image_stream = BytesIO(image_data)

    # Step 3: Open the image using PIL
    img = Image.open(image_stream)

    ocr_data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)

    lines = group_words_into_lines(ocr_data)

    # Prepare the final result
    result = {
        "page": 1,
        "width": img.width,
        "height": img.height,
        "unit": "pixel",
        "lines": []
    }

    for line in lines:
        line_text = " ".join([word['text'] for word in line])
        line_bounding_box = [
            min([word['bounding_box'][0] for word in line]),  # left-most (x1)
            min([word['bounding_box'][1] for word in line]),  # top-most (y1)
            max([word['bounding_box'][2] for word in line]),  # right-most (x2)
            max([word['bounding_box'][5] for word in line])   # bottom-most (y2)
        ]
        # Convert the bounding box for the line to the required format
        line_bounding_box = [
            line_bounding_box[0], line_bounding_box[1],  # top-left
            line_bounding_box[2], line_bounding_box[1],  # top-right
            line_bounding_box[2], line_bounding_box[3],  # bottom-right
            line_bounding_box[0], line_bounding_box[3]   # bottom-left
        ]

        result['lines'].append({
            "boundingBox": line_bounding_box,
            "text": line_text,
            "words": [
                {
                    "boundingBox": word['bounding_box'],
                    "text": word['text']
                } for word in line
            ]
        })
    
    # Save OCR result to a JSON file if save_ocr is True
    if save_ocr:
        json_filename = os.path.splitext(image_path)[0] + ".json"
        with open(json_filename, 'w') as json_file:
            json.dump(result, json_file, indent=4)
        print(f"OCR result saved to {json_filename}")

    return result

# Example usage
if __name__ == '__main__':
    image_path = ''  # Replace with your image path
    ocr_result = ocr_image_with_custom_line_detection(image_path, save_ocr=True)
    # print(ocr_result)
