


# ================================================== data generation ==============================================================
import random
import json
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageOps
from math import floor
import io
import numpy as np
import cv2
from faker import Faker
import os
import bittensor as bt
import uuid

script_dir = os.path.dirname(os.path.abspath(__file__))
fake = Faker()

class GenerateCheckboxTextPair:
    def __init__(self, url, uid):
        self.url = ""
        self.uid = uid
        self.text_color = random.choice([
            (70, 68, 66), (55, 58, 65), (72, 64, 60), (63, 55, 70),
            (67, 72, 62), (58, 66, 72), (75, 60, 65), (65, 58, 73),
            (68, 63, 57), (73, 70, 62)
        ])
        self.checkbox_color = self.text_color
        self.font = ""

    def generate_scanned_document(self):

        """
        Generates a synthetic scanned document image with random text, noise, and scanned effects.

        Returns:
            PIL.Image.Image: The final scanned document image in RGB format.
        """
        try:
            # Step 1: Create a blank image with random size
            sizes = [
                (1200, 900), (1500, 1000), (1300, 950), (1800, 1200),
                (1600, 1100), (1400, 1050), (1700, 1150), (1900, 1250),
                (1450, 950), (1550, 1020), (1650, 1080), (1850, 1220),
                (900, 1200), (1000, 1500), (950, 1300), (1200, 1800),
                (1100, 1600), (1050, 1400), (1150, 1700), (1250, 1900),
                (950, 1450), (1020, 1550), (1080, 1650), (1220, 1850)
            ]

            width, height = random.choice(sizes)
            image = Image.new("RGB", (width, height), "white")
            draw = ImageDraw.Draw(image)

            # Step 2: Generate paragraphs using Faker
            paragraphs = [fake.text(max_nb_chars=170) for _ in range(7)]  # Generate 7 random paragraphs

            # Step 3: Add text to random locations
            text_size = random.choice([22, 24, 26, 28])
            
            try:
                font = random.choice([
                    ImageFont.truetype(os.path.join(script_dir, "fonts/Arial.ttf"), text_size),
                    ImageFont.truetype(os.path.join(script_dir, "fonts/Arial_Bold_Italic.ttf"), text_size),
                    ImageFont.truetype(os.path.join(script_dir, "fonts/Courier_New.ttf"), text_size),
                    ImageFont.truetype(os.path.join(script_dir, "fonts/DroidSans-Bold.ttf"), text_size),
                    ImageFont.truetype(os.path.join(script_dir, "fonts/FiraMono-Regular.ttf"), text_size),
                    ImageFont.truetype(os.path.join(script_dir, "fonts/Times New Roman.ttf"), text_size),
                    ImageFont.truetype(os.path.join(script_dir, "fonts/Vera.ttf"), text_size),
                    ImageFont.truetype(os.path.join(script_dir, "fonts/Verdana_Bold_Italic.ttf"), text_size),
                    ImageFont.truetype(os.path.join(script_dir, "fonts/Verdana.ttf"), text_size),
                    ImageFont.truetype(os.path.join(script_dir, "fonts/DejaVuSansMono-Bold.ttf"), text_size),
                    ImageFont.truetype(os.path.join(script_dir, "handwritten_fonts/Mayonice.ttf"), text_size)
                ])
                self.font=font
            except IOError:
                font = ImageFont.load_default()  # Default font
            
            used_centre = []
            total_retries = 100
            retried = 0
            for _ in range(7):
                text = random.choice(paragraphs)

                while retried <= total_retries:  # Keep generating positions until a valid one is found
                    retried += 1
                    x = random.randint(10, width - 700)  # Avoid edges
                    y = random.randint(10, height - 200)  # Avoid bottom edge

                    # Check if y is not within ±40 of any used y
                    if all(abs(y - prev_y) > 40 for _, prev_y in used_centre):
                        used_centre.append((x, y))
                        break

                draw.text((x, y), text, fill=self.text_color, font=font)

            bt.logging.info(f"[{self.uid}] Document Generated Successfully!")
            return image

        except Exception as e:
            bt.logging.error(f"[generate_scanned_document] An error occurred: {e}")
            return None

    def is_window_empty(self, window, threshold=240, empty_percentage=0.9999):
        """Check if the window region is mostly empty (white)."""
        pixels = list(window.getdata())
        white_pixels = sum(1 for pixel in pixels if sum(pixel[:3]) > threshold * 3)
        return white_pixels >= empty_percentage * len(pixels)

    def find_empty_region(self, image, window_width, window_height, h_stride=15, v_stride=10):
        """Find an empty region in the image by moving a search window with defined strides."""
        width, height = image.size
        # starting_points = [(0, 0), (50, 50), (100, 1000), (0, 500), (100, 500), (150, 500), (200, 500)]
        starting_points = (0, 0)
        start_x, start_y = starting_points
        # Ensure starting point is within image bounds
        start_x = min(start_x, width - window_width)
        start_y = min(start_y, height - window_height)

        for y in range(start_y, height - window_height, v_stride):
            for x in range(start_x, width - window_width, h_stride):
                window = image.crop((x, y, x + window_width, y + window_height))
                if self.is_window_empty(window):
                    return x + 5, y + 5  # Add 5 pixels padding for checkbox and text
        return None, None  # No empty region found

    def get_random_metadata(self):
        
        checkbox_text_size = random.choice([
            (20, 10), (22, 11), (24, 12), (26, 13), (28, 14), (30, 15), (32, 16), (34, 18), (36, 20) 
        ])
        
        padding = random.choice([6, 7, 8, 9, 10])
        
        checkbox_stroke = random.choice([2, 3, 4, 5])
        
        return {
            "text_color": self.text_color,
            "checkbox_text_size": checkbox_text_size,
            "padding": padding,
            "checkbox_stroke": checkbox_stroke,
            "font": self.font
        }

    def draw_random_checkbox(self, draw, x, y, checkbox_size, checkbox_color, shape_drawn):
        """
        Draws a random tick or cross with slight imperfections to simulate natural human checks.
        """
        # Randomly decide whether to draw a tick or a cross
        fill_checkbox = random.choice([True, False])
        if fill_checkbox:
            if shape_drawn == "rectangle":
                if random.choice([True, False]):  # Draw tick
                    line_width = random.randint(2, 4)
                    tick_variation = random.randint(1, 5)
                    
                    if tick_variation == 1:  # Basic tick
                        draw.line((x, y + checkbox_size // 2, x + checkbox_size // 2, y + checkbox_size), fill=checkbox_color, width=line_width)
                        draw.line((x + checkbox_size // 2, y + checkbox_size, x + checkbox_size, y), fill=checkbox_color, width=line_width)
                    
                    elif tick_variation == 2:  # Slightly imperfect tick
                        draw.line((x + random.randint(-2, 2), y + checkbox_size // 2, x + checkbox_size // 2 + random.randint(-2, 2), y + checkbox_size), fill=checkbox_color, width=line_width)
                        draw.line((x + checkbox_size // 2 + random.randint(-2, 2), y + checkbox_size, x + checkbox_size + random.randint(-2, 2), y), fill=checkbox_color, width=line_width)
                    
                    elif tick_variation == 3:  # Off-center tick
                        draw.line((x + random.randint(0, 2), y + checkbox_size // 2, x + checkbox_size // 2 + 2, y + checkbox_size - 3), fill=checkbox_color, width=line_width)
                        draw.line((x + checkbox_size // 2, y + checkbox_size, x + checkbox_size - 2, y - 3), fill=checkbox_color, width=line_width)
                    
                    elif tick_variation == 4:  # Tilted tick
                        draw.line((x, y + checkbox_size // 2, x + checkbox_size // 3, y + checkbox_size + 2), fill=checkbox_color, width=line_width)
                        draw.line((x + checkbox_size // 3, y + checkbox_size, x + checkbox_size, y - 1), fill=checkbox_color, width=line_width)
                    
                    elif tick_variation == 5:  # Tick going slightly out of the box
                        draw.line((x - 1, y + checkbox_size // 2, x + checkbox_size // 2, y + checkbox_size + 2), fill=checkbox_color, width=line_width)
                        draw.line((x + checkbox_size // 2, y + checkbox_size + 1, x + checkbox_size + 2, y - 1), fill=checkbox_color, width=line_width)

                else:  # Draw cross
                    line_width = random.randint(2, 4)
                    cross_variation = random.randint(1, 5)
                    
                    if cross_variation == 1:  # Basic cross
                        draw.line((x, y, x + checkbox_size, y + checkbox_size), fill=checkbox_color, width=line_width)
                        draw.line((x + checkbox_size, y, x, y + checkbox_size), fill=checkbox_color, width=line_width)
                    
                    elif cross_variation == 2:  # Slightly imperfect cross
                        draw.line((x + random.randint(-2, 2), y, x + checkbox_size + random.randint(-2, 2), y + checkbox_size), fill=checkbox_color, width=line_width)
                        draw.line((x + checkbox_size + random.randint(-2, 2), y, x + random.randint(-2, 2), y + checkbox_size), fill=checkbox_color, width=line_width)
                    
                    elif cross_variation == 3:  # Cross tilted outward
                        draw.line((x - 1, y - 1, x + checkbox_size + 1, y + checkbox_size + 1), fill=checkbox_color, width=line_width)
                        draw.line((x + checkbox_size + 1, y - 1, x - 1, y + checkbox_size + 1), fill=checkbox_color, width=line_width)
                    
                    elif cross_variation == 4:  # Cross slightly going out of the box
                        draw.line((x - 1, y, x + checkbox_size + 2, y + checkbox_size), fill=checkbox_color, width=2)
                        draw.line((x + checkbox_size + 2, y - 1, x - 2, y + checkbox_size + 1), fill=checkbox_color, width=line_width)
                    
                    elif cross_variation == 5:  # Irregular cross with random shifts
                        draw.line((x + random.randint(-2, 2), y + random.randint(-2, 2), x + checkbox_size + random.randint(-2, 2), y + checkbox_size + random.randint(-2, 2)), fill=checkbox_color, width=line_width)
                        draw.line((x + checkbox_size + random.randint(-2, 2), y + random.randint(-2, 2), x + random.randint(-2, 2), y + checkbox_size + random.randint(-2, 2)), fill=checkbox_color, width=line_width)
            
            elif shape_drawn == "circle":
                fill_task = random.choice(["full", "inner"])

                if fill_task == "full":
                    draw.ellipse(checkbox_coords, fill=checkbox_color)
                
                elif fill_task == "inner":
                    padding = random.randint(3, 6)
                    inner_coords = [
                        x + padding,
                        y + padding,
                        x + checkbox_size - padding,
                        y + checkbox_size - padding
                    ]
                    draw.ellipse(inner_coords, fill=checkbox_color)
            else:
                pass

            return True
        else:
            # Draw 2 or 3 random small dots inside the checkbox
            num_dots = random.randint(2, 4)
            for _ in range(num_dots):
                dot_radius = 1  # Small dot
                dot_x = random.randint(x + 2, x + checkbox_size - 2)
                dot_y = random.randint(y + 2, y + checkbox_size - 2)
                draw.ellipse(
                    (dot_x - dot_radius, dot_y - dot_radius, dot_x + dot_radius, dot_y + dot_radius),
                    fill=checkbox_color
                )
            return False


    def put_text_randomly(self, draw, x, y, checkbox_size, text, font, text_color, width, height, padding):
        """
        Draws the given text randomly at either the right or bottom of the checkbox.
        Ensures that the text fits within the image bounds.
        """
        # Randomly choose whether to place text to the right or at the bottom
        choice = random.choice(["right", "bottom"])

        # padding = 10  # Space between checkbox and text

        if choice == "right":
            # Place text to the right of the checkbox
            text_x, text_y = x + checkbox_size + padding, y + 5
            text_bbox = draw.textbbox((0, 0), text, font=font)  # Get the bounding box of the text
            text_width, text_height = text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1]

            # Ensure text fits within image bounds
            if text_x + text_width > width:
                text_x = width - text_width - 1
            if text_y + text_height > height:
                text_y = height - text_height - 1

            if text_x < 0:
                text_x = 1
            if text_y < 0:
                text_y = 1

            draw.text((text_x, text_y), text, fill=text_color, font=font)

        elif choice == "bottom":
            # Place text below the checkbox
            text_bbox = draw.textbbox((0, 0), text, font=font)
            text_width, text_height = text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1]
            text_x, text_y = x + checkbox_size // 2 - text_width // 2, y + checkbox_size + padding

            # Ensure text fits within image bounds
            if text_x + text_width > width:
                text_x = width - text_width - 1
            if text_y + text_height > height:
                text_y = height - text_height - 1

            if text_x < 0:
                text_x = 1
            if text_y < 0:
                text_y = 1

            draw.text((text_x, text_y), text, fill=text_color, font=font)
        return text_x, text_y, text_width, text_height

    def generate_random_words(self):
        # Randomly choose the number of words between 2 and 4
        num_words = random.randint(2, 4)
        # Generate the words
        words = [fake.word() for _ in range(num_words)]
        # Join the words into a single string
        return " ".join(words)


    def add_grain_noise(self, image):
        """Adds random noise to an image."""
        np_image = np.array(image)
        noise = np.random.normal(0, 0.5, np_image.shape).astype(np.uint8)
        noisy_image = np.clip(np_image + noise, 0, 255)
        return Image.fromarray(noisy_image)


    def add_noise(self, pil_img):
        # Convert to grayscale
        img = pil_img.convert("L")
        img_np = np.array(img).astype(np.float32)

        h, w = img_np.shape

        # --- 1. Strong Gaussian Noise ---
        mean = 0
        stddev = random.uniform(15, 40)  # Increased range
        noise = np.random.normal(mean, stddev, (h, w))
        img_np += noise

        # --- 2. Stronger Gradient Light Effect ---
        for _ in range(random.randint(1, 2)):
            gradient = np.tile(np.linspace(0, random.randint(10, 60), w), (h, 1))
            if random.choice([True, False]):
                gradient = np.flip(gradient, axis=1)
            img_np += gradient * random.uniform(0.5, 1.5)

        # --- 3. More and Heavier Blotches ---
        for _ in range(random.randint(5, 10)):
            cx, cy = random.randint(0, w), random.randint(0, h)
            radius = random.randint(20, 120)
            strength = random.randint(20, 60)
            Y, X = np.ogrid[:h, :w]
            dist_from_center = np.sqrt((X - cx)**2 + (Y - cy)**2)
            mask = dist_from_center <= radius
            img_np[mask] += strength

        # --- 4. Add Horizontal Scanner Lines ---
        for _ in range(random.randint(5, 10)):
            y_line = random.randint(0, h - 1)
            thickness = random.randint(1, 2)
            intensity = random.randint(10, 40)
            img_np[y_line:y_line+thickness, :] += intensity

        # --- 5. Clip to valid pixel range ---
        img_np = np.clip(img_np, 0, 255).astype(np.uint8)
        img_noisy = Image.fromarray(img_np)

        # --- 6. Subtle Blur ---
        if random.random() < 0.7:
            img_noisy = img_noisy.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.7, 1.5)))

        # --- 7. Yellowish Paper Tint ---
        img_noisy = ImageOps.colorize(img_noisy, black="black", white="#fdfbf0")

        # --- 8. Optional JPEG Compression Artifacts ---
        if random.random() < 0.7:
            buffer = io.BytesIO()
            img_noisy.save(buffer, format="JPEG", quality=random.randint(20, 60))
            buffer.seek(0)
            img_noisy = Image.open(buffer)

        return self.add_grain_noise(img_noisy)

    def transform_bounding_boxes(self, ner_annotations, angle, image):
        """Transforms bounding boxes inside ner_annotations to 8-point format after rotation."""
        
        # Get image dimensions
        w, h = image.size
        center = (w // 2, h // 2)

        # Compute rotation matrix
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated_img = image.rotate(angle, expand=True)
        new_w, new_h = rotated_img.size

        def rotate_bbox(bbox):
            """Rotates a bounding box and converts it into an 8-point polygon."""
            if not bbox:
                return []
            x1, y1, x2, y2 = bbox  # Original bounding box

            # Get all 4 corner points
            points = np.array([
                [x1, y1],  # Top-left
                [x2, y1],  # Top-right
                [x2, y2],  # Bottom-right
                [x1, y2]   # Bottom-left
            ])

            # Apply rotation transformation
            ones = np.ones((4, 1))
            points = np.hstack([points, ones])  # Convert to homogeneous coordinates
            rotated_points = M.dot(points.T).T  # Apply transformation

            # Adjust coordinates to fit new image size
            x_offset = (new_w - w) // 2
            y_offset = (new_h - h) // 2
            rotated_points[:, 0] += x_offset
            rotated_points[:, 1] += y_offset

            # Convert to list (flatten to store as 8 points)
            return rotated_points.flatten().tolist()

        def update_annotations(data):
            """Recursively updates bounding boxes in the annotations."""
            if isinstance(data, dict):
                for key, value in data.items():
                    if isinstance(value, dict):
                        if "bounding_box" in value:
                            value["bounding_box"] = rotate_bbox(value["bounding_box"])
                        else:
                            update_annotations(value)
                    elif isinstance(value, list):
                        for item in value:
                            if isinstance(item, dict) and "bounding_box" in item:
                                item["bounding_box"] = rotate_bbox(item["bounding_box"])
                            elif isinstance(item, dict):
                                for key2, value2 in item.items():
                                    if isinstance(value2, list):
                                        bbox_with_2_points = [value2[0], value2[1], value2[4], value2[5]]
                                        item[key2] = rotate_bbox(bbox_with_2_points)


        # Apply transformations to bounding boxes
        update_annotations(ner_annotations)

        return rotated_img, ner_annotations  # Return updated image and annotations`

    def draw_checkbox_text_pairs(self):
        total_checkboxes_drawn = 0
        max_attempts = 50  # Limit the number of retries to prevent infinite loops
        
        while total_checkboxes_drawn == 0 and max_attempts > 0:
            # Fetch a random image
            image = self.generate_scanned_document()
            metadata = self.get_random_metadata()
            font = metadata.get("font", ImageFont.load_default())
            number_of_pairs = random.choice([2, 3, 4])
            json_data = {"checkboxes": []}
            total_checkboxes_drawn = 0  # Reset count for this attempt

            if image:
                draw = ImageDraw.Draw(image)
                width, height = image.size

                # Define search window dimensions based on image size
                window_width = int(0.25 * width)
                window_height = floor(0.13 * height)
                
                for _ in range(number_of_pairs):
                    # Find an empty region in the image
                    x, y = self.find_empty_region(image, window_width, window_height)
                    if x is None and y is None:
                        break

                    # Define checkbox size and color
                    checkbox_size = metadata.get("checkbox_text_size", (20, 10))[0]
                    checkbox_color = metadata.get("text_color", (60, 60, 60))
                    text_color = metadata.get("text_color", (60, 60, 60))

                    text = self.generate_random_words()

                    # Draw checkbox
                    checkbox_coords = [x, y, x + checkbox_size, y + checkbox_size]
                    shape_drawn = ""
                    if random.choice([True, False]):
                        draw.rectangle(checkbox_coords, outline=checkbox_color, width=metadata.get("checkbox_stroke", 2))
                        shape_drawn = "rectangle"
                    else:
                        draw.ellipse(checkbox_coords, outline=checkbox_color, width=metadata.get("checkbox_stroke", 2))
                        shape_drawn = "circle"
                    

                    # Draw tick or cross within the checkbox
                    checkbox_filled = self.draw_random_checkbox(draw, x, y, checkbox_size, checkbox_color, shape_drawn)

                    text_x, text_y, text_width, text_height = self.put_text_randomly(draw, x, y, checkbox_size, text, font, text_color, width, height, metadata.get("padding", 10))

                    # Save annotation in JSON format
                    if checkbox_filled:
                        json_data["checkboxes"].append(
                            {
                                "checkbox_boundingBox": [x, y, x + checkbox_size, y, x + checkbox_size, y + checkbox_size, x, y + checkbox_size],
                                "boundingBox": [
                                    text_x, text_y,
                                    text_x + text_width, text_y,
                                    text_x + text_width, text_y + text_height,
                                    text_x, text_y + text_height
                                ],
                                "text": text
                            }
                        )
                    total_checkboxes_drawn += 1

                # If no checkboxes were drawn, retry
                if total_checkboxes_drawn == 0:
                    bt.logging.info(f"[{self.uid}] No synthetic image generated, retrying a new image.")
                    max_attempts -= 1
                    continue

                angle = random.randint(-5, 5)
                noisy_image = self.add_noise(image)
                updated_image, updated_json_data = self.transform_bounding_boxes(json_data, angle, noisy_image)

                bt.logging.info(f"[{self.uid}] Synthetic image generated successfully")
                return updated_json_data, updated_image

        bt.logging.info(f"[{self.uid}] Failed to draw any checkboxes after multiple attempts.")
        return None, None
# ---------------------------------------------------------------------------------------------------------------------------------


if __name__ == "__main__":
    unique_id = str(uuid.uuid4())
    class_object = GenerateCheckboxTextPair("", unique_id)
    json_metadata, generated_doc = class_object.draw_checkbox_text_pairs()

    # Define filenames
    base_filename = f"checkbox_{unique_id}"
    image_filename = f"{base_filename}.png"
    json_filename = f"{base_filename}.json"

    # Save image
    cv2.imwrite(image_filename, np.array(generated_doc))

    # Save JSON metadata
    with open(json_filename, "w") as json_file:
        json.dump(json_metadata, json_file, indent=4)

    print(f"Saved: {image_filename} and {json_filename}")