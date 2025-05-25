import random
import numpy as np
import os
import json
import uuid
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance
from math import floor, sin, cos, radians
import textwrap
import cv2
import logging
from faker import Faker
import io

class DocumentWithEncircledTextGenerator:
    def __init__(self, uid=None, 
                 fonts_dir="fonts", 
                 background_dir="backgrounds"):
        """
        Initialize the document generator with encircled text.
        
        Args:
            uid: Unique identifier for logging
            fonts_dir: Directory containing font files
            background_dir: Directory containing background textures
        """
        self.uid = uid if uid else str(random.randint(1000, 9999))
        self.fonts_dir = fonts_dir
        self.background_dir = background_dir
        self.ensure_directories()
        
        # Available fonts list
        self.available_fonts = self._get_available_fonts()
        logging.info(f"[{self.uid}] Initialized with {len(self.available_fonts)} fonts")
        
        # Initialize Faker for text generation
        self.faker = Faker()

    
    def ensure_directories(self):
        """Create necessary directories if they don't exist."""
        for directory in [self.fonts_dir, self.background_dir]:
            os.makedirs(directory, exist_ok=True)
    
    def _get_available_fonts(self):
        """Get list of available font files."""
        if not os.path.exists(self.fonts_dir):
            # If no fonts dir, return default font
            return [None]
        
        fonts = []
        for font_file in os.listdir(self.fonts_dir):
            if font_file.endswith(('.ttf', '.otf')):
                font_path = os.path.join(self.fonts_dir, font_file)
                fonts.append(font_path)
        
        # Always include default font
        if not fonts:
            fonts = [None]
        
        return fonts
    
    def get_random_metadata(self):
        """
        Generate random metadata for document styling.
        """
        font_path = random.choice(self.available_fonts)
        font_size = random.randint(16, 30)
        
        # Try to load custom font, fall back to default
        try:
            if font_path:
                font = ImageFont.truetype(font_path, font_size)
            else:
                font = ImageFont.load_default()
        except Exception as e:
            logging.warning(f"[{self.uid}] Font loading error: {e}. Using default font.")
            font = ImageFont.load_default()
        
        # Color variations
        text_color = (
            random.randint(10, 80),  # Dark text for readability
            random.randint(10, 80),
            random.randint(10, 80)
        )
        
        circle_color = (
            random.randint(170, 255) if random.random() < 0.5 else random.randint(0, 100),  # Often red or blue
            random.randint(0, 100),
            random.randint(170, 255) if random.random() > 0.5 else random.randint(0, 100)
        )
        
        # Circle properties
        circle_width = random.randint(2, 5)
        padding = random.randint(8, 15)
        
        return {
            "font": font,
            "font_size": font_size,
            "text_color": text_color,
            "circle_color": circle_color,
            "circle_width": circle_width,
            "padding": padding
        }
    
    def generate_random_text(self):
        """Generate random text for the document using Faker."""
        # Choose a random number of paragraphs
        num_paragraphs = random.randint(3, 7)
        paragraphs = []
        
        for _ in range(num_paragraphs):
            # Use Faker to generate random paragraphs
            paragraph_type = random.choice(['text', 'sentence', 'paragraph'])
            
            if paragraph_type == 'text':
                paragraph = self.faker.text(max_nb_chars=random.randint(150, 300))
            elif paragraph_type == 'sentence':
                # Generate a paragraph of 3-8 sentences
                num_sentences = random.randint(3, 8)
                paragraph = " ".join(self.faker.sentences(nb=num_sentences))
            else:
                paragraph = self.faker.paragraph(nb_sentences=random.randint(3, 8))
                
            paragraphs.append(paragraph.strip())
        
        return paragraphs
    
    def generate_random_words(self, num_words=None):
        """Generate random words for text using Faker."""
        if num_words is None:
            num_words = random.randint(3, 8)
            
        # Use Faker to generate random words
        return " ".join(self.faker.words(nb=num_words))
    
    def generate_scanned_document(self):
        """
        Generate a document that looks like a scanned paper.
        """
        # Document dimensions - standard A4 at 150 DPI
        width = 1240
        height = 1754
        
        # Create a blank white image
        image = Image.new('RGB', (width, height), (255, 255, 255))
        
        # Add a subtle texture to simulate paper
        self.add_paper_texture(image)
        
        return image
    
    def add_paper_texture(self, image):
        """Add texture to make the image look like paper."""
        # Create noise
        noise = np.random.randint(245, 256, image.size[::-1], dtype=np.uint8)
        noise_img = Image.fromarray(noise, mode='L')
        noise_img = noise_img.convert('RGB')
        
        # Blend the noise with the original image
        alpha = random.uniform(0.92, 0.98)  # Subtle effect
        blended = Image.blend(image, noise_img, 1 - alpha)
        
        # Copy the blended image back to the original
        image.paste(blended)
        
        # Simulate paper edges - slightly darker around edges
        draw = ImageDraw.Draw(image)
        width, height = image.size
        
        # Add vignette effect
        for i in range(15):
            outline_color = (250 - i, 250 - i, 250 - i)
            draw.rectangle(
                [(i, i), (width - i, height - i)],
                outline=outline_color
            )
            
        return image
    
    def get_word_groups_from_line(self, line, draw, font, x_pos, y_pos):
        """
        Split a line into word groups and return their positions and dimensions.
        Each group contains 2-5 words.
        """
        words = line.split()
        if not words:
            return []
        
        groups = []
        current_group = []
        
        # Group words together (2-5 words per group)
        for word in words:
            current_group.append(word)
            if len(current_group) >= random.randint(2, 5) or word == words[-1]:
                if current_group:
                    groups.append(current_group)
                    current_group = []
        
        # If there's a remaining group
        if current_group:
            groups.append(current_group)
        
        # Calculate positions for each word group
        word_groups = []
        current_x = x_pos
        
        for group in groups:
            group_text = " ".join(group)
            
            # Get text dimensions
            text_bbox = draw.textbbox((current_x, y_pos), group_text, font=font)
            
            word_groups.append({
                "text": group_text,
                "bbox": text_bbox,
                "position": (current_x, y_pos)
            })
            
            # Move to next group position (add space after this group)
            group_width = text_bbox[2] - text_bbox[0]
            current_x += group_width + draw.textlength(" ", font=font)
        
        return word_groups
    
    def check_circle_bounds(self, center_x, center_y, radius_x, radius_y, image_width, image_height):
        """
        Check if a circle with given center and radii would fit within the image bounds.
        Returns True if the circle fits, False otherwise.
        """
        margin = 20  # Safety margin from edges
        
        if (center_x - radius_x < margin or 
            center_x + radius_x > image_width - margin or
            center_y - radius_y < margin or 
            center_y + radius_y > image_height - margin):
            return False
        
        return True
    
    def calculate_safe_circle_size(self, text_bbox, image_width, image_height):
        """
        Calculate a safe circle size that won't go outside the page boundaries.
        Returns adjusted radius values.
        """
        x1, y1, x2, y2 = text_bbox
        width = x2 - x1
        height = y2 - y1
        
        center_x = x1 + width / 2
        center_y = y1 + height / 2
        
        # Calculate minimum padding needed to contain text
        min_padding_x = max(15, width * 0.25)
        min_padding_y = max(15, height * 0.25)
        
        # Calculate desired radius
        desired_radius_x = width / 2 + min_padding_x
        desired_radius_y = height / 2 + min_padding_y
        
        # Check bounds and adjust if necessary
        margin = 25  # Safety margin from page edges
        
        # Maximum allowed radius based on position and page bounds
        max_radius_x = min(
            center_x - margin,  # Distance to left edge
            image_width - center_x - margin,  # Distance to right edge
            desired_radius_x
        )
        
        max_radius_y = min(
            center_y - margin,  # Distance to top edge
            image_height - center_y - margin,  # Distance to bottom edge
            desired_radius_y
        )
        
        # Ensure we don't make the circle too small
        final_radius_x = max(width / 2 + 10, max_radius_x)
        final_radius_y = max(height / 2 + 10, max_radius_y)
        
        return final_radius_x, final_radius_y
    
    def check_circle_overlap_with_text(self, center_x, center_y, radius_x, radius_y, text_regions, target_text_bbox):
        """
        Check if a circle would overlap with any other text regions (excluding the target text).
        Returns True if there's overlap, False otherwise.
        """
        # Create circle bounding box with some buffer
        circle_bbox = [
            center_x - radius_x - 10,
            center_y - radius_y - 10,
            center_x + radius_x + 10,
            center_y + radius_y + 10
        ]
        
        for text_info in text_regions:
            text_bbox = text_info["region"]
            
            # Skip if this is the target text we're trying to circle
            if text_bbox == target_text_bbox:
                continue
            
            # Check for overlap with other text
            if not (circle_bbox[2] < text_bbox[0] or circle_bbox[0] > text_bbox[2] or
                    circle_bbox[3] < text_bbox[1] or circle_bbox[1] > text_bbox[3]):
                return True  # Overlap detected
        
        return False  # No overlap
    
    def draw_human_like_circle(self, draw, center_x, center_y, radius_x, radius_y, color, line_width):
        """
        Draw a circle that looks like it was drawn by a human (imperfect).
        Uses pre-calculated center and radii to ensure proper bounds.
        """
        # Deterministic random generator for this specific circle
        rng = random.Random(f"{center_x}{center_y}{radius_x}{radius_y}")
        
        # Number of control points for the circle - more points for smoother circle
        num_points = rng.randint(40, 60)
        
        # Generate points with some random variation
        points = []
        for i in range(num_points):
            angle = 2 * np.pi * i / num_points
            
            # Reduced wobble for smoother appearance
            wobble = rng.uniform(0.95, 1.05)
            
            # Reduced angle wobble for smoother appearance
            angle_wobble = rng.uniform(-0.02, 0.02)
            angle += angle_wobble
            
            # Calculate point position
            px = center_x + radius_x * wobble * np.cos(angle)
            py = center_y + radius_y * wobble * np.sin(angle)
            
            points.append((px, py))
        
        # Close the circle
        points.append(points[0])
        
        # Helper function to draw a smooth curve through points
        def draw_smooth_curve(points, color, width):
            # Duplicate first and last points for the spline calculation
            expanded_points = [points[0]] + points + [points[-1]]
            
            smooth_points = []
            steps = 5  # Number of interpolation steps between points
            
            # Generate smooth curve through the points
            for i in range(1, len(expanded_points) - 2):
                p0 = expanded_points[i-1]
                p1 = expanded_points[i]
                p2 = expanded_points[i+1]
                p3 = expanded_points[i+2]
                
                # Add original point
                smooth_points.append(p1)
                
                # Add interpolated points
                for t in range(1, steps):
                    t = t / steps
                    
                    # Catmull-Rom interpolation
                    t2 = t * t
                    t3 = t2 * t
                    
                    q_x = 0.5 * ((2 * p1[0]) +
                                 (-p0[0] + p2[0]) * t +
                                 (2*p0[0] - 5*p1[0] + 4*p2[0] - p3[0]) * t2 +
                                 (-p0[0] + 3*p1[0] - 3*p2[0] + p3[0]) * t3)
                    
                    q_y = 0.5 * ((2 * p1[1]) +
                                 (-p0[1] + p2[1]) * t +
                                 (2*p0[1] - 5*p1[1] + 4*p2[1] - p3[1]) * t2 +
                                 (-p0[1] + 3*p1[1] - 3*p2[1] + p3[1]) * t3)
                    
                    smooth_points.append((q_x, q_y))
            
            # Draw line segments between all the smooth points
            for i in range(len(smooth_points) - 1):
                draw.line([smooth_points[i], smooth_points[i+1]], fill=color, width=width)
        
        # Draw the smooth circle
        draw_smooth_curve(points, color, line_width)
        
        # Return the bounding box of the circle
        return [
            center_x - radius_x - 10,
            center_y - radius_y - 10,
            center_x + radius_x + 10,
            center_y + radius_y + 10
        ]
    
    def add_noise(self, image):
        """Add realistic scan noise to the image."""
        # Convert to numpy array for easier manipulation
        img_array = np.array(image)
        
        # Add random noise
        noise_factor = random.uniform(3, 8)
        noise = np.random.normal(0, noise_factor, img_array.shape)
        noisy_img = np.clip(img_array + noise, 0, 255).astype(np.uint8)
        
        # Convert back to PIL Image
        noisy_image = Image.fromarray(noisy_img)
        
        # Apply a slight blur to simulate scanner/copier
        blur_radius = random.uniform(0.3, 0.7)
        noisy_image = noisy_image.filter(ImageFilter.GaussianBlur(radius=blur_radius))
        
        # Adjust contrast and brightness
        contrast = ImageEnhance.Contrast(noisy_image)
        contrast_factor = random.uniform(0.9, 1.1)
        noisy_image = contrast.enhance(contrast_factor)
        
        brightness = ImageEnhance.Brightness(noisy_image)
        brightness_factor = random.uniform(0.95, 1.05)
        noisy_image = brightness.enhance(brightness_factor)
        
        # Add JPEG compression artifacts
        quality = random.randint(85, 95)
        output = io.BytesIO()
        noisy_image.save(output, format='JPEG', quality=quality)
        noisy_image = Image.open(output)
        
        return noisy_image
    
    def transform_bounding_boxes(self, json_data, angle, image):
        """
        Transform bounding boxes when rotating the image.
        """
        if angle == 0:
            return image, json_data
            
        # Get image dimensions
        width, height = image.size
        
        # Rotate the image
        rotated_image = image.rotate(angle, expand=True, resample=Image.BICUBIC, fillcolor=(255, 255, 255))
        new_width, new_height = rotated_image.size
        
        # Calculate the translation required after rotation
        translate_x = (new_width - width) / 2
        translate_y = (new_height - height) / 2
        
        # Update each bounding box
        for encircle in json_data.get("encircles", []):
            new_box = []
            for i in range(0, len(encircle["boundingBox"]), 2):
                x = encircle["boundingBox"][i]
                y = encircle["boundingBox"][i+1]
                
                # Calculate rotation around the center of the original image
                angle_rad = radians(-angle)
                cx, cy = width / 2, height / 2
                
                # Translate to origin, rotate, translate back, and add the expansion offset
                new_x = (x - cx) * cos(angle_rad) - (y - cy) * sin(angle_rad) + cx + translate_x
                new_y = (x - cx) * sin(angle_rad) + (y - cy) * cos(angle_rad) + cy + translate_y
                
                new_box.extend([new_x, new_y])
                
            encircle["boundingBox"] = new_box
            
        return rotated_image, json_data
    
    def draw_encircled_text(self):
        """
        Draw text and encircle some word groups randomly.
        """
        # Get a fresh document image
        image = self.generate_scanned_document()
        
        if not image:
            logging.error(f"[{self.uid}] Failed to generate document image")
            return None, None
            
        # Get random styling metadata
        metadata = self.get_random_metadata()
        font = metadata.get("font", ImageFont.load_default())
        
        # Generate paragraphs of text
        paragraphs = self.generate_random_text()
        
        # Create a drawing object
        draw = ImageDraw.Draw(image)
        
        # Get image dimensions
        img_width, img_height = image.size
        
        # Top margin
        y_pos = random.randint(50, 100)
        
        # Track all used regions to avoid overlap
        used_regions = []
        
        # Track all word groups for possible encircling
        all_word_groups = []
        
        # Place paragraphs
        for paragraph in paragraphs:
            # Left margin with some randomness
            x_pos = random.randint(50, 100)
            
            # Break paragraph into lines that fit the page width
            max_width = img_width - x_pos - 50
            font_size = metadata["font_size"]
            
            # Calculate average character width for this font
            avg_char_width = draw.textlength("x", font=font)
            
            # Estimate max chars per line
            chars_per_line = int(max_width / avg_char_width)
            
            # Wrap text
            lines = textwrap.wrap(paragraph, width=chars_per_line)
            
            for line in lines:
                # Skip if we're getting too close to the bottom
                if y_pos > img_height - 100:
                    continue
                
                # Get word groups from this line
                word_groups = self.get_word_groups_from_line(line, draw, font, x_pos, y_pos)
                
                # Check each word group for overlaps and draw them
                line_word_groups = []
                for group_info in word_groups:
                    text_bbox = group_info["bbox"]
                    
                    # Create region with buffer for overlap checking
                    buffer = 20  # Buffer to prevent text crowding
                    new_region = [
                        text_bbox[0] - buffer, 
                        text_bbox[1] - buffer, 
                        text_bbox[2] + buffer, 
                        text_bbox[3] + buffer
                    ]
                    
                    # Check for overlaps and ensure text stays within page boundaries
                    overlaps = False
                    if (new_region[0] < 30 or new_region[2] > img_width - 30 or 
                        new_region[1] < 30 or new_region[3] > img_height - 30):
                        overlaps = True
                    else:
                        for region in used_regions:
                            if not (new_region[2] < region[0] or new_region[0] > region[2] or
                                    new_region[3] < region[1] or new_region[1] > region[3]):
                                overlaps = True
                                break
                    
                    # Skip this word group if overlap detected
                    if overlaps:
                        continue
                    
                    # Add the region to the used regions list
                    used_regions.append(new_region)
                    
                    # Store word group for possible encircling
                    word_group_data = {
                        "region": list(text_bbox),  # [x1, y1, x2, y2]
                        "text": group_info["text"],
                        "position": group_info["position"]
                    }
                    line_word_groups.append(word_group_data)
                    all_word_groups.append(word_group_data)
                    
                    # Draw the word group
                    draw.text(group_info["position"], group_info["text"], 
                             fill=metadata["text_color"], font=font)
                
                # Move to next line with proper spacing
                if line_word_groups:  # Only advance if we drew something
                    line_height = max([group["region"][3] - group["region"][1] for group in line_word_groups])
                    y_pos += line_height + random.randint(3, 8)
            
            # Add paragraph spacing
            y_pos += random.randint(15, 25)
        
        # Now encircle some random word groups
        json_data = {"encircles": []}
        num_groups = len(all_word_groups)
        
        # Determine how many word groups to encircle (1 to 4, or max available)
        num_to_encircle = min(random.randint(1, 4), num_groups)
        
        # Randomly select which word groups to encircle
        if num_groups > 0:
            indices_to_encircle = random.sample(range(num_groups), num_to_encircle)
            
            # Track circle regions to prevent overlap
            used_circle_regions = []
            
            for idx in indices_to_encircle:
                group_info = all_word_groups[idx]
                text_bbox = group_info["region"]
                text = group_info["text"]
                
                # Calculate circle center and safe radius
                x1, y1, x2, y2 = text_bbox
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                
                # Calculate safe circle size that won't go outside page
                radius_x, radius_y = self.calculate_safe_circle_size(text_bbox, img_width, img_height)
                
                # Check if circle would overlap with other text
                if self.check_circle_overlap_with_text(center_x, center_y, radius_x, radius_y, 
                                                     all_word_groups, text_bbox):
                    continue  # Skip this word group if it would overlap other text
                
                # Check if circle overlaps with existing circles
                circle_bbox = [
                    center_x - radius_x - 10,
                    center_y - radius_y - 10,
                    center_x + radius_x + 10,
                    center_y + radius_y + 10
                ]
                
                overlaps_circle = False
                for region in used_circle_regions:
                    if not (circle_bbox[2] < region[0] or circle_bbox[0] > region[2] or
                            circle_bbox[3] < region[1] or circle_bbox[1] > region[3]):
                        overlaps_circle = True
                        break
                
                if overlaps_circle:
                    continue  # Skip if overlaps with existing circle
                
                # Draw the circle
                circle_color = metadata["circle_color"]
                circle_width = metadata["circle_width"]
                
                final_circle_bbox = self.draw_human_like_circle(
                    draw, center_x, center_y, radius_x, radius_y, circle_color, circle_width
                )
                
                # Add to used circle regions
                used_circle_regions.append(final_circle_bbox)
                
                # Save the annotation with rectangular bounding box format for ground truth
                json_data["encircles"].append({
                    "boundingBox": [
                        text_bbox[0], text_bbox[1],  # Top-left
                        text_bbox[2], text_bbox[1],  # Top-right
                        text_bbox[2], text_bbox[3],  # Bottom-right
                        text_bbox[0], text_bbox[3]   # Bottom-left
                    ],
                    "text": text,
                    "encircled": True
                })
        
        # Apply random rotation (slight)
        angle = random.randint(-5, 5)
        
        # Add noise and distortion
        noisy_image = self.add_noise(image)
        
        # Apply rotation and update bounding boxes
        final_image, updated_json_data = self.transform_bounding_boxes(json_data, angle, noisy_image)
        
        logging.info(f"[{self.uid}] Generated document with {len(updated_json_data['encircles'])} encircled word groups")
        
        return updated_json_data, final_image
    

class DocumentWithEncircledLineGenerator:
    def __init__(self, uid=None, 
                 fonts_dir="fonts", 
                 background_dir="backgrounds"):
        """
        Initialize the document generator with encircled text.
        
        Args:
            uid: Unique identifier for logging
            fonts_dir: Directory containing font files
            background_dir: Directory containing background textures
        """
        self.uid = uid if uid else str(random.randint(1000, 9999))
        self.fonts_dir = fonts_dir
        self.background_dir = background_dir
        self.ensure_directories()
        
        # Available fonts list
        self.available_fonts = self._get_available_fonts()
        logging.info(f"[{self.uid}] Initialized with {len(self.available_fonts)} fonts")
        
        # Initialize Faker for text generation
        self.faker = Faker()

    
    def ensure_directories(self):
        """Create necessary directories if they don't exist."""
        for directory in [self.fonts_dir, self.background_dir]:
            os.makedirs(directory, exist_ok=True)
    
    def _get_available_fonts(self):
        """Get list of available font files."""
        if not os.path.exists(self.fonts_dir):
            # If no fonts dir, return default font
            return [None]
        
        fonts = []
        for font_file in os.listdir(self.fonts_dir):
            if font_file.endswith(('.ttf', '.otf')):
                font_path = os.path.join(self.fonts_dir, font_file)
                fonts.append(font_path)
        
        # Always include default font
        if not fonts:
            fonts = [None]
        
        return fonts
    
    def get_random_metadata(self):
        """
        Generate random metadata for document styling.
        """
        font_path = random.choice(self.available_fonts)
        font_size = random.randint(16, 30)
        
        # Try to load custom font, fall back to default
        try:
            if font_path:
                font = ImageFont.truetype(font_path, font_size)
            else:
                font = ImageFont.load_default()
        except Exception as e:
            logging.warning(f"[{self.uid}] Font loading error: {e}. Using default font.")
            font = ImageFont.load_default()
        
        # Color variations
        text_color = (
            random.randint(10, 80),  # Dark text for readability
            random.randint(10, 80),
            random.randint(10, 80)
        )
        
        circle_color = (
            random.randint(170, 255) if random.random() < 0.5 else random.randint(0, 100),  # Often red or blue
            random.randint(0, 100),
            random.randint(170, 255) if random.random() > 0.5 else random.randint(0, 100)
        )
        
        # Circle properties
        circle_width = random.randint(2, 5)
        padding = random.randint(8, 15)
        
        return {
            "font": font,
            "font_size": font_size,
            "text_color": text_color,
            "circle_color": circle_color,
            "circle_width": circle_width,
            "padding": padding
        }
    
    def generate_random_text(self):
        """Generate random text for the document using Faker."""
        # Choose a random number of paragraphs
        num_paragraphs = random.randint(3, 7)
        paragraphs = []
        
        for _ in range(num_paragraphs):
            # Use Faker to generate random paragraphs
            paragraph_type = random.choice(['text', 'sentence', 'paragraph'])
            
            if paragraph_type == 'text':
                paragraph = self.faker.text(max_nb_chars=random.randint(150, 300))
            elif paragraph_type == 'sentence':
                # Generate a paragraph of 3-8 sentences
                num_sentences = random.randint(3, 8)
                paragraph = " ".join(self.faker.sentences(nb=num_sentences))
            else:
                paragraph = self.faker.paragraph(nb_sentences=random.randint(3, 8))
                
            paragraphs.append(paragraph.strip())
        
        return paragraphs
    
    def generate_random_words(self, num_words=None):
        """Generate random words for text using Faker."""
        if num_words is None:
            num_words = random.randint(3, 8)
            
        # Use Faker to generate random words
        return " ".join(self.faker.words(nb=num_words))
    
    def generate_scanned_document(self):
        """
        Generate a document that looks like a scanned paper.
        """
        # Document dimensions - standard A4 at 150 DPI
        width = 1240
        height = 1754
        
        # Create a blank white image
        image = Image.new('RGB', (width, height), (255, 255, 255))
        
        # Add a subtle texture to simulate paper
        self.add_paper_texture(image)
        
        return image
    
    def add_paper_texture(self, image):
        """Add texture to make the image look like paper."""
        # Create noise
        noise = np.random.randint(245, 256, image.size[::-1], dtype=np.uint8)
        noise_img = Image.fromarray(noise, mode='L')
        noise_img = noise_img.convert('RGB')
        
        # Blend the noise with the original image
        alpha = random.uniform(0.92, 0.98)  # Subtle effect
        blended = Image.blend(image, noise_img, 1 - alpha)
        
        # Copy the blended image back to the original
        image.paste(blended)
        
        # Simulate paper edges - slightly darker around edges
        draw = ImageDraw.Draw(image)
        width, height = image.size
        
        # Add vignette effect
        for i in range(15):
            outline_color = (250 - i, 250 - i, 250 - i)
            draw.rectangle(
                [(i, i), (width - i, height - i)],
                outline=outline_color
            )
            
        return image
    
    def find_empty_region(self, image, width, height, used_regions=None):
        """
        Find an empty region in the image where we can place text.
        """
        if used_regions is None:
            used_regions = []
            
        img_width, img_height = image.size
        
        # Define a grid to search for empty spots
        grid_step = max(20, min(width, height) // 5)
        
        # Try up to 50 random positions to ensure we find a good spot
        for _ in range(50):
            # Choose a random position on the grid with proper margins
            # Ensure we stay well within page boundaries
            x = random.randint(50, img_width - width - 50)
            y = random.randint(50, img_height - height - 50)
            
            # Create candidate region with buffer
            buffer = 30  # Increased buffer to avoid text getting too close
            candidate_region = [x - buffer, y - buffer, x + width + buffer, y + height + buffer]
            
            # Check if this region overlaps with any used regions
            overlaps = False
            for region in used_regions:
                # Check for overlap
                if not (candidate_region[2] < region[0] or candidate_region[0] > region[2] or
                        candidate_region[3] < region[1] or candidate_region[1] > region[3]):
                    overlaps = True
                    break
                    
            if not overlaps:
                return x, y
                
        # If we couldn't find an empty spot, return None
        return None, None
    
    def draw_human_like_circle(self, draw, x, y, width, height, color, line_width, used_circle_regions=None, image_width=None, image_height=None):
        """
        Draw a circle that looks like it was drawn by a human (imperfect).
        Ensures that the circle fully contains the text without cutting through it.
        Returns the circle's bounding box and if it overlaps with existing circles.
        """
        if used_circle_regions is None:
            used_circle_regions = []
            
        # Deterministic random generator for this specific circle
        rng = random.Random(f"{x}{y}{width}{height}")
        
        # Number of control points for the circle - more points for smoother circle
        num_points = rng.randint(40, 60)  # Increased from 20-30 to 40-60 for smoothness
        
        # Calculate center and radius
        center_x = x + width / 2
        center_y = y + height / 2
        
        # Make the circle larger than the text bounding box to fully contain it
        # Use a minimum padding to ensure text is fully contained
        min_padding_x = max(15, width * 0.2)  # At least 15px or 20% of width
        min_padding_y = max(15, height * 0.2)  # At least 15px or 20% of height
        
        radius_x = width / 2 + rng.randint(int(min_padding_x), int(min_padding_x * 1.5))
        radius_y = height / 2 + rng.randint(int(min_padding_y), int(min_padding_y * 1.5))
        
        # ADDED: Check if circle would go outside page bounds and adjust if necessary
        if image_width and image_height:
            margin = 25  # Safety margin from page edges
            
            # Calculate maximum allowed radius based on position and page bounds
            max_radius_x = min(
                center_x - margin,  # Distance to left edge
                image_width - center_x - margin,  # Distance to right edge
                radius_x
            )
            
            max_radius_y = min(
                center_y - margin,  # Distance to top edge
                image_height - center_y - margin,  # Distance to bottom edge
                radius_y
            )
            
            # Use the smaller of desired or maximum allowed radius
            radius_x = max(width / 2 + 10, max_radius_x)  # Ensure minimum size
            radius_y = max(height / 2 + 10, max_radius_y)  # Ensure minimum size
        
        # Calculate approximate circle bounding box
        circle_bbox = [
            center_x - radius_x - 10,
            center_y - radius_y - 10,
            center_x + radius_x + 10,
            center_y + radius_y + 10
        ]
        
        # Check for overlap with existing circles
        for region in used_circle_regions:
            # If circles overlap, return None
            if not (circle_bbox[2] < region[0] or circle_bbox[0] > region[2] or
                    circle_bbox[3] < region[1] or circle_bbox[1] > region[3]):
                return None
        
        # Generate points with some random variation - but limit the variation 
        # to ensure the circle is smooth and fully contains the text
        points = []
        for i in range(num_points):
            angle = 2 * np.pi * i / num_points
            
            # Reduced wobble for smoother appearance
            # Now the wobble is within 5% instead of 15%
            wobble = rng.uniform(0.95, 1.05)
            
            # Reduced angle wobble for smoother appearance
            angle_wobble = rng.uniform(-0.02, 0.02)  # Reduced from ±0.05 to ±0.02
            angle += angle_wobble
            
            # Calculate point position
            px = center_x + radius_x * wobble * np.cos(angle)
            py = center_y + radius_y * wobble * np.sin(angle)
            
            points.append((px, py))
        
        # Close the circle
        points.append(points[0])
        
        # Draw the circle with the specified line width
        # Use a Catmull-Rom spline to create a smoother curve
        
        # Helper function to draw a smooth curve through points
        def draw_smooth_curve(points, color, width):
            # We'll use a simplified approach for smooth curves
            # by drawing short line segments between interpolated points
            
            # Duplicate first and last points for the spline calculation
            expanded_points = [points[0]] + points + [points[-1]]
            
            smooth_points = []
            steps = 5  # Number of interpolation steps between points
            
            # Generate smooth curve through the points
            for i in range(1, len(expanded_points) - 2):
                p0 = expanded_points[i-1]
                p1 = expanded_points[i]
                p2 = expanded_points[i+1]
                p3 = expanded_points[i+2]
                
                # Add original point
                smooth_points.append(p1)
                
                # Add interpolated points
                for t in range(1, steps):
                    t = t / steps
                    
                    # Catmull-Rom interpolation
                    t2 = t * t
                    t3 = t2 * t
                    
                    q_x = 0.5 * ((2 * p1[0]) +
                                 (-p0[0] + p2[0]) * t +
                                 (2*p0[0] - 5*p1[0] + 4*p2[0] - p3[0]) * t2 +
                                 (-p0[0] + 3*p1[0] - 3*p2[0] + p3[0]) * t3)
                    
                    q_y = 0.5 * ((2 * p1[1]) +
                                 (-p0[1] + p2[1]) * t +
                                 (2*p0[1] - 5*p1[1] + 4*p2[1] - p3[1]) * t2 +
                                 (-p0[1] + 3*p1[1] - 3*p2[1] + p3[1]) * t3)
                    
                    smooth_points.append((q_x, q_y))
            
            # Draw line segments between all the smooth points
            for i in range(len(smooth_points) - 1):
                draw.line([smooth_points[i], smooth_points[i+1]], fill=color, width=width)
        
        # Draw the smooth circle
        draw_smooth_curve(points, color, line_width)
        
        # Return the approximate bounding box of the circle
        return circle_bbox
    
    def add_noise(self, image):
        """Add realistic scan noise to the image."""
        # Convert to numpy array for easier manipulation
        img_array = np.array(image)
        
        # Add random noise
        noise_factor = random.uniform(3, 8)
        noise = np.random.normal(0, noise_factor, img_array.shape)
        noisy_img = np.clip(img_array + noise, 0, 255).astype(np.uint8)
        
        # Convert back to PIL Image
        noisy_image = Image.fromarray(noisy_img)
        
        # Apply a slight blur to simulate scanner/copier
        blur_radius = random.uniform(0.3, 0.7)
        noisy_image = noisy_image.filter(ImageFilter.GaussianBlur(radius=blur_radius))
        
        # Adjust contrast and brightness
        contrast = ImageEnhance.Contrast(noisy_image)
        contrast_factor = random.uniform(0.9, 1.1)
        noisy_image = contrast.enhance(contrast_factor)
        
        brightness = ImageEnhance.Brightness(noisy_image)
        brightness_factor = random.uniform(0.95, 1.05)
        noisy_image = brightness.enhance(brightness_factor)
        
        # Add JPEG compression artifacts
        quality = random.randint(85, 95)
        output = io.BytesIO()
        noisy_image.save(output, format='JPEG', quality=quality)
        noisy_image = Image.open(output)
        
        return noisy_image
    
    def transform_bounding_boxes(self, json_data, angle, image):
        """
        Transform bounding boxes when rotating the image.
        """
        if angle == 0:
            return image, json_data
            
        # Get image dimensions
        width, height = image.size
        
        # Rotate the image
        rotated_image = image.rotate(angle, expand=True, resample=Image.BICUBIC, fillcolor=(255, 255, 255))
        new_width, new_height = rotated_image.size
        
        # Calculate the translation required after rotation
        translate_x = (new_width - width) / 2
        translate_y = (new_height - height) / 2
        
        # Update each bounding box
        for encircle in json_data.get("encircles", []):
            new_box = []
            for i in range(0, len(encircle["boundingBox"]), 2):
                x = encircle["boundingBox"][i]
                y = encircle["boundingBox"][i+1]
                
                # Calculate rotation around the center of the original image
                angle_rad = radians(-angle)
                cx, cy = width / 2, height / 2
                
                # Translate to origin, rotate, translate back, and add the expansion offset
                new_x = (x - cx) * cos(angle_rad) - (y - cy) * sin(angle_rad) + cx + translate_x
                new_y = (x - cx) * sin(angle_rad) + (y - cy) * cos(angle_rad) + cy + translate_y
                
                new_box.extend([new_x, new_y])
                
            encircle["boundingBox"] = new_box
            
        return rotated_image, json_data
    
    def draw_encircled_text(self):
        """
        Draw text and encircle some of them randomly.
        """
        # Get a fresh document image
        image = self.generate_scanned_document()
        
        if not image:
            logging.error(f"[{self.uid}] Failed to generate document image")
            return None, None
            
        # Get random styling metadata
        metadata = self.get_random_metadata()
        font = metadata.get("font", ImageFont.load_default())
        
        # Generate paragraphs of text
        paragraphs = self.generate_random_text()
        
        # Create a drawing object
        draw = ImageDraw.Draw(image)
        
        # Get image dimensions
        width, height = image.size
        
        # Top margin
        y_pos = random.randint(50, 100)
        
        # Track all used regions to avoid overlap
        used_regions = []
        
        # Track all text regions for possible encircling
        text_regions = []
        
        # Place paragraphs
        for paragraph in paragraphs:
            # Left margin with some randomness
            x_pos = random.randint(50, 100)
            
            # Break paragraph into lines that fit the page width
            max_width = width - x_pos - 50
            font_size = metadata["font_size"]
            
            # Calculate average character width for this font
            avg_char_width = draw.textlength("x", font=font)
            
            # Estimate max chars per line
            chars_per_line = int(max_width / avg_char_width)
            
            # Wrap text
            lines = textwrap.wrap(paragraph, width=chars_per_line)
            
            for line in lines:
                # Skip if we're getting too close to the bottom
                if y_pos > height - 100:
                    continue
                    
                # Check line width
                line_width = draw.textlength(line, font=font)
                
                # Get text bounding box
                text_bbox = draw.textbbox((x_pos, y_pos), line, font=font)
                text_width = text_bbox[2] - text_bbox[0]
                text_height = text_bbox[3] - text_bbox[1]
                
                # Check if this text region would overlap with existing ones
                new_region = [text_bbox[0], text_bbox[1], text_bbox[2], text_bbox[3]]
                
                # Check for overlaps and ensure text stays within page boundaries
                overlaps = False
                if (new_region[0] < 30 or new_region[2] > width - 30 or 
                    new_region[1] < 30 or new_region[3] > height - 30):
                    overlaps = True
                else:
                    for region in used_regions:
                        if not (new_region[2] < region[0] or new_region[0] > region[2] or
                                new_region[3] < region[1] or new_region[1] > region[3]):
                            overlaps = True
                            break
                
                # Skip this line if overlap detected
                if overlaps:
                    continue
                
                # Add the region to the used regions list
                used_regions.append(new_region)
                
                # Store text region for possible encircling
                text_regions.append({
                    "region": new_region,
                    "text": line
                })
                
                # Draw the text
                draw.text((x_pos, y_pos), line, fill=metadata["text_color"], font=font)
                
                # MODIFIED: Increase line spacing to prevent circles from overlapping nearby text
                # Move to next line with increased spacing
                y_pos += text_height + random.randint(15, 25)  # Increased from (3, 8) to (15, 25)
            
            # MODIFIED: Increase paragraph spacing
            y_pos += random.randint(25, 40)  # Increased from (15, 25) to (25, 40)
        
        # Now encircle some random text portions
        json_data = {"encircles": []}
        num_texts = len(text_regions)
        
        # Determine how many text regions to encircle (1 to 4, or max available)
        num_to_encircle = min(random.randint(1, 4), num_texts)
        
        # Randomly select which text regions to encircle
        indices_to_encircle = random.sample(range(num_texts), num_to_encircle)
        
        # Track circle regions to prevent overlap
        used_circle_regions = []
        
        for idx in indices_to_encircle:
            text_info = text_regions[idx]
            region = text_info["region"]
            text = text_info["text"]
            
            x, y, x2, y2 = region
            region_width = x2 - x
            region_height = y2 - y
            
            # MODIFIED: Draw the circle with page bounds checking
            circle_color = metadata["circle_color"]
            circle_width = metadata["circle_width"]
            circle_bbox = self.draw_human_like_circle(
                draw, x, y, region_width, region_height, circle_color, circle_width, 
                used_circle_regions, width, height  # Pass image dimensions
            )
            
            # If circle couldn't be drawn due to overlap or bounds, skip this text
            if circle_bbox is None:
                continue
            
            # Add to used circle regions
            used_circle_regions.append(circle_bbox)
            
            # Save the annotation with rectangular bounding box format for ground truth
            json_data["encircles"].append({
                "boundingBox": [
                    region[0], region[1],  # Top-left
                    region[2], region[1],  # Top-right
                    region[2], region[3],  # Bottom-right
                    region[0], region[3]   # Bottom-left
                ],
                "text": text,
                "encircled": True
            })
        
        # Apply random rotation (slight)
        angle = random.randint(-5, 5)
        
        # Add noise and distortion
        noisy_image = self.add_noise(image)
        
        # Apply rotation and update bounding boxes
        final_image, updated_json_data = self.transform_bounding_boxes(json_data, angle, noisy_image)
        
        logging.info(f"[{self.uid}] Generated document with {len(updated_json_data['encircles'])} encircled text regions")
        
        return updated_json_data, final_image
    

# Example usage
if __name__ == "__main__":
    
    unique_id = str(uuid.uuid4())
    generator = random.choice([DocumentWithEncircledTextGenerator(), DocumentWithEncircledLineGenerator()])
    
    # Generate a document with encircled text
    json_metadata, generated_doc = generator.draw_encircled_text()
    
    # Define filenames
    base_filename = f"encircled_{unique_id}"
    image_filename = f"{base_filename}.png"
    json_filename = f"{base_filename}.json"

    # Save image
    cv2.imwrite(image_filename, np.array(generated_doc))

    # Save JSON metadata
    with open(json_filename, "w") as json_file:
        json.dump(json_metadata, json_file, indent=4)

    print(f"Saved: {image_filename} and {json_filename}")