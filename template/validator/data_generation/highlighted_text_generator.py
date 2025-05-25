import random
import json
from math import floor, cos, sin, radians
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance
import numpy as np
import textwrap
import lorem
import os
import io
import bittensor as bt
from typing import Tuple, List, Dict, Any, Optional
import uuid
import cv2

class HighlightedDocumentGenerator:
    def __init__(self, uid="generator"):
        self.uid = uid
        self.fonts_path = "./fonts/"
        self.backgrounds_path = "./backgrounds/"
        # Ensure these directories exist or use system fonts
        self.load_resources()
        
    def load_resources(self):
        """Load or create necessary resources for document generation."""
        # Default fallback fonts
        self.available_fonts = [
            ImageFont.truetype("arial.ttf", 14) if os.path.exists("arial.ttf") else ImageFont.load_default(),
            ImageFont.truetype("times.ttf", 14) if os.path.exists("times.ttf") else ImageFont.load_default(),
        ]
        
        # Try to load custom fonts if directory exists
        if os.path.exists(self.fonts_path):
            try:
                for font_file in os.listdir(self.fonts_path):
                    if font_file.endswith(('.ttf', '.otf')):
                        font_path = os.path.join(self.fonts_path, font_file)
                        for size in [12, 14, 16, 18]:
                            self.available_fonts.append(ImageFont.truetype(font_path, size))
            except Exception as e:
                bt.logging.error(f"[{self.uid}] Error loading fonts: {e}")
        
        # Create some basic backgrounds if none exist
        self.background_images = []
        if os.path.exists(self.backgrounds_path):
            try:
                for bg_file in os.listdir(self.backgrounds_path):
                    if bg_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        bg_path = os.path.join(self.backgrounds_path, bg_file)
                        self.background_images.append(Image.open(bg_path))
            except Exception as e:
                bt.logging.error(f"[{self.uid}] Error loading backgrounds: {e}")
        
        # If no backgrounds loaded, generate some basic ones
        if not self.background_images:
            self.background_images = [self.create_basic_background() for _ in range(5)]
    
    def create_basic_background(self, width=800, height=1100):
        """Create a basic document background with subtle texture."""
        # Create a white/off-white base
        color = random.randint(245, 255)
        img = Image.new('RGB', (width, height), (color, color, color))
        draw = ImageDraw.Draw(img)
        
        # Add subtle noise
        for _ in range(width * height // 100):
            x = random.randint(0, width - 1)
            y = random.randint(0, height - 1)
            shade = random.randint(color - 10, color)
            draw.point((x, y), fill=(shade, shade, shade))
        
        # Add subtle lines to simulate paper texture
        for _ in range(20):
            x1, y1 = random.randint(0, width), random.randint(0, height)
            x2, y2 = random.randint(0, width), random.randint(0, height)
            opacity = random.randint(color - 5, color)
            draw.line([(x1, y1), (x2, y2)], fill=(opacity, opacity, opacity), width=1)
        
        return img.filter(ImageFilter.GaussianBlur(radius=0.5))
    
    def get_random_metadata(self):
        """Generate random metadata for document styling."""
        font = random.choice(self.available_fonts)
        
        return {
            "font": font,
            "text_color": (random.randint(0, 80), random.randint(0, 80), random.randint(0, 80)),
            "line_spacing": random.uniform(1.1, 1.5),
            "paragraph_spacing": random.randint(10, 25),
            "margin_top": random.randint(50, 100),
            "margin_left": random.randint(50, 100),
            "margin_right": random.randint(50, 100),
            "line_length": random.randint(60, 80),  # characters per line
            "highlight_colors": [
                (0, 255, 100, 90),  # Green with alpha
                (255, 255, 0, 90),  # Yellow with alpha
                (255, 150, 255, 90)  # Pink with alpha
            ]
        }
    
    def generate_scanned_document(self, width=800, height=1100):
        """Generate a basic document image to work with."""
        try:
            if self.background_images and random.random() < 0.8:
                # Use a pre-existing background
                bg = random.choice(self.background_images).copy()
                # Resize if needed
                if bg.size != (width, height):
                    bg = bg.resize((width, height), Image.LANCZOS)
                return bg
            else:
                # Create a new basic background
                return self.create_basic_background(width, height)
        except Exception as e:
            bt.logging.error(f"[{self.uid}] Error generating document: {e}")
            # Fallback to simple white background
            return Image.new('RGB', (width, height), (255, 255, 255))

    def generate_random_text(self, paragraphs=3, sentences_per_paragraph=None):
        """Generate random Lorem Ipsum text for the document."""
        if sentences_per_paragraph is None:
            sentences_per_paragraph = [random.randint(3, 8) for _ in range(paragraphs)]
        
        text = []
        for i in range(paragraphs):
            try:
                if random.random() < 0.2:
                    # Sometimes use a heading-like sentence
                    paragraph = lorem.sentence().upper()
                else:
                    paragraph = ' '.join(lorem.sentence() for _ in range(sentences_per_paragraph[i]))
                text.append(paragraph)
            except Exception:
                # Fallback if lorem fails
                words = ["lorem", "ipsum", "dolor", "sit", "amet", "consectetur", 
                         "adipiscing", "elit", "sed", "do", "eiusmod", "tempor"]
                paragraph = ' '.join(random.choice(words) for _ in range(40))
                text.append(paragraph)
        
        return text
    
    def find_empty_region(self, image, width, height):
        """Find an empty region in the image where we can place a text element."""
        img_width, img_height = image.size
        max_attempts = 20
        
        for _ in range(max_attempts):
            # Try to find a position that's within margins
            x = random.randint(50, max(51, img_width - width - 50))
            y = random.randint(50, max(51, img_height - height - 50))
            
            # For simplicity, we're not checking if the region is truly empty
            # But in a more sophisticated version, we could check pixel values
            return x, y
        
        return None, None
    
    def draw_text_with_highlights(self, image, metadata):
        """Draw paragraphs of text and randomly highlight some portions."""
        width, height = image.size
        draw = ImageDraw.Draw(image, 'RGBA')  # Use RGBA to support transparency for highlights
        
        font = metadata["font"]
        text_color = metadata["text_color"]
        line_spacing = metadata["line_spacing"]
        paragraph_spacing = metadata["paragraph_spacing"]
        margin_top = metadata["margin_top"]
        margin_left = metadata["margin_left"]
        margin_right = metadata["margin_right"]
        line_length = metadata["line_length"]
        highlight_colors = metadata["highlight_colors"]
        
        # Generate random text
        paragraphs = self.generate_random_text(paragraphs=random.randint(3, 7))
        
        # Track all text elements and highlights for ground truth
        text_elements = []
        highlights = []
        
        current_y = margin_top
        for paragraph in paragraphs:
            # Wrap text to fit within margins
            max_width = width - margin_left - margin_right
            wrapped_lines = textwrap.wrap(paragraph, width=line_length)
            
            for line in wrapped_lines:
                # Get line dimensions
                text_bbox = draw.textbbox((0, 0), line, font=font)
                text_width = text_bbox[2] - text_bbox[0]
                text_height = text_bbox[3] - text_bbox[1]
                
                # Skip if we're too close to the bottom
                if current_y + text_height > height - margin_top:
                    break
                
                # Decide whether to highlight this line
                should_highlight = random.random() < 0.3  # 30% chance to highlight
                highlight_color = random.choice(highlight_colors) if should_highlight else None
                
                # Draw the highlight first if needed
                if should_highlight:
                    # Determine if we highlight the whole line or just a part
                    if random.random() < 0.7:  # 70% chance for whole line
                        highlight_start = 0
                        highlight_end = len(line)
                    else:
                        # Highlight a random portion
                        highlight_start = random.randint(0, max(0, len(line) - 5))
                        highlight_end = random.randint(highlight_start + 1, len(line))
                    
                    # Get the text portion to highlight
                    highlight_text = line[highlight_start:highlight_end]
                    
                    # Calculate the bounding box for the highlighted portion
                    prefix_bbox = draw.textbbox((0, 0), line[:highlight_start], font=font)
                    highlight_bbox = draw.textbbox((0, 0), highlight_text, font=font)
                    
                    # Calculate the position of the highlight
                    x1 = margin_left + (prefix_bbox[2] if highlight_start > 0 else 0)
                    y1 = current_y
                    x2 = x1 + (highlight_bbox[2] - highlight_bbox[0])
                    y2 = y1 + text_height
                    
                    # Draw the highlight with slight padding
                    padding = 2
                    highlight_box = [
                        x1 - padding, y1 - padding,
                        x2 + padding, y2 + padding
                    ]
                    draw.rectangle(highlight_box, fill=highlight_color)
                    
                    # Store highlight info for ground truth
                    highlights.append({
                        "boundingBox": [
                            x1, y1,
                            x2, y1,
                            x2, y2,
                            x1, y2
                        ],
                        "text": highlight_text,
                        "highlighted": True
                    })
                
                # Draw the text
                draw.text((margin_left, current_y), line, fill=text_color, font=font)
                
                # Store text element info
                text_elements.append({
                    "boundingBox": [
                        margin_left, current_y,
                        margin_left + text_width, current_y,
                        margin_left + text_width, current_y + text_height,
                        margin_left, current_y + text_height
                    ],
                    "text": line
                })
                
                # Move to next line
                current_y += int(text_height * line_spacing)
            
            # Add paragraph spacing
            current_y += paragraph_spacing
        
        return image, {"text_elements": text_elements, "highlights": highlights}
    
    def add_noise(self, image):
        """Add realistic scanner noise to the image."""
        # Convert to numpy array for manipulation
        img_array = np.array(image)
        
        # Add random noise
        noise_level = random.uniform(2, 8)
        noise = np.random.normal(0, noise_level, img_array.shape)
        noisy_img_array = np.clip(img_array + noise, 0, 255).astype(np.uint8)
        
        # Convert back to PIL Image
        noisy_image = Image.fromarray(noisy_img_array)
        
        # Slightly blur to simulate scanner resolution limits
        blur_radius = random.uniform(0.2, 0.7)
        noisy_image = noisy_image.filter(ImageFilter.GaussianBlur(radius=blur_radius))
        
        # Adjust contrast and brightness to simulate scanner effects
        contrast_factor = random.uniform(0.9, 1.1)
        brightness_factor = random.uniform(0.95, 1.05)
        
        enhancer = ImageEnhance.Contrast(noisy_image)
        noisy_image = enhancer.enhance(contrast_factor)
        
        enhancer = ImageEnhance.Brightness(noisy_image)
        noisy_image = enhancer.enhance(brightness_factor)
        
        # Sometimes add a very subtle color cast
        if random.random() < 0.3:
            r, g, b = [random.uniform(0.98, 1.02) for _ in range(3)]
            noisy_img_array = np.array(noisy_image)
            noisy_img_array = noisy_img_array * [r, g, b]
            noisy_img_array = np.clip(noisy_img_array, 0, 255).astype(np.uint8)
            noisy_image = Image.fromarray(noisy_img_array)
        
        return noisy_image
    
    def transform_bounding_boxes(self, json_data, angle, image):
        """Rotate the image and update the bounding boxes."""
        width, height = image.size
        center_x, center_y = width / 2, height / 2
        
        # Rotate image
        rotated_image = image.rotate(angle, resample=Image.BICUBIC, expand=False)
        
        # Updated JSON with rotated coordinates
        updated_json = {"highlights": []}
        
        # Transform highlight bounding boxes
        for highlight in json_data["highlights"]:
            rotated_bbox = []
            bbox = highlight["boundingBox"]
            
            # Rotate each corner of the bounding box
            for i in range(0, len(bbox), 2):
                x, y = bbox[i], bbox[i+1]
                
                # Translate to origin
                x -= center_x
                y -= center_y
                
                # Rotate
                angle_rad = radians(-angle)
                new_x = x * cos(angle_rad) - y * sin(angle_rad)
                new_y = x * sin(angle_rad) + y * cos(angle_rad)
                
                # Translate back
                new_x += center_x
                new_y += center_y
                
                rotated_bbox.extend([new_x, new_y])
            
            # Add to updated JSON
            updated_json["highlights"].append({
                "boundingBox": rotated_bbox,
                "text": highlight["text"],
                "highlighted": highlight["highlighted"]
            })
        
        return rotated_image, updated_json
    
    def add_random_artifacts(self, image):
        """Add random scanner artifacts like fold lines, smudges, or specks."""
        width, height = image.size
        draw = ImageDraw.Draw(image, 'RGBA')
        
        # Add random artifacts
        artifacts_type = random.choice(["fold", "smudge", "specks", "streak", "none"])
        
        if artifacts_type == "fold":
            # Add a fold line
            start_x = random.randint(0, width)
            end_x = random.randint(0, width)
            y = random.randint(height // 4, 3 * height // 4)
            for i in range(3):
                offset = random.randint(-2, 2)
                shadow_color = (100, 100, 100, random.randint(10, 40))
                draw.line([(start_x, y + offset), (end_x, y + offset)], fill=shadow_color, width=1)
        
        elif artifacts_type == "smudge":
            # Add a random smudge
            x = random.randint(width // 4, 3 * width // 4)
            y = random.randint(height // 4, 3 * height // 4)
            size = random.randint(5, 20)
            for _ in range(50):
                smudge_x = x + random.randint(-size, size)
                smudge_y = y + random.randint(-size // 2, size // 2)
                opacity = random.randint(5, 30)
                draw.point((smudge_x, smudge_y), fill=(100, 100, 100, opacity))
        
        elif artifacts_type == "specks":
            # Add random specks
            for _ in range(random.randint(10, 50)):
                x = random.randint(0, width - 1)
                y = random.randint(0, height - 1)
                size = random.randint(1, 3)
                color = random.randint(0, 50)
                draw.ellipse([x, y, x + size, y + size], fill=(color, color, color))
        
        elif artifacts_type == "streak":
            # Add a streak like from a dirty scanner
            start_y = random.randint(0, height)
            for x in range(0, width, 2):
                opacity = random.randint(5, 20)
                streak_width = random.randint(1, 3)
                y_offset = random.randint(-2, 2)
                draw.line([(x, start_y + y_offset), (x + streak_width, start_y + y_offset)], 
                         fill=(100, 100, 100, opacity), width=1)
        
        return image
    
    def generate_document_with_highlights(self):
        """Main method to generate a document with highlighted text."""
        max_attempts = 10
        
        for _ in range(max_attempts):
            try:
                # Generate the base document
                image = self.generate_scanned_document()
                metadata = self.get_random_metadata()
                
                # Draw text with highlights
                image, json_data = self.draw_text_with_highlights(image, metadata)
                
                # If no highlights were generated, try again
                if not json_data["highlights"]:
                    continue
                
                # Add scanner artifacts
                image = self.add_random_artifacts(image)
                
                # Add noise to make it look scanned
                image = self.add_noise(image)
                
                # Apply a slight rotation (between -5 and 5 degrees)
                angle = random.randint(-5, 5)
                
                # Rotate image and update bounding boxes
                rotated_image, updated_json = self.transform_bounding_boxes(json_data, angle, image)
                
                bt.logging.info(f"[{self.uid}] Synthetic highlighted document generated successfully with {len(updated_json['highlights'])} highlights")
                return updated_json, rotated_image
            
            except Exception as e:
                bt.logging.error(f"[{self.uid}] Error generating document: {e}")
        
        bt.logging.error(f"[{self.uid}] Failed to generate document after {max_attempts} attempts")
        return None, None

# Example usage
if __name__ == "__main__":
    unique_id = str(uuid.uuid4())
    generator = HighlightedDocumentGenerator()
    
    # Generate a document with highlighted text
    json_metadata, generated_doc = generator.generate_document_with_highlights()
    
    # Define filenames
    base_filename = f"highlighted_{unique_id}"
    image_filename = f"{base_filename}.png"
    json_filename = f"{base_filename}.json"

    # Save image
    cv2.imwrite(image_filename, np.array(generated_doc))

    # Save JSON metadata
    with open(json_filename, "w") as json_file:
        json.dump(json_metadata, json_file, indent=4)

    print(f"Saved: {image_filename} and {json_filename}")