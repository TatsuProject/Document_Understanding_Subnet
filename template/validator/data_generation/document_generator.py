

import random
import json
from PIL import Image, ImageDraw, ImageFont
from math import floor
import io
import numpy as np
import cv2
from faker import Faker
import os
script_dir = os.path.dirname(os.path.abspath(__file__))
fake = Faker()


class GenerateDocument:
    def __init__(self, url, uid):
        self.url = ""
        self.uid = uid


    def advertisement(self, FONTS):
        # Choose a random image size
        IMAGE_SIZES = [(800, 600), (900, 700), (1000, 750), (1100, 800), (1200, 900), (1300, 950)]
        img_size = random.choice(IMAGE_SIZES)
        img = Image.new('RGB', img_size, 'white')
        draw = ImageDraw.Draw(img)

        # Generate random advertisement details
        metadata = {
            "organization": fake.company(),
            "product": fake.catch_phrase(),
            "offer": f"{random.randint(10, 50)}% OFF",
            "valid_until": fake.future_date().strftime("%Y-%m-%d"),
            "contact": fake.phone_number(),
            "website": fake.url()
        }

        # Define starting positions
        x, y = 50, 50

        # Store bounding boxes for NER
        ner_annotations = []

        # Draw "Advertisement" title
        title_font = ImageFont.truetype(random.choice(FONTS), 40)
        draw.text((x, y), "ADVERTISEMENT", font=title_font, fill="black")
        bbox = draw.textbbox((x, y), "ADVERTISEMENT", font=title_font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        y += text_height + 20  # Move down

        # Function to add text & store bounding box
        def add_text(label, content, font_size=25, offset=10):
            nonlocal y
            font = ImageFont.truetype(random.choice(FONTS), font_size)
            text = f"{label}: {content}"
            draw.text((x, y), text, font=font, fill="black")
            bbox = draw.textbbox((x, y), text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            bounding_box = [x, y, x + text_width, y + text_height]
            ner_annotations.append({"label": label.lower(), "content": content, "bounding_box": bounding_box})
            y += text_height + offset  # Move down

        # Add organization, product, offer, validity, and contact info
        add_text("Organization", metadata["organization"])
        add_text("Product", metadata["product"])
        add_text("Offer", metadata["offer"], font_size=30, offset=15)
        add_text("Valid Until", metadata["valid_until"])
        add_text("Contact", metadata["contact"])
        add_text("Website", metadata["website"], font_size=22, offset=15)

        # Convert to OpenCV for noise & rotation
        image_cv = np.array(img)

        # Add Gaussian noise
        noise = np.random.normal(0, 15, image_cv.shape).astype(np.uint8)
        noisy_image = cv2.add(image_cv, noise)

        # Rotate image slightly
        angle = random.uniform(-5, 5)
        center = (img_size[0] // 2, img_size[1] // 2)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated_image = cv2.warpAffine(noisy_image, matrix, (img_size[0], img_size[1]), borderMode=cv2.BORDER_REPLICATE)

        # Save NER annotations
        GT_json = {
            "document_class": "advertisement",
            "NER": ner_annotations
        }
        return GT_json, rotated_image

    def budget(self, FONTS):
        IMAGE_SIZES = [(800, 600), (900, 700), (1000, 750), (1100, 800), (1200, 900), (1300, 950)]
        # Choose a random image size
        img_size = random.choice(IMAGE_SIZES)
        img = Image.new('RGB', img_size, 'white')
        draw = ImageDraw.Draw(img)

        # Generate random budget details
        metadata = {
            "date": fake.date_this_year().strftime("%m/%d/%y"),
            "organization": fake.company(),
            "department": fake.job(),
            "budget_items": [
                {"item": fake.word(), "cost": round(random.uniform(100, 5000), 2)} for _ in range(5)
            ],
            "total_budget": round(random.uniform(5000, 50000), 2)
        }

        # Define starting positions
        x, y = 50, 50

        # Store bounding boxes for NER
        ner_annotations = []

        # Draw "BUDGET REPORT" title
        title_font = ImageFont.truetype(random.choice(FONTS), 40)
        draw.text((x, y), "BUDGET REPORT", font=title_font, fill="black")
        bbox = draw.textbbox((x, y), "BUDGET REPORT", font=title_font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        y += text_height + 20  # Move down

        # Function to add text & store bounding box
        def add_text(label, content, font_size=25, offset=10):
            nonlocal y
            font = ImageFont.truetype(random.choice(FONTS), font_size)
            text = f"{label}: {content}"
            draw.text((x, y), text, font=font, fill="black")
            bbox = draw.textbbox((x, y), text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            bounding_box = [x, y, x + text_width, y + text_height]
            ner_annotations.append({"label": label.lower(), "content": content, "bounding_box": bounding_box})
            y += text_height + offset  # Move down

        # Add budget metadata
        add_text("Date", metadata["date"])
        add_text("Organization", metadata["organization"])
        add_text("Department", metadata["department"])

        # Add budget items
        for item in metadata["budget_items"]:
            add_text("Item", f"{item['item']} - ${item['cost']}")

        # Add total budget
        add_text("Total Budget", f"${metadata['total_budget']}", font_size=28, offset=15)

        # Convert to OpenCV for noise & rotation
        image_cv = np.array(img)

        # Add Gaussian noise
        noise = np.random.normal(0, 15, image_cv.shape).astype(np.uint8)
        noisy_image = cv2.add(image_cv, noise)

        # Rotate image slightly
        angle = random.uniform(-5, 5)
        center = (img_size[0] // 2, img_size[1] // 2)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated_image = cv2.warpAffine(noisy_image, matrix, (img_size[0], img_size[1]), borderMode=cv2.BORDER_REPLICATE)

        # Save NER annotations
        GT_json = {
            "document_class": "budget",
            "NER": ner_annotations
        }
        return GT_json, rotated_image
        
    
    def email(self, FONTS):
        IMAGE_SIZES = [(800, 1000), (850, 1100), (900, 1200), (1000, 1300), (1100, 1400), (1200, 1500)]
        EMAIL_TYPES = ["Personal Email", "Business Email", "Notification Email", "Marketing Email"]

        # Choose random image size
        img_size = random.choice(IMAGE_SIZES)
        img = Image.new('RGB', img_size, 'white')
        draw = ImageDraw.Draw(img)

        # Select a random email type
        email_type = random.choice(EMAIL_TYPES)

        # Generate random email fields
        email_fields = {
            "From": fake.email(),
            "To": fake.email(),
            "Date": fake.date(),
            "Subject": fake.sentence(nb_words=6),
            "Body": fake.paragraph(nb_sentences=5),
        }

        # Define starting positions
        x, y = 50, 50

        # Store bounding boxes for NER
        ner_annotations = []

        # Load font safely
        try:
            font_path = random.choice(FONTS)
            title_font = ImageFont.truetype(font_path, 30)
        except IOError:
            title_font = ImageFont.load_default()

        # Draw email header
        draw.text((x, y), email_type, font=title_font, fill="black")
        y += 50  # Move Y down

        # Draw each email field on the image
        for label, value in email_fields.items():
            try:
                font_path = random.choice(FONTS)
                font_size = random.randint(18, 28)
                font = ImageFont.truetype(font_path, font_size)
            except IOError:
                font = ImageFont.load_default()

            # Draw field label
            draw.text((x, y), f"{label}:", font=font, fill="black")

            # Calculate bounding box for value
            text_bbox = draw.textbbox((x + 150, y), value, font=font)
            bounding_box = [text_bbox[0], text_bbox[1], text_bbox[2], text_bbox[3]]

            # Draw text content
            draw.text((x + 150, y), value, font=font, fill="black")

            # Save bounding box with label
            ner_annotations.append({"label": label.lower().replace(" ", "_"), "content": value, "bounding_box": bounding_box})

            # Move Y position for next field
            y += (text_bbox[3] - text_bbox[1]) + 20

        # Convert to OpenCV for noise & rotation
        image_cv = np.array(img)

        # Add Gaussian noise
        noise = np.random.normal(0, 15, image_cv.shape).astype(np.uint8)
        noisy_image = cv2.add(image_cv, noise)

        # Rotate image slightly
        angle = random.uniform(-5, 5)
        center = (img_size[0] // 2, img_size[1] // 2)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated_image = cv2.warpAffine(noisy_image, matrix, (img_size[0], img_size[1]), borderMode=cv2.BORDER_REPLICATE)

        GT_json = {
            "document_class": "email",
            "NER": ner_annotations
        }

        return GT_json, rotated_image

    def file_folder(self, FONTS):
        IMAGE_SIZES = [(800, 600), (1024, 768), (1280, 720), (1280, 1024), (1600, 900), (1920, 1080)]

        # Apply noise and rotation to simulate a scanned document
        def apply_scan_effects(img):
            # Convert to grayscale
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

            # Add Gaussian noise
            noise = np.random.normal(0, 25, img.shape).astype(np.uint8)
            img = cv2.add(img, noise)

            # Rotate slightly
            angle = random.uniform(-3, 3)
            h, w = img.shape
            matrix = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1)
            img = cv2.warpAffine(img, matrix, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))

            return img

        # Choose random image size
        img_size = random.choice(IMAGE_SIZES)
        img = Image.new("RGB", img_size, "white")
        draw = ImageDraw.Draw(img)

        # Generate random file-folder related metadata
        metadata = {
            "Folder Name": fake.company(),
            "File Name": f"{fake.word()}.pdf",
            "Document Type": random.choice(["Invoice", "Contract", "Report", "Memo"]),
            "Creation Date": fake.date(),
            "Reference Number": fake.uuid4()[:8],
            "Owner": fake.name(),
        }

        # List to store bounding boxes
        ner_annotations = []

        # Start position for text
        x, y = 50, 50
        line_spacing = 30  # Adjusted for better spacing

        # Load font safely
        try:
            font_path = random.choice(FONTS)
            font = ImageFont.truetype(font_path, 24)
        except IOError:
            font = ImageFont.load_default()

        # Draw the metadata on the image
        for label, content in metadata.items():
            text = f"{label}: {content}"
            draw.text((x, y), text, fill="black", font=font)

            # Generate bounding box using textbbox
            text_bbox = draw.textbbox((x, y), text, font=font)
            bounding_box = [text_bbox[0], text_bbox[1], text_bbox[2], text_bbox[3]]

            ner_annotations.append({
                "label": label.lower().replace(" ", "_"),
                "content": content,
                "bounding_box": bounding_box
            })

            y += line_spacing

        # Convert to OpenCV format and add noise/rotation
        img = np.array(img)
        img = apply_scan_effects(img)

        # JSON Output
        GT_json = {
            "document_class": "file_folder",
            "NER": ner_annotations
        }

        return GT_json, img

    def form(self, FONTS):
        IMAGE_SIZES = [(800, 1000), (850, 1100), (900, 1200), (1000, 1300), (1100, 1400), (1200, 1500)]
        TEMPLATES = ["simple", "header_footer"]
        FORM_TYPES = ["Admission Form", "Feedback Form", "Registration Form", "Application Form", "Survey Form"]
        
        img_size = random.choice(IMAGE_SIZES)
        img = Image.new('RGB', img_size, 'white')
        draw = ImageDraw.Draw(img)
        
        if random.choice(TEMPLATES) == "header_footer":
            draw.rectangle([(0, 0), (img_size[0], 80)], fill="black")
            draw.rectangle([(0, img_size[1] - 80), (img_size[0], img_size[1])], fill="black")
        
        form_title = random.choice(FORM_TYPES)
        font_path = random.choice(FONTS)
        title_font = ImageFont.truetype(font_path, 40)
        draw.text((50, 100), form_title, font=title_font, fill="black")
        y = 160
        
        form_fields = {
            "Name": fake.name(),
            "Date of Birth": fake.date_of_birth(minimum_age=18, maximum_age=60).strftime("%d/%m/%Y"),
            "Email": fake.email(),
            "Phone Number": fake.phone_number(),
            "Address": fake.address(),
            "City": fake.city(),
            "Postal Code": fake.postcode(),
            "Date": fake.date(),
            "Comments": fake.sentence(nb_words=10)
        }
        
        ner_annotations = []
        for label, value in form_fields.items():
            font_path = random.choice(FONTS)
            font = ImageFont.truetype(font_path, random.randint(20, 30))
            draw.text((50, y), f"{label}:", font=font, fill="black")
            
            bbox = draw.textbbox((250, y), value, font=font)
            draw.text((250, y), value, font=font, fill="black")
            draw.line([(250, bbox[3] + 3), (bbox[2], bbox[3] + 3)], fill="black", width=2)
            
            ner_annotations.append({"label": label.lower().replace(" ", "_"), "content": value, "bounding_box": list(bbox)})
            y += (bbox[3] - bbox[1]) + 30
        
        image_cv = np.array(img)
        noise = np.random.normal(0, 15, image_cv.shape).astype(np.uint8)
        noisy_image = cv2.add(image_cv, noise)
        angle = random.uniform(-5, 5)
        matrix = cv2.getRotationMatrix2D((img_size[0] // 2, img_size[1] // 2), angle, 1.0)
        rotated_image = cv2.warpAffine(noisy_image, matrix, (img_size[0], img_size[1]), borderMode=cv2.BORDER_REPLICATE)
        
        return {"document_class": "form", "NER": ner_annotations}, rotated_image


    def handwritten(self, FONTS):
        def generate_handwritten_text(text, font_path, font_size):
            try:
                font = ImageFont.truetype(font_path, font_size)
            except Exception as e:
                print(f"Error loading font: {e}")
                return None, (0, 0, 0, 0)
            
            bbox = font.getbbox(text)
            img = Image.new('RGBA', (bbox[2] + 10, bbox[3] + 10), (255, 255, 255, 0))
            draw = ImageDraw.Draw(img)
            draw.text((5, 5), text, font=font, fill=(0, 0, 0, 255))
            img = img.rotate(random.uniform(-5, 5), expand=True)
            return img, bbox

        IMAGE_SIZES = [(1000, 1400), (1200, 1600), (1400, 1800), (1600, 2000), (1800, 2200), (2000, 2400)]
        img_size = random.choice(IMAGE_SIZES)
        img = Image.new('RGB', img_size, 'white')
        draw = ImageDraw.Draw(img)
        draw.rectangle([0, 0, img_size[0], 100], fill="lightgray")
        draw.rectangle([0, img_size[1] - 80, img_size[0], img_size[1]], fill="lightgray")
        
        publication_data = {
            "title": fake.sentence(nb_words=6),
            "author": fake.name(),
            "date": fake.date_this_year().strftime("%m/%d/%y"),
            "abstract": "\n".join([fake.sentence() for _ in range(3)])
        }
        
        x, y = 100, 150
        ner_annotations = []
        
        def add_handwritten_text(label, content, font_size=32, offset=20):
            nonlocal y
            text_img, bbox = generate_handwritten_text(content, random.choice(FONTS), font_size)
            if text_img:
                img.paste(text_img, (x, y), text_img)
                ner_annotations.append({"label": label.lower(), "content": content, "bounding_box": [x, y, x + bbox[2], y + bbox[3]]})
                y += bbox[3] + offset
        
        add_handwritten_text("title", publication_data["title"], 40)
        add_handwritten_text("author", f"Author: {publication_data['author']}", 30)
        add_handwritten_text("date", f"Date: {publication_data['date']}", 30)
        add_handwritten_text("abstract", "Abstract:", 35)
        
        for line in publication_data["abstract"].split("\n"):
            add_handwritten_text("abstract_content", line, 28)
        
        image_cv = np.array(img)
        noise = np.random.normal(0, 15, image_cv.shape).astype(np.uint8)
        noisy_image = cv2.add(image_cv, noise)
        angle = random.uniform(-5, 5)
        matrix = cv2.getRotationMatrix2D((img_size[0] // 2, img_size[1] // 2), angle, 1.0)
        rotated_image = cv2.warpAffine(noisy_image, matrix, (img_size[0], img_size[1]), borderMode=cv2.BORDER_REPLICATE)
        
        return {"document_class": "handwritten", "NER": ner_annotations}, rotated_image


    def invoice(self, FONTS):
        IMAGE_SIZES = [
            (600, 800), (800, 1000), (1000, 1200),
            (1200, 1400), (1400, 1600), (1600, 1800)
        ]

        FONT_PATH = random.choice(FONTS)
        FONT_SIZES = [20, 24, 28, 32]

        def add_noise(image):
            """Adds random noise to an image."""
            np_image = np.array(image)
            noise = np.random.normal(0, 25, np_image.shape).astype(np.uint8)
            noisy_image = np.clip(np_image + noise, 0, 255)  # Ensure values stay in valid range
            return Image.fromarray(noisy_image)

        def rotate_image(image):
            """Rotates the image slightly to mimic a scanned document."""
            angle = random.randint(-5, 5)  # Small rotation
            return image.rotate(angle, expand=True)

        def generate_invoice_data(draw, img_width):
            """
            Generates invoice data using Faker and draws it on the image.
            Returns a dictionary containing bounding boxes for NER.
            """
            ner_annotations = []
            y_offset = 50
            font_size = random.choice(FONT_SIZES)
            font = ImageFont.truetype(FONT_PATH, font_size)

            # Generate fake invoice details
            invoice_number = fake.uuid4()[:8]
            company_name = fake.company()
            payee_name = fake.name()
            invoice_date = fake.date()
            total_amount = f"${random.randint(100, 1000)}"

            # Draw and record bounding boxes
            def draw_text(label, text):
                """Helper function to draw text and record bounding box."""
                nonlocal y_offset
                draw.text((50, y_offset), text, fill="black", font=font)
                bbox = draw.textbbox((50, y_offset), text, font=font)  # Corrected for Pillow 10.x.x
                ner_annotations.append({"label": label, "content": text, "bounding_box": bbox})
                y_offset += font_size + 10  # Adjust line spacing

            # Draw invoice fields
            draw_text("invoice_number", f"Invoice #: {invoice_number}")
            draw_text("date", f"Date: {invoice_date}")
            draw_text("company_name", f"From: {company_name}")
            draw_text("payee_name", f"To: {payee_name}")

            # Draw table headers
            y_offset += 20
            draw_text("items_header", "Item      Qty      Price")

            # Generate random invoice items
            for _ in range(random.randint(3, 5)):  # 3-5 items
                item = fake.word().capitalize()
                qty = random.randint(1, 5)
                price = f"${random.randint(10, 100)}"
                draw_text("item", f"{item}      {qty}      {price}")

            y_offset += 10
            draw_text("total", f"Total: {total_amount}")

            return ner_annotations

        img_width, img_height = random.choice(IMAGE_SIZES)
        img = Image.new("RGB", (img_width, img_height), "white")
        draw = ImageDraw.Draw(img)

        # Generate invoice data and get NER annotations
        ner_annotations = generate_invoice_data(draw, img_width)

        # Add noise and rotate
        img = add_noise(img)
        img = rotate_image(img)
        image_cv = np.array(img)
        # Save annotations as JSON
        GT_json = {
            "document_class": "invoice",
            "NER": ner_annotations
        }
        
        return GT_json, image_cv


    def letter(self, FONTS):
        IMAGE_SIZES = [(800, 1000), (850, 1100), (900, 1200), (1000, 1300), (1100, 1400), (1200, 1500)]
        TEMPLATES = ["simple", "header_footer"]

        def random_text(length=10):
            return ''.join(random.choices(string.ascii_uppercase + string.digits, k=length))

        img_size = random.choice(IMAGE_SIZES)
        img = Image.new('RGB', img_size, 'white')
        draw = ImageDraw.Draw(img)

        template = random.choice(TEMPLATES)

        if template == "header_footer":
            draw.rectangle([(0, 0), (img_size[0], 80)], fill="black")  # Header
            draw.rectangle([(0, img_size[1] - 80), (img_size[0], img_size[1])], fill="black")  # Footer

        sender = fake.company() + "\n" + fake.address()
        receiver = fake.name() + "\n" + fake.address()
        date = fake.date()
        subject = "Subject: " + fake.sentence(nb_words=5)
        body = "\n".join(fake.paragraphs(nb=5))

        content = {
            "Sender": sender,
            "Receiver": receiver,
            "Date": date,
            "Subject": subject,
            "Body": body
        }

        x, y = 50, 100
        ner_annotations = []

        for label, text in content.items():
            font_path = random.choice(FONTS)
            font_size = random.randint(20, 30)
            font = ImageFont.truetype(font_path, font_size)

            bbox = draw.textbbox((x, y), text, font=font)  # Updated from `textsize()`
            draw.text((x, y), text, font=font, fill="black")

            ner_annotations.append({"label": label.lower(), "content": text, "bounding_box": bbox})
            y += bbox[3] - bbox[1] + 20  # Adjust Y position based on text height

        # Convert to OpenCV for noise & rotation
        image_cv = np.array(img)

        noise = np.random.normal(0, 15, image_cv.shape).astype(np.uint8)
        noisy_image = np.clip(image_cv + noise, 0, 255)  # Ensure valid pixel values

        angle = random.uniform(-5, 5)
        center = (img_size[0] // 2, img_size[1] // 2)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated_image = cv2.warpAffine(noisy_image, matrix, (img_size[0], img_size[1]), borderMode=cv2.BORDER_REPLICATE)

        GT_json = {
            "document_class": "letter",
            "NER": ner_annotations
        }
        
        return GT_json, rotated_image


    def memo(self, FONTS):
        IMAGE_SIZES = [(800, 1000), (850, 1100), (900, 1200), (1000, 1300), (1100, 1400), (1200, 1500)]
        img_size = random.choice(IMAGE_SIZES)
        img = Image.new('RGB', img_size, 'white')
        draw = ImageDraw.Draw(img)

        metadata = {
            "sender": fake.name(),
            "receiver": fake.name(),
            "date": fake.date(),
            "subject": fake.sentence(nb_words=6),
            "body": fake.paragraph(nb_sentences=5),
        }

        x, y = 50, 50
        ner_annotations = []

        title_font = ImageFont.truetype(random.choice(FONTS), 40)
        draw.text((x, y), "MEMO", font=title_font, fill="black")
        text_bbox = draw.textbbox((x, y), "MEMO", font=title_font)
        y += text_bbox[3] - text_bbox[1] + 20

        def add_text(label, content, font_size=25, offset=10):
            nonlocal y
            font = ImageFont.truetype(random.choice(FONTS), font_size)
            draw.text((x, y), f"{label}: {content}", font=font, fill="black")
            text_bbox = draw.textbbox((x, y), f"{label}: {content}", font=font)
            bounding_box = [x, y, text_bbox[2], text_bbox[3]]
            ner_annotations.append({"label": label.lower(), "content": content, "bounding_box": bounding_box})
            y += text_bbox[3] - text_bbox[1] + offset

        add_text("Sender", metadata["sender"])
        add_text("Receiver", metadata["receiver"])
        add_text("Date", metadata["date"])
        add_text("Subject", metadata["subject"], font_size=28, offset=15)

        body_font = ImageFont.truetype(random.choice(FONTS), 22)
        draw.text((x, y), metadata["body"], font=body_font, fill="black")
        text_bbox = draw.textbbox((x, y), metadata["body"], font=body_font)
        bounding_box = [x, y, text_bbox[2], text_bbox[3]]
        ner_annotations.append({"label": "body", "content": metadata["body"], "bounding_box": bounding_box})

        image_cv = np.array(img)
        noise = np.random.normal(0, 15, image_cv.shape).astype(np.uint8)
        noisy_image = cv2.add(image_cv, noise)

        angle = random.uniform(-5, 5)
        center = (img_size[0] // 2, img_size[1] // 2)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated_image = cv2.warpAffine(noisy_image, matrix, (img_size[0], img_size[1]), borderMode=cv2.BORDER_REPLICATE)

        GT_json = {"document_class": "memo", "NER": ner_annotations}
        return GT_json, rotated_image

        

    def news_article(self, FONTS):
        IMAGE_SIZES = [(800, 600), (900, 700), (1000, 750), (1100, 800), (1200, 900), (1300, 950)]
        img_size = random.choice(IMAGE_SIZES)
        img = Image.new('RGB', img_size, 'white')
        draw = ImageDraw.Draw(img)

        metadata = {
            "headline": fake.sentence(nb_words=6),
            "author": fake.name(),
            "date": fake.date_this_decade().strftime("%Y-%m-%d"),
            "location": fake.city(),
            "organization": fake.company(),
            "content": fake.paragraph(nb_sentences=5)
        }

        x, y = 50, 50
        ner_annotations = []

        title_font = ImageFont.truetype(random.choice(FONTS), 40)
        draw.text((x, y), "NEWS ARTICLE", font=title_font, fill="black")
        text_width, text_height = draw.textbbox((x, y), "NEWS ARTICLE", font=title_font)[2:]  # Use textbbox for Pillow 10
        y += text_height + 20

        def add_text(label, content, font_size=25, offset=10):
            nonlocal y
            font = ImageFont.truetype(random.choice(FONTS), font_size)
            text = f"{label}: {content}"
            draw.text((x, y), text, font=font, fill="black")
            bbox = draw.textbbox((x, y), text, font=font)  # Use textbbox for accurate dimensions
            ner_annotations.append({"label": label.lower(), "content": content, "bounding_box": bbox})
            y += bbox[3] - bbox[1] + offset

        add_text("Headline", metadata["headline"], font_size=28, offset=15)
        add_text("Author", metadata["author"])
        add_text("Date", metadata["date"])
        add_text("Location", metadata["location"])
        add_text("Organization", metadata["organization"])
        add_text("Content", metadata["content"], font_size=22, offset=15)

        image_cv = np.array(img)
        noise = np.random.normal(0, 15, image_cv.shape).astype(np.uint8)
        noisy_image = cv2.add(image_cv, noise)

        angle = random.uniform(-5, 5)
        center = (img_size[0] // 2, img_size[1] // 2)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated_image = cv2.warpAffine(noisy_image, matrix, (img_size[0], img_size[1]), borderMode=cv2.BORDER_REPLICATE)

        GT_json = {
            "document_class": "news_article",
            "NER": ner_annotations
        }
        
        return GT_json, rotated_image


    def presentation(self, FONTS):
        IMAGE_SIZES = [(1000, 700), (1200, 800), (1400, 900), (1600, 1000), (1800, 1100), (2000, 1200)]
        img_size = random.choice(IMAGE_SIZES)
        img = Image.new('RGB', img_size, 'white')
        draw = ImageDraw.Draw(img)

        border_thickness = 20
        draw.rectangle([border_thickness, border_thickness, img_size[0] - border_thickness, img_size[1] - border_thickness], outline="black", width=border_thickness)

        slide_data = {
            "title": fake.sentence(nb_words=6),
            "content": "\n".join([fake.sentence() for _ in range(5)]),
            "date": fake.date_this_year().strftime("%m/%d/%y"),
            "presenter": fake.name(),
        }

        x, y = 100, 120
        ner_annotations = []

        title_font = ImageFont.truetype(random.choice(FONTS), 50)
        draw.text((x, y), slide_data["title"], font=title_font, fill="black")
        x1, y1, x2, y2 = draw.textbbox((x, y), slide_data["title"], font=title_font)
        title_bbox = [x1, y1, x2, y2]
        ner_annotations.append({"label": "title", "content": slide_data["title"], "bounding_box": title_bbox})
        y = y2 + 40

        def add_text(label, content, font_size=30, offset=15):
            nonlocal y
            font = ImageFont.truetype(random.choice(FONTS), font_size)
            draw.text((x, y), content, font=font, fill="black")
            x1, y1, x2, y2 = draw.textbbox((x, y), content, font=font)
            ner_annotations.append({"label": label.lower(), "content": content, "bounding_box": [x1, y1, x2, y2]})
            y = y2 + offset

        for line in slide_data["content"].split("\n"):
            add_text("content", line)

        add_text("date", f"Date: {slide_data['date']}", font_size=28, offset=20)
        add_text("presenter", f"Presenter: {slide_data['presenter']}", font_size=28, offset=20)

        image_cv = np.array(img)
        noise = np.random.normal(0, 15, image_cv.shape).astype(np.uint8)
        noisy_image = cv2.add(image_cv, noise)

        angle = random.uniform(-5, 5)
        center = (img_size[0] // 2, img_size[1] // 2)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated_image = cv2.warpAffine(noisy_image, matrix, (img_size[0], img_size[1]), borderMode=cv2.BORDER_REPLICATE)

        GT_json = {"document_class": "presentation", "NER": ner_annotations}
        return GT_json, rotated_image


    def questionnaire(self, FONTS):
        IMAGE_SIZES = [(800, 1000), (850, 1100), (900, 1200), (1000, 1300), (1100, 1400), (1200, 1500)]
        QUESTIONNAIRE_TYPES = ["Customer Feedback", "Medical Questionnaire", "Employee Survey", "Product Review"]
        img_size = random.choice(IMAGE_SIZES)
        img = Image.new('RGB', img_size, 'white')
        draw = ImageDraw.Draw(img)

        questionnaire_type = random.choice(QUESTIONNAIRE_TYPES)

        metadata = {
            "Title": questionnaire_type,
            "Date": fake.date()
        }

        questions = []
        for _ in range(random.randint(5, 10)):
            question_text = fake.sentence(nb_words=8)
            answers = [fake.word() for _ in range(4)]
            questions.append({"question": question_text, "answers": answers})

        x, y = 50, 50
        ner_annotations = []

        font_path = random.choice(FONTS)
        title_font = ImageFont.truetype(font_path, 30)
        draw.text((x, y), metadata["Title"], font=title_font, fill="black")
        y += 50

        font = ImageFont.truetype(random.choice(FONTS), 20)
        draw.text((x, y), f"Date: {metadata['Date']}", font=font, fill="black")
        x1, y1, x2, y2 = draw.textbbox((x + 70, y), metadata['Date'], font=font)
        ner_annotations.append({"label": "date", "content": metadata['Date'], "bounding_box": [x1, y1, x2, y2]})
        y += 40

        for idx, q in enumerate(questions):
            font = ImageFont.truetype(random.choice(FONTS), 22)
            draw.text((x, y), f"Q{idx + 1}: {q['question']}", font=font, fill="black")
            x1, y1, x2, y2 = draw.textbbox((x + 50, y), q['question'], font=font)
            ner_annotations.append({"label": "question", "content": q['question'], "bounding_box": [x1, y1, x2, y2]})
            y += y2 - y1 + 10

            font = ImageFont.truetype(random.choice(FONTS), 20)
            for ans in q["answers"]:
                draw.rectangle([x, y, x + 20, y + 20], outline="black")
                draw.text((x + 30, y), ans, font=font, fill="black")
                x1, y1, x2, y2 = draw.textbbox((x + 30, y), ans, font=font)
                ner_annotations.append({"label": "answer", "content": ans, "bounding_box": [x1, y1, x2, y2]})
                y += y2 - y1 + 5
            y += 15

        image_cv = np.array(img)
        noise = np.random.normal(0, 15, image_cv.shape).astype(np.uint8)
        noisy_image = cv2.add(image_cv, noise)

        angle = random.uniform(-5, 5)
        center = (img_size[0] // 2, img_size[1] // 2)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated_image = cv2.warpAffine(noisy_image, matrix, (img_size[0], img_size[1]), borderMode=cv2.BORDER_REPLICATE)

        GT_json = {"document_class": "questionnaire", "NER": ner_annotations}
        return GT_json, rotated_image


    def resume(self, FONTS):
        IMAGE_SIZES = [(800, 1000), (850, 1100), (900, 1200), (1000, 1300), (1100, 1400), (1200, 1500)]
        SECTIONS = ["Summary", "Experience", "Education", "Skills", "Certifications", "Projects"]
        img_size = random.choice(IMAGE_SIZES)
        img = Image.new('RGB', img_size, 'white')
        draw = ImageDraw.Draw(img)

        metadata = {
            "person_name": fake.name(),
            "address": fake.address(),
            "phone": fake.phone_number(),
            "email": fake.email(),
            "summary": fake.paragraph(nb_sentences=3),
            "skills": [fake.job() for _ in range(5)],
            "experience": [{"company": fake.company(), "position": fake.job(), "years": f"{random.randint(1, 10)} years"} for _ in range(3)],
            "education": [{"degree": fake.catch_phrase(), "institution": fake.company(), "year": random.randint(2000, 2022)}],
            "certifications": [fake.bs() for _ in range(2)]
        }

        x, y = 50, 50
        ner_annotations = []

        font_path = random.choice(FONTS)
        name_font = ImageFont.truetype(font_path, 35)
        draw.text((x, y), metadata["person_name"], font=name_font, fill="black")
        bbox = draw.textbbox((x, y), metadata['person_name'], font=name_font)
        ner_annotations.append({"label": "person_name", "content": metadata['person_name'], "bounding_box": bbox})
        y = bbox[3] + 10

        font = ImageFont.truetype(random.choice(FONTS), 20)
        contact_info = f"{metadata['address']} | {metadata['phone']} | {metadata['email']}"
        draw.text((x, y), contact_info, font=font, fill="black")
        bbox = draw.textbbox((x, y), contact_info, font=font)
        ner_annotations.append({"label": "contact_info", "content": contact_info, "bounding_box": bbox})
        y = bbox[3] + 20

        for section in SECTIONS:
            draw.text((x, y), section.upper(), font=ImageFont.truetype(font_path, 25), fill="black")
            y += 30
            
            if section == "Summary":
                text = metadata["summary"]
            elif section == "Skills":
                text = ", ".join(metadata["skills"])
            elif section == "Experience":
                text = "\n".join([f"{exp['position']} at {exp['company']} ({exp['years']})" for exp in metadata["experience"]])
            elif section == "Education":
                text = "\n".join([f"{edu['degree']} from {edu['institution']} ({edu['year']})" for edu in metadata["education"]])
            elif section == "Certifications":
                text = ", ".join(metadata["certifications"])
            else:
                continue

            draw.text((x, y), text, font=ImageFont.truetype(random.choice(FONTS), 20), fill="black")
            bbox = draw.textbbox((x, y), text, font=font)
            ner_annotations.append({"label": section.lower(), "content": text, "bounding_box": bbox})
            y = bbox[3] + 20

        image_cv = np.array(img)
        noise = np.random.normal(0, 15, image_cv.shape).astype(np.uint8)
        noisy_image = cv2.add(image_cv, noise)
        
        angle = random.uniform(-5, 5)
        center = (img_size[0] // 2, img_size[1] // 2)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated_image = cv2.warpAffine(noisy_image, matrix, (img_size[0], img_size[1]), borderMode=cv2.BORDER_REPLICATE)
        
        GT_json = {"document_class": "resume", "NER": ner_annotations}
        return GT_json, rotated_image


    def scientific_publication(self, FONTS):
        SCIENTIFIC_TERMS = [
            "Neural Networks", "Quantum Computing", "DNA Sequencing", "Machine Learning", 
            "Black Hole Physics", "Thermodynamics", "Gene Editing", "Nanotechnology",
            "CRISPR-Cas9", "Protein Folding", "String Theory", "Artificial Intelligence"
        ]

        # Define random image sizes (A4-like dimensions)
        IMAGE_SIZES = [(1000, 1400), (1200, 1600), (1400, 1800), (1600, 2000), (1800, 2200), (2000, 2400)]

        # Choose a random image size
        img_size = random.choice(IMAGE_SIZES)
        img = Image.new('RGB', img_size, 'white')
        draw = ImageDraw.Draw(img)

        # Add header and footer
        header_height, footer_height = 100, 80
        draw.rectangle([0, 0, img_size[0], header_height], fill="lightgray")  # Header
        draw.rectangle([0, img_size[1] - footer_height, img_size[0], img_size[1]], fill="lightgray")  # Footer

        # Generate random publication details
        publication_data = {
            "title": fake.sentence(nb_words=6),
            "abstract": "\n".join([fake.sentence() for _ in range(3)]),
            "author": fake.name(),
            "date": fake.date_this_year().strftime("%m/%d/%y"),
            "keywords": ", ".join(random.sample(SCIENTIFIC_TERMS, 4)),
            "doi": f"10.{random.randint(1000, 9999)}/{random.randint(10000, 99999)}"
        }

        # Define starting positions
        x, y = 100, header_height + 50

        # Store bounding boxes for NER
        ner_annotations = []

        # Draw Title
        title_font = ImageFont.truetype(random.choice(FONTS), 50)
        draw.text((x, y), publication_data["title"], font=title_font, fill="black")
        title_bbox = draw.textbbox((x, y), publication_data["title"], font=title_font)
        ner_annotations.append({"label": "title", "content": publication_data["title"], "bounding_box": title_bbox})
        y += title_bbox[3] - title_bbox[1] + 40  # Move down

        # Function to add text & store bounding box
        def add_text(label, content, font_size=30, offset=15):
            nonlocal y
            font = ImageFont.truetype(random.choice(FONTS), font_size)
            draw.text((x, y), content, font=font, fill="black")
            text_bbox = draw.textbbox((x, y), content, font=font)
            ner_annotations.append({"label": label.lower(), "content": content, "bounding_box": text_bbox})
            y += text_bbox[3] - text_bbox[1] + offset  # Move down

        # Add content sections
        add_text("author", f"Author: {publication_data['author']}", font_size=28, offset=20)
        add_text("date", f"Date: {publication_data['date']}", font_size=28, offset=20)
        add_text("doi", f"DOI: {publication_data['doi']}", font_size=28, offset=20)
        add_text("keywords", f"Keywords: {publication_data['keywords']}", font_size=28, offset=20)

        # Abstract section
        add_text("abstract", "Abstract:", font_size=32, offset=10)
        for line in publication_data["abstract"].split("\n"):
            add_text("abstract_content", line, font_size=26, offset=10)

        # Convert to OpenCV for noise & rotation
        image_cv = np.array(img)

        # Add Gaussian noise (ensure correct dtype handling)
        noise = np.random.normal(0, 15, image_cv.shape).astype(np.int16)  # Use int16 to prevent overflow
        noisy_image = np.clip(image_cv + noise, 0, 255).astype(np.uint8)  # Clip and convert back

        # Rotate the image slightly
        angle = random.uniform(-5, 5)
        center = (img_size[0] // 2, img_size[1] // 2)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated_image = cv2.warpAffine(noisy_image, matrix, (img_size[0], img_size[1]), borderMode=cv2.BORDER_REPLICATE)

        # Save NER annotations
        GT_json = {
            "document_class": "scientific_publication",
            "NER": ner_annotations
        }

        return GT_json, rotated_image
        

    def scientific_report(self, FONTS):
        # Scientific Keywords
        SCIENTIFIC_TERMS = [
            "Quantum Mechanics", "Neural Networks", "DNA Sequencing", "Photosynthesis", 
            "Machine Learning", "Protein Folding", "Nanotechnology", "Gene Editing", 
            "CRISPR-Cas9", "Black Hole", "String Theory", "Thermodynamics", "Biochemical Pathways"
        ]

        # Define random image sizes (A4-like formats)
        IMAGE_SIZES = [(1000, 1400), (1200, 1600), (1400, 1800), (1600, 2000), (1800, 2200), (2000, 2400)]

        # Choose a random image size
        img_size = random.choice(IMAGE_SIZES)
        img = Image.new('RGB', img_size, 'white')
        draw = ImageDraw.Draw(img)

        # Draw a simple header and footer
        header_height, footer_height = 100, 80
        draw.rectangle([0, 0, img_size[0], header_height], fill="lightgray")  # Header
        draw.rectangle([0, img_size[1] - footer_height, img_size[0], img_size[1]], fill="lightgray")  # Footer

        # Generate random report content
        report_data = {
            "title": fake.sentence(nb_words=6),
            "abstract": "\n".join([fake.sentence() for _ in range(3)]),
            "author": fake.name(),
            "date": fake.date_this_year().strftime("%m/%d/%y"),
            "keywords": ", ".join(random.sample(SCIENTIFIC_TERMS, 4))
        }

        # Define starting positions
        x, y = 100, header_height + 50

        # Store bounding boxes for NER
        ner_annotations = []

        # Draw Report Title
        title_font = ImageFont.truetype(random.choice(FONTS), 50)
        draw.text((x, y), report_data["title"], font=title_font, fill="black")
        
        # Use textbbox() instead of textsize()
        text_bbox = draw.textbbox((x, y), report_data["title"], font=title_font)
        title_bbox = [text_bbox[0], text_bbox[1], text_bbox[2], text_bbox[3]]
        ner_annotations.append({"label": "title", "content": report_data["title"], "bounding_box": title_bbox})
        
        y += text_bbox[3] - text_bbox[1] + 40  # Move down

        # Function to add text & store bounding box
        def add_text(label, content, font_size=30, offset=15):
            nonlocal y
            font = ImageFont.truetype(random.choice(FONTS), font_size)
            draw.text((x, y), content, font=font, fill="black")
            
            # Use textbbox() instead of textsize()
            text_bbox = draw.textbbox((x, y), content, font=font)
            bounding_box = [text_bbox[0], text_bbox[1], text_bbox[2], text_bbox[3]]
            ner_annotations.append({"label": label.lower(), "content": content, "bounding_box": bounding_box})
            
            y += text_bbox[3] - text_bbox[1] + offset  # Move down

        # Add report content
        add_text("author", f"Author: {report_data['author']}", font_size=28, offset=20)
        add_text("date", f"Date: {report_data['date']}", font_size=28, offset=20)
        add_text("keywords", f"Keywords: {report_data['keywords']}", font_size=28, offset=20)

        # Add abstract section
        add_text("abstract", "Abstract:", font_size=32, offset=10)
        for line in report_data["abstract"].split("\n"):
            add_text("abstract_content", line, font_size=26, offset=10)

        # Convert to OpenCV for noise & rotation
        image_cv = np.array(img)

        # Add Gaussian noise
        noise = np.random.normal(0, 15, image_cv.shape).astype(np.uint8)
        noisy_image = cv2.add(image_cv, noise)

        # Rotate image slightly
        angle = random.uniform(-5, 5)
        center = (img_size[0] // 2, img_size[1] // 2)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated_image = cv2.warpAffine(noisy_image, matrix, (img_size[0], img_size[1]), borderMode=cv2.BORDER_REPLICATE)

        # Save NER annotations
        GT_json = {
            "document_class": "scientific_report",
            "NER": ner_annotations
        }

        return GT_json, rotated_image


    def specifications(self, FONTS):
        IMAGE_SIZES = [(1000, 1400), (1200, 1600), (1400, 1800), (1600, 2000), (1800, 2200), (2000, 2400)]
        DEFAULT_FONT = random.choice(FONTS)

        def generate_text(text, font_size):
            try:
                font = ImageFont.truetype(DEFAULT_FONT, font_size)
            except Exception as e:
                print(f"Error loading font: {e}")
                return None, (0, 0, 0, 0)

            # Create a dummy image to get text size
            temp_img = Image.new("RGBA", (10, 10), (255, 255, 255, 0))
            temp_draw = ImageDraw.Draw(temp_img)
            
            # Get text bounding box (Pillow 10+)
            bbox = temp_draw.textbbox((0, 0), text, font=font)  # Returns (x0, y0, x1, y1)
            text_width, text_height = bbox[2] - bbox[0], bbox[3] - bbox[1]

            # Create a blank image with transparent background
            img = Image.new("RGBA", (text_width + 10, text_height + 10), (255, 255, 255, 0))
            draw = ImageDraw.Draw(img)

            # Write text
            draw.text((5, 5), text, font=font, fill=(0, 0, 0, 255))

            # Rotate text slightly
            angle = random.uniform(-5, 5)
            img = img.rotate(angle, expand=True)

            return img, (0, 0, text_width, text_height)

        # Select a random image size
        img_size = random.choice(IMAGE_SIZES)
        img = Image.new("RGB", img_size, "white")
        draw = ImageDraw.Draw(img)

        # Add header and footer
        header_height, footer_height = 100, 80
        draw.rectangle([0, 0, img_size[0], header_height], fill="lightgray")  # Header
        draw.rectangle([0, img_size[1] - footer_height, img_size[0], img_size[1]], fill="lightgray")  # Footer

        # Generate random specifications
        specification_data = {
            "title": f"{fake.company()} - Product Specifications",
            "date": fake.date_this_year().strftime("%m/%d/%y"),
            "model": fake.bothify(text="Model-###X"),
            "manufacturer": fake.company(),
            "weight": f"{random.uniform(0.5, 50):.2f} kg",
            "dimensions": f"{random.randint(10, 200)}x{random.randint(10, 200)}x{random.randint(10, 200)} cm",
            "battery": f"{random.randint(2000, 10000)} mAh",
            "processor": f"{random.choice(['Intel i7', 'AMD Ryzen 9', 'Apple M2', 'Snapdragon 888'])}",
            "storage": f"{random.choice(['128GB SSD', '256GB SSD', '512GB NVMe', '1TB HDD'])}",
            "display": f"{random.uniform(5.0, 17.0):.1f} inch {random.choice(['LCD', 'OLED', 'AMOLED'])}",
        }

        # Define text positions
        x, y = 50, header_height + 40  # Start slightly lower from the header
        ner_annotations = []

        # Function to add text & store bounding box
        def add_text(label, content, font_size=32):
            nonlocal y
            text_img, bbox = generate_text(content, font_size)
            if text_img is not None:
                img.paste(text_img, (x, y), text_img)
                bbox = [x, y, x + bbox[2], y + bbox[3]]  # Convert to absolute position
                ner_annotations.append({"label": label.lower(), "content": content, "bounding_box": bbox})
                y += bbox[3] + int(font_size * 0.4)  # Dynamically adjust spacing (smaller gap)

        # Add text fields
        add_text("title", specification_data["title"], font_size=40)
        add_text("date", f"Date: {specification_data['date']}", font_size=30)
        add_text("model", f"Model: {specification_data['model']}", font_size=30)
        add_text("manufacturer", f"Manufacturer: {specification_data['manufacturer']}", font_size=30)
        add_text("weight", f"Weight: {specification_data['weight']}", font_size=30)
        add_text("dimensions", f"Dimensions: {specification_data['dimensions']}", font_size=30)
        add_text("battery", f"Battery: {specification_data['battery']}", font_size=30)
        add_text("processor", f"Processor: {specification_data['processor']}", font_size=30)
        add_text("storage", f"Storage: {specification_data['storage']}", font_size=30)
        add_text("display", f"Display: {specification_data['display']}", font_size=30)

        # Convert to OpenCV for noise & rotation
        image_cv = np.array(img)

        # Ensure correct dtype for OpenCV processing
        if image_cv.dtype != np.uint8:
            image_cv = image_cv.astype(np.uint8)

        # Add Gaussian noise
        noise = np.random.normal(0, 15, image_cv.shape).astype(np.uint8)
        noisy_image = cv2.add(image_cv, noise)

        # Rotate the image slightly
        angle = random.uniform(-5, 5)
        center = (img_size[0] // 2, img_size[1] // 2)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated_image = cv2.warpAffine(noisy_image, matrix, (img_size[0], img_size[1]), borderMode=cv2.BORDER_REPLICATE)

        # Save NER annotations
        GT_json = {
            "document_class": "specifications",
            "NER": ner_annotations
        }

        return GT_json, rotated_image
        

        
    def generate_document(self):
        FONTS = [os.path.join(script_dir, "fonts/Arial.ttf"), 
                 os.path.join(script_dir, "fonts/Arial_Bold_Italic.ttf"),
                 os.path.join(script_dir, "fonts/Courier_New.ttf"), 
                 os.path.join(script_dir, "fonts/DroidSans-Bold.ttf"), 
                 os.path.join(script_dir, "fonts/FiraMono-Regular.ttf"), 
                 os.path.join(script_dir, "fonts/Times New Roman.ttf"), 
                 os.path.join(script_dir, "fonts/Vera.ttf"), 
                 os.path.join(script_dir, "fonts/Verdana_Bold_Italic.ttf"), 
                 os.path.join(script_dir, "fonts/Verdana.ttf"), 
                 os.path.join(script_dir, "fonts/DejaVuSansMono-Bold.ttf")
                ]

        # Define HANDWRITTEN_FONTS
        HANDWRITTEN_FONTS = [
            os.path.join(script_dir, "handwritten_fonts/Mayonice.ttf")
        ]

        # Map functions to their respective argument lists
        function_map = {
            self.advertisement: FONTS,
            self.budget:FONTS,
            self.email:FONTS,
            self.file_folder:FONTS,
            self.form:FONTS,
            self.handwritten:HANDWRITTEN_FONTS,
            self.invoice:FONTS,
            self.letter:FONTS,
            self.memo:FONTS,
            self.news_article:FONTS,
            self.presentation:FONTS,
            self.questionnaire:FONTS,
            self.resume:FONTS,
            self.scientific_publication:FONTS,
            self.scientific_report:FONTS,
            self.specifications:FONTS,
        }

        # Randomly select a function
        selected_function = random.choice(list(function_map.keys()))

        # Call the selected function with its corresponding argument
        GT_json, image = selected_function(function_map[selected_function])
        if image.ndim == 2:  # Grayscale
            final_image = Image.fromarray(image)
        else:  # Convert from RGB to BGR for OpenCV compatibility
            final_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))


        return GT_json, final_image