import random
import json
from PIL import Image, ImageDraw, ImageFont
import math
from math import floor
import io
import numpy as np
import cv2
from faker import Faker
import os
import uuid
from datetime import date, timedelta

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
            "advertisement_title": "LIMITED TIME OFFER!",
            "company_name": fake.company(),
            "contact_phone": fake.phone_number(),
            "contact_email": fake.email(),
            "website": fake.url(),
            "product_service_name": fake.catch_phrase(),
            "description": fake.sentence(),
            "features_benefits": fake.sentence(),
            "pricing": f"${random.randint(10, 500)}",
            "promotional_offers": f"{random.randint(10, 50)}% OFF",
            "call_to_action": random.choice(["Call Now!", "Visit Us Today!", "Order Now!"]),
            "advertisement_date": fake.future_date().strftime("%Y-%m-%d"),
            "location": fake.address(),
            "social_media_links": [fake.url() for _ in range(random.randint(0, 3))],
            "legal_disclaimers": random.choice(["Limited stock available.", "Terms and conditions apply.", "No refunds on promotional items."])
        }

        # Define starting positions
        x, y = 50, 50

        # Store bounding boxes for NER
        ner_annotations = {
            "advertisement_title": {"text": "", "bounding_box": []},
            "advertiser_information": {
                "company_name": {"text": "", "bounding_box": []},
                "contact_information": {
                    "phone": {"text": "", "bounding_box": []},
                    "email": {"text": "", "bounding_box": []},
                    "website": {"text": "", "bounding_box": []}
                }
            },
            "product_service_details": {
                "product_service_name": {"text": "", "bounding_box": []},
                "description": {"text": "", "bounding_box": []},
                "features_benefits": {"text": "", "bounding_box": []},
                "pricing": {"text": "", "bounding_box": []}
            },
            "promotional_offers": {"text": "", "bounding_box": []},
            "call_to_action": {"text": "", "bounding_box": []},
            "advertisement_date": {"text": "", "bounding_box": []},
            "location_information": {"text": "", "bounding_box": []},
            "social_media_links": [],
            "legal_disclaimers": {"text": "", "bounding_box": []}
        }

        # Draw "Advertisement" title
        title_font = ImageFont.truetype(random.choice(FONTS), 40)
        draw.text((x, y), metadata["advertisement_title"], font=title_font, fill="black")
        bbox = draw.textbbox((x, y), metadata["advertisement_title"], font=title_font)
        ner_annotations["advertisement_title"] = {"text": metadata["advertisement_title"], "bounding_box": bbox}
        y += (bbox[3] - bbox[1]) + 20  # Move down

        # Function to add text & store bounding box
        def add_text(field_path, label, content, font_size=25, offset=10):
            nonlocal y
            font = ImageFont.truetype(random.choice(FONTS), font_size)
            text = f"{label}: {content}"
            draw.text((x, y), text, font=font, fill="black")
            bbox = draw.textbbox((x, y), text, font=font)

            # Assign to the correct nested field
            temp = ner_annotations
            for key in field_path[:-1]:
                temp = temp[key]
            temp[field_path[-1]] = {"text": content, "bounding_box": bbox}

            y += (bbox[3] - bbox[1]) + offset  # Move down

        # Randomly include some fields to add variety
        add_text(["advertiser_information", "company_name"], "Company", metadata["company_name"])
        add_text(["advertiser_information", "contact_information", "phone"], "Phone", metadata["contact_phone"])
        if random.choice([True, False]):
            add_text(["advertiser_information", "contact_information", "email"], "Email", metadata["contact_email"])
        add_text(["advertiser_information", "contact_information", "website"], "Website", metadata["website"])
        
        add_text(["product_service_details", "product_service_name"], "Product", metadata["product_service_name"])
        add_text(["product_service_details", "description"], "Description", metadata["description"])
        
        if random.choice([True, False]):
            add_text(["product_service_details", "features_benefits"], "Features", metadata["features_benefits"])
        
        add_text(["product_service_details", "pricing"], "Price", metadata["pricing"])
        add_text(["promotional_offers"], "Offer", metadata["promotional_offers"], font_size=30, offset=15)
        add_text(["call_to_action"], "Action", metadata["call_to_action"])
        
        if random.choice([True, False]):
            add_text(["advertisement_date"], "Date", metadata["advertisement_date"])
        
        if random.choice([True, False]):
            add_text(["location_information"], "Location", metadata["location"])
        
        if metadata["social_media_links"]:
            for link in metadata["social_media_links"]:
                add_text(["social_media_links"], "Social Media", link, font_size=22, offset=15)
        
        if random.choice([True, False]):
            add_text(["legal_disclaimers"], "Disclaimer", metadata["legal_disclaimers"])

        # Convert to OpenCV for noise & rotation
        image_cv = np.array(img)

        # Add Gaussian noise
        noise = np.random.normal(0, 15, image_cv.shape).astype(np.uint8)
        noisy_image = cv2.add(image_cv, noise)

        # Rotate image slightly
        # Convert OpenCV BGR image to RGB format
        noisy_image_rgb = cv2.cvtColor(noisy_image, cv2.COLOR_BGR2RGB)

        # Convert NumPy array to PIL Image
        noisy_pil_image = Image.fromarray(noisy_image_rgb)

        # Rotate image slightly
        angle = random.uniform(-3, 3)
        rotated_image, ner_annotations = self.transform_bounding_boxes(ner_annotations, angle, noisy_pil_image)
        rotated_image = np.array(rotated_image)

        # Save NER annotations
        GT_json = {
            "document_class": "advertisement",
            "NER": ner_annotations
        }
        return GT_json, rotated_image

    def budget(self, FONTS):
        IMAGE_SIZES = [(800, 600), (900, 700), (1000, 750), (1100, 800), (1200, 900), (1300, 950)]
        img_size = random.choice(IMAGE_SIZES)
        img = Image.new('RGB', img_size, 'white')
        draw = ImageDraw.Draw(img)

        # Randomly decide which fields to include
        include_budget_name = random.choice([True, False])
        include_currency = random.choice([True, False])

        metadata = {
            "budget_name": fake.catch_phrase() if include_budget_name else None,
            "date": fake.date_this_year().strftime("%m/%d/%y"),
            "total_budget": round(random.uniform(5000, 50000), 2),
            "currency": random.choice(["USD", "EUR", "GBP", "CAD"]) if include_currency else None,
            "allocations": [
                {"category": fake.word(), "amount": round(random.uniform(100, 5000), 2)} for _ in range(random.randint(3, 6))
            ]
        }

        x, y = 50, 50
        ner_annotations = {}

        title_font = ImageFont.truetype(random.choice(FONTS), 40)
        draw.text((x, y), "BUDGET REPORT", font=title_font, fill="black")
        y += title_font.size + 20

        def add_text(key, content, font_size=25, offset=10, return_result=False):
            nonlocal y
            if content is None:
                return
            font = ImageFont.truetype(random.choice(FONTS), font_size)
            draw.text((x, y), f"{content}", font=font, fill="black")
            bbox = draw.textbbox((x, y), content, font=font)
            if return_result:
                y += (bbox[3] - bbox[1]) + offset
                return {"text": content, "bounding_box": list(bbox)}
            else:
                ner_annotations[key] = {"text": content, "bounding_box": list(bbox)}
            y += (bbox[3] - bbox[1]) + offset

        add_text("budget_name", metadata["budget_name"], font_size=30)
        add_text("date", metadata["date"])
        add_text("total_budget", f"Total Budget: ${metadata['total_budget']}", font_size=28, offset=15)
        add_text("currency", metadata["currency"], font_size=24)

        # Adding allocations
        allocations = []
        for allocation in metadata["allocations"]:
            category_bbox = add_text("category", allocation["category"], font_size=22, offset=5, return_result=True)
            amount_bbox = add_text("amount", f"${allocation['amount']}", font_size=22, offset=15, return_result=True)
            allocations.append({"category": category_bbox, "amount": amount_bbox})
        
        ner_annotations["allocations"] = allocations

        image_cv = np.array(img)
        noise = np.random.normal(0, 15, image_cv.shape).astype(np.uint8)
        noisy_image = cv2.add(image_cv, noise)
        # Convert OpenCV BGR image to RGB format
        noisy_image_rgb = cv2.cvtColor(noisy_image, cv2.COLOR_BGR2RGB)

        # Convert NumPy array to PIL Image
        noisy_pil_image = Image.fromarray(noisy_image_rgb)

        # Rotate image slightly
        angle = random.uniform(-3, 3)
        rotated_image, ner_annotations = self.transform_bounding_boxes(ner_annotations, angle, noisy_pil_image)
        rotated_image = np.array(rotated_image)

        GT_json = {"document_class": "budget", "NER": ner_annotations}
        return GT_json, rotated_image
        
    
    def email(self, FONTS):
        IMAGE_SIZES = [(800, 1000), (850, 1100), (900, 1200), (1000, 1300), (1100, 1400), (1200, 1500)]
        EMAIL_TYPES = ["Personal Email", "Business Email", "Notification Email", "Marketing Email"]

        img_size = random.choice(IMAGE_SIZES)
        img = Image.new('RGB', img_size, 'white')
        draw = ImageDraw.Draw(img)

        email_type = random.choice(EMAIL_TYPES)

        email_fields = {
            "sender_name": fake.name() if random.random() > 0.3 else "",
            "sender_email": fake.email(),
            "recipient_name": fake.name() if random.random() > 0.3 else "",
            "recipient_email": fake.email(),
            "date": fake.date(),
            "time": fake.time() if random.random() > 0.5 else "",
            "subject": fake.sentence(nb_words=6),
            "signature": fake.name() if random.random() > 0.6 else "",
        }
        
        if random.random() > 0.5:
            email_fields["cc"] = [fake.email() for _ in range(random.randint(1, 3))]
        if random.random() > 0.5:
            email_fields["bcc"] = [fake.email() for _ in range(random.randint(1, 3))]
        if random.random() > 0.4:
            email_fields["attachments"] = [fake.word() + ".pdf" for _ in range(random.randint(1, 2))]

        x, y = 50, 50
        ner_annotations = {}

        try:
            font_path = random.choice(FONTS)
            title_font = ImageFont.truetype(font_path, 30)
        except IOError:
            title_font = ImageFont.load_default()

        draw.text((x, y), email_type, font=title_font, fill="black")
        y += 50

        for label, value in email_fields.items():
            if not value:
                continue  # Skip empty fields

            try:
                font_path = random.choice(FONTS)
                font_size = random.randint(18, 28)
                font = ImageFont.truetype(font_path, font_size)
            except IOError:
                font = ImageFont.load_default()

            draw.text((x, y), f"{label.replace('_', ' ').title()}:", font=font, fill="black")
            text_x = x + 150
            
            if isinstance(value, list):
                bounding_boxes = []
                for item in value:
                    bbox = draw.textbbox((text_x, y), item, font=font)
                    draw.text((text_x, y), item, font=font, fill="black")
                    bounding_boxes.append([bbox[0], bbox[1], bbox[2], bbox[3]])
                    y += (bbox[3] - bbox[1]) + 10
                ner_annotations[label] = [{"text": v, "bounding_box": b} for v, b in zip(value, bounding_boxes)]
            else:
                bbox = draw.textbbox((text_x, y), value, font=font)
                draw.text((text_x, y), value, font=font, fill="black")
                ner_annotations[label] = {"text": value, "bounding_box": [bbox[0], bbox[1], bbox[2], bbox[3]]}
                y += (bbox[3] - bbox[1]) + 20

        image_cv = np.array(img)
        noise = np.random.normal(0, 15, image_cv.shape).astype(np.uint8)
        noisy_image = cv2.add(image_cv, noise)
        # Convert OpenCV BGR image to RGB format
        noisy_image_rgb = cv2.cvtColor(noisy_image, cv2.COLOR_BGR2RGB)

        # Convert NumPy array to PIL Image
        noisy_pil_image = Image.fromarray(noisy_image_rgb)

        # Rotate image slightly
        angle = random.uniform(-3, 3)
        rotated_image, ner_annotations = self.transform_bounding_boxes(ner_annotations, angle, noisy_pil_image)
        rotated_image = np.array(rotated_image)

        GT_json = {
            "document_class": "email",
            "NER": ner_annotations
        }
        
        return GT_json, rotated_image

    def file_folder(self, FONTS):
        IMAGE_SIZES = [(800, 600), (1024, 768), (1280, 720), (1280, 1024), (1600, 900), (1920, 1080)]
        
        def apply_scan_effects(img):
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            noise = np.random.normal(0, 25, img.shape).astype(np.uint8)
            img = cv2.add(img, noise)
            # Removing the rotation here - we'll rotate consistently at the end
            return img
        
        img_size = random.choice(IMAGE_SIZES)
        img = Image.new("RGB", img_size, "white")
        draw = ImageDraw.Draw(img)
        
        optional_fields = {
            "folder_title": fake.company() if random.random() > 0.2 else None,
            "folder_id": fake.uuid4()[:8] if random.random() > 0.3 else None,
            "creation_date": fake.date() if random.random() > 0.3 else None,
            "owner": fake.name() if random.random() > 0.4 else None,
            "department": fake.word().capitalize() if random.random() > 0.5 else None,
        }
        
        contained_documents = []
        for _ in range(random.randint(1, 3)):
            if random.random() > 0.3:
                contained_documents.append({
                    "document_title": fake.sentence(nb_words=3).rstrip("."),
                    "document_id": fake.uuid4()[:8],
                    "date_added": fake.date()
                })
        
        tags = []
        for _ in range(random.randint(1, 4)):
            if random.random() > 0.3:
                tags.append(fake.word())
        
        x, y = 50, 50
        line_spacing = 30
        bounding_boxes = {}
        
        try:
            font_path = random.choice(FONTS)
            font = ImageFont.truetype(font_path, 24)
        except IOError:
            font = ImageFont.load_default()
        
        for label, content in optional_fields.items():
            if content:
                text = f"{label.replace('_', ' ').capitalize()}: {content}"
                draw.text((x, y), text, fill="black", font=font)
                text_bbox = draw.textbbox((x, y), text, font=font)
                bounding_boxes[label] = {"text": content, "bounding_box": list(text_bbox)}
                y += line_spacing
        
        contained_doc_boxes = []
        for doc in contained_documents:
            doc_entry = {}
            for key, content in doc.items():
                text = f"{key.replace('_', ' ').capitalize()}: {content}"
                draw.text((x, y), text, fill="black", font=font)
                text_bbox = draw.textbbox((x, y), text, font=font)
                doc_entry[key] = {"text": content, "bounding_box": list(text_bbox)}
                y += line_spacing
            contained_doc_boxes.append(doc_entry)
        
        tag_boxes = []
        for tag in tags:
            text = f"Tag: {tag}"
            draw.text((x, y), text, fill="black", font=font)
            text_bbox = draw.textbbox((x, y), text, font=font)
            tag_boxes.append({"text": tag, "bounding_box": list(text_bbox)})
            y += line_spacing
        
        # Apply noise effects (without rotation)
        img_np = np.array(img)
        img_np = apply_scan_effects(img_np)
        
        ner_annotations = {
            "folder_title": bounding_boxes.get("folder_title", {"text": "", "bounding_box": []}),
            "folder_id": bounding_boxes.get("folder_id", {"text": "", "bounding_box": []}),
            "creation_date": bounding_boxes.get("creation_date", {"text": "", "bounding_box": []}),
            "owner": bounding_boxes.get("owner", {"text": "", "bounding_box": []}),
            "department": bounding_boxes.get("department", {"text": "", "bounding_box": []}),
            "contained_documents": contained_doc_boxes,
            "tags": tag_boxes
        }

        # Convert back to PIL for final rotation
        if len(img_np.shape) == 2:  # If grayscale
            pil_image = Image.fromarray(img_np)
        else:  # If RGB/BGR
            pil_image = Image.fromarray(cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB))

        # Rotate image and transform bounding boxes
        angle = random.uniform(-3, 3)
        rotated_image, updated_annotations = self.transform_bounding_boxes(ner_annotations, angle, pil_image)

        return {
            "document_class": "file_folder",
            "NER": updated_annotations
        }, np.array(rotated_image)

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
        title_bbox = draw.textbbox((50, 100), form_title, font=title_font)
        y = 160
        
        sections = {
            "applicant_details": {
                "full_name": fake.name(),
                "date_of_birth": fake.date_of_birth(minimum_age=18, maximum_age=60).strftime("%d/%m/%Y"),
                "gender": random.choice(["Male", "Female", "Other"]),
                "nationality": fake.country()
            },
            "contact_information": {
                "phone_number": fake.phone_number(),
                "email_address": fake.email(),
                "home_address": fake.address()
            },
            "identification_details": {
                "id_number": fake.ssn(),
                "social_security_number": fake.ssn()
            },
            "employment_details": {
                "company_name": fake.company(),
                "job_title": fake.job(),
                "work_address": fake.address()
            },
            "financial_details": {
                "account_number": fake.bban(),
                "taxpayer_id": fake.ssn(),
                "salary_information": f"${random.randint(30000, 150000)}"
            },
            "submission_date": fake.date(),
            "reference_number": fake.uuid4()[:8]
        }
        
        selected_sections = random.sample(list(sections.keys()), random.randint(3, len(sections)))
        gt_json = {"form_title": {"text": form_title, "bounding_box": list(title_bbox)}}
        
        for section in selected_sections:
            if isinstance(sections[section], dict):
                gt_json[section] = {}
                for field, value in sections[section].items():
                    font_path = random.choice(FONTS)
                    font = ImageFont.truetype(font_path, random.randint(20, 30))
                    draw.text((50, y), f"{field.replace('_', ' ').title()}:", font=font, fill="black")
                    bbox = draw.textbbox((250, y), value, font=font)
                    draw.text((250, y), value, font=font, fill="black")
                    draw.line([(250, bbox[3] + 3), (bbox[2], bbox[3] + 3)], fill="black", width=2)
                    gt_json[section][field] = {"text": value, "bounding_box": list(bbox)}
                    y += (bbox[3] - bbox[1]) + 30
            else:
                font_path = random.choice(FONTS)
                font = ImageFont.truetype(font_path, random.randint(20, 30))
                draw.text((50, y), f"{section.replace('_', ' ').title()}:", font=font, fill="black")
                bbox = draw.textbbox((250, y), sections[section], font=font)
                draw.text((250, y), sections[section], font=font, fill="black")
                draw.line([(250, bbox[3] + 3), (bbox[2], bbox[3] + 3)], fill="black", width=2)
                gt_json[section] = {"text": sections[section], "bounding_box": list(bbox)}
                y += (bbox[3] - bbox[1]) + 30
        
        image_cv = np.array(img)
        noise = np.random.normal(0, 15, image_cv.shape).astype(np.uint8)
        noisy_image = cv2.add(image_cv, noise)
        # Convert OpenCV BGR image to RGB format
        noisy_image_rgb = cv2.cvtColor(noisy_image, cv2.COLOR_BGR2RGB)

        # Convert NumPy array to PIL Image
        noisy_pil_image = Image.fromarray(noisy_image_rgb)

        # Rotate image slightly
        angle = random.uniform(-3, 3)
        rotated_image, ner_annotations = self.transform_bounding_boxes(gt_json, angle, noisy_pil_image)
        rotated_image = np.array(rotated_image)
        
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
            # We're no longer rotating individual text elements
            return img, bbox

        IMAGE_SIZES = [(1000, 1400), (1200, 1600), (1400, 1800), (1600, 2000), (1800, 2200), (2000, 2400)]
        img_size = random.choice(IMAGE_SIZES)
        img = Image.new('RGB', img_size, 'white')
        
        draw = ImageDraw.Draw(img)
        draw.rectangle([0, 0, img_size[0], 100], fill="lightgray")  # Header
        draw.rectangle([0, img_size[1] - 80, img_size[0], img_size[1]], fill="lightgray")  # Footer
        
        x, y = 100, 150
        ner_annotations = {"person_names": [], "dates": []}

        def add_handwritten_text(label, content, font_size=32, offset=20):
            nonlocal y
            text_img, bbox = generate_handwritten_text(content, random.choice(FONTS), font_size)
            if text_img:
                img.paste(text_img, (x, y), text_img)
                bounding_box = [x, y, x + bbox[2], y + bbox[3]]
                if label in ner_annotations.keys():
                    ner_annotations[label].append({"text": content, "bounding_box": bounding_box})
                y += bbox[3] + offset

        # Add text elements
        add_handwritten_text("person_names", fake.name(), 40)
        text1 = "\n".join(fake.sentences(nb=random.randint(4, 6)))
        add_handwritten_text("text1", text1)
        add_handwritten_text("dates", fake.date_this_year().strftime("%m/%d/%y"), 30)
        if random.choice([True, False]):
            text2 = "\n".join(fake.sentences(nb=random.randint(4, 6)))
            add_handwritten_text("text2", text2)

        # Add noise
        image_cv = np.array(img)
        noise = np.random.normal(0, 15, image_cv.shape).astype(np.uint8)
        noisy_image = cv2.add(image_cv, noise)
        
        # Convert OpenCV BGR image to RGB format
        noisy_image_rgb = cv2.cvtColor(noisy_image, cv2.COLOR_BGR2RGB)

        # Convert NumPy array to PIL Image
        noisy_pil_image = Image.fromarray(noisy_image_rgb)

        # Now rotate the entire image and transform bounding boxes
        angle = random.uniform(-3, 3)
        rotated_image, updated_annotations = self.transform_bounding_boxes(ner_annotations, angle, noisy_pil_image)
        
        return {"document_class": "handwritten", "NER": updated_annotations}, np.array(rotated_image)


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
            noise = np.random.normal(0, 0.5, np_image.shape).astype(np.uint8)
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
            y_offset = 50
            x_offset = 50
            font_size = random.choice(FONT_SIZES)
            font = ImageFont.truetype(FONT_PATH, font_size)

            # Initialize ground truth structure
            gt_template = {
                "organization": None,
                "date": None,
                "invoice_number": None,
                "payee_name": None,
                "purchased_item": [],
                "total_amount": None,
                "discount_amount": None,
                "tax_amount": None,
                "final_amount": None
            }

            # Draw and record bounding boxes
            def draw_text(label, text):
                """Helper function to draw text and record bounding box."""
                nonlocal y_offset
                nonlocal x_offset
                draw.text((x_offset, y_offset), text, fill="black", font=font)
                bbox = draw.textbbox((x_offset, y_offset), text, font=font)  # Corrected for Pillow 10.x.x

                if label in ["item", "quantity"]:
                    x_offset += 250
                else:
                    x_offset = 50
                    y_offset += font_size + 10  # Adjust line spacing
                return {"text": text, "bounding_box": bbox}


            gt_template["invoice_number"] = draw_text("invoice_number", f"Invoice #: {fake.uuid4()[:8]}")
            gt_template["date"] = draw_text("date", f"Date: {fake.date()}")
            gt_template["organization"] = draw_text("organization", f"{fake.company()}")
            gt_template["payee_name"] = draw_text("payee_name", f"Buyer: {fake.name()}")

            # Draw table headers
            y_offset += 20
            draw_text("items_header", "Item      Qty      Price")

            # Generate random invoice items
            for _ in range(random.randint(2, 5)):  # 2-5 items
                item = fake.word().capitalize()
                qty = random.randint(1, 5)
                price = f"${random.randint(10, 100)}"

                # Add item to GT structure
                gt_template["purchased_item"].append({
                    "item": draw_text("item", f"{item}"),
                    "quantity": draw_text("quantity", f"{qty}"),
                    "price": draw_text("price", f"{price}")
                })

            y_offset += 10

            gt_template["total_amount"] = draw_text("total_amount", f"Total: ${random.randint(100, 1000)}")

            if random.choice([True, False]):
                gt_template["discount_amount"] = draw_text("discount_amount", f"Discount: ${random.randint(5, 50)}")

            if random.choice([True, False]):
                gt_template["tax_amount"] = draw_text("tax_amount", f"Tax: ${random.randint(5, 50)}")

            gt_template["final_amount"] = draw_text("final_amount", f"Final Total: ${random.randint(150, 2000)}")

            return gt_template

        img_width, img_height = random.choice(IMAGE_SIZES)
        img = Image.new("RGB", (img_width, img_height), "white")
        draw = ImageDraw.Draw(img)

        # Generate invoice data and get GT annotations
        gt_annotations = generate_invoice_data(draw, img_width)

        # Add noise and rotate
        img = add_noise(img)
        angle = random.uniform(-3, 3)
        rotated_image, ner_annotations = self.transform_bounding_boxes(gt_annotations, angle, img)
        rotated_image = np.array(rotated_image)

        # Save annotations as JSON
        GT_json = {
            "document_class": "invoice",
            "NER": ner_annotations
        }

        return GT_json, rotated_image


    def letter(self, FONTS):
        IMAGE_SIZES = [(800, 1000), (850, 1100), (900, 1200), (1000, 1300), (1100, 1400), (1200, 1500)]
        TEMPLATES = ["simple", "header_footer"]

        font_path = random.choice(FONTS)
        font_size = random.randint(20, 30)

        img_size = random.choice(IMAGE_SIZES)
        img = Image.new('RGB', img_size, 'white')
        draw = ImageDraw.Draw(img)

        template = random.choice(TEMPLATES)
        
        if template == "header_footer":
            draw.rectangle([(0, 0), (img_size[0], 80)], fill="black")  # Header
            draw.rectangle([(0, img_size[1] - 80), (img_size[0], img_size[1])], fill="black")  # Footer
        
        # Generate document fields with optional inclusion
        sender_name = fake.name() 
        sender_address = fake.address() 
        sender_contact = fake.phone_number() 
        receiver_name = fake.name() 
        receiver_address = fake.address() 
        date = fake.date()
        attachments = [fake.word() for _ in range(random.randint(0, 3))]  # 0 to 3 attachments
        
        # Content mapping for drawing text and generating bounding boxes
        content = {
            "receiver_address": receiver_address,
            "date": date,
            "receiver_name": receiver_name,
            "body": "\n".join(fake.sentences(nb=random.randint(5, 8))),
            "sender_name": sender_name,
            "sender_address": sender_address,
            "sender_contact": sender_contact,
        }
        
        x, y = 50, 100
        ner_annotations = {}
        
        for label, text in content.items():
            if text:  # Only include if the field has content
                font = ImageFont.truetype(font_path, font_size)

                if label == "body":
                    y+=20
                if label == "receiver_name":
                    text = f"Dear {text}, "
                if label == "sender_name":
                    ending_word = random.choice(["Well-wisher", "Benefactor", "Patron", "Supporter"])
                    text = f"Your {ending_word}: {text}"
                
                bbox = draw.textbbox((x, y), text, font=font)
                draw.text((x, y), text, font=font, fill="black")
                y += bbox[3] - bbox[1] + 20  # Adjust Y position based on text height
                
                # Map label to GT key format
                if label == "body":
                    x += 300
                    y += 20
                    continue

                gt_key = label.lower().replace(" ", "_")
                if gt_key != "body":
                    ner_annotations[gt_key] = {"text": text, "bounding_box": bbox}
        
        # Convert to OpenCV for noise & rotation
        image_cv = np.array(img)
        noise = np.random.normal(0, 0.5, image_cv.shape).astype(np.uint8)
        noisy_image = np.clip(image_cv + noise, 0, 255)
        
        # Convert OpenCV BGR image to RGB format
        noisy_image_rgb = cv2.cvtColor(noisy_image, cv2.COLOR_BGR2RGB)

        # Convert NumPy array to PIL Image
        noisy_pil_image = Image.fromarray(noisy_image_rgb)

        # Rotate image slightly
        angle = random.uniform(-3, 3)
        rotated_image, ner_annotations = self.transform_bounding_boxes(ner_annotations, angle, noisy_pil_image)
        rotated_image = np.array(rotated_image)
        
        # Final GT JSON structure
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
            "sender_name": fake.name(),
            "sender_position": fake.job() if random.random() > 0.5 else "",
            "recipient_name": fake.name(),
            "recipient_position": fake.job() if random.random() > 0.5 else "",
            "cc": [fake.name() for _ in range(random.randint(0, 2))],
            "date": fake.date(),
            "subject": fake.sentence(nb_words=6),
            "reference_number": fake.uuid4() if random.random() > 0.7 else "",
            "attachments": [fake.word() for _ in range(random.randint(0, 2))],
            "body": "\n".join(fake.sentences(nb=random.randint(5, 7)))
        }

        x, y = 50, 50
        ner_annotations = {
            "sender_name": {"text": "", "bounding_box": []},
            "sender_position": {"text": "", "bounding_box": []},
            "recipient_name": {"text": "", "bounding_box": []},
            "recipient_position": {"text": "", "bounding_box": []},
            "cc": [],
            "date": {"text": "", "bounding_box": []},
            "subject": {"text": "", "bounding_box": []},
            "reference_number": {"text": "", "bounding_box": []},
            "attachments": []
        }

        title_font = ImageFont.truetype(random.choice(FONTS), 40)
        draw.text((x, y), "MEMO", font=title_font, fill="black")
        text_bbox = draw.textbbox((x, y), "MEMO", font=title_font)
        y += text_bbox[3] - text_bbox[1] + 20

        def add_text(label, content, font_size=25, offset=10):
            nonlocal y
            if not content:
                return
            font = ImageFont.truetype(random.choice(FONTS), font_size)
            draw.text((x, y), f"{label}: {content}", font=font, fill="black")
            text_bbox = draw.textbbox((x, y), f"{label}: {content}", font=font)
            bounding_box = [x, y, text_bbox[2], text_bbox[3]]
            
            if label in ner_annotations:
                if label in ["cc", "attachments"]:
                    ner_annotations[label].append({"text": content, "bounding_box": bounding_box})
                else:
                    ner_annotations[label] = {"text": content, "bounding_box": bounding_box}
            
            y += text_bbox[3] - text_bbox[1] + offset
        
        for key, value in metadata.items():
            if isinstance(value, list):
                for item in value:
                    add_text(key, item, font_size=22, offset=5)
            else:
                add_text(key, value)
        
        image_cv = np.array(img)
        noise = np.random.normal(0, 15, image_cv.shape).astype(np.uint8)
        noisy_image = cv2.add(image_cv, noise)
        
        # Convert OpenCV BGR image to RGB format
        noisy_image_rgb = cv2.cvtColor(noisy_image, cv2.COLOR_BGR2RGB)

        # Convert NumPy array to PIL Image
        noisy_pil_image = Image.fromarray(noisy_image_rgb)

        # Rotate image slightly
        angle = random.uniform(-3, 3)
        rotated_image, ner_annotations = self.transform_bounding_boxes(ner_annotations, angle, noisy_pil_image)
        rotated_image = np.array(rotated_image)
        
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
            "category": fake.word() if random.random() > 0.5 else None,
            "source": fake.company() if random.random() > 0.5 else None,
            "content": fake.paragraph(nb_sentences=5)
        }

        x, y = 50, 50
        ner_annotations = {
            "headline": {},
            "author": {},
            "date": {},
            "category": {},
            "source": {},
            "content": {}
        }

        title_font = ImageFont.truetype(random.choice(FONTS), 40)
        draw.text((x, y), "NEWS ARTICLE", font=title_font, fill="black")
        text_bbox = draw.textbbox((x, y), "NEWS ARTICLE", font=title_font)
        y += text_bbox[3] - text_bbox[1] + 20

        def add_text(label, content, font_size=25, offset=10):
            nonlocal y
            if content:
                font = ImageFont.truetype(random.choice(FONTS), font_size)
                text = f"{label}: {content}"
                draw.text((x, y), text, font=font, fill="black")
                bbox = draw.textbbox((x, y), text, font=font)
                ner_annotations[label.lower()] = {"text": content, "bounding_box": bbox}
                y += bbox[3] - bbox[1] + offset

        add_text("Headline", metadata["headline"], font_size=28, offset=15)
        add_text("Author", metadata["author"])
        add_text("Date", metadata["date"])
        add_text("Category", metadata["category"])
        add_text("Source", metadata["source"])
        add_text("Content", metadata["content"], font_size=22, offset=15)

        image_cv = np.array(img)
        noise = np.random.normal(0, 15, image_cv.shape).astype(np.uint8)
        noisy_image = cv2.add(image_cv, noise)

        # Convert OpenCV BGR image to RGB format
        noisy_image_rgb = cv2.cvtColor(noisy_image, cv2.COLOR_BGR2RGB)

        # Convert NumPy array to PIL Image
        noisy_pil_image = Image.fromarray(noisy_image_rgb)

        # Rotate image slightly
        angle = random.uniform(-3, 3)
        rotated_image, ner_annotations = self.transform_bounding_boxes(ner_annotations, angle, noisy_pil_image)
        rotated_image = np.array(rotated_image)

        GT_json = {"document_class": "news_article", "NER": ner_annotations}
        
        return GT_json, rotated_image


    def presentation(self, FONTS):
        IMAGE_SIZES = [(1000, 700), (1200, 800), (1400, 900), (1600, 1000), (1800, 1100), (2000, 1200)]
        img_size = random.choice(IMAGE_SIZES)
        img = Image.new('RGB', img_size, 'white')
        draw = ImageDraw.Draw(img)

        border_thickness = 20
        draw.rectangle([border_thickness, border_thickness, img_size[0] - border_thickness, img_size[1] - border_thickness], outline="black", width=border_thickness)

        slide_data = {}

        slide_data["slide_title"] = fake.sentence(nb_words=6)
        slide_data["content"] = "\n".join([fake.sentence() for _ in range(random.randint(3, 6))])
        if random.choice([True, False]):
            slide_data["date"] = fake.date_this_year().strftime("%m/%d/%y")
        if random.choice([True, False]):
            slide_data["presenter"] = fake.name()
        
        x, y = 100, 120
        ner_annotations = {}

        def add_text(label, content, font_size=30, offset=15):
            nonlocal y
            font = ImageFont.truetype(random.choice(FONTS), font_size)
            draw.text((x, y), content, font=font, fill="black")
            x1, y1, x2, y2 = draw.textbbox((x, y), content, font=font)
            if label not in ner_annotations:
                ner_annotations[label] = [] if isinstance(content, list) else {}
            if isinstance(content, list):
                ner_annotations[label].append({"text": content, "bounding_box": [x1, y1, x2, y2]})
            else:
                if label == "content":
                    if ner_annotations[label].get("text", "") and ner_annotations[label].get("bounding_box", []):
                        ner_annotations[label]["text"] += " " + content

                        # Expand the bounding box to include the new text
                        prev_x1, prev_y1, prev_x2, prev_y2 = ner_annotations[label]["bounding_box"]
                        new_x1 = min(prev_x1, x1)
                        new_y1 = min(prev_y1, y1)
                        new_x2 = max(prev_x2, x2)
                        new_y2 = max(prev_y2, y2)

                        ner_annotations[label]["bounding_box"] = [new_x1, new_y1, new_x2, new_y2]
                    else:
                        ner_annotations[label] = {"text": content, "bounding_box": [x1, y1, x2, y2]}
                else:
                    ner_annotations[label] = {"text": content, "bounding_box": [x1, y1, x2, y2]}
            y = y2 + offset
        
        if "slide_title" in slide_data:
            add_text("slide_title", slide_data["slide_title"], font_size=50, offset=40)
        if "content" in slide_data:
            for line in slide_data["content"].split("\n"):
                add_text("content", line)
        if "date" in slide_data:
            add_text("date", f"Date: {slide_data['date']}", font_size=28, offset=20)
        if "presenter" in slide_data:
            add_text("presenter", f"Presenter: {slide_data['presenter']}", font_size=28, offset=20)
        
        image_cv = np.array(img)
        noise = np.random.normal(0, 15, image_cv.shape).astype(np.uint8)
        noisy_image = cv2.add(image_cv, noise)

        # Convert OpenCV BGR image to RGB format
        noisy_image_rgb = cv2.cvtColor(noisy_image, cv2.COLOR_BGR2RGB)

        # Convert NumPy array to PIL Image
        noisy_pil_image = Image.fromarray(noisy_image_rgb)

        # Rotate image slightly
        angle = random.uniform(-3, 3)
        rotated_image, ner_annotations = self.transform_bounding_boxes(ner_annotations, angle, noisy_pil_image)
        rotated_image = np.array(rotated_image)

        GT_json = {"document_class": "presentation", "NER": ner_annotations}
        return GT_json, rotated_image


    def questionnaire(self, FONTS):
        IMAGE_SIZES = [(800, 1000), (850, 1100), (900, 1200), (1000, 1300), (1100, 1400), (1200, 1500)]
        QUESTIONNAIRE_TYPES = ["Customer Feedback", "Medical Questionnaire", "Employee Survey", "Product Review"]
        img_size = random.choice(IMAGE_SIZES)
        img = Image.new('RGB', img_size, 'white')
        draw = ImageDraw.Draw(img)

        questionnaire_type = random.choice(QUESTIONNAIRE_TYPES)

        metadata = {"Title": questionnaire_type, "Date": fake.date()}
        include_respondent = random.choice([True, False])
        if include_respondent:
            metadata["Respondent Name"] = fake.name()
            metadata["Respondent ID"] = fake.uuid4()[:8]

        questions = []
        for _ in range(random.randint(5, 10)):
            question_text = fake.sentence(nb_words=8)
            answers = [fake.word() for _ in range(4)]
            questions.append({"question": question_text, "answers": answers})

        x, y = 50, 50
        ner_annotations = {}
        font_path = random.choice(FONTS)
        title_font = ImageFont.truetype(font_path, 30)
        draw.text((x, y), metadata["Title"], font=title_font, fill="black")
        x1, y1, x2, y2 = draw.textbbox((x, y), metadata["Title"], font=title_font)
        ner_annotations["title"] = {"text": metadata["Title"], "bounding_box": [x1, y1, x2, y2]}
        y += 50

        font = ImageFont.truetype(random.choice(FONTS), 20)
        draw.text((x, y), f"Date: {metadata['Date']}", font=font, fill="black")
        x1, y1, x2, y2 = draw.textbbox((x + 70, y), metadata['Date'], font=font)
        ner_annotations["date"] = {"text": metadata['Date'], "bounding_box": [x1, y1, x2, y2]}
        y += 40

        if include_respondent:
            draw.text((x, y), f"Name: {metadata['Respondent Name']}", font=font, fill="black")
            x1, y1, x2, y2 = draw.textbbox((x + 70, y), metadata['Respondent Name'], font=font)
            ner_annotations["respondent_name"] = {"text": metadata['Respondent Name'], "bounding_box": [x1, y1, x2, y2]}
            y += 40

            draw.text((x, y), f"ID: {metadata['Respondent ID']}", font=font, fill="black")
            x1, y1, x2, y2 = draw.textbbox((x + 50, y), metadata['Respondent ID'], font=font)
            ner_annotations["respondent_id"] = {"text": metadata['Respondent ID'], "bounding_box": [x1, y1, x2, y2]}
            y += 40

        # ner_annotations["questions"] = []
        for idx, q in enumerate(questions):
            font = ImageFont.truetype(random.choice(FONTS), 22)
            draw.text((x, y), f"Q{idx + 1}: {q['question']}", font=font, fill="black")
            x1, y1, x2, y2 = draw.textbbox((x + 50, y), q['question'], font=font)
            question_entry = {"text": q['question'], "bounding_box": [x1, y1, x2, y2], "answers": []}
            y += y2 - y1 + 10

            font = ImageFont.truetype(random.choice(FONTS), 20)
            for ans in q["answers"]:
                draw.rectangle([x, y, x + 20, y + 20], outline="black")
                draw.text((x + 30, y), ans, font=font, fill="black")
                x1, y1, x2, y2 = draw.textbbox((x + 30, y), ans, font=font)
                question_entry["answers"].append({"text": ans, "bounding_box": [x1, y1, x2, y2]})
                y += y2 - y1 + 5
            y += 15

            # ner_annotations["questions"].append(question_entry)

        image_cv = np.array(img)
        noise = np.random.normal(0, 15, image_cv.shape).astype(np.uint8)
        noisy_image = cv2.add(image_cv, noise)

        # Convert OpenCV BGR image to RGB format
        noisy_image_rgb = cv2.cvtColor(noisy_image, cv2.COLOR_BGR2RGB)

        # Convert NumPy array to PIL Image
        noisy_pil_image = Image.fromarray(noisy_image_rgb)

        # Rotate image slightly
        angle = random.uniform(-3, 3)
        rotated_image, ner_annotations = self.transform_bounding_boxes(ner_annotations, angle, noisy_pil_image)
        rotated_image = np.array(rotated_image)

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
            "skills": [fake.job() for _ in range(random.randint(3, 6))],
            "experience": [
                {"company": fake.company(), "position": fake.job(), "years": f"{random.randint(1, 10)} years"} 
                for _ in range(random.randint(1, 3))
            ],
            "education": [
                {"degree": fake.catch_phrase(), "institution": fake.company(), "year": random.randint(2000, 2022)}
                for _ in range(random.randint(1, 2))
            ],
            "certifications": [fake.bs() for _ in range(random.randint(1, 2))]
        }

        x, y = 50, 50
        ner_annotations = {}
        
        font_path = random.choice(FONTS)
        name_font = ImageFont.truetype(font_path, 35)
        draw.text((x, y), metadata["person_name"], font=name_font, fill="black")
        bbox = draw.textbbox((x, y), metadata['person_name'], font=name_font)
        ner_annotations["name"] = {"text": metadata['person_name'], "bounding_box": bbox}
        y = bbox[3] + 10

        font = ImageFont.truetype(random.choice(FONTS), 20)
        contact_info = f"{metadata['address']} | {metadata['phone']} | {metadata['email']}"
        draw.text((x, y), contact_info, font=font, fill="black")
        bbox = draw.textbbox((x, y), contact_info, font=font)
        ner_annotations["contact_info"] = {
            "email": {"text": metadata["email"], "bounding_box": bbox},
            "phone": {"text": metadata["phone"], "bounding_box": bbox},
            "address": {"text": metadata["address"], "bounding_box": bbox}
        }
        y = bbox[3] + 20

        for section in SECTIONS:
            draw.text((x, y), section.upper(), font=ImageFont.truetype(font_path, 25), fill="black")
            y += 30
            section_data = []

            if section == "Summary":
                text = metadata["summary"]
                draw.text((x, y), text, font=font, fill="black")
                bbox = draw.textbbox((x, y), text, font=font)
                # ner_annotations["summary"] = {"text": text, "bounding_box": bbox}
                y = bbox[3] + 20
                continue

            elif section == "Skills":
                for skill in metadata["skills"]:
                    draw.text((x, y), skill, font=font, fill="black")
                    bbox = draw.textbbox((x, y), skill, font=font)
                    section_data.append({"text": skill, "bounding_box": bbox})
                    y = bbox[3] + 10
                ner_annotations["skills"] = section_data
                continue

            elif section == "Experience":
                for exp in metadata["experience"]:
                    exp_text = f"{exp['position']} at {exp['company']} ({exp['years']})"
                    draw.text((x, y), exp_text, font=font, fill="black")
                    bbox = draw.textbbox((x, y), exp_text, font=font)
                    section_data.append({
                        "job_title": {"text": exp['position'], "bounding_box": bbox},
                        "company": {"text": exp['company'], "bounding_box": bbox},
                        "years": {"text": exp['years'], "bounding_box": bbox}
                    })
                    y = bbox[3] + 10
                ner_annotations["work_experience"] = section_data
                continue

            elif section == "Education":
                for edu in metadata["education"]:
                    edu_text = f"{edu['degree']} from {edu['institution']} ({edu['year']})"
                    draw.text((x, y), edu_text, font=font, fill="black")
                    bbox = draw.textbbox((x, y), edu_text, font=font)
                    section_data.append({
                        "degree": {"text": edu['degree'], "bounding_box": bbox},
                        "institution": {"text": edu['institution'], "bounding_box": bbox},
                        "year": {"text": str(edu['year']), "bounding_box": bbox}
                    })
                    y = bbox[3] + 10
                ner_annotations["education"] = section_data
                continue

        image_cv = np.array(img)
        noise = np.random.normal(0, 15, image_cv.shape).astype(np.uint8)
        noisy_image = cv2.add(image_cv, noise)
        
        # Convert OpenCV BGR image to RGB format
        noisy_image_rgb = cv2.cvtColor(noisy_image, cv2.COLOR_BGR2RGB)

        # Convert NumPy array to PIL Image
        noisy_pil_image = Image.fromarray(noisy_image_rgb)

        # Rotate image slightly
        angle = random.uniform(-3, 3)
        rotated_image, ner_annotations = self.transform_bounding_boxes(ner_annotations, angle, noisy_pil_image)
        rotated_image = np.array(rotated_image)
        
        GT_json = {"document_class": "resume", "NER": ner_annotations}
        return GT_json, rotated_image


    def scientific_publication(self, FONTS):
        SCIENTIFIC_TERMS = [
            "Neural Networks", "Quantum Computing", "DNA Sequencing", "Machine Learning", 
            "Black Hole Physics", "Thermodynamics", "Gene Editing", "Nanotechnology",
            "CRISPR-Cas9", "Protein Folding", "String Theory", "Artificial Intelligence",
            "Graph Theory", "Statistical Mechanics", "Bioinformatics", "Computational Neuroscience",
            "Cybernetics", "Photonics", "Astrobiology", "Synthetic Biology", "Cognitive Computing",
            "Quantum Cryptography", "Deep Reinforcement Learning", "Cosmology",
            "Evolutionary Algorithms", "Genomic Data Science", "Robotics", "Renewable Energy Technology"
        ]
        
        IMAGE_SIZES = [(1000, 1400), (1200, 1600), (1400, 1800), (1600, 2000), (1800, 2200), (2000, 2400)]
        img_size = random.choice(IMAGE_SIZES)
        img = Image.new('RGB', img_size, 'white')
        draw = ImageDraw.Draw(img)
        
        header_height, footer_height = 100, 80
        draw.rectangle([0, 0, img_size[0], header_height], fill="lightgray")
        draw.rectangle([0, img_size[1] - footer_height, img_size[0], img_size[1]], fill="lightgray")
        
        publication_data = {
            "title": fake.sentence(nb_words=6),
            "abstract": "\n".join([fake.sentence() for _ in range(3)]),
            "author": fake.name(),
            "affiliation": fake.company() if random.random() > 0.5 else "",
            "date": fake.date_this_year().strftime("%m/%d/%y"),
            "keywords": ", ".join(random.sample(SCIENTIFIC_TERMS, 4)),
            "doi": f"10.{random.randint(1000, 9999)}/{random.randint(10000, 99999)}",
            "journal_conference": fake.company() if random.random() > 0.7 else ""
        }
        
        x, y = 100, header_height + 50
        ner_annotations = {
            "title": {},
            "authors": [],
            "publication_date": {},
            "abstract": {"text": "", "bounding_box": []},
            "journal_conference_name": {"text": "", "bounding_box": []},
        }
        
        def add_text(label, content, font_size=30, offset=15, entry=None):
            nonlocal y
            font = ImageFont.truetype(random.choice(FONTS), font_size)
            draw.text((x, y), content, font=font, fill="black")
            text_bbox = draw.textbbox((x, y), content, font=font)

            if entry is not None:
                entry[label] = {"text": content, "bounding_box": text_bbox}
            else:
                ner_annotations[label] = {"text": content, "bounding_box": text_bbox}

            y += text_bbox[3] - text_bbox[1] + offset

        # Title
        add_text("title", publication_data["title"], font_size=50, offset=40)

        # Authors
        author_entry = {}
        add_text("name", publication_data["author"], font_size=28, offset=20, entry=author_entry)

        if publication_data["affiliation"]:
            add_text("affiliation", publication_data["affiliation"], font_size=28, offset=20, entry=author_entry)

        ner_annotations["authors"].append(author_entry)

        # Publication Date
        add_text("publication_date", publication_data["date"], font_size=28, offset=20)

        # Journal/Conference (if present)
        if publication_data["journal_conference"]:
            ner_annotations["journal_conference_name"] = {}
            add_text("journal_conference_name", publication_data["journal_conference"], font_size=28, offset=20)

        # Abstract
        add_text("abstract", "Abstract:", font_size=32, offset=10)
        for line in publication_data["abstract"].split("\n"):
            add_text("abstract", line, font_size=26, offset=10)

        # Convert Image to Noisy and Rotated Version
        image_cv = np.array(img)
        noise = np.random.normal(0, 15, image_cv.shape).astype(np.int16)
        noisy_image = np.clip(image_cv + noise, 0, 255).astype(np.uint8)

        # Convert OpenCV BGR image to RGB format
        noisy_image_rgb = cv2.cvtColor(noisy_image, cv2.COLOR_BGR2RGB)

        # Convert NumPy array to PIL Image
        noisy_pil_image = Image.fromarray(noisy_image_rgb)

        # Rotate image slightly
        angle = random.uniform(-3, 3)
        rotated_image, ner_annotations = self.transform_bounding_boxes(ner_annotations, angle, noisy_pil_image)
        rotated_image = np.array(rotated_image)

        GT_json = {
            "document_class": "scientific_publication",
            "NER": ner_annotations
        }

        return GT_json, rotated_image

        

    def scientific_report(self, FONTS):
        SCIENTIFIC_TERMS = [
            "Quantum Mechanics", "Neural Networks", "DNA Sequencing", "Photosynthesis",
            "Machine Learning", "Protein Folding", "Nanotechnology", "Gene Editing",
            "CRISPR-Cas9", "Black Hole", "String Theory", "Thermodynamics", "Biochemical Pathways",
            "Artificial Intelligence", "Blockchain in Healthcare", "Deep Learning", "Evolutionary Biology",
            "Metabolic Engineering", "Synthetic Biology", "Astronomical Spectroscopy", "Computational Linguistics",
            "Cybersecurity Threats", "Quantum Cryptography", "Exoplanet Detection", "Dark Matter Research"
        ]
        
        IMAGE_SIZES = [(1000, 1400), (1200, 1600), (1400, 1800), (1600, 2000), (1800, 2200), (2000, 2400)]
        img_size = random.choice(IMAGE_SIZES)
        img = Image.new('RGB', img_size, 'white')
        draw = ImageDraw.Draw(img)
        header_height, footer_height = 100, 80
        draw.rectangle([0, 0, img_size[0], header_height], fill="lightgray")
        draw.rectangle([0, img_size[1] - footer_height, img_size[0], img_size[1]], fill="lightgray")
        
        report_data = {
            "title": fake.sentence(nb_words=6),
            "author": [
                    {"name": fake.name(), "affiliation": fake.company()}
                    for _ in range(random.randint(2, 4))
                ],
            "date": fake.date_this_year().strftime("%m/%d/%y"),
            "keywords": random.sample(SCIENTIFIC_TERMS, random.randint(2, 4)),
            "report_id": fake.uuid4(),
            "funding_source": fake.company() if random.random() > 0.5 else None,
        }
        
        x, y = 100, header_height + 50
        ner_annotations = {}
        
        title_font = ImageFont.truetype(random.choice(FONTS), 50)
        draw.text((x, y), report_data["title"], font=title_font, fill="black")
        text_bbox = draw.textbbox((x, y), report_data["title"], font=title_font)
        ner_annotations["title"] = {"text": report_data["title"], "bounding_box": list(text_bbox)}
        y += text_bbox[3] - text_bbox[1] + 40

        auth_dict = {}
        auth_list = []
        def add_text(label, content, font_size=30, offset=15, is_list=False):
            nonlocal y
            nonlocal auth_dict
            nonlocal auth_list
            font = ImageFont.truetype(random.choice(FONTS), font_size)
            draw.text((x, y), content, font=font, fill="black")
            text_bbox = draw.textbbox((x, y), content, font=font)

            if label != "authors":
                if is_list:
                    if label in ["name", "affiliation"]:
                        if label not in auth_dict:
                            auth_dict[label] = {"text": content, "bounding_box": list(text_bbox)}
                        if label=="affiliation":
                            auth_list.append(auth_dict)
                            auth_dict = {}
                    else:
                        if label not in ner_annotations:
                            ner_annotations[label] = []
                        ner_annotations[label].append({"text": content, "bounding_box": list(text_bbox)})
                else:
                    ner_annotations[label] = {"text": content, "bounding_box": list(text_bbox)}
            y += text_bbox[3] - text_bbox[1] + offset
        
        add_text("authors", f"Authors: ", font_size=28, offset=20)
        for auth_info in report_data['author']:
            add_text("name", f"name: {auth_info['name']}", font_size=28, offset=20, is_list=True)
            add_text("affiliation", f"Affiliation: {auth_info['affiliation']}", font_size=28, offset=20, is_list=True)
        ner_annotations["authors"] = auth_list
        

        add_text("date", f"Date: {report_data['date']}", font_size=28, offset=20)
        for keyword in report_data["keywords"]:
            add_text("keywords", keyword, font_size=28, offset=20, is_list=True)
        add_text("report_id", f"Report ID: {report_data['report_id']}", font_size=28, offset=20)
        if report_data["funding_source"]:
            add_text("funding_source", f"Funding Source: {report_data['funding_source']}", font_size=28, offset=20)
        
        image_cv = np.array(img)
        noise = np.random.normal(0, 15, image_cv.shape).astype(np.uint8)
        noisy_image = cv2.add(image_cv, noise)
        # Convert OpenCV BGR image to RGB format
        noisy_image_rgb = cv2.cvtColor(noisy_image, cv2.COLOR_BGR2RGB)

        # Convert NumPy array to PIL Image
        noisy_pil_image = Image.fromarray(noisy_image_rgb)

        # Rotate image slightly
        angle = random.uniform(-3, 3)
        rotated_image, ner_annotations = self.transform_bounding_boxes(ner_annotations, angle, noisy_pil_image)
        rotated_image = np.array(rotated_image)
        
        GT_json = {"document_class": "scientific_report", "NER": ner_annotations}
        return GT_json, rotated_image


    def specifications(self, FONTS):
        IMAGE_SIZES = [(1000, 1400), (1200, 1600), (1400, 1800), (1600, 2000), (1800, 2200), (2000, 2400)]
        DEFAULT_FONT = random.choice(FONTS)

        def generate_text(text, font_size):
            """ Generates text as an image with correct bounding box """
            try:
                font = ImageFont.truetype(DEFAULT_FONT, font_size)
            except Exception as e:
                print(f"Error loading font: {e}")
                return None, (0, 0, 0, 0)

            temp_img = Image.new("RGBA", (10, 10), (255, 255, 255, 0))
            temp_draw = ImageDraw.Draw(temp_img)
            bbox = temp_draw.textbbox((0, 0), text, font=font)
            text_width, text_height = bbox[2] - bbox[0], bbox[3] - bbox[1]

            # Create image with padding
            padding = 8
            img = Image.new("RGBA", (text_width + padding, text_height + padding), (255, 255, 255, 0))
            draw = ImageDraw.Draw(img)
            draw.text((padding // 2, padding // 2), text, font=font, fill=(0, 0, 0, 255))

            # Apply slight rotation
            # angle = random.uniform(-3, 3)
            # img = img.rotate(angle, expand=True)

            return img, (0, 0, text_width, text_height)

        img_size = random.choice(IMAGE_SIZES)
        img = Image.new("RGB", img_size, "white")
        draw = ImageDraw.Draw(img)

        # Add header & footer
        header_height, footer_height = 100, 80
        draw.rectangle([0, 0, img_size[0], header_height], fill="lightgray")
        draw.rectangle([0, img_size[1] - footer_height, img_size[0], img_size[1]], fill="lightgray")

        specification_data = {
            "title": fake.company(),
            "date": fake.date_this_year().strftime("%m/%d/%y"),
            "organization": fake.company(),
            "key_sections": [{
                "section_title": fake.catch_phrase(),
                "section_number": fake.bothify(text="##.##")
            } for _ in range(random.randint(2, 4))],
            "regulatory_compliance": [fake.sentence() for _ in range(random.randint(0, 2))],
            "key_requirements": [fake.sentence() for _ in range(random.randint(1, 3))]
        }

        x, y = 50, header_height + 40
        line_spacing_factor = 1.5  # Increased line spacing factor

        ner_annotations = {
            "title": {"text": "", "bounding_box": []},
            "date": {"text": "", "bounding_box": []},
            "organization": {"text": "", "bounding_box": []},
            "key_sections": [],
            "regulatory_compliance": [],
            "key_requirements": []
        }

        def add_text(label, content, font_size=32):
            """ Adds text to the image with proper spacing and bounding box handling """
            nonlocal y

            text_img, bbox = generate_text(content, font_size)
            
            if text_img is not None:
                img.paste(text_img, (x, y), text_img)

                text_height = bbox[3]
                absolute_bbox = [x, y, x + bbox[2], y + text_height]

                if label in ["title", "date", "organization"]:
                    ner_annotations[label]["text"] = content
                    ner_annotations[label]["bounding_box"] = absolute_bbox
                elif label in ["section_title", "section_number"]:
                    y = y + int(text_height * line_spacing_factor)
                    return absolute_bbox
                else:
                    ner_annotations[label].append({"text": content, "bounding_box": absolute_bbox})

                # Update y-coordinate with proper spacing
                y = y + int(text_height * line_spacing_factor)
                return absolute_bbox
            return []

        # Add main elements
        add_text("title", specification_data["title"], font_size=40)
        add_text("date", specification_data["date"], font_size=30)
        add_text("organization", specification_data["organization"], font_size=30)

        # Add key sections with clear spacing
        key_section_spacing = 0
        for section in specification_data["key_sections"]:
            section_data = {
                "section_title": {"text": section["section_title"], "bounding_box": add_text("section_title", section["section_title"], font_size=28)},
                "section_number": {"text": section["section_number"], "bounding_box": add_text("section_number", section["section_number"], font_size=28)}  # Extra spacing added
            }
            ner_annotations["key_sections"].append(section_data)
            key_section_spacing+=140
        # Add regulatory compliance
        for compliance in specification_data["regulatory_compliance"]:
            ner_annotations["regulatory_compliance"].append({
                "text": compliance,
                "bounding_box": add_text("regulatory_compliance", compliance, font_size=28)
            })

        # Add key requirements
        for requirement in specification_data["key_requirements"]:
            ner_annotations["key_requirements"].append({
                "text": requirement,
                "bounding_box": add_text("key_requirements", requirement, font_size=28)
            })

        # Convert to OpenCV format
        image_cv = np.array(img)
        if image_cv.dtype != np.uint8:
            image_cv = image_cv.astype(np.uint8)

        # Add noise
        noise = np.random.normal(0, 15, image_cv.shape).astype(np.uint8)
        noisy_image = cv2.add(image_cv, noise)

        # Convert OpenCV BGR image to RGB format
        noisy_image_rgb = cv2.cvtColor(noisy_image, cv2.COLOR_BGR2RGB)

        # Convert NumPy array to PIL Image
        noisy_pil_image = Image.fromarray(noisy_image_rgb)

        # Rotate image slightly
        angle = random.uniform(-3, 3)
        rotated_image, ner_annotations = self.transform_bounding_boxes(ner_annotations, angle, noisy_pil_image)
        image_cv = np.array(rotated_image)

        GT_json = {
            "document_class": "specification",
            "NER": ner_annotations
        }

        return GT_json, image_cv


    def medical_document(self, FONTS):
        IMAGE_SIZES = [(1000, 1400), (1200, 1600), (1400, 1800), (1600, 2000), (1800, 2200), (2000, 2400)]
        DEFAULT_FONT = random.choice(FONTS)
        FONT_SIZES = [24, 28, 32, 36]

        def add_noise(image):
            """Adds random noise to an image."""
            np_image = np.array(image)
            noise = np.random.normal(0, 0.5, np_image.shape).astype(np.uint8)
            noisy_image = np.clip(np_image + noise, 0, 255)
            return Image.fromarray(noisy_image)

        def rotate_image(image):
            """Rotates the image slightly to mimic a scanned document."""
            angle = random.randint(-5, 5)
            return image.rotate(angle, expand=True)

        def draw_text(draw, text, x, y, font):
            """Helper function to draw text and return bounding box."""
            draw.text((x, y), text, fill="black", font=font)
            bbox = draw.textbbox((x, y), text, font=font)  
            return {"text": text, "bounding_box": bbox}

        # Create image
        img_width, img_height = random.choice(IMAGE_SIZES)
        img = Image.new("RGB", (img_width, img_height), "white")
        draw = ImageDraw.Draw(img)
        font_size = random.choice(FONT_SIZES)
        font = ImageFont.truetype(DEFAULT_FONT, font_size)

        # Initialize annotations
        ner_annotations = {"hospital": {}, "patient_information": {}, "physician_information": {},
                        "procedures": [], "diagnosis": [], "medical_history": {}, 
                        "medication": {}, "lab_tests": []}

        x_offset, y_offset = 50, 50

        # Hospital/Facility Name
        hospital_name = fake.company()
        hospital_keyword = random.choice(["Hospital", "Clinic", "Medical Facility"])
        hospital_name = f"{hospital_name} {hospital_keyword}"
        ner_annotations["hospital"]["name"] = draw_text(draw, hospital_name, x_offset, y_offset, font)
        y_offset += font_size + 20

        # Patient Information
        patient_info = {
            "patient name": fake.name(),
            "DOB": fake.date_of_birth(minimum_age=18, maximum_age=90).strftime("%Y-%m-%d"),
            "member_id": fake.uuid4()[:10].upper()
        }
        for key, value in patient_info.items():
            ner_annotations["patient_information"][key] = draw_text(draw, f"{key.capitalize()}: {value}", x_offset, y_offset, font)
            y_offset += font_size + 10

        # Physician Information
        y_offset += 20
        physician_info = {
            "physician name": fake.name(),
            "tax_id": fake.random_number(9, True),
            "provider_id": fake.uuid4()[:8].upper()
        }
        for key, value in physician_info.items():
            ner_annotations["physician_information"][key] = draw_text(draw, f"{key.capitalize()}: {value}", x_offset, y_offset, font)
            y_offset += font_size + 10

        y_offset += 20  # Space before next section

        # Procedures Section
        y_offset += 20
        for _ in range(random.randint(2, 4)):
            procedure = {
                "cpt_code": fake.random_int(10000, 99999),
                "reason": fake.sentence(nb_words=6)
            }
            proc_annotations = {
                "cpt_code": draw_text(draw, f"CPT: {procedure['cpt_code']}", x_offset, y_offset, font),
                "reason": draw_text(draw, f"Reason: {procedure['reason']}", x_offset, y_offset + font_size + 5, font)
            }
            ner_annotations["procedures"].append(proc_annotations)
            y_offset += font_size * 2 + 10

        y_offset += 20  # Space before next section

        # Diagnosis Section
        y_offset += 20
        for _ in range(random.randint(2, 4)):
            diagnosis = {
                "icd_code": f"{random.randint(1, 99)}.{random.randint(0, 99)}",
                "diagnosis": fake.sentence(nb_words=6)
            }
            diag_annotations = {
                "icd_code": draw_text(draw, f"ICD: {diagnosis['icd_code']}", x_offset, y_offset, font),
                "diagnosis": draw_text(draw, f"Diagnosis: {diagnosis['diagnosis']}", x_offset, y_offset + font_size + 5, font)
            }
            ner_annotations["diagnosis"].append(diag_annotations)
            y_offset += font_size * 2 + 10

        y_offset += 20  # Space before next section

        # Optional Sections
        if random.choice([True, False]):  # Medical History
            y_offset += 20
            history_text = "\n".join(fake.sentences(nb=random.randint(3, 5)))
            ner_annotations["medical_history"] = draw_text(draw, f"Medical History: {history_text}", x_offset, y_offset, font)
            y_offset += font_size * 5 + 10

        if random.choice([True, False]):  # Medication
            y_offset += 20
            ner_annotations["medication"] = {
                "past": draw_text(draw, f"Past Medications: {fake.sentence(nb_words=5)}", x_offset, y_offset, font),
                "prescribed": draw_text(draw, f"Prescribed Medications: {fake.sentence(nb_words=5)}", x_offset, y_offset + font_size + 5, font)
            }
            y_offset += font_size * 2 + 10

        if random.choice([True, False]):  # Lab Tests
            y_offset += 20
            for _ in range(random.randint(2, 4)):
                lab_test = {
                    "test_name": fake.word().capitalize(),
                    "normal_values": f"{random.randint(10, 100)}-{random.randint(101, 200)}",
                    "test_result": f"{random.randint(10, 200)} {random.choice(['Normal', 'High', 'Low'])}"
                }
                lab_annotations = {
                    "test_name": draw_text(draw, f"Test: {lab_test['test_name']}", x_offset, y_offset, font),
                    "normal_values": draw_text(draw, f"Normal: {lab_test['normal_values']}", x_offset, y_offset + font_size + 5, font),
                    "test_result": draw_text(draw, f"Result: {lab_test['test_result']}", x_offset, y_offset + font_size * 2 + 10, font)
                }
                ner_annotations["lab_tests"].append(lab_annotations)
                y_offset += font_size * 3 + 10

        # Add noise and rotate the image
        img = add_noise(img)
        angle = random.randint(-5, 5)
        rotated_image, ner_annotations = self.transform_bounding_boxes(ner_annotations, angle, img)
        image_cv = np.array(rotated_image)

        GT_json = {"document_class": "medical_document", "NER": ner_annotations}
        return GT_json, image_cv
    
    
    def driving_license(self, FONTS):
        """Generate realistic driving license documents (both scanned and digital versions)"""
        
        # Different license layouts and sizes with more variety
        DIGITAL_SIZES = [(800, 500), (850, 530), (900, 560), (780, 490), (820, 520)]
        SCANNED_SIZES = [(1000, 700), (1100, 750), (1200, 800), (950, 680), (1050, 720)]
        
        # Realistic US States data
        US_STATES = [
            ("California", "CA"), ("Texas", "TX"), ("Florida", "FL"), ("New York", "NY"),
            ("Pennsylvania", "PA"), ("Illinois", "IL"), ("Ohio", "OH"), ("Georgia", "GA"),
            ("North Carolina", "NC"), ("Michigan", "MI"), ("New Jersey", "NJ"), ("Virginia", "VA"),
            ("Washington", "WA"), ("Arizona", "AZ"), ("Massachusetts", "MA"), ("Tennessee", "TN"),
            ("Indiana", "IN"), ("Missouri", "MO"), ("Maryland", "MD"), ("Wisconsin", "WI"),
            ("Colorado", "CO"), ("Minnesota", "MN"), ("South Carolina", "SC"), ("Alabama", "AL"),
            ("Louisiana", "LA"), ("Kentucky", "KY"), ("Oregon", "OR"), ("Oklahoma", "OK"),
            ("Connecticut", "CT"), ("Utah", "UT"), ("Iowa", "IA"), ("Nevada", "NV"),
            ("Arkansas", "AR"), ("Mississippi", "MS"), ("Kansas", "KS"), ("New Mexico", "NM")
        ]
        
        # Realistic license data
        REAL_LICENSE_CLASSES = ["A", "B", "C", "CDL-A", "CDL-B", "CDL-C", "M", "D"]
        REAL_ENDORSEMENTS = ["NONE", "M", "H", "N", "P", "S", "T", "X", "P/S", "H/N"]
        REAL_RESTRICTIONS = ["NONE", "A", "B", "C", "D", "E", "F", "G", "A/B", "C/D"]
        REAL_EYE_COLORS = ["BLU", "BRN", "GRN", "HZL", "GRY", "BLK", "AMB", "DIC"]
        
        # Randomly decide if it's digital or scanned
        is_digital = random.choice([True, False])
        
        if is_digital:
            img_size = random.choice(DIGITAL_SIZES)
            bg_colors = ["white", "#f8f9fa", "#fefefe", "#f5f5f5", "#fbfbfb"]
            bg_color = random.choice(bg_colors)
        else:
            img_size = random.choice(SCANNED_SIZES)
            bg_color = random.choice(["white", "#fffffe", "#fefefe"])
        
        img = Image.new('RGB', img_size, bg_color)
        draw = ImageDraw.Draw(img)
        
        # Generate realistic license metadata
        state_name, state_abbr = random.choice(US_STATES)
        
        # Generate realistic dates with proper logic
        from datetime import date
        current_date = date.today()
        birth_date = fake.date_between(start_date='-75y', end_date='-16y')
        
        # Issue date should be after 18th birthday and before current date
        min_issue_age = birth_date.replace(year=birth_date.year + 16)
        earliest_issue = max(min_issue_age, current_date.replace(year=current_date.year - 8))
        issue_date = fake.date_between(start_date=earliest_issue, end_date=current_date)
        
        # Expiration date is typically 4-8 years from issue date
        exp_years = random.choice([4, 5, 6, 8])
        exp_date = issue_date.replace(year=issue_date.year + exp_years)
        
        # Generate realistic license number patterns by state
        license_patterns = {
            'CA': f"{fake.random_letter()}{fake.random_number(digits=7)}",
            'TX': f"{fake.random_number(digits=8)}",
            'FL': f"{fake.random_letter()}{fake.random_number(digits=3)}-{fake.random_number(digits=3)}-{fake.random_number(digits=2)}-{fake.random_number(digits=3)}-{fake.random_number(digits=1)}",
            'NY': f"{fake.random_number(digits=3)} {fake.random_number(digits=3)} {fake.random_number(digits=3)}",
            'default': f"{fake.random_letter()}{fake.random_number(digits=2)}{fake.random_number(digits=6)}"
        }
        
        license_number = license_patterns.get(state_abbr, license_patterns['default'])
        
        # Generate realistic height and weight combinations
        is_male = random.choice([True, False])
        if is_male:
            height_inches = random.randint(64, 78)  # 5'4" to 6'6"
            weight = random.randint(140, 280)
            sex = "M"
        else:
            height_inches = random.randint(60, 72)  # 5'0" to 6'0"
            weight = random.randint(110, 220)
            sex = "F"
        
        height_feet = height_inches // 12
        height_remaining = height_inches % 12
        height_str = f"{height_feet}'-{height_remaining:02d}\""
        
        metadata = {
            "full_name": fake.name_male() if is_male else fake.name_female(),
            "license_number": license_number,
            "date_of_birth": birth_date.strftime("%m/%d/%Y"),
            "address": fake.street_address() + ", " + fake.city() + ", " + state_abbr + " " + fake.zipcode(),
            "state": state_name,
            "state_abbr": state_abbr,
            "issue_date": issue_date.strftime("%m/%d/%Y"),
            "expiration_date": exp_date.strftime("%m/%d/%Y"),
            "license_class": random.choice(REAL_LICENSE_CLASSES),
            "endorsements": random.choice(REAL_ENDORSEMENTS),
            "restrictions": random.choice(REAL_RESTRICTIONS),
            "height": height_str,
            "weight": f"{weight} lbs",
            "eye_color": random.choice(REAL_EYE_COLORS),
            "sex": sex,
            "organ_donor": random.choice(["YES", "NO"]),
            "veteran": random.choice(["YES", "NO"]) if random.random() > 0.85 else None
        }
        
        # Initialize coordinates and annotations
        ner_annotations = {}
        
        # Choose fonts with more variety
        try:
            available_fonts = FONTS if FONTS else []
            if available_fonts:
                header_font = ImageFont.truetype(random.choice(available_fonts), random.randint(22, 26))
                regular_font = ImageFont.truetype(random.choice(available_fonts), random.randint(14, 18))
                small_font = ImageFont.truetype(random.choice(available_fonts), random.randint(10, 14))
                large_font = ImageFont.truetype(random.choice(available_fonts), random.randint(18, 22))
                name_font = ImageFont.truetype(random.choice(available_fonts), random.randint(16, 20))
            else:
                raise Exception("No fonts available")
        except:
            header_font = ImageFont.load_default()
            regular_font = ImageFont.load_default()
            small_font = ImageFont.load_default()
            large_font = ImageFont.load_default()
            name_font = ImageFont.load_default()
        
        # More diverse color schemes
        if is_digital:
            color_schemes = [
                {"header": "#1e40af", "text": "#000000", "accent": "#3b82f6"},
                {"header": "#166534", "text": "#000000", "accent": "#22c55e"},
                {"header": "#991b1b", "text": "#000000", "accent": "#ef4444"},
                {"header": "#6b21a8", "text": "#000000", "accent": "#a855f7"},
                {"header": "#ea580c", "text": "#000000", "accent": "#f97316"},
                {"header": "#0f766e", "text": "#000000", "accent": "#14b8a6"},
                {"header": "#7c2d12", "text": "#000000", "accent": "#f59e0b"}
            ]
            colors = random.choice(color_schemes)
            header_color = colors["header"]
            text_color = colors["text"]
            accent_color = colors["accent"]
        else:
            header_color = random.choice(["black", "#1a1a1a", "#2a2a2a"])
            text_color = random.choice(["black", "#0a0a0a", "#1a1a1a"])
            accent_color = random.choice(["black", "#333333", "#444444"])
        
        # FIXED: More consistent layout parameters for better spacing
        y_start = random.randint(20, 30)
        x_left = random.randint(30, 45)
        # Reduced spacing variation to prevent overlap
        base_spacing = 22  # Consistent base spacing
        spacing_variation = random.randint(2, 6)  # Much smaller variation
        
        y = y_start
        
        # More diverse header layouts
        header_styles = ["standard", "compact", "spaced", "centered"]
        header_style = random.choice(header_styles)
        
        if header_style == "centered":
            # Centered header
            state_header = f"{state_name.upper()} DRIVER LICENSE"
            header_bbox = draw.textbbox((0, 0), state_header, font=header_font)
            header_width = header_bbox[2] - header_bbox[0]
            header_x = (img_size[0] - header_width) // 2
            draw.text((header_x, y), state_header, font=header_font, fill=header_color)
            bbox = draw.textbbox((header_x, y), state_header, font=header_font)
            ner_annotations["state"] = {"text": state_name, "bounding_box": list(bbox)}
            y = bbox[3] + base_spacing + spacing_variation
        else:
            # Standard left-aligned header
            header_variants = [
                f"{state_name.upper()} DRIVER LICENSE",
                f"{state_name.upper()} DRIVER'S LICENSE", 
                f"{state_name.upper()} DRIVERS LICENSE",
                f"STATE OF {state_name.upper()} DRIVER LICENSE"
            ]
            state_header = random.choice(header_variants)
            draw.text((x_left, y), state_header, font=header_font, fill=header_color)
            bbox = draw.textbbox((x_left, y), state_header, font=header_font)
            ner_annotations["state"] = {"text": state_name, "bounding_box": list(bbox)}
            y = bbox[3] + base_spacing + spacing_variation
        
        # State abbreviation placement with variation
        abbr_positions = [
            (img_size[0] - 80, 20),
            (img_size[0] - 100, 25),
            (img_size[0] - 90, 30),
            (img_size[0] - 70, 15)
        ]
        abbr_x, abbr_y = random.choice(abbr_positions)
        draw.text((abbr_x, abbr_y), state_abbr, font=large_font, fill=accent_color)
        
        # License number with varied prefixes
        license_prefixes = ["DL", "LIC", "LICENSE", "ID", state_abbr, ""]
        prefix = random.choice(license_prefixes)
        if prefix:
            lic_text = f"{prefix} {metadata['license_number']}"
        else:
            lic_text = metadata['license_number']
        
        draw.text((x_left, y), lic_text, font=large_font, fill=text_color)
        bbox = draw.textbbox((x_left, y), lic_text, font=large_font)
        ner_annotations["license_number"] = {"text": metadata['license_number'], "bounding_box": list(bbox)}
        y = bbox[3] + base_spacing + 5
        
        # FIXED: Better photo placeholder positioning and size calculation
        photo_sizes = [(120, 150), (110, 140), (130, 160), (125, 155)]
        photo_size = random.choice(photo_sizes)
        photo_margin = random.randint(30, 45)
        photo_x = img_size[0] - photo_size[0] - photo_margin
        photo_y = y
        
        # Calculate photo boundaries for overlap detection
        photo_left = photo_x
        photo_right = photo_x + photo_size[0]
        photo_top = photo_y
        photo_bottom = photo_y + photo_size[1]
        
        # Varied photo placeholder styles
        photo_styles = ["standard", "rounded", "bordered"]
        photo_style = random.choice(photo_styles)
        
        if photo_style == "rounded":
            # Simulate rounded corners with multiple rectangles
            draw.rectangle([photo_x+2, photo_y, photo_x + photo_size[0]-2, photo_y + photo_size[1]], 
                        outline="gray", fill="#e5e5e5")
            draw.rectangle([photo_x, photo_y+2, photo_x + photo_size[0], photo_y + photo_size[1]-2], 
                        outline="gray", fill="#e5e5e5")
        elif photo_style == "bordered":
            draw.rectangle([photo_x-2, photo_y-2, photo_x + photo_size[0]+2, photo_y + photo_size[1]+2], 
                        outline=accent_color, fill=accent_color, width=2)
            draw.rectangle([photo_x, photo_y, photo_x + photo_size[0], photo_y + photo_size[1]], 
                        outline="gray", fill="#e0e0e0")
        else:
            draw.rectangle([photo_x, photo_y, photo_x + photo_size[0], photo_y + photo_size[1]], 
                        outline="gray", fill="gray")
        
        # Varied photo text
        photo_texts = ["PHOTO", "PICTURE", "IMAGE", "ID PHOTO", ""]
        photo_text = random.choice(photo_texts)
        if photo_text:
            text_bbox = draw.textbbox((0, 0), photo_text, font=small_font)
            text_width = text_bbox[2] - text_bbox[0]
            draw.text((photo_x + (photo_size[0] - text_width)//2, photo_y + 70), 
                    photo_text, font=small_font, fill="gray")
        
        # FIXED: Main information section with better overlap prevention
        info_layouts = ["standard", "compact", "spaced", "two_column"]
        info_layout = random.choice(info_layouts)
        
        # Field label variations
        field_labels = {
            "full_name": random.choice(["Name:", "Full Name:", "NAME:", "Licensee:"]),
            "address": random.choice(["Address:", "ADDRESS:", "Residence:", "Home Address:"]),
            "date_of_birth": random.choice(["Date of Birth:", "DOB:", "Born:", "Birth Date:", "DATE OF BIRTH:"]),
            "sex": random.choice(["Sex:", "Gender:", "SEX:", "M/F:"]),
            "height": random.choice(["Height:", "HGT:", "HEIGHT:", "Ht:"]),
            "weight": random.choice(["Weight:", "WGT:", "WEIGHT:", "Wt:"]),
            "eye_color": random.choice(["Eyes:", "Eye Color:", "EYE:", "EYES:"]),
            "license_class": random.choice(["Class:", "LICENSE CLASS:", "Type:", "Category:"]),
            "endorsements": random.choice(["Endorsements:", "ENDORSE:", "End:", "Special:"]),
            "restrictions": random.choice(["Restrictions:", "RESTRICT:", "Rest:", "Limits:"])
        }
        
        info_items = [
            (field_labels["full_name"], metadata['full_name'], "full_name"),
            (field_labels["address"], metadata['address'], "address"),
            (field_labels["date_of_birth"], metadata['date_of_birth'], "date_of_birth"),
            (field_labels["sex"], metadata['sex'], "sex"),
            (field_labels["height"], metadata['height'], "height"),
            (field_labels["weight"], metadata['weight'], "weight"),
            (field_labels["eye_color"], metadata['eye_color'], "eye_color"),
            (field_labels["license_class"], metadata['license_class'], "license_class"),
            (field_labels["endorsements"], metadata['endorsements'], "endorsements"),
            (field_labels["restrictions"], metadata['restrictions'], "restrictions")
        ]
        
        if info_layout == "two_column":
            # FIXED: Better two column layout with proper spacing
            col1_items = info_items[:5]
            col2_items = info_items[5:]
            
            # Calculate safe column positions
            max_label_width = max([draw.textbbox((0, 0), item[0], font=regular_font)[2] for item in info_items])
            col1_value_x = x_left + max_label_width + 10
            col2_x = max(col1_value_x + 200, x_left + 320)  # Ensure minimum distance
            col2_value_x = col2_x + max_label_width + 10
            
            # Ensure column 2 doesn't overlap with photo
            if col2_value_x + 150 > photo_left:  # 150px estimated max value width
                # Fall back to single column if two columns would overlap
                info_layout = "standard"
            else:
                # Column 1
                col_y = y
                for label, value, key in col1_items:
                    draw.text((x_left, col_y), label, font=regular_font, fill=text_color)
                    draw.text((col1_value_x, col_y), str(value), font=name_font if key == "full_name" else regular_font, fill=text_color)
                    value_bbox = draw.textbbox((col1_value_x, col_y), str(value), font=name_font if key == "full_name" else regular_font)
                    ner_annotations[key] = {"text": str(value), "bounding_box": list(value_bbox)}
                    col_y += base_spacing + random.randint(0, spacing_variation)
                
                # Column 2
                col_y = y
                for label, value, key in col2_items:
                    draw.text((col2_x, col_y), label, font=regular_font, fill=text_color)
                    draw.text((col2_value_x, col_y), str(value), font=regular_font, fill=text_color)
                    value_bbox = draw.textbbox((col2_value_x, col_y), str(value), font=regular_font)
                    ner_annotations[key] = {"text": str(value), "bounding_box": list(value_bbox)}
                    col_y += base_spacing + random.randint(0, spacing_variation)
        
        if info_layout != "two_column":  # Single column layouts (including fallback)
            # FIXED: Consistent value alignment
            max_label_width = max([draw.textbbox((0, 0), item[0], font=regular_font)[2] for item in info_items])
            value_x = x_left + max_label_width + 15  # Consistent alignment
            
            for label, value, key in info_items:
                # FIXED: Consistent spacing instead of random
                y += base_spacing + random.randint(0, spacing_variation)
                
                # FIXED: Better overlap detection and handling
                if y >= photo_top and y <= photo_bottom and value_x + 200 > photo_left:
                    # Skip to below photo if there would be overlap
                    y = max(y, photo_bottom + 15)
                
                draw.text((x_left, y), label, font=regular_font, fill=text_color)
                
                font_to_use = name_font if key == "full_name" else regular_font
                draw.text((value_x, y), str(value), font=font_to_use, fill=text_color)
                value_bbox = draw.textbbox((value_x, y), str(value), font=font_to_use)
                
                ner_annotations[key] = {"text": str(value), "bounding_box": list(value_bbox)}
        
        # FIXED: Better date positioning to avoid overlap
        date_y = max(y + base_spacing * 2, photo_bottom + 20, img_size[1] - 80)
        
        # Date label variations
        issue_labels = ["Issued:", "ISS:", "Issue Date:", "ISSUED:"]
        exp_labels = ["Expires:", "EXP:", "Expiration:", "EXPIRES:"]
        
        issue_text = f"{random.choice(issue_labels)} {metadata['issue_date']}"
        draw.text((x_left, date_y), issue_text, font=small_font, fill=text_color)
        bbox = draw.textbbox((x_left, date_y), issue_text, font=small_font)
        ner_annotations["issue_date"] = {"text": metadata['issue_date'], "bounding_box": list(bbox)}
        
        # FIXED: Better expiration date spacing
        issue_width = bbox[2] - bbox[0]
        exp_x = x_left + issue_width + 30  # Consistent spacing
        exp_text = f"{random.choice(exp_labels)} {metadata['expiration_date']}"
        draw.text((exp_x, date_y), exp_text, font=small_font, fill=text_color)
        bbox = draw.textbbox((exp_x, date_y), exp_text, font=small_font)
        ner_annotations["expiration_date"] = {"text": metadata['expiration_date'], "bounding_box": list(bbox)}
        
        # Optional veteran status
        if metadata['veteran']:
            vet_labels = ["VETERAN", "VET", "MILITARY VETERAN", "U.S. VETERAN"]
            vet_text = random.choice(vet_labels)
            exp_width = bbox[2] - bbox[0]
            vet_x = exp_x + exp_width + 25  # Consistent spacing
            if vet_x + 80 < img_size[0]:  # Only add if it fits
                draw.text((vet_x, date_y), vet_text, font=small_font, fill=accent_color)
                bbox = draw.textbbox((vet_x, date_y), vet_text, font=small_font)
                ner_annotations["veteran"] = {"text": metadata['veteran'], "bounding_box": list(bbox)}
        
        # Organ donor status with varied labels
        date_y += base_spacing
        donor_labels = [
            f"Organ Donor: {metadata['organ_donor']}",
            f"DONOR: {metadata['organ_donor']}",
            f"Organ/Tissue Donor: {metadata['organ_donor']}",
            f"ANATOMICAL GIFT: {metadata['organ_donor']}"
        ]
        donor_text = random.choice(donor_labels)
        draw.text((x_left, date_y), donor_text, font=small_font, fill=text_color)
        bbox = draw.textbbox((x_left, date_y), donor_text, font=small_font)
        ner_annotations["organ_donor"] = {"text": metadata['organ_donor'], "bounding_box": list(bbox)}
        
        # Add varied decorative elements for digital licenses
        if is_digital:
            decoration_styles = ["minimal", "standard", "ornate"]
            decoration_style = random.choice(decoration_styles)
            
            if decoration_style == "ornate":
                # Multiple design elements
                draw.line([(x_left, 60), (img_size[0] - 30, 60)], fill=accent_color, width=2)
                draw.line([(x_left, 62), (img_size[0] - 30, 62)], fill=header_color, width=1)
                draw.rectangle([x_left - 15, 10, img_size[0] - 15, img_size[1] - 10], outline=header_color, width=3)
                draw.rectangle([x_left - 12, 13, img_size[0] - 18, img_size[1] - 13], outline=accent_color, width=1)
            elif decoration_style == "standard":
                # Standard elements
                draw.line([(x_left, 60), (img_size[0] - 30, 60)], fill=accent_color, width=2)
                draw.rectangle([x_left - 10, 15, img_size[0] - 20, img_size[1] - 15], outline=header_color, width=2)
            # Minimal has no extra decorations
            
            # State seal with varied designs
            seal_designs = ["circle", "square", "hexagon"]
            seal_design = random.choice(seal_designs)
            seal_size = random.randint(35, 45)
            seal_x, seal_y = img_size[0] - seal_size - random.randint(70, 90), img_size[1] - seal_size - random.randint(50, 70)
            
            if seal_design == "circle":
                draw.ellipse([seal_x, seal_y, seal_x + seal_size, seal_y + seal_size], 
                            outline=accent_color, width=2)
            elif seal_design == "square":
                draw.rectangle([seal_x, seal_y, seal_x + seal_size, seal_y + seal_size], 
                            outline=accent_color, width=2)
            else:  # hexagon approximation
                points = []
                for i in range(6):
                    angle = i * 60 * 3.14159 / 180
                    x = seal_x + seal_size//2 + (seal_size//2 - 2) * math.cos(angle)
                    y = seal_y + seal_size//2 + (seal_size//2 - 2) * math.sin(angle)
                    points.extend([x, y])
                draw.polygon(points, outline=accent_color, width=2)
            
            # Seal text variations
            seal_texts = [state_abbr, "SEAL", "STATE", fake.lexify(text="????", letters="ABCDEFGHIJKLMNOPQRSTUVWXYZ")]
            seal_text = random.choice(seal_texts)
            text_bbox = draw.textbbox((0, 0), seal_text, font=small_font)
            text_width = text_bbox[2] - text_bbox[0]
            draw.text((seal_x + (seal_size - text_width)//2, seal_y + seal_size//2 - 6), 
                    seal_text, font=small_font, fill=accent_color)
        
        # Apply realistic effects based on document type
        image_cv = np.array(img)
        
        if is_digital:
            # Digital: varied noise levels and effects
            noise_level = random.randint(3, 12)
            noise = np.random.normal(0, noise_level, image_cv.shape).astype(np.uint8)
            noisy_image = cv2.add(image_cv, noise)
            
            # Occasional slight blur
            if random.random() > 0.6:
                blur_kernel = random.choice([(3, 3), (5, 5)])
                noisy_image = cv2.GaussianBlur(noisy_image, blur_kernel, 0)
            
            # Occasional compression artifacts
            if random.random() > 0.7:
                encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), random.randint(75, 95)]
                _, encimg = cv2.imencode('.jpg', noisy_image, encode_param)
                noisy_image = cv2.imdecode(encimg, 1)
        else:
            # Scanned: more varied artifacts
            noise_level = random.randint(15, 30)
            noise = np.random.normal(0, noise_level, image_cv.shape).astype(np.uint8)
            noisy_image = cv2.add(image_cv, noise)
            
            # Scanning lines with variation
            if random.random() > 0.5:
                line_spacing = random.randint(40, 120)
                line_intensity = random.randint(180, 220)
                for i in range(0, img_size[1], line_spacing):
                    cv2.line(noisy_image, (0, i), (img_size[0], i), 
                            (line_intensity, line_intensity, line_intensity), random.randint(1, 2))
            
            # Occasional dust spots
            if random.random() > 0.8:
                for _ in range(random.randint(3, 8)):
                    spot_x = random.randint(0, img_size[0])
                    spot_y = random.randint(0, img_size[1])
                    spot_size = random.randint(1, 3)
                    cv2.circle(noisy_image, (spot_x, spot_y), spot_size, (200, 200, 200), -1)
        
        # Convert to RGB
        if len(noisy_image.shape) == 3:
            noisy_image_rgb = cv2.cvtColor(noisy_image, cv2.COLOR_BGR2RGB)
        else:
            noisy_image_rgb = noisy_image
        
        # Convert to PIL for rotation
        noisy_pil_image = Image.fromarray(noisy_image_rgb)
        
        # Apply rotation with more variation
        if is_digital:
            angle = random.uniform(-1.5, 1.5)
        else:
            angle = random.uniform(-5, 5)
        
        rotated_image, ner_annotations = self.transform_bounding_boxes(ner_annotations, angle, noisy_pil_image)
        rotated_image = np.array(rotated_image)
        
        # Create document type identifier
        doc_type = "driving_license"
        
        GT_json = {
            "document_class": doc_type,
            "NER": ner_annotations
        }
        
        return GT_json, rotated_image


    def insurance_card(self, FONTS):
        """Generate realistic insurance card documents (both scanned and digital versions)"""

        # FIXED: Significantly increased card sizes to prevent any overflow
        DIGITAL_SIZES = [(800, 800), (850, 850), (900, 900), (780, 780), (820, 820), (880, 880)]
        SCANNED_SIZES = [(1000, 1000), (1100, 1100), (1200, 1200), (950, 950), (1050, 1050), (1150, 1150)]
        
        # Real insurance company names and types
        REAL_INSURANCE_COMPANIES = [
            "Blue Cross Blue Shield", "Aetna", "Cigna", "Humana", "UnitedHealthcare",
            "Kaiser Permanente", "Anthem", "Health Net", "Molina Healthcare", 
            "WellCare", "Centene", "Independence Blue Cross", "Highmark",
            "Medical Mutual", "Priority Health", "Blue Shield of California"
        ]
        
        REAL_PLAN_TYPES = [
            "HMO", "PPO", "EPO", "POS", "HDHP", "Gold Plan", "Silver Plan", 
            "Bronze Plan", "Platinum Plan", "Premier", "Essential", "Advantage",
            "Choice", "Select", "Complete", "Basic", "Plus"
        ]
        
        REAL_NETWORKS = [
            "In-Network", "Preferred", "Standard", "Extended", "National", 
            "Regional", "Local Plus", "Premier Network", "Select Network",
            "Advantage Network", "Complete Care", "Essential Network"
        ]
        
        # Randomly decide if it's digital or scanned
        is_digital = random.choice([True, False])
        
        if is_digital:
            img_size = random.choice(DIGITAL_SIZES)
            bg_colors = ["white", "#f8f9fa", "#ffffff", "#fefefe", "#fbfbfb", "#f5f5f5"]
            bg_color = random.choice(bg_colors)
        else:
            img_size = random.choice(SCANNED_SIZES)
            bg_color = random.choice(["white", "#fffffe", "#fefefe", "#fcfcfc"])
        
        img = Image.new('RGB', img_size, bg_color)
        draw = ImageDraw.Draw(img)
        
        # ADDED: Helper function to check if text fits within bounds
        def can_fit_text(y_pos, font, margin_bottom=60):
            """Check if text at y_pos will fit within image bounds with margin"""
            text_height = draw.textbbox((0, 0), "Test", font=font)[3]
            return y_pos + text_height + margin_bottom <= img_size[1]
        
        def safe_draw_text(x, y, text, font, fill, key=None, metadata_value=None):
            """Draw text only if it fits within bounds and add to annotations if successful"""
            if can_fit_text(y, font):
                draw.text((x, y), text, font=font, fill=fill)
                if key and metadata_value is not None:
                    bbox = draw.textbbox((x, y), text, font=font)
                    ner_annotations[key] = {"text": str(metadata_value), "bounding_box": list(bbox)}
                return True
            return False
        
        # Generate realistic insurance metadata
        current_date = date.today()
        
        # Generate realistic coverage dates
        plan_start_dates = [
            date(current_date.year, 1, 1),  # Jan 1st (most common)
            date(current_date.year, 7, 1),  # July 1st 
            date(current_date.year - 1, 1, 1),  # Last year Jan 1st
            current_date - timedelta(days=random.randint(30, 365))  # Random recent date
        ]
        effective_date = random.choice(plan_start_dates)
        
        # Expiration is typically end of year or one year from effective
        if effective_date.month == 1 and effective_date.day == 1:
            expiration_date = date(effective_date.year, 12, 31)
        else:
            expiration_date = date(effective_date.year + 1, effective_date.month, effective_date.day) - timedelta(days=1)
        
        # Generate realistic member ID patterns
        member_id_patterns = [
            f"{fake.random_letter()}{fake.random_letter()}{fake.random_number(digits=9)}",  # AA123456789
            f"{fake.random_number(digits=3)}{fake.random_letter()}{fake.random_number(digits=8)}",  # 123A45678901
            f"{fake.random_letter()}{fake.random_number(digits=2)}-{fake.random_number(digits=3)}-{fake.random_number(digits=4)}",  # A12-345-6789
            f"{fake.random_number(digits=9)}",  # 123456789
            f"{fake.random_letter()}{fake.random_number(digits=8)}"  # A12345678
        ]
        
        # Generate realistic group numbers (employer-based)
        group_patterns = [
            f"{fake.random_number(digits=6)}",  # 123456
            f"{fake.random_letter()}{fake.random_letter()}{fake.random_number(digits=4)}",  # AB1234
            f"{fake.random_number(digits=4)}-{fake.random_number(digits=2)}",  # 1234-56
            f"{fake.random_number(digits=8)}",  # 12345678
            f"{fake.lexify(text='????', letters='ABCDEFGHIJKLMNOPQRSTUVWXYZ')}{fake.random_number(digits=4)}"  # COMP1234
        ]
        
        # Policy number patterns
        policy_patterns = [
            f"POL-{fake.random_number(digits=8)}",
            f"{fake.random_letter()}{fake.random_letter()}-{fake.random_number(digits=6)}",
            f"{fake.random_number(digits=3)}-{fake.random_letter()}{fake.random_letter()}-{fake.random_number(digits=4)}",
            f"P{fake.random_number(digits=9)}",
            f"{fake.random_number(digits=4)}{fake.random_letter()}{fake.random_number(digits=4)}"
        ]
        
        # Select company and customize
        insurance_company = random.choice(REAL_INSURANCE_COMPANIES)
        
        # Generate realistic copay amounts based on plan type
        plan_type = random.choice(REAL_PLAN_TYPES)
        
        # HMO/EPO typically have lower copays, PPO higher, HDHP very high deductibles
        if plan_type in ["HMO", "EPO"]:
            primary_copay = random.randint(15, 35)
            specialist_copay = random.randint(40, 70)
            deductible = random.randint(500, 2000)
        elif plan_type == "PPO":
            primary_copay = random.randint(25, 50)
            specialist_copay = random.randint(50, 90)
            deductible = random.randint(750, 3000)
        elif plan_type == "HDHP":
            primary_copay = 0  # HDHP usually no copay until deductible met
            specialist_copay = 0
            deductible = random.randint(3000, 7000)
        else:  # Other plan types
            primary_copay = random.randint(20, 45)
            specialist_copay = random.randint(45, 85)
            deductible = random.randint(1000, 4000)
        
        metadata = {
            "insurance_company": insurance_company,
            "member_name": fake.name(),
            "member_id": random.choice(member_id_patterns),
            "group_number": random.choice(group_patterns),
            "policy_number": random.choice(policy_patterns),
            "plan_type": plan_type,
            "network": random.choice(REAL_NETWORKS),
            "effective_date": effective_date.strftime("%m/%d/%Y"),
            "expiration_date": expiration_date.strftime("%m/%d/%Y"),
            "copay_primary": f"${primary_copay}" if primary_copay > 0 else "After Deductible",
            "copay_specialist": f"${specialist_copay}" if specialist_copay > 0 else "After Deductible",
            "deductible": f"${deductible:,}",
            "out_of_pocket_max": f"${random.randint(deductible * 2, 15000):,}",
            "rx_generic": f"${random.randint(5, 25)}",
            "rx_brand": f"${random.randint(25, 75)}",
            "emergency_room": f"${random.randint(150, 500)}",
            "urgent_care": f"${random.randint(50, 150)}",
            "customer_service": fake.phone_number(),
            "provider_phone": fake.phone_number(),
            "website": f"www.{insurance_company.lower().replace(' ', '').replace('blue', 'blue')}.com",
            "address": fake.street_address() + ", " + fake.city() + ", " + fake.state_abbr() + " " + fake.zipcode()
        }
        
        # Initialize coordinates and annotations
        ner_annotations = {}
        
        # Choose fonts with more variety
        try:
            available_fonts = FONTS if FONTS else []
            if available_fonts:
                company_font = ImageFont.truetype(random.choice(available_fonts), random.randint(24, 32))
                header_font = ImageFont.truetype(random.choice(available_fonts), random.randint(18, 24))
                regular_font = ImageFont.truetype(random.choice(available_fonts), random.randint(14, 18))
                small_font = ImageFont.truetype(random.choice(available_fonts), random.randint(10, 14))
                medium_font = ImageFont.truetype(random.choice(available_fonts), random.randint(16, 20))
                tiny_font = ImageFont.truetype(random.choice(available_fonts), random.randint(8, 12))
            else:
                raise Exception("No fonts available")
        except:
            company_font = ImageFont.load_default()
            header_font = ImageFont.load_default()
            regular_font = ImageFont.load_default()
            small_font = ImageFont.load_default()
            medium_font = ImageFont.load_default()
            tiny_font = ImageFont.load_default()
        
        # More diverse color schemes based on real insurance companies
        if is_digital:
            company_colors = {
                "Blue Cross Blue Shield": {"primary": "#1e40af", "secondary": "#3b82f6", "accent": "#60a5fa"},
                "Aetna": {"primary": "#7c3aed", "secondary": "#a855f7", "accent": "#c084fc"},
                "Cigna": {"primary": "#059669", "secondary": "#10b981", "accent": "#34d399"},
                "UnitedHealthcare": {"primary": "#1e3a8a", "secondary": "#2563eb", "accent": "#3b82f6"},
                "Humana": {"primary": "#dc2626", "secondary": "#ef4444", "accent": "#f87171"},
                "default": {"primary": "#1f2937", "secondary": "#374151", "accent": "#6b7280"}
            }
            
            colors = company_colors.get(insurance_company, company_colors["default"])
            primary_color = colors["primary"]
            secondary_color = colors["secondary"]
            accent_color = colors["accent"]
            text_color = "black"
        else:
            primary_color = random.choice(["black", "#1a1a1a", "#2d2d2d"])
            secondary_color = random.choice(["#333333", "#404040", "#555555"])
            accent_color = random.choice(["#666666", "#777777", "#888888"])
            text_color = random.choice(["black", "#0a0a0a", "#1a1a1a"])
        
        # Layout parameters
        y_start = random.randint(20, 30)
        x_left = random.randint(25, 40)
        spacing_base = 20
        spacing_variation = random.randint(1, 4)
        
        y = y_start
        
        # Diverse header layouts
        header_styles = ["standard", "centered", "logo_right", "compact"]
        header_style = random.choice(header_styles)
        
        # FIXED: Header with bounds checking
        if header_style == "centered":
            if can_fit_text(y, company_font):
                company_bbox = draw.textbbox((0, 0), metadata['insurance_company'], font=company_font)
                company_width = company_bbox[2] - company_bbox[0]
                company_x = (img_size[0] - company_width) // 2
                draw.text((company_x, y), metadata['insurance_company'], font=company_font, fill=primary_color)
                bbox = draw.textbbox((company_x, y), metadata['insurance_company'], font=company_font)
                ner_annotations["insurance_company"] = {"text": metadata['insurance_company'], "bounding_box": list(bbox)}
                y = bbox[3] + spacing_base
            else:
                return None, None  # Exit early if header doesn't fit
        elif header_style == "logo_right":
            if can_fit_text(y, company_font):
                draw.text((x_left, y), metadata['insurance_company'], font=company_font, fill=primary_color)
                bbox = draw.textbbox((x_left, y), metadata['insurance_company'], font=company_font)
                ner_annotations["insurance_company"] = {"text": metadata['insurance_company'], "bounding_box": list(bbox)}
                
                # Logo placeholder
                logo_size = random.randint(50, 70)
                logo_x = img_size[0] - logo_size - 30
                logo_y = y - 5
                draw.rectangle([logo_x, logo_y, logo_x + logo_size, logo_y + logo_size], 
                            outline=primary_color, fill="#f8f9fa", width=2)
                draw.text((logo_x + 20, logo_y + 25), "LOGO", font=tiny_font, fill=primary_color)
                y = max(bbox[3], logo_y + logo_size) + spacing_base
            else:
                return None, None
        else:
            if can_fit_text(y, company_font):
                draw.text((x_left, y), metadata['insurance_company'], font=company_font, fill=primary_color)
                bbox = draw.textbbox((x_left, y), metadata['insurance_company'], font=company_font)
                ner_annotations["insurance_company"] = {"text": metadata['insurance_company'], "bounding_box": list(bbox)}
                y = bbox[3] + spacing_base
            else:
                return None, None
        
        # Insurance type subtitle
        insurance_subtitles = [
            "Health Insurance Plan", "Medical Coverage", "Healthcare Plan", 
            "Medical Insurance", "Health Plan", "Medical Benefits Plan",
            "Comprehensive Health Coverage", "Health & Medical Insurance"
        ]
        insurance_type_text = random.choice(insurance_subtitles)
        if can_fit_text(y, regular_font):
            draw.text((x_left, y), insurance_type_text, font=regular_font, fill=secondary_color)
            y += spacing_base + 10
        
        # Member information section
        member_section_titles = [
            "Member Information", "Subscriber Details", "Member Info", 
            "Primary Member", "Policyholder Information", "Member Data"
        ]
        member_section_title = random.choice(member_section_titles)
        if can_fit_text(y, header_font):
            draw.text((x_left, y), member_section_title, font=header_font, fill=primary_color)
            y += spacing_base + spacing_variation
        
        # Field label variations
        field_labels = {
            "member_name": random.choice(["Member Name:", "Name:", "Subscriber:", "Primary Member:", "Insured:"]),
            "member_id": random.choice(["Member ID:", "ID Number:", "Subscriber ID:", "Member #:", "ID#:"]),
            "group_number": random.choice(["Group Number:", "Group #:", "Employer Group:", "Group ID:", "Grp:"]),
            "policy_number": random.choice(["Policy Number:", "Policy #:", "Contract #:", "Plan ID:", "Pol:"]),
            "plan_type": random.choice(["Plan Type:", "Plan:", "Coverage Type:", "Product:", "Plan Name:"]),
            "network": random.choice(["Network:", "Provider Network:", "Network Type:", "Coverage Area:", "Net:"])
        }
        
        # FIXED: Member details with bounds checking
        member_items = [
            (field_labels["member_name"], metadata['member_name'], "member_name"),
            (field_labels["member_id"], metadata['member_id'], "member_id"),
            (field_labels["group_number"], metadata['group_number'], "group_number"),
            (field_labels["policy_number"], metadata['policy_number'], "policy_number"),
            (field_labels["plan_type"], metadata['plan_type'], "plan_type"),
            (field_labels["network"], metadata['network'], "network")
        ]
        
        # Calculate consistent column alignment
        max_label_width = max([draw.textbbox((0, 0), label, font=regular_font)[2] for label, _, _ in member_items])
        value_x = x_left + max_label_width + 15
        
        for label, value, key in member_items:
            if can_fit_text(y, regular_font):
                draw.text((x_left, y), label, font=regular_font, fill=text_color)
                draw.text((value_x, y), str(value), font=regular_font, fill=text_color)
                value_bbox = draw.textbbox((value_x, y), str(value), font=regular_font)
                ner_annotations[key] = {"text": str(value), "bounding_box": list(value_bbox)}
                y += spacing_base + spacing_variation
            else:
                break  # Stop adding member items if they don't fit
        
        # FIXED: Coverage dates section with bounds checking
        if can_fit_text(y + spacing_base, header_font):
            y += spacing_base
            date_section_titles = [
                "Coverage Period", "Plan Dates", "Coverage Dates", 
                "Effective Period", "Plan Period", "Coverage Term"
            ]
            dates_title = random.choice(date_section_titles)
            draw.text((x_left, y), dates_title, font=header_font, fill=primary_color)
            y += spacing_base + spacing_variation
            
            # Date labels
            effective_labels = ["Effective Date:", "Start Date:", "From:", "Effective:", "Coverage Begins:"]
            expiration_labels = ["Expiration Date:", "End Date:", "To:", "Expires:", "Coverage Ends:"]
            
            if can_fit_text(y, regular_font):
                eff_text = f"{random.choice(effective_labels)} {metadata['effective_date']}"
                draw.text((x_left, y), eff_text, font=regular_font, fill=text_color)
                bbox = draw.textbbox((x_left, y), eff_text, font=regular_font)
                ner_annotations["effective_date"] = {"text": metadata['effective_date'], "bounding_box": list(bbox)}
                
                # Expiration date
                eff_width = bbox[2] - bbox[0]
                exp_x = x_left + eff_width + 30
                exp_text = f"{random.choice(expiration_labels)} {metadata['expiration_date']}"
                if exp_x + 200 < img_size[0]:  # Check if expiration date fits horizontally
                    draw.text((exp_x, y), exp_text, font=regular_font, fill=text_color)
                    bbox = draw.textbbox((exp_x, y), exp_text, font=regular_font)
                    ner_annotations["expiration_date"] = {"text": metadata['expiration_date'], "bounding_box": list(bbox)}
                y += spacing_base + spacing_base
        
        # FIXED: Benefits section with bounds checking
        if can_fit_text(y, header_font):
            benefit_section_titles = [
                "Benefits Summary", "Coverage Benefits", "Plan Benefits", 
                "Copayments & Deductibles", "Cost Sharing", "Benefit Details"
            ]
            benefits_title = random.choice(benefit_section_titles)
            draw.text((x_left, y), benefits_title, font=header_font, fill=primary_color)
            y += spacing_base + spacing_variation
            
            # Benefit labels
            benefit_labels = {
                "copay_primary": random.choice(["Primary Care:", "PCP Visit:", "Primary Doctor:", "Family Doctor:", "PCP:"]),
                "copay_specialist": random.choice(["Specialist:", "Specialist Visit:", "Specialist Care:", "Spec:", "Specialist Copay:"]),
                "deductible": random.choice(["Annual Deductible:", "Deductible:", "Individual Deductible:", "Ded:", "Annual Ded:"]),
                "out_of_pocket_max": random.choice(["Out-of-Pocket Max:", "Max OOP:", "Annual Max:", "OOP Maximum:", "Max Out-of-Pocket:"]),
                "rx_generic": random.choice(["Generic Rx:", "Generic Drugs:", "Generic:", "Tier 1 Rx:", "Generic Prescription:"]),
                "rx_brand": random.choice(["Brand Rx:", "Brand Drugs:", "Preferred Brand:", "Tier 2 Rx:", "Brand Name Rx:"]),
                "emergency_room": random.choice(["Emergency Room:", "ER Visit:", "Emergency Care:", "ER Copay:", "Emergency:"]),
                "urgent_care": random.choice(["Urgent Care:", "Urgent Care Visit:", "Walk-in Clinic:", "Urgent Care Copay:", "UC:"])
            }
            
            # Try two-column layout first, fall back to single column
            benefits_layout = random.choice(["single_column", "two_column"])
            
            if benefits_layout == "two_column" and img_size[0] > 700:  # Only try two columns on wider images
                benefits_col1 = [
                    (benefit_labels["copay_primary"], metadata['copay_primary'], "copay_primary"),
                    (benefit_labels["copay_specialist"], metadata['copay_specialist'], "copay_specialist"),
                    (benefit_labels["deductible"], metadata['deductible'], "deductible"),
                    (benefit_labels["rx_generic"], metadata['rx_generic'], "rx_generic")
                ]
                
                benefits_col2 = [
                    (benefit_labels["out_of_pocket_max"], metadata['out_of_pocket_max'], "out_of_pocket_max"),
                    (benefit_labels["rx_brand"], metadata['rx_brand'], "rx_brand"),
                    (benefit_labels["emergency_room"], metadata['emergency_room'], "emergency_room"),
                    (benefit_labels["urgent_care"], metadata['urgent_care'], "urgent_care")
                ]
                
                col1_x = x_left
                col2_x = x_left + int(img_size[0] * 0.5)
                col_y = y
                
                # Draw columns with bounds checking
                for i, (label, value, key) in enumerate(benefits_col1):
                    if can_fit_text(col_y + 12, medium_font):  # Check for both label and value
                        draw.text((col1_x, col_y), label, font=small_font, fill=text_color)
                        draw.text((col1_x, col_y + 12), str(value), font=medium_font, fill=primary_color)
                        value_bbox = draw.textbbox((col1_x, col_y + 12), str(value), font=medium_font)
                        ner_annotations[key] = {"text": str(value), "bounding_box": list(value_bbox)}
                        col_y += spacing_base + spacing_base
                    else:
                        break
                
                col_y = y
                for i, (label, value, key) in enumerate(benefits_col2):
                    if can_fit_text(col_y + 12, medium_font) and i < len(benefits_col2):
                        if col2_x + 200 < img_size[0]:  # Check horizontal bounds
                            draw.text((col2_x, col_y), label, font=small_font, fill=text_color)
                            draw.text((col2_x, col_y + 12), str(value), font=medium_font, fill=primary_color)
                            value_bbox = draw.textbbox((col2_x, col_y + 12), str(value), font=medium_font)
                            ner_annotations[key] = {"text": str(value), "bounding_box": list(value_bbox)}
                        col_y += spacing_base + spacing_base
                    else:
                        break
                
                y = col_y
            else:
                # Single column benefit layout
                all_benefits = [
                    (benefit_labels["copay_primary"], metadata['copay_primary'], "copay_primary"),
                    (benefit_labels["copay_specialist"], metadata['copay_specialist'], "copay_specialist"),
                    (benefit_labels["deductible"], metadata['deductible'], "deductible"),
                    (benefit_labels["out_of_pocket_max"], metadata['out_of_pocket_max'], "out_of_pocket_max"),
                    (benefit_labels["rx_generic"], metadata['rx_generic'], "rx_generic"),
                    (benefit_labels["rx_brand"], metadata['rx_brand'], "rx_brand"),
                    (benefit_labels["emergency_room"], metadata['emergency_room'], "emergency_room"),
                    (benefit_labels["urgent_care"], metadata['urgent_care'], "urgent_care")
                ]
                
                for label, value, key in all_benefits:
                    if can_fit_text(y, regular_font):
                        text_content = f"{label} {value}"
                        draw.text((x_left, y), text_content, font=regular_font, fill=text_color)
                        bbox = draw.textbbox((x_left, y), text_content, font=regular_font)
                        ner_annotations[key] = {"text": str(value), "bounding_box": list(bbox)}
                        y += spacing_base + spacing_variation
                    else:
                        break  # Stop adding benefits if they don't fit
        
        # FIXED: Contact information - draw each entity separately so all appear on image
        contact_y = y + spacing_base
        
        # Try to add contact section with header
        if can_fit_text(contact_y, header_font, margin_bottom=80):  # Need more space for 3 lines
            contact_titles = [
                "Contact Information", "Customer Service", "Questions? Call Us", 
                "Need Help?", "Contact Details", "Customer Support"
            ]
            contact_title = random.choice(contact_titles)
            draw.text((x_left, contact_y), contact_title, font=header_font, fill=primary_color)
            contact_y += spacing_base
            
            # Draw each contact entity on separate lines
            if can_fit_text(contact_y + (spacing_base * 2), small_font, margin_bottom=15):  # Check space for all 3 lines
                # Customer Service
                customer_service_text = f"Customer Service: {metadata['customer_service']}"
                draw.text((x_left, contact_y), customer_service_text, font=small_font, fill=text_color)
                bbox = draw.textbbox((x_left, contact_y), customer_service_text, font=small_font)
                ner_annotations["customer_service"] = {"text": metadata['customer_service'], "bounding_box": list(bbox)}
                contact_y += spacing_base
                
                # Provider Phone
                provider_phone_text = f"Provider Services: {metadata['provider_phone']}"
                draw.text((x_left, contact_y), provider_phone_text, font=small_font, fill=text_color)
                bbox = draw.textbbox((x_left, contact_y), provider_phone_text, font=small_font)
                ner_annotations["provider_phone"] = {"text": metadata['provider_phone'], "bounding_box": list(bbox)}
                contact_y += spacing_base
                
                # Website
                website_text = f"Website: {metadata['website']}"
                draw.text((x_left, contact_y), website_text, font=small_font, fill=text_color)
                bbox = draw.textbbox((x_left, contact_y), website_text, font=small_font)
                ner_annotations["website"] = {"text": metadata['website'], "bounding_box": list(bbox)}
        elif can_fit_text(contact_y + (spacing_base * 2), small_font, margin_bottom=15):
            # Try without header if header doesn't fit but we have space for 3 contact lines
            # Customer Service
            customer_service_text = f"Customer Service: {metadata['customer_service']}"
            draw.text((x_left, contact_y), customer_service_text, font=small_font, fill=text_color)
            bbox = draw.textbbox((x_left, contact_y), customer_service_text, font=small_font)
            ner_annotations["customer_service"] = {"text": metadata['customer_service'], "bounding_box": list(bbox)}
            contact_y += spacing_base
            
            # Provider Phone
            provider_phone_text = f"Provider Services: {metadata['provider_phone']}"
            draw.text((x_left, contact_y), provider_phone_text, font=small_font, fill=text_color)
            bbox = draw.textbbox((x_left, contact_y), provider_phone_text, font=small_font)
            ner_annotations["provider_phone"] = {"text": metadata['provider_phone'], "bounding_box": list(bbox)}
            contact_y += spacing_base
            
            # Website
            website_text = f"Website: {metadata['website']}"
            draw.text((x_left, contact_y), website_text, font=small_font, fill=text_color)
            bbox = draw.textbbox((x_left, contact_y), website_text, font=small_font)
            ner_annotations["website"] = {"text": metadata['website'], "bounding_box": list(bbox)}
        # If contact info doesn't fit, simply don't include it in annotations
        
        # Add decorative elements (only if we have space)
        if is_digital and y < img_size[1] - 100:
            decoration_styles = ["minimal", "standard", "corporate", "modern"]
            decoration_style = random.choice(decoration_styles)
            
            if decoration_style == "corporate":
                draw.rectangle([10, 10, img_size[0] - 10, img_size[1] - 10], outline=primary_color, width=4)
                draw.rectangle([15, 15, img_size[0] - 15, img_size[1] - 15], outline=accent_color, width=1)
                draw.line([(x_left, 90), (img_size[0] - 30, 90)], fill=secondary_color, width=3)
                draw.line([(x_left, 93), (img_size[0] - 30, 93)], fill=accent_color, width=1)
            elif decoration_style == "modern":
                for i in range(5):
                    draw.rectangle([15 + i, 15 + i, img_size[0] - 15 - i, img_size[1] - 15 - i], 
                                outline=primary_color, width=1)
                draw.line([(x_left, 85), (img_size[0] - 30, 85)], fill=accent_color, width=2)
            elif decoration_style == "standard":
                draw.rectangle([15, 15, img_size[0] - 15, img_size[1] - 15], outline=primary_color, width=3)
                draw.line([(x_left, 80), (img_size[0] - 30, 80)], fill=secondary_color, width=2)
            
            # Company-specific design elements
            if "Blue" in insurance_company:
                for i in range(3):
                    y_pos = 40 + (i * 15)
                    draw.line([(img_size[0] - 50, y_pos), (img_size[0] - 20, y_pos)], fill=accent_color, width=2)
        
        # Apply realistic effects
        image_cv = np.array(img)
        
        if is_digital:
            # Digital effects with variation
            noise_level = random.randint(2, 8)
            noise = np.random.normal(0, noise_level, image_cv.shape).astype(np.uint8)
            noisy_image = cv2.add(image_cv, noise)
            
            # Occasional compression artifacts
            if random.random() > 0.6:
                encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), random.randint(80, 95)]
                _, encimg = cv2.imencode('.jpg', noisy_image, encode_param)
                noisy_image = cv2.imdecode(encimg, 1)
            
            # Slight blur for phone camera captures
            if random.random() > 0.7:
                noisy_image = cv2.GaussianBlur(noisy_image, (3, 3), 0)
        else:
            # Scanned effects with more variety
            noise_level = random.randint(12, 25)
            noise = np.random.normal(0, noise_level, image_cv.shape).astype(np.uint8)
            noisy_image = cv2.add(image_cv, noise)
            
            # Scanner artifacts
            if random.random() > 0.4:
                line_spacing = random.randint(60, 120)
                for i in range(0, img_size[1], line_spacing):
                    intensity = random.randint(210, 240)
                    cv2.line(noisy_image, (0, i), (img_size[0], i), 
                            (intensity, intensity, intensity), random.randint(1, 2))
            
            # Add scanner edge darkening
            if random.random() > 0.6:
                h, w = noisy_image.shape[:2]
                # Create gradient mask for edge darkening
                mask = np.ones((h, w), dtype=np.float32)
                edge_width = random.randint(20, 50)
                
                # Darken edges
                mask[:edge_width, :] *= 0.9
                mask[-edge_width:, :] *= 0.9
                mask[:, :edge_width] *= 0.9
                mask[:, -edge_width:] *= 0.9
                
                # Apply mask
                for c in range(3):
                    noisy_image[:, :, c] = (noisy_image[:, :, c] * mask).astype(np.uint8)
            
            # Add occasional wrinkles or fold lines
            if random.random() > 0.8:
                fold_y = random.randint(img_size[1] // 4, 3 * img_size[1] // 4)
                fold_intensity = random.randint(180, 220)
                cv2.line(noisy_image, (0, fold_y), (img_size[0], fold_y), 
                        (fold_intensity, fold_intensity, fold_intensity), random.randint(2, 4))
            
            # Add dust spots or scanning artifacts
            if random.random() > 0.7:
                num_spots = random.randint(2, 6)
                for _ in range(num_spots):
                    spot_x = random.randint(0, img_size[0])
                    spot_y = random.randint(0, img_size[1])
                    spot_size = random.randint(1, 4)
                    spot_intensity = random.randint(200, 250)
                    cv2.circle(noisy_image, (spot_x, spot_y), spot_size, 
                            (spot_intensity, spot_intensity, spot_intensity), -1)
            
            # Add slight perspective distortion for scanned cards
            if random.random() > 0.8:
                rows, cols = noisy_image.shape[:2]
                # Create slight perspective transformation
                distortion = random.randint(3, 8)
                pts1 = np.float32([[0, 0], [cols, 0], [0, rows], [cols, rows]])
                pts2 = np.float32([
                    [random.randint(-distortion, distortion), random.randint(-distortion, distortion)], 
                    [cols + random.randint(-distortion, distortion), random.randint(-distortion, distortion)], 
                    [random.randint(-distortion, distortion), rows + random.randint(-distortion, distortion)], 
                    [cols + random.randint(-distortion, distortion), rows + random.randint(-distortion, distortion)]
                ])
                matrix = cv2.getPerspectiveTransform(pts1, pts2)
                noisy_image = cv2.warpPerspective(noisy_image, matrix, (cols, rows), 
                                                borderMode=cv2.BORDER_CONSTANT, 
                                                borderValue=(255, 255, 255))
        
        # Convert to RGB format
        if len(noisy_image.shape) == 3:
            noisy_image_rgb = cv2.cvtColor(noisy_image, cv2.COLOR_BGR2RGB)
        else:
            noisy_image_rgb = cv2.cvtColor(noisy_image, cv2.COLOR_GRAY2RGB)
        
        # Convert to PIL for rotation
        noisy_pil_image = Image.fromarray(noisy_image_rgb)
        
        # Apply rotation with realistic variation
        if is_digital:
            # Digital cards: minimal rotation (phone camera slight tilt)
            angle = random.uniform(-1.0, 1.0)
        else:
            # Scanned cards: more rotation variation
            angle = random.uniform(-4.0, 4.0)
        
        # Apply the rotation and bounding box transformation
        rotated_image, ner_annotations = self.transform_bounding_boxes(ner_annotations, angle, noisy_pil_image)
        rotated_image = np.array(rotated_image)
        
        # Create document type identifier
        doc_type = "insurance_card"
        
        GT_json = {
            "document_class": doc_type,
            "NER": ner_annotations
        }
        
        return GT_json, rotated_image

    def other(self, FONTS):
        IMAGE_SIZES = [(1000, 1400), (1200, 1600), (1400, 1800), (1600, 2000), (1800, 2200), (2000, 2400)]
        DEFAULT_FONT = random.choice(FONTS)

        # Randomly select an image size
        img_size = random.choice(IMAGE_SIZES)

        # Create a white background image
        img = Image.new('RGB', img_size, 'white')
        draw = ImageDraw.Draw(img)

        # Select a font
        font_size = 28
        font = ImageFont.truetype(DEFAULT_FONT, font_size)

        # Generate 10–20 random sentences
        num_sentences = random.randint(10, 20)
        sentences = [fake.sentence(nb_words=random.randint(6, 12)) for _ in range(num_sentences)]

        # Draw each sentence on a new line
        x, y = 100, 150
        for sentence in sentences:
            draw.text((x, y), sentence, font=font, fill="black")
            text_bbox = draw.textbbox((x, y), sentence, font=font)
            line_height = text_bbox[3] - text_bbox[1] + 20
            y += line_height
            if y > img_size[1] - 100:
                break  # Avoid overflow beyond the bottom

        # Convert PIL to OpenCV format
        image_cv = np.array(img)
        noise = np.random.normal(0, 15, image_cv.shape).astype(np.uint8)
        noisy_image = cv2.add(image_cv, noise)

        # Rotate the image randomly between -5 to 5 degrees
        angle = random.uniform(-5, 5)
        center = (img_size[0] // 2, img_size[1] // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated_image = cv2.warpAffine(noisy_image, M, (img_size[0], img_size[1]))

        # Return empty NER JSON
        GT_json = {"document_class": "other", "NER": {}}
        return GT_json, rotated_image



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
                                update_annotations(item)

        # Apply transformations to bounding boxes
        update_annotations(ner_annotations)

        return rotated_img, ner_annotations  # Return updated image and annotations`


    def transform_bounding_boxes_2(self, ner_annotations, angle, image):
        """Transforms bounding boxes inside ner_annotations to 8-point format after rotation."""
        
        # Get image dimensions
        w, h = image.size
        center = (w // 2, h // 2)
        
        # Calculate new dimensions after rotation
        # Important: PIL rotates counterclockwise, OpenCV rotates clockwise
        rangle = math.radians(-angle)  # Negate angle to match PIL's direction
        new_w = int(abs(w * math.cos(rangle)) + abs(h * math.sin(rangle)))
        new_h = int(abs(h * math.cos(rangle)) + abs(w * math.sin(rangle)))
        
        # Compute rotation matrix (use negative angle to match PIL's rotation direction)
        M = cv2.getRotationMatrix2D(center, -angle, 1.0)
        
        # Adjust the rotation matrix for the expanded image size
        M[0, 2] += (new_w - w) / 2
        M[1, 2] += (new_h - h) / 2
        
        # Rotate the image using PIL (which handles the expand parameter properly)
        rotated_img = image.rotate(angle, expand=True)

        def rotate_bbox(bbox):
            """Rotates a bounding box and converts it into an 8-point polygon."""
            if not bbox or len(bbox) != 4:
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

            # Convert to list (flatten to store as 8 points)
            return rotated_points.flatten().tolist()

        def update_annotations(data):
            """Recursively updates bounding boxes in the annotations."""
            if isinstance(data, dict):
                for key, value in data.items():
                    if isinstance(value, dict) and "bounding_box" in value:
                        value["bounding_box"] = rotate_bbox(value["bounding_box"])
                    elif isinstance(value, dict):
                        update_annotations(value)
                    elif isinstance(value, list):
                        for item in value:
                            if isinstance(item, dict) and "bounding_box" in item:
                                item["bounding_box"] = rotate_bbox(item["bounding_box"])
                            elif isinstance(item, dict):
                                update_annotations(item)
            return data

        # Apply transformations to bounding boxes
        ner_annotations = update_annotations(ner_annotations)

        return rotated_img, ner_annotations
        

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
            self.medical_document:FONTS,
            self.driving_license:FONTS,
            self.insurance_card:FONTS,
            self.other:FONTS
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


if __name__ == "__main__":
    unique_id = str(uuid.uuid4())
    class_object = GenerateDocument("", unique_id)
    json_metadata, generated_doc = class_object.generate_document()

    # Define filenames
    base_filename = f"{json_metadata['document_class']}_{unique_id}"
    image_filename = f"{base_filename}.png"
    json_filename = f"{base_filename}.json"

    # Save image
    cv2.imwrite(image_filename, np.array(generated_doc))

    # Save JSON metadata
    with open(json_filename, "w") as json_file:
        json.dump(json_metadata, json_file, indent=4)

    print(f"Saved: {image_filename} and {json_filename}")