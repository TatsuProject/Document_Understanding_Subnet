import random
import json
from PIL import Image, ImageDraw, ImageFont
from math import floor
import io
import numpy as np
import cv2
from faker import Faker
import os
import uuid

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

        def add_text(key, content, font_size=25, offset=10):
            nonlocal y
            if content is None:
                return
            font = ImageFont.truetype(random.choice(FONTS), font_size)
            draw.text((x, y), f"{content}", font=font, fill="black")
            bbox = draw.textbbox((x, y), content, font=font)
            ner_annotations[key] = {"text": content, "bounding_box": list(bbox)}
            y += (bbox[3] - bbox[1]) + offset

        add_text("budget_name", metadata["budget_name"], font_size=30)
        add_text("date", metadata["date"])
        add_text("total_budget", f"Total Budget: ${metadata['total_budget']}", font_size=28, offset=15)
        add_text("currency", metadata["currency"], font_size=24)

        # Adding allocations
        allocations = []
        for allocation in metadata["allocations"]:
            category_bbox = add_text("category", allocation["category"], font_size=22, offset=5)
            amount_bbox = add_text("amount", f"${allocation['amount']}", font_size=22, offset=15)
            allocations.append({"category": category_bbox, "amount": amount_bbox})
        
        ner_annotations["allocations"] = allocations

        image_cv = np.array(img)
        noise = np.random.normal(0, 15, image_cv.shape).astype(np.uint8)
        noisy_image = cv2.add(image_cv, noise)
        angle = random.uniform(-5, 5)
        center = (img_size[0] // 2, img_size[1] // 2)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated_image = cv2.warpAffine(noisy_image, matrix, (img_size[0], img_size[1]), borderMode=cv2.BORDER_REPLICATE)

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
        
        def apply_scan_effects(img):
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            noise = np.random.normal(0, 25, img.shape).astype(np.uint8)
            img = cv2.add(img, noise)
            angle = random.uniform(-3, 3)
            h, w = img.shape
            matrix = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1)
            img = cv2.warpAffine(img, matrix, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
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
        
        img = np.array(img)
        img = apply_scan_effects(img)
        
        ner_annotations = {
            "folder_title": bounding_boxes.get("folder_title", {"text": "", "bounding_box": []}),
            "folder_id": bounding_boxes.get("folder_id", {"text": "", "bounding_box": []}),
            "creation_date": bounding_boxes.get("creation_date", {"text": "", "bounding_box": []}),
            "owner": bounding_boxes.get("owner", {"text": "", "bounding_box": []}),
            "department": bounding_boxes.get("department", {"text": "", "bounding_box": []}),
            "contained_documents": contained_doc_boxes,
            "tags": tag_boxes
        }

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
        angle = random.uniform(-5, 5)
        matrix = cv2.getRotationMatrix2D((img_size[0] // 2, img_size[1] // 2), angle, 1.0)
        rotated_image = cv2.warpAffine(noisy_image, matrix, (img_size[0], img_size[1]), borderMode=cv2.BORDER_REPLICATE)
        
        return {"document_class": "form", "NER": gt_json}, rotated_image


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
                ner_annotations[label].append({"text": content, "bounding_box": bounding_box})
                y += bbox[3] + offset

        # Randomly decide to add names and dates
        if random.choice([True, False]):
            add_handwritten_text("person_names", fake.name(), 40)
        
        if random.choice([True, False]):
            add_handwritten_text("dates", fake.date_this_year().strftime("%m/%d/%y"), 30)

        # Add noise and rotation for realism
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
            y_offset = 50
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
                draw.text((50, y_offset), text, fill="black", font=font)
                bbox = draw.textbbox((50, y_offset), text, font=font)  # Corrected for Pillow 10.x.x
                y_offset += font_size + 10  # Adjust line spacing
                return {"text": text, "bounding_box": bbox}

            # Generate fields dynamically based on a random selection
            if random.choice([True, False]):
                gt_template["invoice_number"] = draw_text("invoice_number", f"Invoice #: {fake.uuid4()[:8]}")

            if random.choice([True, False]):
                gt_template["date"] = draw_text("date", f"Date: {fake.date()}")

            if random.choice([True, False]):
                gt_template["organization"] = draw_text("organization", f"From: {fake.company()}")

            if random.choice([True, False]):
                gt_template["payee_name"] = draw_text("payee_name", f"To: {fake.name()}")

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
                    "item": draw_text("item", f"{item}      {qty}"),
                    "price": draw_text("price", f"{price}")
                })

            y_offset += 10

            if random.choice([True, False]):
                gt_template["total_amount"] = draw_text("total_amount", f"Total: ${random.randint(100, 1000)}")

            if random.choice([True, False]):
                gt_template["discount_amount"] = draw_text("discount_amount", f"Discount: ${random.randint(5, 50)}")

            if random.choice([True, False]):
                gt_template["tax_amount"] = draw_text("tax_amount", f"Tax: ${random.randint(5, 50)}")

            if random.choice([True, False]):
                gt_template["final_amount"] = draw_text("final_amount", f"Final Total: ${random.randint(150, 2000)}")

            return gt_template

        img_width, img_height = random.choice(IMAGE_SIZES)
        img = Image.new("RGB", (img_width, img_height), "white")
        draw = ImageDraw.Draw(img)

        # Generate invoice data and get GT annotations
        gt_annotations = generate_invoice_data(draw, img_width)

        # Add noise and rotate
        img = add_noise(img)
        img = rotate_image(img)
        image_cv = np.array(img)

        # Save annotations as JSON
        GT_json = {
            "document_class": "invoice",
            "NER": gt_annotations
        }

        return GT_json, image_cv


    def letter(self, FONTS):
        IMAGE_SIZES = [(800, 1000), (850, 1100), (900, 1200), (1000, 1300), (1100, 1400), (1200, 1500)]
        TEMPLATES = ["simple", "header_footer"]

        img_size = random.choice(IMAGE_SIZES)
        img = Image.new('RGB', img_size, 'white')
        draw = ImageDraw.Draw(img)

        template = random.choice(TEMPLATES)
        
        if template == "header_footer":
            draw.rectangle([(0, 0), (img_size[0], 80)], fill="black")  # Header
            draw.rectangle([(0, img_size[1] - 80), (img_size[0], img_size[1])], fill="black")  # Footer
        
        # Generate document fields with optional inclusion
        sender_name = fake.company() if random.choice([True, False]) else ""
        sender_address = fake.address() if random.choice([True, False]) else ""
        sender_contact = fake.phone_number() if random.choice([True, False]) else ""
        receiver_name = fake.name() if random.choice([True, False]) else ""
        receiver_address = fake.address() if random.choice([True, False]) else ""
        date = fake.date() if random.choice([True, False]) else ""
        attachments = [fake.word() for _ in range(random.randint(0, 3))]  # 0 to 3 attachments
        
        # Content mapping for drawing text and generating bounding boxes
        content = {
            "Sender Name": sender_name,
            "Sender Address": sender_address,
            "Sender Contact": sender_contact,
            "Date": date,
            "Receiver Name": receiver_name,
            "Receiver Address": receiver_address,
            "Attachments": "\n".join(attachments) if attachments else ""
        }
        
        x, y = 50, 100
        ner_annotations = {}
        
        for label, text in content.items():
            if text:  # Only include if the field has content
                font_path = random.choice(FONTS)
                font_size = random.randint(20, 30)
                font = ImageFont.truetype(font_path, font_size)
                
                bbox = draw.textbbox((x, y), text, font=font)
                draw.text((x, y), text, font=font, fill="black")
                y += bbox[3] - bbox[1] + 20  # Adjust Y position based on text height
                
                # Map label to GT key format
                gt_key = label.lower().replace(" ", "_")
                if gt_key == "attachments":
                    ner_annotations[gt_key] = [{"text": text, "bounding_box": bbox}] if text else []
                else:
                    ner_annotations[gt_key] = {"text": text, "bounding_box": bbox}
        
        # Convert to OpenCV for noise & rotation
        image_cv = np.array(img)
        noise = np.random.normal(0, 15, image_cv.shape).astype(np.uint8)
        noisy_image = np.clip(image_cv + noise, 0, 255)
        
        angle = random.uniform(-5, 5)
        center = (img_size[0] // 2, img_size[1] // 2)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated_image = cv2.warpAffine(noisy_image, matrix, (img_size[0], img_size[1]), borderMode=cv2.BORDER_REPLICATE)
        
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
            "body": fake.paragraph(nb_sentences=5),
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
            "category": fake.word() if random.random() > 0.5 else None,
            "source": fake.company() if random.random() > 0.5 else None,
            "content": fake.paragraph(nb_sentences=5)
        }

        x, y = 50, 50
        ner_annotations = []

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
                ner_annotations.append({"label": label.lower(), "content": content, "bounding_box": bbox})
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

        angle = random.uniform(-5, 5)
        center = (img_size[0] // 2, img_size[1] // 2)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated_image = cv2.warpAffine(noisy_image, matrix, (img_size[0], img_size[1]), borderMode=cv2.BORDER_REPLICATE)

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
        if random.choice([True, False]):
            slide_data["slide_title"] = fake.sentence(nb_words=6)
        if random.choice([True, False]):
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
                ner_annotations["summary"] = {"text": text, "bounding_box": bbox}
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

        angle = random.uniform(-5, 5)
        center = (img_size[0] // 2, img_size[1] // 2)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated_image = cv2.warpAffine(noisy_image, matrix, (img_size[0], img_size[1]), borderMode=cv2.BORDER_REPLICATE)

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
            "author": fake.name(),
            "affiliation": fake.company(),
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
        
        def add_text(label, content, font_size=30, offset=15, is_list=False):
            nonlocal y
            font = ImageFont.truetype(random.choice(FONTS), font_size)
            draw.text((x, y), content, font=font, fill="black")
            text_bbox = draw.textbbox((x, y), content, font=font)
            if is_list:
                if label not in ner_annotations:
                    ner_annotations[label] = []
                ner_annotations[label].append({"text": content, "bounding_box": list(text_bbox)})
            else:
                ner_annotations[label] = {"text": content, "bounding_box": list(text_bbox)}
            y += text_bbox[3] - text_bbox[1] + offset
        
        ner_annotations["authors"] = [{
            "name": {"text": report_data["author"], "bounding_box": []},
            "affiliation": {"text": report_data["affiliation"], "bounding_box": []}
        }]
        add_text("authors", f"Author: {report_data['author']}", font_size=28, offset=20)
        add_text("authors", f"Affiliation: {report_data['affiliation']}", font_size=28, offset=20)
        add_text("date", f"Date: {report_data['date']}", font_size=28, offset=20)
        for keyword in report_data["keywords"]:
            add_text("keywords", keyword, font_size=28, offset=20, is_list=True)
        add_text("report_id", f"Report ID: {report_data['report_id']}", font_size=28, offset=20)
        if report_data["funding_source"]:
            add_text("funding_source", f"Funding Source: {report_data['funding_source']}", font_size=28, offset=20)
        
        image_cv = np.array(img)
        noise = np.random.normal(0, 15, image_cv.shape).astype(np.uint8)
        noisy_image = cv2.add(image_cv, noise)
        angle = random.uniform(-5, 5)
        center = (img_size[0] // 2, img_size[1] // 2)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated_image = cv2.warpAffine(noisy_image, matrix, (img_size[0], img_size[1]), borderMode=cv2.BORDER_REPLICATE)
        
        GT_json = {"document_class": "scientific_report", "NER": ner_annotations}
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

            temp_img = Image.new("RGBA", (10, 10), (255, 255, 255, 0))
            temp_draw = ImageDraw.Draw(temp_img)
            bbox = temp_draw.textbbox((0, 0), text, font=font)
            text_width, text_height = bbox[2] - bbox[0], bbox[3] - bbox[1]

            img = Image.new("RGBA", (text_width + 10, text_height + 10), (255, 255, 255, 0))
            draw = ImageDraw.Draw(img)
            draw.text((5, 5), text, font=font, fill=(0, 0, 0, 255))
            angle = random.uniform(-5, 5)
            img = img.rotate(angle, expand=True)

            return img, (0, 0, text_width, text_height)

        img_size = random.choice(IMAGE_SIZES)
        img = Image.new("RGB", img_size, "white")
        draw = ImageDraw.Draw(img)
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
            } for _ in range(random.randint(1, 3))],
            "regulatory_compliance": [fake.sentence() for _ in range(random.randint(0, 2))],
            "key_requirements": [fake.sentence() for _ in range(random.randint(1, 3))]
        }

        mobile_specs = {
            "section_title": "Mobile Specs",
            "section_number": f"RAM: {random.choice(['4GB', '6GB', '8GB'])}, "
                                f"Storage: {random.choice(['64GB', '128GB', '256GB'])}, "
                                f"Camera: {random.choice(['12MP', '48MP', '64MP'])}, "
                                f"Battery: {random.choice(['3000mAh', '4000mAh', '5000mAh'])}"
        }
        specification_data["key_sections"].append(mobile_specs)

        x, y = 50, header_height + 40
        ner_annotations = {
            "title": {"text": "", "bounding_box": []},
            "date": {"text": "", "bounding_box": []},
            "organization": {"text": "", "bounding_box": []},
            "key_sections": [],
            "regulatory_compliance": [],
            "key_requirements": []
        }

        def add_text(label, content, font_size=32):
            nonlocal y
            text_img, bbox = generate_text(content, font_size)
            if text_img is not None:
                img.paste(text_img, (x, y), text_img)
                absolute_bbox = [x, y, x + bbox[2], y + bbox[3]]
                
                if label in ner_annotations:
                    ner_annotations[label]["text"] = content
                    ner_annotations[label]["bounding_box"] = absolute_bbox
                
                y += bbox[3] + int(font_size * 0.4)
                return absolute_bbox
            return []

        add_text("title", specification_data["title"], font_size=40)
        add_text("date", specification_data["date"], font_size=30)
        add_text("organization", specification_data["organization"], font_size=30)

        for section in specification_data["key_sections"]:
            section_data = {}
            section_data["section_title"] = {"text": section["section_title"], "bounding_box": add_text("key_sections", section["section_title"], font_size=28)}
            section_data["section_number"] = {"text": section["section_number"], "bounding_box": add_text("key_sections", section["section_number"], font_size=28)}
            ner_annotations["key_sections"].append(section_data)

        for compliance in specification_data["regulatory_compliance"]:
            ner_annotations["regulatory_compliance"].append({"text": compliance, "bounding_box": add_text("regulatory_compliance", compliance, font_size=28)})

        for requirement in specification_data["key_requirements"]:
            ner_annotations["key_requirements"].append({"text": requirement, "bounding_box": add_text("key_requirements", requirement, font_size=28)})

        image_cv = np.array(img)
        if image_cv.dtype != np.uint8:
            image_cv = image_cv.astype(np.uint8)

        noise = np.random.normal(0, 15, image_cv.shape).astype(np.uint8)
        noisy_image = cv2.add(image_cv, noise)

        angle = random.uniform(-5, 5)
        center = (img_size[0] // 2, img_size[1] // 2)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated_image = cv2.warpAffine(noisy_image, matrix, (img_size[0], img_size[1]), borderMode=cv2.BORDER_REPLICATE)

        GT_json = {
            "document_class": "specification",
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
