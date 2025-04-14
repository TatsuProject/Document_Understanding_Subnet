
# Docs-Insights Subnet 

The Docs-Insights is a decentralized system built for advanced document processing tasks. It combines multiple AI models—including vision models, language models, vision-language models (VLMs), and OCR engines—to accurately understand and extract information from documents. This subnet aims to offer a powerful, open-source alternative to proprietary tools, making document comprehension more accessible and efficient. By delivering key insights with a single click, it significantly reduces the time and effort required for document review.

### Key Capabilities:
1. **Checkbox and Associated Text Detection** - Currently live and operational on SN-84, outperforming industry standards like GPT-4 Vision and Azure Form Recognizer.
2. **Highlighted and Encircled Text Detection** - Detects and extracts highlighted or circled text segments accurately (Under Development).
3. **Document Classification** - Automatically classifies documents by type (e.g., receipts, forms, letters). This feature is live on SN-84 and powered by the Donut model, a cutting-edge, OCR-free architecture.
4. **Document Parsing** - Leverages powerful LLMs to extract key entities like names, addresses, phone numbers, and monetary values. Documents are intelligently segmented into logical sections for improved clarity. Live on SN-84.
5. **JSON Data Structuring** - Compiles and formats extracted data into a concise, readable JSON file, significantly reducing document review time.


## Table of Contents

- [Architecture](#architecture)
- [Reward Mechanism](#reward-mechanism)
- [Installation](#installation)
- [Usage](#usage)
- [Technical Guide](#technical-guide)
- [License](#license)

## Architecture

The system consists of two primary components:

1. **Validator**
   - Equipped with synthetic data generation:
     - The validator first decide the task out of three: "checkbox", "doc-class", "doc-parse" 
     - Then validator randomly generates an image along with its corresponding ground truth data related to the decided task.
     - This image is then sent to the miner for processing.

2. **Miner**
   - checkbox
      - **Vision Model**: Processes the image to detect checkboxes, returning their coordinates.
      - **OCR Engine and Preprocessor**: Extracts text from the image, organizes it into lines, and records the coordinates for each line.
      - **Post-Processor**: Integrates the checkbox and text coordinates to associate text with each checkbox.
   - doc-class
      - **VLM (Donut)**: Processes the image to find out the class of the document. No OCR/postprocessor needed here.
   - doc-parse
      - **OCR Engine**: Extracts text from the image, organizes it into lines.
      - **LLMs**: are used to carefully analyze the text, parse them into main sections, fill the sections with necessary information.

## Reward Mechanism

1. The **Validator** generates an image and its ground truth, keeping the ground truth file and sending the image to the miner.
2. The **Miner** processes the image using models and post-processors, then returns the output to the validator.
3. The **Validator** evaluates the result based on:
   - **Accuracy**: Scores based on the
      - overlap of detected bounding box coordinates with the ground truth,
      - and text content matching.
   - Validator then rewards top performing miner when it is 5% better than the second best miner.

## Installation

To set up the Document Understanding project:

1. **Clone the repository:**
   ```bash
   git clone https://github.com/TatsuProject/Document_Understanding_Subnet.git
   cd Document_Understanding_Subnet
   ```

2. **Install required dependencies:**
   ```bash
   pip install -r requirements.txt
   pip install -e .
   ```

3. **Install Tesseract (for miners only):**
   ```bash
   sudo apt-get install tesseract-ocr
   ```

4. **Install and run AI models (for miners only):**  
   Follow the steps in the link below to install the service:  
   ```bash
   https://github.com/TatsuProject/document_insights_base_model 
   ```
   After installation, ensure the service is running on the same machine as the miner.

## Usage

### On Testnet:

1. **Start the Validator:**
   ```bash
   python3 neurons/validator.py --netuid 236 --subtensor.network test --wallet.name validator --wallet.hotkey default --logging.debug 
   ```

2. **Start the Miner:**
   ```bash
   python3 neurons/miner.py --netuid 236 --subtensor.network test --wallet.name miner --wallet.hotkey default --logging.debug 
   ```

### On Mainnet:

1. **Start the Validator:**
   ```bash
   python3 neurons/validator.py --netuid 84 --subtensor.network finney --wallet.name validator --wallet.hotkey default --logging.debug 
   ```

2. **Start the Miner:**
   ```bash
   python3 neurons/miner.py --netuid 84 --subtensor.network finney --wallet.name miner --wallet.hotkey default --logging.debug 
   ```


## Technical Guide

For more in-depth information, refer to the [Technical Guide](docs/Technical_Guide.md).

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
