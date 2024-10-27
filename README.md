
# Document Understanding Subnet 📃 - Testnet 236

The **Document Understanding Subnet** is a pioneering, decentralized system dedicated to advanced document understanding tasks, designed to streamline document processing. Leveraging a multi-model architecture of vision, text models, and OCR engines, it aims to set a new standard in document comprehension while providing an open and accessible alternative to proprietary solutions.

### 🚀 Key Capabilities in Development:
1. **✅ Checkbox and Associated Text Detection** - Currently live and operational on testnet 236, outperforming industry standards like GPT-4 Vision and Azure Form Recognizer.
2. **🖍️ Highlighted and Encircled Text Detection** - Detects and extracts highlighted or circled text segments accurately.
3. **📂 Document Classification** - Automatically identifies document types (e.g., receipts, forms, letters).
4. **🔍 Entity Detection** - Extracts key details such as names, addresses, phone numbers, and costs.
5. **📄 JSON Data Structuring** - Compiles and formats extracted data into a concise, readable JSON file, significantly reducing document review time.

This system will bring efficiency to document processing workflows by combining these capabilities, enabling faster, more efficient, and decentralized document analysis. Currently, checkbox and associated text detection are fully operational on testnet 236, with additional features in development.

*Update: Now live on testnet as subnet-236.*

## 📑 Table of Contents

- [⚙️ Architecture](#architecture)
- [💰 Reward Mechanism](#reward-mechanism)
- [📥 Installation](#installation)
- [▶️ Usage](#usage)
- [📘 Technical Guide](#technical-guide)
- [📜 License](#license)

## ⚙️ Architecture

The system consists of two primary components:

1. **Validator**
   - Equipped with a **Dataset with Ground Truths**:
     - The validator randomly selects an image along with its corresponding ground truth data.
     - This image is then sent to the miner for processing.

2. **Miner**
   - **Vision Model**: Processes the image to detect checkboxes, returning their coordinates.
   - **OCR Engine and Preprocessor**: Extracts text from the image, organizes it into lines, and records the coordinates for each line.
   - **Post-Processor**: Integrates the checkbox and text coordinates to associate text with each checkbox.

## 💰 Reward Mechanism

1. The **Validator** retrieves an image and its ground truth, keeping the ground truth file and sending the image to the miner.
2. The **Miner** processes the image using models and a post-processor, then returns the output to the validator.
3. The **Validator** evaluates the result based on:
   - **⏱️ Time Efficiency**: Scores the miner based on processing time, benchmarked against a low-end machine (8 GB RAM, dual-core).
   - **📏 Accuracy**: Scores based on the overlap of detected checkbox and text coordinates with the ground truth, along with text content matching.

## 📥 Installation

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

4. **Install the YOLO Checkbox Service (for miners only):**  
   Follow the steps in the link below to install the service:  
   ```bash
   https://github.com/TatsuProject/yolo_checkbox_detector
   ```
   After installation, ensure the service is running on the same machine as the miner.

## ▶️ Usage

### ⚙️ On Testnet:

1. **Start the Validator:**
   ```bash
   python3 neurons/validator.py --netuid 236 --subtensor.network test --wallet.name validator --wallet.hotkey default --logging.debug 
   ```

2. **Start the Miner:**
   ```bash
   python3 neurons/miner.py --netuid 236 --subtensor.network test --wallet.name miner --wallet.hotkey default --logging.debug 
   ```

### 🌐 On Mainnet:
   ```bash
   coming soon
   ```

## 📘 Technical Guide

For more in-depth information, refer to the [Technical Guide](docs/Technical_Guide.md).

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
