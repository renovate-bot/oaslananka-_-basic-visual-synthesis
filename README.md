# Basic Visual Synthesis

Basic Visual Synthesis is a project that performs image analysis and generates detailed commentary by integrating YOLO-based object detection with Anything LLM (Deepseek R114B) using a RAG (Retrieval Augmented Generation) approach. The pipeline detects objects in an image, retrieves additional context from a built-in knowledge base, and then leverages the LLM to produce comprehensive commentary on the image.

> **Project Description:**
> "This project aims to perform image analysis and generate detailed commentary by integrating YOLO-based object detection with the Deepseek LLM using a RAG (Retrieval Augmented Generation) approach."

## Table of Contents

- [Features](#features)
- [Installation &amp; Requirements](#installation--requirements)

## Features

- **Object Detection:** Utilizes [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) for detecting objects in images.
- **Context Retrieval:** Provides additional context via a built-in knowledge base.
- **RAG Approach:** Integrates [Anything LLM](https://anythingllm.com/) for detailed commentary generation. with  [Deepseek](https://www.deepseek.com/)  R114B to generate detailed commentary using a Retrieval Augmented Generation method.
- **Colored Logging:** Uses [Colorama](https://pypi.org/project/colorama/) with Python's `logging` module for enhanced, colored log output.

## Installation & Requirements

1. **Python Version:** Tested with Python 3.8 and above.
2. **Clone the Repository:**

   ```bash
   git clone https://github.com/oaslananka/BasicVisualSynthesis.git
   cd BasicVisualSynthesis
   ```

3. **Create a Virtual Environment (Optional but Recommended):**

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install ultralytics colorama requests
   python VisualSynthesisRAG.py
   ```

4. **Install Required Packages:**

   ```bash
   pip install ultralytics colorama requests
   ```

5. **Configure API Keys and Model Paths:**

   - Update the API_KEY and BASE_URL constants in the code as needed.
