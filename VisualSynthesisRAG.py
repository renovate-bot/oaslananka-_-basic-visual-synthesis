import os
import requests
import json
import logging
from colorama import init, Fore, Style
from typing import Dict, Any


# pip install ultralytics
from ultralytics import YOLO

# Configuration constants
API_KEY = "ANYTHING_LL_API_KEY"  # Replace with your ANYTHING LLM API key
BASE_URL = "http://localhost:3001/api"


# Initialize Colorama
init(autoreset=True)

# Custom colored logging formatter


class ColoredFormatter(logging.Formatter):
    LOG_COLORS = {
        logging.DEBUG: Fore.CYAN,
        logging.INFO: Fore.GREEN,
        logging.WARNING: Fore.YELLOW,
        logging.ERROR: Fore.RED,
        logging.CRITICAL: Fore.MAGENTA,
    }

    def format(self, record: logging.LogRecord) -> str:
        color = self.LOG_COLORS.get(record.levelno, Fore.WHITE)
        record.levelname = f"{color}{record.levelname}{Style.RESET_ALL}"
        record.msg = f"{color}{record.msg}{Style.RESET_ALL}"
        return super().format(record)


# Logging configuration
handler = logging.StreamHandler()
formatter = ColoredFormatter("%(asctime)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger = logging.getLogger(__name__)
logger.handlers.clear()  # Clear any existing handlers
logger.addHandler(handler)
logger.setLevel(logging.INFO)


class ImageAnalyzer:
    """
    Analyzes an image using a YOLO model to detect objects.
    """

    def __init__(self, model_path: str) -> None:
        logger.info("Loading YOLO model from %s...", model_path)
        self.model = YOLO(model_path)

    def analyze(self, image_path: str) -> str:
        """
        Detects objects in the image and returns an analysis report.

        :param image_path: Path to the image file.
        :return: Analysis report summarizing detected objects.
        """
        if not os.path.exists(image_path):
            logger.error("Image file not found: %s", image_path)
            return "Image analysis failed."

        logger.info("Starting image analysis...")
        results = self.model.predict(source=image_path)
        detected_objects: Dict[str, int] = {}

        for result in results:
            for box in result.boxes:
                cls_id = int(box.cls[0].item())
                cls_name = self.model.names[cls_id]
                detected_objects[cls_name] = detected_objects.get(cls_name, 0) + 1

        if not detected_objects:
            analysis_report = "No significant objects detected in the image."
        else:
            lines = ["Detected objects (YOLO):"]
            for label, count in detected_objects.items():
                lines.append(f"{label}: {count}")
            analysis_report = "\n".join(lines)

        logger.info("Image analysis completed.\n%s", analysis_report)
        return analysis_report


class RetrievalEngine:
    """
    Retrieves additional context information based on the image analysis results.
    (Dummy knowledge base for demonstration)
    """

    def __init__(self) -> None:
        self.knowledge_base = {
            "person": (
                "People are complex beings with diverse roles and emotions. In many paintings, "
                "they can represent different symbolic or cultural elements."
            ),
            "table": (
                "A table is a piece of furniture used for dining, working, or other activities. "
                "In historical or religious art, a table can symbolize community or gathering."
            ),
            "the last supper": (
                "The Last Supper is a late 15th-century mural painting by Leonardo da Vinci. "
                "It is one of the world's most recognizable paintings. It depicts Jesus and his disciples "
                "at the final meal before his crucifixion."
            )
        }

    def retrieve_info(self, analysis: str) -> str:
        """
        Extracts object names from the analysis and retrieves additional context from the knowledge base.

        :param analysis: The analysis report from the ImageAnalyzer.
        :return: Retrieved additional context as a string.
        """
        additional_contexts = []
        lines = analysis.splitlines()
        # lines[0] is "Detected objects (YOLO):", subsequent lines have 'label: count'
        for line in lines[1:]:
            parts = line.split(":")
            if len(parts) >= 1:
                obj = parts[0].strip().lower()
                if obj in self.knowledge_base:
                    context = self.knowledge_base[obj]
                    additional_contexts.append(f"{obj.capitalize()}: {context}")

        if additional_contexts:
            retrieved_info = "\n".join(additional_contexts)
        else:
            retrieved_info = "No additional context available."

        logger.info("Retrieved additional context:\n%s", retrieved_info)
        return retrieved_info


class LLMClient:
    """
    Handles communication with a language model API to generate detailed commentary
    based on image analysis results and retrieved context.
    """

    def __init__(self, api_key: str, base_url: str) -> None:
        self.api_key = api_key
        self.base_url = base_url
        self.chat_endpoint = f"{self.base_url}/v1/openai/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "accept": "*/*"
        }
        logger.info("LLMClient initialized with endpoint: %s", self.chat_endpoint)

    def build_request_payload(self, analysis: str, retrieved_info: str) -> Dict[str, Any]:
        """
        Constructs the JSON payload for the LLM API request using both analysis and retrieved context.

        :param analysis: The analysis report from the ImageAnalyzer.
        :param retrieved_info: Additional context from the RetrievalEngine.
        :return: JSON payload for the API request.
        """
        system_message = (
            "You are an experienced assistant who analyzes all elements detected in an image. "
            "Evaluate the relationships between objects, the overall atmosphere, emotional impact, and composition. "
            "Below are the image analysis results and additional context retrieved from external sources. "
            "Based on this information, provide a detailed explanation of the image content and commentary."
        )
        user_message = (
            f"Image analysis results:\n{analysis}\n\n"
            f"Additional context:\n{retrieved_info}\n\n"
            "Based on this information, please explain what is depicted in the image and provide detailed commentary."
        )

        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message}
        ]

        payload = {
            "messages": messages,
            "model": "generative-ai",  # Adjust the model name as needed
            "stream": True,
            "temperature": 0.7,
            "max_tokens": 200,
            "stop": ["\n"]
        }
        logger.debug("Request payload created: %s", payload)
        return payload

    def stream_chat_completion(self, payload: Dict[str, Any]) -> str:
        """
        Sends the payload to the LLM API and streams the response.

        :param payload: JSON payload for the request.
        :return: Complete response text from the LLM.
        """
        try:
            response = requests.post(self.chat_endpoint, headers=self.headers, json=payload, stream=True)
            response.raise_for_status()
            logger.info("Chat completion request sent successfully.")
        except requests.RequestException as error:
            logger.error("HTTP request failed: %s", error)
            raise

        full_response = ""
        try:
            for line in response.iter_lines(decode_unicode=True):
                if line:
                    if line.startswith("data: "):
                        line = line[6:]
                    json_data = json.loads(line)
                    choices = json_data.get("choices")
                    if choices:
                        delta = choices[0].get("delta", {})
                        content = delta.get("content")
                        if content:
                            full_response += content
                            print(f"{Fore.CYAN}{content}{Style.RESET_ALL}", end="", flush=True)
        except Exception as error:
            logger.error("Error processing streaming response: %s", error)
            raise

        return full_response


class ImageAnalysisPipeline:
    """
    Coordinates the image analysis, retrieval, and commentary generation.
    """

    def __init__(self, analyzer: ImageAnalyzer, retrieval_engine: RetrievalEngine, llm_client: LLMClient) -> None:
        self.analyzer = analyzer
        self.retrieval_engine = retrieval_engine
        self.llm_client = llm_client

    def run(self, image_path: str) -> str:
        """
        Executes the analysis pipeline on the given image.

        :param image_path: Path to the image file.
        :return: Commentary generated by the LLM.
        """
        logger.info("Initiating image analysis...")
        analysis = self.analyzer.analyze(image_path)

        logger.info("Retrieving additional context...")
        retrieved_info = self.retrieval_engine.retrieve_info(analysis)

        logger.info("Preparing LLM request payload...")
        payload = self.llm_client.build_request_payload(analysis, retrieved_info)

        logger.info("Sending request to LLM API...")
        return self.llm_client.stream_chat_completion(payload)


def main() -> None:
    model_path = "yolo12x.pt"           # Update the model file path if necessary
    image_path = "the_last_supper.jpg"  # Update the image file path accordingly

    try:
        analyzer = ImageAnalyzer(model_path)
        retrieval_engine = RetrievalEngine()
        llm_client = LLMClient(API_KEY, BASE_URL)
        pipeline = ImageAnalysisPipeline(analyzer, retrieval_engine, llm_client)
        final_response = pipeline.run(image_path)
        logger.info("\nFull Response:\n%s", final_response)
    except Exception as error:
        logger.error("An error occurred during pipeline execution: %s", error)


if __name__ == "__main__":
    main()
