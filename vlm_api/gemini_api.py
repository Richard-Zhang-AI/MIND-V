import os
from google.cloud import aiplatform
from google.oauth2 import service_account
from vertexai.generative_models import GenerativeModel, Image, Part
import urllib.request
import http.client
import pathlib
import time
import typing
from loguru import logger

class GeminiChat:
    def __init__(self, model="gemini-2.5-flash", region="us-central1",
                 project="captioner-test-1017", credential_file="captioner-test-1017-b3fa56e15267.json",
                 temperature=0.1, seed=None):
        """
        Initialize a Gemini chat session for multi-turn conversations.

        Args:
            model (str): The Gemini model to use
            region (str): Google Cloud region
            project (str): Google Cloud project ID
            credential_file (str): Path to credentials JSON file
            temperature (float): Sampling temperature
            seed (int): Random seed for reproducibility
        """
        self.model_name = model
        self.region = region
        self.project = project
        self.credential_file = credential_file
        self.temperature = temperature
        self.seed = seed

        # Initialize Google Cloud credentials and AI Platform
        credentials = service_account.Credentials.from_service_account_file(credential_file)
        aiplatform.init(project=project, credentials=credentials, location=region)

        # Initialize the model
        self.model = GenerativeModel(model)

        # Initialize conversation history
        self.history = []

    def add_message(self, role, content, image_path=None):
        """
        Add a message to the conversation history.

        Args:
            role (str): "user" or "assistant"
            content (str): Text content of the message
            image_path (str, optional): Path or URL to an image file
        """
        message = {"role": role, "content": content}
        if image_path:
            message["image_path"] = image_path
        self.history.append(message)

    def format_history_for_api(self):
        """
        Format conversation history for Gemini API call.

        The Vertex AI GenerativeModel expects `contents` to be either a single
        string/Part or a *flat* list of strings/Parts representing one
        conversation turn. Previously this method returned a list of lists,
        which caused errors like:
        "Unexpected item type: ['...']".

        Returns:
            list: Flat list of Image/str parts suitable for `generate_content`.
        """
        contents = []
        for msg in self.history:
            if msg.get("image_path"):
                # Handle image
                try:
                    if msg["image_path"].startswith("http"):
                        with urllib.request.urlopen(msg["image_path"]) as response:
                            response = typing.cast(http.client.HTTPResponse, response)
                            image_bytes = response.read()
                    else:
                        image_bytes = pathlib.Path(msg["image_path"]).read_bytes()
                    image_data = Image.from_bytes(image_bytes)
                    contents.append(image_data)
                except Exception as e:
                    logger.error(f"Failed to load image: {e}")
                    # Skip this image but still keep the text part if any
                    pass

            if msg["content"]:
                contents.append(msg["content"])

        return contents

    def chat(self, message, image_path=None, retry_attempts=5, return_usage=False):
        """
        Send a message to the model and get a response, maintaining conversation context.

        Args:
            message (str): The user's message
            image_path (str, optional): Path or URL to an image file
            retry_attempts (int): Number of retry attempts for failed API calls
            return_usage (bool): Whether to return token usage information

        Returns:
            str or tuple: Response text (and usage info if return_usage=True)
        """
        # Add user message to history
        self.add_message("user", message, image_path)

        # Configure generation parameters
        generation_config = {"temperature": self.temperature}
        if self.seed is not None:
            generation_config["seed"] = int(self.seed)

        # Format history for API call (flat list of Parts/strings)
        contents = self.format_history_for_api()

        # Make API call with retries
        for attempt in range(retry_attempts):
            try:
                response = self.model.generate_content(
                    contents=contents,
                    generation_config=generation_config
                )
                if response.candidates and response.candidates[0].content.parts:
                    response_text = response.candidates[0].content.parts[0].text

                    # Add assistant response to history
                    self.add_message("assistant", response_text)

                    if return_usage:
                        # Extract usage information
                        usage_info = {
                            'prompt_token_count': 0,
                            'candidates_token_count': 0,
                            'total_token_count': 0
                        }

                        if hasattr(response, 'usage_metadata') and response.usage_metadata:
                            usage_info['prompt_token_count'] = getattr(response.usage_metadata, 'prompt_token_count', 0)
                            usage_info['candidates_token_count'] = getattr(response.usage_metadata, 'candidates_token_count', 0)
                            usage_info['total_token_count'] = getattr(response.usage_metadata, 'total_token_count', 0)

                        return response_text, usage_info
                    else:
                        return response_text
                else:
                    logger.error("No valid candidates in response")
            except Exception as e:
                logger.error(f"API call failed: {e}, Retry attempt {attempt + 1}/{retry_attempts}")
                time.sleep(0.2)

        logger.error("Failed to get response after all retries")
        return (None, None) if return_usage else None

    def start_interactive_chat(self):
        """
        Start an interactive chat session with the model.
        """
        print("🤖 Gemini Interactive Chat Started!")
        print("Type 'quit', 'exit', or press Ctrl+C to end the conversation.")
        print("Type 'image: <path>' to include an image with your message.")
        print("Type 'history' to see conversation history.")
        print("Type 'clear' to clear conversation history.")
        print("-" * 50)

        try:
            while True:
                # Get user input
                user_input = input("\nYou: ").strip()

                # Handle special commands
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("👋 Goodbye!")
                    break
                elif user_input.lower() == 'history':
                    self.show_history()
                    continue
                elif user_input.lower() == 'clear':
                    self.clear_history()
                    print("🧹 Conversation history cleared.")
                    continue
                elif not user_input:
                    print("⚠️  Please enter a message.")
                    continue

                # Check for image in message
                image_path = None
                message = user_input
                if user_input.lower().startswith('image:'):
                    parts = user_input.split(':', 1)
                    if len(parts) > 1:
                        image_path = parts[1].strip()
                        message = input(f"You (with image {image_path}): ").strip()

                if not message:
                    print("⚠️  Please enter a message.")
                    continue

                print("🤖 Thinking...")
                # Get model response
                response = self.chat(message, image_path)
                if response:
                    print(f"\nGemini: {response}")
                else:
                    print("❌ Failed to get a response. Please try again.")

        except KeyboardInterrupt:
            print("\n👋 Goodbye!")

    def show_history(self):
        """Display conversation history."""
        if not self.history:
            print("📝 No conversation history yet.")
            return

        print("\n📝 Conversation History:")
        print("-" * 30)
        for i, msg in enumerate(self.history, 1):
            role = "👤 User" if msg["role"] == "user" else "🤖 Assistant"
            content = msg["content"]
            if msg.get("image_path"):
                content += f" [📷 Image: {msg['image_path']}]"
            print(f"{i}. {role}: {content}")

    def clear_history(self):
        """Clear conversation history."""
        self.history = []

def call_gemini_api(prompt, image_path=None, model="gemini-2.5-flash", region="us-central1", 
                    project="captioner-test-1017", credential_file="captioner-test-1017-b3fa56e15267.json", 
                    temperature=0.1, seed=None, retry_attempts=5, return_usage=False):
    """
    Calls the Gemini API to generate a response for a given prompt, optionally with an image.

    Args:
        prompt (str): The text prompt to send to the Gemini model.
        image_path (str, optional): Path or URL to an image file. Defaults to None.
        model (str): The Gemini model to use (e.g., "gemini-2.0-flash"). Defaults to "gemini-2.0-flash".
        region (str): The Google Cloud region. Defaults to "us-central1".
        project (str): The Google Cloud project ID. Defaults to "captioner-test-1017".
        credential_file (str): Path to the Google Cloud credentials JSON file. 
                              Defaults to "captioner-test-1017-b3fa56e15267.json".
        temperature (float): Sampling temperature for the model. Defaults to 0.01.
        seed (int): Random seed for reproducibility. Defaults to 42.
        retry_attempts (int): Number of retry attempts for failed API calls. Defaults to 5.
        return_usage (bool): Whether to return token usage information. Defaults to False.

    Returns:
        str or tuple: If return_usage is False, returns the response text or None.
                     If return_usage is True, returns (response_text, usage_dict) or (None, None).
    """
    # Initialize Google Cloud credentials and AI Platform
    credentials = service_account.Credentials.from_service_account_file(credential_file)
    aiplatform.init(project=project, credentials=credentials, location=region)

    # Prepare the payload
    payload = []
    if image_path:
        # Handle image input (local file or URL)
        try:
            if image_path.startswith("http"):
                with urllib.request.urlopen(image_path) as response:
                    response = typing.cast(http.client.HTTPResponse, response)
                    image_bytes = response.read()
            else:
                image_bytes = pathlib.Path(image_path).read_bytes()
            image_data = Image.from_bytes(image_bytes)
            payload.append(image_data)
        except Exception as e:
            logger.error(f"Failed to load image: {e}")
            return (None, None) if return_usage else None
    payload.append(prompt)

    # Initialize the model
    model_instance = GenerativeModel(model)

    # Configure generation parameters
    generation_config = {"temperature": temperature}
    if seed is not None:
        generation_config["seed"] = int(seed)

    # Make API call with retries
    for attempt in range(retry_attempts):
        try:
            response = model_instance.generate_content(
                contents=payload,
                generation_config=generation_config
            )
            if response.candidates and response.candidates[0].content.parts:
                response_text = response.candidates[0].content.parts[0].text
                
                if return_usage:
                    # Extract usage information
                    usage_info = {
                        'prompt_token_count': 0,
                        'candidates_token_count': 0,
                        'total_token_count': 0
                    }
                    
                    if hasattr(response, 'usage_metadata') and response.usage_metadata:
                        usage_info['prompt_token_count'] = getattr(response.usage_metadata, 'prompt_token_count', 0)
                        usage_info['candidates_token_count'] = getattr(response.usage_metadata, 'candidates_token_count', 0)
                        usage_info['total_token_count'] = getattr(response.usage_metadata, 'total_token_count', 0)
                    
                    return response_text, usage_info
                else:
                    return response_text
            else:
                logger.error("No valid candidates in response")
        except Exception as e:
            logger.error(f"API call failed: {e}, Retry attempt {attempt + 1}/{retry_attempts}")
            time.sleep(0.2)
    
    logger.error("Failed to get response after all retries")
    return (None, None) if return_usage else None

# Example usage
if __name__ == "__main__":
    import sys

    # Check if interactive mode is requested
    if len(sys.argv) > 1 and sys.argv[1].lower() in ['interactive', 'chat', '-i', '-c']:
        # Start interactive chat session
        chat_session = GeminiChat()
        chat_session.start_interactive_chat()
    else:
        # Original single Q&A example
        prompt = "Describe the picture. How many avocados are there in this picture? Can you locate the position of the avocado on the right in the picture? Please output the (x, y) coordinates of the center of the avocado on the right, where x and y are the proportions of the width and height of the picture respectively (decimal numbers between 0 and 1). By default, the top-left corner of the picture is (0,0) and the bottom-right corner is (1,1)."
        image_path = "/data/rczhang/MIND-V/demos/avocado.png"

        # Call the function with token usage information
        result = call_gemini_api(prompt=prompt, image_path=image_path, return_usage=True)
        if result[0]:
            response_text, usage_info = result
            print("Response:", response_text)
            print("Token Usage:")
            print(f"  Input tokens: {usage_info['prompt_token_count']}")
            print(f"  Output tokens: {usage_info['candidates_token_count']}")
            print(f"  Total tokens: {usage_info['total_token_count']}")
        else:
            print("Failed to get a response from the Gemini API.")

        print("\n💡 To start an interactive chat session, run:")
        print("python gemini_api.py interactive")
