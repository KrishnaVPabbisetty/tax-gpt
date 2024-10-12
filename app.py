from flask import Flask, request, jsonify
import requests
import os
from dotenv import load_dotenv
from flask_cors import CORS  # Import CORS from flask_cors

# Load environment variables from the .env file
load_dotenv()

# Initialize Flask app
app = Flask(__name__)

# Enable CORS for all origins, methods, and headers
CORS(
    app,
    resources={r"/*": {"origins": "*"}},
    allow_headers=["Content-Type", "Authorization", "Access-Control-Allow-Credentials"],
    supports_credentials=True,
)

# Hugging Face API URL for text generation (or any other endpoint)
API_URL = "https://api-inference.huggingface.co/models/gpt2"

# Load Hugging Face API key from .env
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")


# Function to call Hugging Face API for text generation
def query_huggingface(prompt):
    API_URL = "https://api-inference.huggingface.co/models/gpt2"
    headers = {"Authorization": f"Bearer {HUGGINGFACE_API_KEY}"}

    payload = {
        "inputs": prompt,
        "parameters": {
            "max_length": 50,
            "temperature": 0.7,
            "top_p": 0.9,
            "top_k": 50,
        },
    }

    response = requests.post(API_URL, headers=headers, json=payload)

    if response.status_code == 200:
        return response.json()
    else:
        return [{"generated_text": "Error calling Hugging Face API"}]  # Error handling


# Function to check if the prompt is tax-related
def is_tax_related(prompt):
    tax_keywords = ["tax", "deductions", "IRS", "income", "refund", "taxes", "audit"]
    return any(keyword in prompt.lower() for keyword in tax_keywords)


# Main route that handles POST and OPTIONS requests
@app.route("/api/check_tax", methods=["POST", "OPTIONS"])
def check_tax_prompt():
    # Handle preflight OPTIONS request
    if request.method == "OPTIONS":
        return _build_cors_preflight_response()

    # Handle actual POST request
    if request.method == "POST":
        data = request.json
        prompt = data.get("prompt")

        # Check if the prompt is tax-related
        if is_tax_related(prompt):
            generated_response = query_huggingface(prompt)
            text_response = generated_response[0]["generated_text"]
            return _corsify_actual_response(
                jsonify({"is_tax_related": True, "response": text_response})
            )
        else:
            return _corsify_actual_response(jsonify({"is_tax_related": False}))


# Preflight CORS response
def _build_cors_preflight_response():
    response = jsonify({"message": "Preflight CORS response"})
    response.headers.add("Access-Control-Allow-Origin", "*")
    response.headers.add("Access-Control-Allow-Headers", "Content-Type,Authorization")
    response.headers.add("Access-Control-Allow-Methods", "GET,POST,OPTIONS")
    return response


# Actual response with CORS headers
def _corsify_actual_response(response):
    response.headers.add("Access-Control-Allow-Origin", "*")
    response.headers.add("Access-Control-Allow-Headers", "Content-Type,Authorization")
    response.headers.add("Access-Control-Allow-Methods", "GET,POST,OPTIONS")
    return response


if __name__ == "__main__":
    app.run(debug=True)
