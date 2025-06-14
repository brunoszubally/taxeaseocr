import os
import json
import time
import logging
import tempfile
import requests
from flask import Flask, request, jsonify
from flask_cors import CORS
from openai import OpenAI
from azure.ai.formrecognizer import DocumentAnalysisClient
from azure.core.credentials import AzureKeyCredential
from dotenv import load_dotenv

# Környezeti változók betöltése
load_dotenv()

# Logging beállítása
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Azure Form Recognizer beállítások
endpoint = os.getenv("AZURE_FORM_RECOGNIZER_ENDPOINT")
key = os.getenv("AZURE_FORM_RECOGNIZER_KEY")

# OpenAI beállítások
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
assistant_id = os.getenv("OPENAI_ASSISTANT_ID")

# Flask alkalmazás létrehozása
app = Flask(__name__)
CORS(app)

def capture_output(local_image_path):
    logging.debug("Kép feldolgozása Azure Form Recognizer-rel.")
    document_analysis_client = DocumentAnalysisClient(endpoint=endpoint, credential=AzureKeyCredential(key))
    
    with open(local_image_path, "rb") as image:
        poller = document_analysis_client.begin_analyze_document("prebuilt-document", image.read())
    result = poller.result()
    
    output = []
    for page in result.pages:
        for line in page.lines:
            output.append(line.content)
    
    return "\n".join(output)

def send_to_assistant(content):
    logging.debug("Üzenet küldése az Assistant-nak.")
    thread = client.beta.threads.create()
    client.beta.threads.messages.create(thread_id=thread.id, role="user", content=content)
    run = client.beta.threads.runs.create(thread_id=thread.id, assistant_id=assistant_id)
    return run.id, thread.id

def check_status(run_id, thread_id):
    logging.debug("Feldolgozás állapotának ellenőrzése.")
    run = client.beta.threads.runs.retrieve(thread_id=thread_id, run_id=run_id)
    return run.status

def process_image(local_image_path):
    logging.debug("Kép feldolgozása megkezdődött.")
    captured_output = capture_output(local_image_path)
    logging.debug("Kimenet: %s", captured_output)

    run_id, thread_id = send_to_assistant(captured_output)

    while True:
        status = check_status(run_id, thread_id)
        logging.debug("Feldolgozás állapota: %s", status)
        if status == "completed":
            break
        elif status == "failed":
            logging.error("Assistant feldolgozás sikertelen.")
            raise Exception("❌ Assistant feldolgozás sikertelen.")
        time.sleep(2)

    messages = client.beta.threads.messages.list(thread_id=thread_id)

    if messages.data:
        for message in messages.data:
            if message.role == "assistant":
                try:
                    if isinstance(message.content, list):
                        text_content = message.content[0].text.value
                        clean_content = text_content.replace("```json", "").replace("```", "").strip()
                        try:
                            response_data = json.loads(clean_content)
                            if isinstance(response_data, list):
                                return response_data[0]
                            else:
                                return response_data
                        except json.JSONDecodeError as e:
                            logging.error(f"JSON dekódolási hiba: {str(e)}")
                            logging.error(f"Problémás tartalom: {clean_content}")
                            raise Exception(f"❌ JSON dekódolási hiba: {str(e)}")
                    else:
                        raise Exception("❌ A válasz nem lista.")
                except json.JSONDecodeError:
                    logging.error("A válasz nem érvényes JSON.")
                    raise Exception("❌ A válasz nem érvényes JSON.")
    else:
        logging.error("Nem érkezett válasz az Assistant-tól.")
        raise Exception("❌ Nem érkezett válasz az Assistant-tól.")

@app.route('/process', methods=['POST'])
def process_image_url():
    data = request.get_json()
    
    if not data or 'url' not in data:
        return jsonify({'error': 'Hiányzó URL a kérésben'}), 400
    
    image_url = data['url']
    
    try:
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_file:
            response = requests.get(image_url)
            if response.status_code != 200:
                return jsonify({'error': 'Kép letöltése sikertelen'}), 400
                
            temp_file.write(response.content)
            temp_file_path = temp_file.name
        
        result = process_image(temp_file_path)
        
        os.unlink(temp_file_path)
        
        return result
        
    except Exception as e:
        if 'temp_file_path' in locals() and os.path.exists(temp_file_path):
            os.unlink(temp_file_path)
        return jsonify({'error': str(e)}), 500

@app.route('/ping', methods=['GET'])
def ping():
    return jsonify({"status": "ok", "message": "pong"}), 200

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5001))
    app.run(host='0.0.0.0', port=port) 