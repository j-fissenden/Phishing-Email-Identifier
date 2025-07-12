from flask import Flask, render_template, request, redirect, url_for, flash
import email
from email.utils import parseaddr
from email.policy import default
import re
import joblib
import pandas as pd
import os
import logging
import requests
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

app = Flask(__name__)

app.secret_key = os.urandom(24)
app.config['MAX_CONTENT_LENGTH'] = 1 * 1024 * 1024
app.config['ALLOWED_EXTENSIONS'] = {'eml'}
logging.basicConfig(level = logging.INFO)

nltk.download(['punkt', 'punkt_tab', 'stopwords', 'wordnet'])
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

MODELS = {
    'Random Forest': joblib.load(os.path.join('models', 'random_forest_detector.pkl')),
    'Decision Tree': joblib.load(os.path.join('models', 'decision_tree_detector.pkl')),
    'Naive Bayes': joblib.load(os.path.join('models', 'naive_bayes_detector.pkl')),
    'KNN': joblib.load(os.path.join('models', 'knn_detector.pkl'))
}

def extract_email_features(eml_content):
    """Extracts key features from an email (.eml file)"""
    try:
        msg = email.message_from_bytes(eml_content, policy=default)
    except Exception as e:
        raise ValueError("Invalid EML file format") from e

    features = {
        'sender': msg['from'] or '',
        'receiver': msg['to'] or '',
        'subject': msg['subject'] or '',
        'body': '',
        'urls': 0
    }

    if msg.is_multipart():
        for part in msg.walk():
            if part.get_content_type() == 'text/plain':
                try:
                    features['body'] += part.get_payload(decode=True).decode(errors='ignore')
                    break
                except UnicodeDecodeError:
                    continue
    else:
        features['body'] = msg.get_payload(decode=True).decode(errors='ignore')

    url_pattern = r'https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+'
    features['urls'] = 1 if re.search(url_pattern, features['body']) else 0
    features['extracted_urls'] = re.findall(url_pattern, features['body'])

    logging.info(f"Extracted URLs: {features['extracted_urls']}")
    logging.info(f"Sender: {features['sender']}")

    return features

def preprocess_features(features):
    """Processes extracted email features for model compatibility"""
    def safe_email_parser(field):
        """Extracts domain from email address safely"""
        try:
            _, email_addr = parseaddr(str(field))
            if not email_addr or '@' not in email_addr:
                return 'invalid_domain'
            return email_addr.split('@')[-1].lower().strip('>')
        except:
            return 'error_domain'

    def clean_text(text):
        """Cleans and tokenises text for NLP processing"""
        text = str(text).lower()
        text = re.sub(r'[^\w\s]', '', text)
        tokens = word_tokenize(text)
        tokens = [lemmatizer.lemmatize(word) for word in tokens 
                 if word not in stop_words and len(word) > 2]
        return ' '.join(tokens)

    processed = {
        'processed_text': (
            f"{safe_email_parser(features['sender'])} "
            f"{safe_email_parser(features['receiver'])} "
            f"{clean_text(features['subject'])} "
            f"{clean_text(features['body'])}"
        ),
        'urls': features['urls']
    }

    return pd.DataFrame([processed])

def allowed_file(filename):
    """Checks if the uploaded file has a valid extension"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def check_url_with_virustotal(url):
    """Verifies URL safety using VirusTotal API"""
    API_KEY = os.getenv("VT_API_KEY")  # Change this to your VirusTotal API Key
    endpoint = f"https://www.virustotal.com/api/v3/urls"
    headers = {
        "x-apikey": API_KEY,
        "Content-Type": "application/x-www-form-urlencoded"
    }
    data = {"url": url}

    try:
        response = requests.post(endpoint, headers=headers, data=data, timeout=20)
        response.raise_for_status()
        json_response = response.json()
        analysis_id = json_response.get("data", {}).get("id")

        if not analysis_id:
            return {"status": "unknown", "message": "Not enough information", "detail_url": None}

        report_endpoint = f"https://www.virustotal.com/api/v3/analyses/{analysis_id}"
        report_response = requests.get(report_endpoint, headers=headers, timeout=10)
        report_response.raise_for_status()
        report_json = report_response.json()
        stats = report_json.get("data", {}).get("attributes", {}).get("stats", {})

        malicious = stats.get("malicious", 0)
        phishing = stats.get("phishing", 0)
        harmless = stats.get("harmless", 0)
        undetected = stats.get("undetected", 0)
        detail_url = f"https://www.virustotal.com/gui/url/{analysis_id}/detection"

        if phishing > 0:
            return {"status": "phishing", "message": "⛔ URL is a phishing attempt", "detail_url": detail_url}
        elif malicious > 0:
            return {"status": "malware", "message": "⛔ URL leads to malware", "detail_url": detail_url}
        elif harmless > 0 and undetected > 0:
            return {"status": "safe", "message": "✅ URL is safe", "detail_url": detail_url}
        else:
            return {"Status": "unknown", "message": "❓ Not enough information", "detail_url": detail_url}

    except Exception as e:
        logging.error(f"VirusTotal lookup error: {str(e)}")
        return {"status": "error", "message": "❓ Unable to verify URL status", "detail_url": detail_url}

@app.route('/')
def index():
    """Renders the home page"""
    return render_template('index.html')

@app.route('/analyse', methods=['POST'])
def analyse():
    """Handles email file uploads and phishing detection"""
    if 'email_file' not in request.files:
        flash('No file uploaded')
        return redirect(url_for('index'))
    
    file = request.files['email_file']
    if not file or not allowed_file(file.filename):
        flash('Invalid file type')
        return redirect(url_for('index'))

    try:
        features = extract_email_features(file.read())
        processed_data = preprocess_features(features)

        results = {}
        for model_name, model in MODELS.items():
            proba = model.predict_proba(processed_data)[0]
            results[model_name] = {
                'prediction': 'Phishing' if model.predict(processed_data)[0] == 1 else 'Legitimate',
                'phishing_confidence': round(proba[1]*100, 1),
                'legitimate_confidence': round(proba[0]*100, 1)
            }

        vt_results = {url: check_url_with_virustotal(url) for url in features.get('extracted_urls', [])}

        return render_template('results.html', results=results, features=features, vt_results=vt_results)

    except Exception as e:
        logging.error(f"Processing error: {str(e)}")
        return render_template('error.html', error = str(e))

if __name__ == '__main__':
    app.run(host ='0.0.0.0', port = 5000, debug = True)
