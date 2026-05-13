import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
import cv2
import numpy as np
import base64
from werkzeug.utils import secure_filename
from tests.app import FaceRecognitor

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'gif'}

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

face_recognitor = FaceRecognitor()


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def index():
    ui_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'ui', 'index.html')
    if os.path.exists(ui_path):
        return send_from_directory(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'ui'), 'index.html')
    return jsonify({
        'service': 'Face Recognition API',
        'version': '1.0.0',
        'endpoints': {
            'health': '/api/health',
            'detect_base64': '/api/detect/base64',
            'detect_file': '/api/detect/file',
            'detect_url': '/api/detect/url'
        }
    })


@app.route('/api/health')
def health():
    return jsonify({
        'status': 'healthy',
        'service': 'face-recognition-api',
        'version': '1.0.0'
    })


@app.route('/api/detect/base64', methods=['POST'])
def detect_base64():
    try:
        data = request.get_json()
        if 'image' not in data:
            return jsonify({'error': 'No image data provided'}), 400

        image_data = data['image']
        if ',' in image_data:
            image_data = image_data.split(',')[1]

        image_bytes = base64.b64decode(image_data)
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            return jsonify({'error': 'Invalid image data'}), 400

        result = face_recognitor.process_image(image, draw_results=False)

        return jsonify({
            'success': True,
            'face_count': result['face_count'],
            'has_face': result['has_face'],
            'faces': result['faces']
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/detect/file', methods=['POST'])
def detect_file():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        if not allowed_file(file.filename):
            return jsonify({'error': 'File type not allowed'}), 400

        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        image = cv2.imread(filepath)
        if image is None:
            os.remove(filepath)
            return jsonify({'error': 'Invalid image file'}), 400

        draw = request.form.get('draw', 'false').lower() == 'true'
        result = face_recognitor.process_image(image, draw_results=draw)

        if draw:
            output_path = os.path.join(app.config['UPLOAD_FOLDER'], 'result_' + filename)
            cv2.imwrite(output_path, image)
            result['result_image'] = '/uploads/' + 'result_' + filename

        os.remove(filepath)

        return jsonify({
            'success': True,
            'filename': filename,
            'face_count': result['face_count'],
            'has_face': result['has_face'],
            'faces': result['faces']
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/detect/url', methods=['POST'])
def detect_url():
    try:
        data = request.get_json()
        if 'url' not in data:
            return jsonify({'error': 'No URL provided'}), 400

        import urllib.request

        url = data['url']
        headers = {'User-Agent': 'Mozilla/5.0'}
        req = urllib.request.Request(url, headers=headers)
        response = urllib.request.urlopen(req, timeout=10)

        image_bytes = response.read()
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            return jsonify({'error': 'Could not download image'}), 400

        result = face_recognitor.process_image(image, draw_results=False)

        return jsonify({
            'success': True,
            'url': url,
            'face_count': result['face_count'],
            'has_face': result['has_face'],
            'faces': result['faces']
        })

    except urllib.error.URLError:
        return jsonify({'error': 'Invalid URL or timeout'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/uploads/<path:filename>')
def serve_upload(filename):
    from flask import send_from_directory
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


if __name__ == '__main__':
    app.run(debug=True, port=5000)