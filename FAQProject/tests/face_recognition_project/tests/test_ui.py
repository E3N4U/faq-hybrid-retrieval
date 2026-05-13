import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import time


class TestWebInterface:
    def test_index_page_loads(self, client):
        response = client.get('/')
        assert response.status_code == 200
        html_content = response.data.decode('utf-8')
        assert 'Face Recognition' in html_content or '人脸识别' in html_content

    def test_health_endpoint_accessible(self, client):
        response = client.get('/api/health')
        assert response.status_code == 200
        data = response.get_json()
        assert data['status'] == 'healthy'


class TestUIInteractions:
    def test_upload_section_exists(self, client):
        response = client.get('/')
        html = response.data.decode('utf-8')
        assert 'upload-section' in html or '上传' in html

    def test_detect_button_exists(self, client):
        response = client.get('/')
        html = response.data.decode('utf-8')
        assert 'detectBtn' in html or '开始检测' in html


class TestUIFlow:
    def test_api_integration_with_ui(self, client):
        import io
        import numpy as np
        import cv2
        from PIL import Image
        import base64

        image = np.zeros((100, 100, 3), dtype=np.uint8)
        cv2.rectangle(image, (20, 20), (80, 80), (255, 255, 255), -1)
        _, buffer = cv2.imencode('.jpg', image)

        response = client.post('/api/detect/file',
                              data={'file': (io.BytesIO(buffer.tobytes()), 'test.jpg')},
                              content_type='multipart/form-data')

        assert response.status_code == 200
        data = response.get_json()
        assert data['success'] is True
        assert 'face_count' in data
        assert 'has_face' in data


class TestUIResponsiveness:
    def test_mobile_viewport(self, client):
        response = client.get('/')
        html = response.data.decode('utf-8')
        assert 'viewport' in html
        assert 'max-width' in html

    def test_css_animations(self, client):
        response = client.get('/')
        html = response.data.decode('utf-8')
        assert 'animation' in html or '@keyframes' in html
