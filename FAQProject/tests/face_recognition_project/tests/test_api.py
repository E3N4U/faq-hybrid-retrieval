import pytest
import io
import numpy as np
import cv2


class TestHealthEndpoint:
    def test_health_check(self, client):
        response = client.get("/api/health")
        assert response.status_code == 200
        data = response.json()
        assert data['status'] == 'healthy'
        assert 'version' in data

    def test_root_endpoint(self, client):
        response = client.get("/")
        assert response.status_code in [200, 404]


class TestDetectBase64Endpoint:
    def test_detect_no_image(self, client):
        response = client.post("/api/detect/base64", json={})
        assert response.status_code == 422

    def test_detect_invalid_base64(self, client):
        response = client.post("/api/detect/base64", json={'image': 'invalid_base64_data'})
        assert response.status_code == 500

    def test_detect_valid_image(self, client, sample_image_base64):
        response = client.post("/api/detect/base64", json={'image': sample_image_base64})
        assert response.status_code == 200
        data = response.json()
        assert data['success'] is True
        assert 'face_count' in data
        assert 'has_face' in data

    def test_detect_base64_with_prefix(self, client, sample_image_base64):
        response = client.post("/api/detect/base64",
                              json={'image': 'data:image/jpeg;base64,' + sample_image_base64})
        assert response.status_code == 200
        data = response.json()
        assert data['success'] is True


class TestDetectFileEndpoint:
    def test_detect_no_file(self, client):
        response = client.post("/api/detect/file")
        assert response.status_code == 422

    def test_detect_invalid_file_type(self, client):
        files = {'file': ('test.txt', io.BytesIO(b'invalid'), 'text/plain')}
        response = client.post("/api/detect/file", files=files)
        assert response.status_code == 400

    def test_detect_valid_jpg_file(self, client):
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        cv2.rectangle(image, (20, 20), (80, 80), (255, 255, 255), -1)
        _, buffer = cv2.imencode('.jpg', image)

        files = {'file': ('test.jpg', io.BytesIO(buffer.tobytes()), 'image/jpeg')}
        response = client.post("/api/detect/file", files=files)

        assert response.status_code == 200
        result = response.json()
        assert result['success'] is True
        assert 'filename' in result

    def test_detect_with_features(self, client):
        image = np.zeros((200, 200, 3), dtype=np.uint8)
        cv2.ellipse(image, (100, 100), (60, 80), 0, 0, 360, (255, 230, 200), -1)
        cv2.circle(image, (80, 85), 8, (50, 50, 50), -1)
        cv2.circle(image, (120, 85), 8, (50, 50, 50), -1)
        _, buffer = cv2.imencode('.jpg', image)

        files = {
            'file': ('test.jpg', io.BytesIO(buffer.tobytes()), 'image/jpeg'),
            'detect_features': (None, 'true')
        }
        response = client.post("/api/detect/file", files=files)

        assert response.status_code == 200
        result = response.json()
        assert result['success'] is True


class TestDetectStreamEndpoint:
    def test_detect_stream(self, client):
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        _, buffer = cv2.imencode('.jpg', image)

        files = {'file': ('test.jpg', io.BytesIO(buffer.tobytes()), 'image/jpeg')}
        response = client.post("/api/detect/stream", files=files)

        assert response.status_code == 200
        data = response.json()
        assert data['success'] is True
        assert 'image' in data


class TestCompareEndpoint:
    def test_compare_faces(self, client):
        encoding1 = [0.1] * 128
        encoding2 = [0.1] * 128

        response = client.post("/api/compare",
                             json={'face1_enc': encoding1, 'face2_enc': encoding2})

        assert response.status_code in [200, 422]


class TestHistoryEndpoint:
    def test_get_history(self, client):
        response = client.get("/api/history")
        assert response.status_code == 200
        data = response.json()
        assert 'records' in data

    def test_get_history_with_limit(self, client):
        response = client.get("/api/history?limit=5")
        assert response.status_code == 200
        data = response.json()
        assert 'records' in data


class TestStatsEndpoint:
    def test_get_stats(self, client):
        response = client.get("/api/stats")
        assert response.status_code == 200
        data = response.json()
        assert 'total_detections' in data
        assert 'total_faces_detected' in data


class TestBatchDetectEndpoint:
    def test_batch_detect(self, client):
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        _, buffer = cv2.imencode('.jpg', image)

        files = [
            ('files', ('test1.jpg', io.BytesIO(buffer.tobytes()), 'image/jpeg')),
            ('files', ('test2.jpg', io.BytesIO(buffer.tobytes()), 'image/jpeg'))
        ]

        response = client.post("/api/batch-detect", files=files)
        assert response.status_code == 200
        data = response.json()
        assert data['total'] == 2
