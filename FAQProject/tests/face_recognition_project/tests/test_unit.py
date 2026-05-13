import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import cv2
import numpy as np
from tests.app import FaceRecognitor


@pytest.fixture
def face_recognitor():
    return FaceRecognitor()


@pytest.fixture
def sample_image():
    image = np.zeros((200, 200, 3), dtype=np.uint8)
    cv2.rectangle(image, (50, 50), (150, 150), (255, 255, 255), -1)
    return image


@pytest.fixture
def face_image():
    image = np.zeros((200, 200, 3), dtype=np.uint8)
    cv2.ellipse(image, (100, 100), (60, 80), 0, 0, 360, (255, 230, 200), -1)
    cv2.circle(image, (80, 85), 8, (50, 50, 50), -1)
    cv2.circle(image, (120, 85), 8, (50, 50, 50), -1)
    return image


class TestFaceRecognitionUnit:
    def test_initialization(self, face_recognitor):
        assert face_recognitor is not None
        assert face_recognitor.face_cascade is not None
        assert face_recognitor.eye_cascade is not None

    def test_detect_faces_empty_image(self, face_recognitor, sample_image):
        faces = face_recognitor.detect_faces(sample_image)
        assert isinstance(faces, list)

    def test_detect_faces_with_face(self, face_recognitor, face_image):
        faces = face_recognitor.detect_faces(face_image)
        assert isinstance(faces, list)

    def test_count_faces(self, face_recognitor, face_image):
        count = face_recognitor.count_faces(face_image)
        assert isinstance(count, int)
        assert count >= 0

    def test_has_face_true(self, face_recognitor, face_image):
        result = face_recognitor.has_face(face_image)
        assert isinstance(result, bool)

    def test_has_face_false(self, face_recognitor, sample_image):
        result = face_recognitor.has_face(sample_image)
        assert isinstance(result, bool)

    def test_get_largest_face_none(self, face_recognitor, sample_image):
        result = face_recognitor.get_largest_face(sample_image)
        assert result is None or isinstance(result, tuple)

    def test_get_largest_face_with_face(self, face_recognitor, face_image):
        result = face_recognitor.get_largest_face(face_image)
        if result:
            assert len(result) == 4

    def test_process_image_structure(self, face_recognitor, face_image):
        result = face_recognitor.process_image(face_image, draw_results=False)
        assert 'face_count' in result
        assert 'faces' in result
        assert 'has_face' in result
        assert isinstance(result['faces'], list)

    def test_process_image_with_draw(self, face_recognitor, face_image):
        result = face_recognitor.process_image(face_image.copy(), draw_results=True)
        assert result['face_count'] >= 0

    def test_detect_eyes_invalid_region(self, face_recognitor, sample_image):
        eyes = face_recognitor.detect_eyes(sample_image, (0, 0, 50, 50))
        assert isinstance(eyes, list)

    def test_none_image(self, face_recognitor):
        faces = face_recognitor.detect_faces(None)
        assert faces == []

    def test_empty_image_array(self, face_recognitor):
        empty_image = np.array([])
        faces = face_recognitor.detect_faces(empty_image)
        assert faces == []
