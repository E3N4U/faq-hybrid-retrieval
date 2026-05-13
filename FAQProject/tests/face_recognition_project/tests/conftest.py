import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from fastapi.testclient import TestClient
from tests.app import app


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def sample_image_base64():
    import base64
    import numpy as np
    import cv2

    image = np.zeros((100, 100, 3), dtype=np.uint8)
    cv2.rectangle(image, (20, 20), (80, 80), (255, 255, 255), -1)
    _, buffer = cv2.imencode('.jpg', image)
    return base64.b64encode(buffer).decode('utf-8')


@pytest.fixture
def sample_face_image_base64():
    import base64
    import numpy as np
    import cv2

    image = np.zeros((200, 200, 3), dtype=np.uint8)
    cv2.ellipse(image, (100, 100), (60, 80), 0, 0, 360, (255, 230, 200), -1)
    cv2.circle(image, (80, 85), 8, (50, 50, 50), -1)
    cv2.circle(image, (120, 85), 8, (50, 50, 50), -1)

    _, buffer = cv2.imencode('.jpg', image)
    return base64.b64encode(buffer).decode('utf-8')
