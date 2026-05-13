import cv2
import numpy as np
from typing import List, Tuple, Optional


class FaceRecognitor:
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        self.eye_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_eye.xml'
        )

    def detect_faces(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        if image is None or image.size == 0:
            return []

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        return faces.tolist() if len(faces) > 0 else []

    def detect_eyes(self, image: np.ndarray, face_region: Tuple[int, int, int, int]) -> List[Tuple[int, int, int, int]]:
        if image is None or len(face_region) != 4:
            return []

        x, y, w, h = face_region
        roi_gray = image[y:y+h, x:x+w]
        eyes = self.eye_cascade.detectMultiScale(roi_gray)

        adjusted_eyes = []
        for (ex, ey, ew, eh) in eyes:
            adjusted_eyes.append((ex + x, ey + y, ew, eh))

        return adjusted_eyes.tolist() if len(eyes) > 0 else []

    def count_faces(self, image: np.ndarray) -> int:
        faces = self.detect_faces(image)
        return len(faces)

    def get_largest_face(self, image: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        faces = self.detect_faces(image)
        if not faces:
            return None

        return max(faces, key=lambda f: f[2] * f[3])

    def has_face(self, image: np.ndarray) -> bool:
        return self.count_faces(image) > 0

    def process_image(self, image: np.ndarray, draw_results: bool = True) -> dict:
        faces = self.detect_faces(image)
        result = {
            'face_count': len(faces),
            'faces': [],
            'has_face': len(faces) > 0
        }

        if draw_results:
            for (x, y, w, h) in faces:
                cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)

                roi_gray = image[y:y+h, x:x+w]
                eyes = self.eye_cascade.detectMultiScale(roi_gray)
                for (ex, ey, ew, eh) in eyes:
                    cv2.rectangle(image, (x+ex, y+ey), (x+ex+ew, y+ey+eh), (0, 255, 0), 2)

        result['faces'] = [
            {'x': int(x), 'y': int(y), 'width': int(w), 'height': int(h)}
            for (x, y, w, h) in faces
        ]

        return result
