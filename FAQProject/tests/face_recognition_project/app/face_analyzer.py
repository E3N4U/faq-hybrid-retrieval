import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
import base64
import os


class FaceAnalyzer:
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        self.eye_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_eye.xml'
        )
        self.smile_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_smile.xml'
        )

        age_net = None
        gender_net = None

        try:
            prototxt_path = "models/deploy_age.prototxt"
            weights_path = "models/age_net.caffemodel"
            if os.path.exists(prototxt_path) and os.path.exists(weights_path):
                age_net = cv2.dnn.readNetFromCaffe(prototxt_path, weights_path)
        except:
            pass

        try:
            prototxt_path = "models/deploy_gender.prototxt"
            weights_path = "models/gender_net.caffemodel"
            if os.path.exists(prototxt_path) and os.path.exists(weights_path):
                gender_net = cv2.dnn.readNetFromCaffe(prototxt_path, weights_path)
        except:
            pass

        self.age_net = age_net
        self.gender_net = gender_net

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

    def detect_smile(self, image: np.ndarray, face_region: Tuple[int, int, int, int]) -> bool:
        if image is None or len(face_region) != 4:
            return False

        x, y, w, h = face_region
        roi_gray = image[y:y+h, x:x+w]
        smiles = self.smile_cascade.detectMultiScale(
            roi_gray,
            scaleFactor=1.7,
            minNeighbors=20
        )
        return len(smiles) > 0

    def estimate_age_gender(self, image: np.ndarray, face_region: Tuple[int, int, int, int]) -> Dict[str, Any]:
        result = {'age': None, 'gender': None, 'confidence': {}}

        if self.age_net is None or self.gender_net is None:
            return result

        x, y, w, h = face_region
        face = image[y:y+h, x:x+w]

        if face.size == 0:
            return result

        blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), (78.4263377603, 87.7689143744, 114.895847746), swapRB=False)

        if self.gender_net is not None:
            self.gender_net.setInput(blob)
            gender_preds = self.gender_net.forward()
            genders = ['Male', 'Female']
            gender = genders[gender_preds[0].argmax()]
            confidence = float(gender_preds[0].max())
            result['gender'] = gender
            result['confidence']['gender'] = confidence

        if self.age_net is not None:
            self.age_net.setInput(blob)
            age_preds = self.age_net.forward()
            ages = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
            age = ages[age_preds[0].argmax()]
            result['age'] = age

        return result

    def get_face_encoding(self, image: np.ndarray, face_region: Tuple[int, int, int, int]) -> Optional[np.ndarray]:
        x, y, w, h = face_region
        face = image[y:y+h, x:x+w]

        if face.size == 0:
            return None

        face_resized = cv2.resize(face, (128, 128))
        gray = cv2.cvtColor(face_resized, cv2.COLOR_BGR2GRAY)

        encoding = gray.flatten()
        encoding = encoding / 255.0

        return encoding

    def compare_faces(self, encoding1: np.ndarray, encoding2: np.ndarray) -> float:
        if encoding1 is None or encoding2 is None:
            return 0.0

        distance = np.linalg.norm(encoding1 - encoding2)
        similarity = 1.0 / (1.0 + distance)

        return float(similarity)

    def process_image(self, image: np.ndarray, draw_results: bool = True, detect_features: bool = False) -> Dict[str, Any]:
        faces = self.detect_faces(image)
        result = {
            'face_count': len(faces),
            'faces': [],
            'has_face': len(faces) > 0
        }

        for (x, y, w, h) in faces:
            face_info = {
                'x': int(x),
                'y': int(y),
                'width': int(w),
                'height': int(h),
                'encoding': None
            }

            if detect_features:
                eyes = self.detect_eyes(image, (x, y, w, h))
                face_info['eyes'] = [{'x': ex, 'y': ey, 'width': ew, 'height': eh} for ex, ey, ew, eh in eyes]
                face_info['smiling'] = self.detect_smile(image, (x, y, w, h))

                age_gender = self.estimate_age_gender(image, (x, y, w, h))
                face_info.update(age_gender)

            encoding = self.get_face_encoding(image, (x, y, w, h))
            if encoding is not None:
                face_info['encoding'] = encoding.tolist()

            result['faces'].append(face_info)

            if draw_results:
                cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)

                if detect_features:
                    label = f"Face"
                    if face_info.get('gender'):
                        label += f" - {face_info['gender']}"
                    if face_info.get('age'):
                        label += f" - {face_info['age']}"

                    cv2.putText(image, label, (x, y-10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                for (ex, ey, ew, eh) in face_info.get('eyes', []):
                    cv2.rectangle(image, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)

        return result

    def count_faces(self, image: np.ndarray) -> int:
        return len(self.detect_faces(image))

    def has_face(self, image: np.ndarray) -> bool:
        return self.count_faces(image) > 0
