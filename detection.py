import cv2
import numpy as np
import pytesseract
from edge_detection import CannyEdgeDetector
from hough import Hough


class Detection:
    def __init__(self, image_path):
        #pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
        self._image = cv2.imread(image_path)
        self._image = cv2.resize(self._image, (500, 500), interpolation=cv2.INTER_AREA)
        self._gray = cv2.cvtColor(self._image, cv2.COLOR_BGR2GRAY)
        self._image = self._gray / 255
        self._width, self._height = self._image.shape[:2]

    def canny_detection(self):
        detector = CannyEdgeDetector(self._image, sigma=3, kernel_size=5, lowthreshold=0.3, highthreshold=0.5,
                                     weak_pixel=100)
        self._image = detector.detect()
        self._image = self._image.astype(np.uint8)

    def hough(self):
        hough = Hough(self._image)
        return hough.line_detection_vectorized()

    def symbols(self, points):
        detected_text = []
        for contour in points:
            cv2.drawContours(self._image, [contour], -1, (0, 0, 255), 3)
            mask = np.zeros(self._gray.shape, np.uint8)
            cv2.drawContours(mask, [contour], 0, 255, -1, )
            cv2.bitwise_and(self._image, self._image, mask=mask)
            (x, y) = np.where(mask == 255)
            (top_x, top_y) = (np.min(x), np.min(y))
            (bottom_x, bottom_y) = (np.max(x), np.max(y))
            cropped = self._gray[top_x:bottom_x + 1, top_y:bottom_y + 1]
            detected_text.append(pytesseract.image_to_string(cropped, config='--psm 11'))
        return detected_text

    def detect(self):
        self.canny_detection()
        found = self.symbols(self.hough())
        found = [x.replace('\x0c', '').replace('\n', '') for x in found]
        cleared = []
        for x in found:
            counter = 0
            for symb in x:
                if symb.isalnum():
                    counter += 1
            if counter >= 4:
                cleared.append((x, counter))
        cleared = sorted(cleared, key=lambda x: 1 / x[1])
        for x in cleared:
            if abs(len(x[0]) - x[1]) < 2:
                return x[0]


if __name__ == "__main__":
    path = input("File path: ")
    d = Detection(path)
    detected = d.detect()
    print(detected)