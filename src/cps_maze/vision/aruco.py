import cv2
from dataclasses import dataclass
import numpy as np
from packaging import version

@dataclass(frozen=True)
class ArucoDetection:
    found : bool
    corners : list|None = None
    ids : list | None = None
    rejected : list | None = None

class ArucoDetector:
    def __init__(self):

        if version.parse(cv2.__version__) >= version.parse("4.7.0"):
            self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_100)
            self.params = cv2.aruco.DetectorParameters()
            self.detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.params)
        else:
            self.aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_100)
            self.params = cv2.aruco.DetectorParameters_create()

        self.params.minMarkerPerimeterRate = 0.01
        self.params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX

    def detect(self, image: np.ndarray) -> ArucoDetection:
        # corners, ids, rejected = cv2.aruco.detectMarkers(image)
        if version.parse(cv2.__version__) >= version.parse("4.7.0"):
            corners, ids, rejected = self.detector.detectMarkers(image)
        else:
            corners, ids, rejected = cv2.aruco.detectMarkers(
                image, 
                self.aruco_dict, 
                parameters=self.params
            )
        
        # print(f"Detected: {len(corners) if corners else 0} markers, "
        #   f"Rejected: {len(rejected) if rejected else 0}")

        return ArucoDetection(
            found=(ids is not None and len(ids)>0),
            corners=corners,
            ids=ids,
            rejected=rejected
        )    
    @staticmethod
    def draw_detection(image_bgr: np.ndarray, detection: ArucoDetection) -> np.ndarray:
        output = image_bgr.copy()
        if detection.found and detection.ids is not None:
            for corners in detection.corners:
                pts = corners[0].astype(int)
                cv2.polylines(output, [pts], True, (0, 160, 160), 2)
            
            if detection.corners is not None:
                for i, corners in enumerate(detection.corners):
                    corner_x = int(corners[0][:, 0].mean())
                    corner_y = int(corners[0][:, 1].mean())
                    cv2.putText(output, str(detection.ids[i][0]), (corner_x, corner_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        return output
    




