import cv2
import numpy as np
import mediapipe as mp


class PostureCorrector:
    def __init__(self):
        # инициализация mediapipe
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.pose = self.mp_pose.Pose(
            static_image_mode = False,
            min_detection_confidence = 0.5,
            min_tracking_confidence = 0.5
        )

        # параметры калибровки и пороги для метрик осанки

        self.is_calibrated = False
        self.calibration_frames = 0
        self.max_calibration_frames = 30

        self.calibration_neck = []
        self.calibration_forward = []
        self.calibration_lateral = []

        self.neck_threshold = 0
        self.forward_threshold = 0
        self.lateral_threshold = 0


    # функции для вычисления положений и углов головы/шеи

    def neck_compression_ratio(self, ear, shoulder, hip):
        neck_height = abs(ear[1] - shoulder[1])
        torso_height = abs(shoulder[1] - hip[1]) + 1e-6
        return neck_height / torso_height

    def forward_head_ratio(self, ear, shoulder, hip):
        horizontal_offset = abs(ear[0] - shoulder[0])
        torso_height = abs(shoulder[1] - hip[1]) + 1e-6
        return horizontal_offset / torso_height

    def lateral_head_tilt(self, left_ear, right_ear, shoulder, hip):
        ear_vertical_diff = abs(left_ear[1] - right_ear[1])
        torso_height = abs(shoulder[1] - hip[1]) + 1e-6
        return ear_vertical_diff / torso_height

    def torso_angle(self, shoulder, hip):
        dx = hip[0] - shoulder[0]
        dy = hip[1] - shoulder[1]
        angle = np.degrees(np.arctan2(dx, dy))
        return abs(angle)

    def midpoint(self, a, b):
        x = (a[0] + b[0]) // 2
        y = (a[1] + b[1]) // 2
        return (x, y)

    # рисуем скелет

    def draw_line(self, frame, point_a, point_b, color):
        cv2.line(frame, point_a, point_b, color, 2)
        cv2.circle(frame, point_a, 4, color, -1)
        cv2.circle(frame, point_b, 4, color, -1)

    # калибруем осанку

    def calibrate(self, neck, forward, lateral, frame):
        if self.calibration_frames < self.max_calibration_frames:
            self.calibration_neck.append(neck)
            self.calibration_forward.append(forward)
            self.calibration_lateral.append(lateral)
            self.calibration_frames += 1

            cv2.putText(
                frame,
                f"Calibrating... {self.calibration_frames}/{self.max_calibration_frames}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 255),
                2
            )
        else:
            # выбор значений:
            # 0.9 - допустиме уменьшение длины шеи на 10% (если больше => сильно втягивается шея)
            # 1.15 - голова выдвинута вперед на 15% (учитываем естественное положение головы, но больше 15% - ухудшение осанки)
            # 1.5 - наклон вправо/влево на 50% (наклон в пределах нормы)

            self.neck_threshold = np.mean(self.calibration_neck) * 0.9
            self.forward_threshold = np.mean(self.calibration_forward) * 1.15
            self.lateral_threshold = np.mean(self.calibration_lateral) * 1.5
            self.is_calibrated = True
            print("Calibration complete")

    # даем ответ - плохая или хорошая осанка

    def posture_feedback(self, neck, forward, lateral, frame):
        bad_neck = neck < self.neck_threshold
        bad_forward = forward > self.forward_threshold
        bad_lateral = lateral > self.lateral_threshold

        if bad_neck or bad_forward or bad_lateral:
            status = "POOR POSTURE"
            color = (0, 0, 255)
        else:
            status = "GOOD POSTURE"
            color = (0, 255, 0)

        cv2.putText(
            frame,
            status,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            color,
            2
        )

    # основной цикл

    def run(self):
        cap = cv2.VideoCapture(0)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                continue

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.pose.process(rgb)

            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                h, w, _ = frame.shape

                left_ear = (
                    int(landmarks[self.mp_pose.PoseLandmark.LEFT_EAR].x * w),
                    int(landmarks[self.mp_pose.PoseLandmark.LEFT_EAR].y * h)
                )
                right_ear = (
                    int(landmarks[self.mp_pose.PoseLandmark.RIGHT_EAR].x * w),
                    int(landmarks[self.mp_pose.PoseLandmark.RIGHT_EAR].y * h)
                )

                left_shoulder = (
                    int(landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER].x * w),
                    int(landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER].y * h)
                )
                right_shoulder = (
                    int(landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER].x * w),
                    int(landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER].y * h)
                )

                left_hip = (
                    int(landmarks[self.mp_pose.PoseLandmark.LEFT_HIP].x * w),
                    int(landmarks[self.mp_pose.PoseLandmark.LEFT_HIP].y * h)
                )
                right_hip = (
                    int(landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP].x * w),
                    int(landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP].y * h)
                )

                mid_ear = self.midpoint(left_ear, right_ear)
                mid_shoulder = self.midpoint(left_shoulder, right_shoulder)
                mid_hip = self.midpoint(left_hip, right_hip)

                # 12 градусов - угол наклона (между плечами и бедрами). Если больше 12, то пользователь сильно повернулся вбок

                if self.torso_angle(mid_shoulder, mid_hip) > 12:
                    cv2.putText(
                        frame,
                        "Align body with camera",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 255),
                        2
                    )
                else:
                    neck = self.neck_compression_ratio(mid_ear, mid_shoulder, mid_hip)
                    forward = self.forward_head_ratio(mid_ear, mid_shoulder, mid_hip)
                    lateral = self.lateral_head_tilt(left_ear, right_ear, mid_shoulder, mid_hip)

                    self.mp_drawing.draw_landmarks(
                        frame,
                        results.pose_landmarks,
                        self.mp_pose.POSE_CONNECTIONS
                    )

                    self.draw_line(frame, mid_ear, mid_shoulder, (255, 0, 0))
                    self.draw_line(frame, mid_shoulder, mid_hip, (0, 255, 0))

                    if not self.is_calibrated:
                        self.calibrate(neck, forward, lateral, frame)
                    else:
                        self.posture_feedback(neck, forward, lateral, frame)

            cv2.imshow("Posture Corrector", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    PostureCorrector().run()
