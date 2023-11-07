import cv2
import numpy as np


def draw_counts(frame, classes, class_counts):
    i = 0
    for class_id, count in class_counts.items():
        cv2.putText(
            frame,
            classes[class_id] + " " + str(count),
            (0, i * 10 + 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.3,
            (255, 0, 0),
            1
        )
        i += 1


def apply_nms(detections, score_threshold=0.5, nms_threshold=0.6):
    boxes = [(x, y, x + w, y + h) for _, _, x, y, w, h in detections]
    scores = [score for _, score, *_ in detections]
    indices = cv2.dnn.NMSBoxes(boxes, scores, score_threshold=score_threshold, nms_threshold=nms_threshold)

    return [detections[i] for i in indices]


def run_yolo(video_path='video.mp4'):
    cap = cv2.VideoCapture(video_path)
    net = cv2.dnn.readNet("yolov4.cfg", "yolov4.weights")
    with open('coco.names', 'r') as f:
        classes = [line.strip() for line in f.readlines()]
    output_layers = net.getUnconnectedOutLayersNames()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        height, width, channels = frame.shape
        blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        outs = net.forward(output_layers)
        detections = list()
        class_counts = dict()

        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                detections.append((class_id, confidence, w, h, x, y))

        filtered_detections = apply_nms(detections, 0.4, 0.85)

        for class_id, confidence, w, h, x, y in filtered_detections:
            class_counts[class_id] = class_counts.get(class_id, 0) + 1
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(
                frame,
                classes[class_id],
                (x, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 0, 0),
                2
            )

        draw_counts(frame, classes, class_counts)

        cv2.imshow("Video", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
