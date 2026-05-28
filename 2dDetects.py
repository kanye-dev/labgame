import cv2
import math

cap = cv2.VideoCapture(0)

SHAPE_3D = {
    "Triangle":  "Cone / Pyramid?",
    "Square":    "Cube?",
    "Rectangle": "Cuboid / Cylinder?",
    "Pentagon":  "Pentagon",
    "Hexagon":   "Prism?",
    "Circle":    "Sphere / Cylinder?"
}

def detect_shape(contour):
    peri = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.05 * peri, True)
    vertices = len(approx)

    if vertices == 3:
        return "Triangle"

    elif vertices == 4:
        x, y, w, h = cv2.boundingRect(approx)
        aspect_ratio = w / float(h)
        return "Square" if 0.95 <= aspect_ratio <= 1.05 else "Rectangle"

    elif vertices == 5:
        return "Pentagon"

    elif vertices == 6:
        return "Hexagon"

    elif vertices > 6:
        area = cv2.contourArea(contour)
        _, radius = cv2.minEnclosingCircle(contour)
        circle_area = math.pi * radius ** 2
        if area / circle_area > 0.85:
            return "Circle"
        else:
            return "Unknown"

    return "Unknown"


while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.resize(frame, (640, 480))

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # binary threshold 
    _, thresh = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for c in contours:
        if cv2.contourArea(c) < 2000:
            continue

        shape = detect_shape(c)

        if shape == "Unknown":
            continue

        x, y, w, h = cv2.boundingRect(c)

        shape_3d = SHAPE_3D.get(shape, "")

        cv2.drawContours(frame, [c], -1, (0, 255, 0), 2)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 200, 255), 1)

        # 2D label
        cv2.putText(frame, shape, (x, y - 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # 3D inference label
        cv2.putText(frame, shape_3d, (x, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)

    cv2.imshow("2D Shape Detector", frame)
    cv2.imshow("Threshold", thresh)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()