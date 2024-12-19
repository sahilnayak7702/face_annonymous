import cv2
import argparse
import os
import mediapipe as mp

def process_img(img, face_detection):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    out = face_detection.process(img_rgb)
    H, W, _ = img.shape
    if out.detections is not None:
        for detection in out.detections:
            location_data = detection.location_data
            bbox = location_data.relative_bounding_box
            x1, y1, w, h = bbox.xmin, bbox.ymin, bbox.width, bbox.height

            x1 = int(x1*W)
            y1 = int(y1*H)
            w = int(w*W)
            h = int(h*H)
            # blur face
            img[y1:y1+h, x1:x1+w,:] = cv2.blur(img[y1:y1+h, x1:x1+w,:],(150,150))
    return img

args = argparse.ArgumentParser()

args.add_argument("--mode", default='Webcam')
args.add_argument("--filepath", default = None)
args = args.parse_args()

# detect faces
mp_face_detection = mp.solutions.face_detection

with mp_face_detection.FaceDetection(model_selection = 0, min_detection_confidence =0.5) as face_detection:
    if args.mode in ["image"]:
        image = cv2.imread(r"C:\Users\91760\Football_analysis\OpenCV_learnings\face_annonymous\img.jpeg")
        img = cv2.resize(image, (512, 650))
        img = process_img(img, face_detection)
        # save image
        cv2.imwrite(os.path.join(r'C:\Users\91760\Football_analysis\OpenCV_learnings\face_annonymous', 'output.jpeg' ), img)

    elif args.mode in ["Video"]:
        cap = cv2.VideoCapture(r"C:\Users\91760\Football_analysis\OpenCV_learnings\face_annonymous\video.mp4")
        ret, frame = cap.read()
        output_video = cv2.VideoWriter(os.path.join(r'C:\Users\91760\Football_analysis\OpenCV_learnings\face_annonymous' ,'output.mp4'), 
                                       cv2.VideoWriter_fourcc(*'MPV4'), 
                                       25, 
                                       (frame.shape[1], frame.shape[0]))
        while ret:
            img = process_img(frame, face_detection)
            output_video.write(frame)
            ret, frame = cap.read()

            

        cap.release()
        output_video.release()

    elif args.mode in ["Webcam"]:
        cap = cv2.VideoCapture(0)
        ret, frame = cap.read()
        while ret:
            img = process_img(frame, face_detection)
            cv2.imshow('frame', frame)
            ret, frame = cap.read()
            cv2.waitKey(30)
        cap.release()

