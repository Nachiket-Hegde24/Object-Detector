import argparse
import os
import time
import cv2
from flask import Flask, render_template, request, redirect, send_file, url_for, Response, send_from_directory
from werkzeug.utils import secure_filename
from ultralytics import YOLO

app = Flask(__name__, template_folder='template', static_folder='static')

if not os.path.exists('uploads'):
    os.makedirs('uploads')


@app.route('/')
def hello_world():
    return render_template('index.html')


@app.route("/upload.html")
def upload_page():
    return render_template('upload.html')


@app.route("/webcam.html")
def webcam_page():
    return render_template('webcam.html')


@app.route("/predict_img", methods=["POST"])
def predict_img():
    if 'file' not in request.files:
        return redirect(request.url)

    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)

    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join('uploads', filename)
        file.save(filepath)

        file_extension = filename.rsplit('.', 1)[1].lower()

        if file_extension in ['jpg', 'jpeg']:
            img = cv2.imread(filepath)
            img = cv2.resize(img, (640, 480))  # Resize image to reduce processing time
            model = YOLO('yolov9c.pt')
            detections = model(img, save=True)
            return display_image(filename)

        elif file_extension == 'mp4':
            video_path = filepath
            cap = cv2.VideoCapture(video_path)

            frame_width = 640
            frame_height = 480

            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            output_path = os.path.join('uploads', 'output.mp4')
            out = cv2.VideoWriter(output_path, fourcc, 30.0, (frame_width, frame_height))

            model = YOLO('yolov9c.pt')

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frame = cv2.resize(frame, (640, 480))  # Resize frame to reduce processing time
                results = model(frame, save=True)
                res_plotted = results[0].plot()

                out.write(res_plotted)

                if cv2.waitKey(1) == ord('q'):
                    break

            cap.release()
            out.release()

            return redirect(url_for('video_feed'))

    return render_template('upload.html')


@app.route('/display_image/<filename>')
def display_image(filename):
    folder_path = 'runs/detect'
    subfolders = [f for f in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, f))]
    latest_subfolder = max(subfolders, key=lambda x: os.path.getctime(os.path.join(folder_path, x)))
    directory = folder_path + '/' + latest_subfolder
    files = os.listdir(directory)
    latest_file = files[0]

    file_extension = latest_file.rsplit('.', 1)[1].lower()

    if file_extension in ['jpg', 'jpeg']:
        return send_from_directory(directory, latest_file)
    else:
        return "Invalid file format"


# Route to stream the processed video
@app.route('/video_feed')
def video_feed():
    print("function called")
    return Response(get_video_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


def get_video_frames():
    video_path = os.path.join('uploads', 'output.mp4')
    video = cv2.VideoCapture(video_path)  # detected video path
    while True:
        success, image = video.read()
        if not success:
            break
        ret, jpeg = cv2.imencode('.jpg', image)

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')
        time.sleep(0.05)  # Reduce delay to display more frames per second


cap = None
stop_webcam = False


@app.route('/webcam_feed')
def webcam_feed():
    global cap, stop_webcam
    stop_webcam = False
    if cap is None:
        cap = cv2.VideoCapture(0)
    return Response(generate_webcam_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


def generate_webcam_frames():
    global cap, stop_webcam
    model = YOLO('yolov9c.pt')

    while True:
        if stop_webcam:
            if cap is not None:
                cap.release()  # Release webcam
                cap = None  # Reset capture object
            break  # Exit the loop

        success, frame = cap.read()
        if not success:
            break

        results = model(frame, save=True)
        res_plotted = results[0].plot()

        ret, jpeg = cv2.imencode('.jpg', res_plotted)
        frame = jpeg.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

        time.sleep(0.1)

    cap.release()
    cap = None


@app.route('/turn_off_webcam', methods=['POST'])
def turn_off_webcam():
    global cap, stop_webcam
    stop_webcam = True  # Stop the loop in `generate_webcam_frames`

    if cap is not None:
        cap.release()  # Release the camera
        cap = None  # Reset capture object

    return '', 204  # Return success


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flask app exposing YOLO models")
    parser.add_argument("--port", default=5000, type=int, help="port number")
    args = parser.parse_args()
    model = YOLO('yolov9c.pt')
    app.run(host="0.0.0.0", port=args.port, debug=True)
