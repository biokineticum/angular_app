import cv2
import numpy as np
import matplotlib.pyplot as plt
import threading
from io import BytesIO

from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.image import Image
from kivy.uix.filechooser import FileChooserIconView
from kivy.uix.label import Label
from kivy.uix.spinner import Spinner
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.core.image import Image as CoreImage

from tensorflow.lite.python.interpreter import Interpreter

JOINTS = {
    "Prawy łokieć": [6, 8, 10],
    "Lewy łokieć": [5, 7, 9],
    "Prawe kolano": [12, 14, 16],
    "Lewe kolano": [11, 13, 15],
}
CONFIDENCE_THRESHOLD = 0.2


def get_angle(a, b, c):
    """Return the angle between points ``a``, ``b`` and ``c`` in degrees.

    Parameters
    ----------
    a, b, c : array-like
        Dwuwymiarowe współrzędne punktów tworzących ramiona kąta przy wierzchołku
        ``b``.

    Returns
    -------
    float
        Wartość kąta w stopniach obliczona na podstawie iloczynu skalarnego.
    """

    ab = np.array(a) - np.array(b)
    cb = np.array(c) - np.array(b)
    cosine = np.dot(ab, cb) / (np.linalg.norm(ab) * np.linalg.norm(cb))
    return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))


def detect_pose(interpreter, input_details, output_details, frame):
    img = cv2.resize(frame, (192, 192))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    interpreter.set_tensor(input_details[0]['index'], [img.astype(np.uint8)])
    interpreter.invoke()
    keypoints = interpreter.get_tensor(output_details[0]['index'])[0, 0]
    return keypoints


class PoseApp(BoxLayout):
    def __init__(self, **kwargs):
        super().__init__(orientation='vertical', **kwargs)

        file_select_layout = BoxLayout(orientation='horizontal', size_hint_y=0.25)
        self.modelchooser = FileChooserIconView(filters=["*.tflite"], size_hint_x=0.5)
        self.videochooser = FileChooserIconView(filters=["*.mp4"], size_hint_x=0.5)
        file_select_layout.add_widget(self.modelchooser)
        file_select_layout.add_widget(self.videochooser)

        control_layout = BoxLayout(size_hint_y=0.07)
        self.joint_spinner = Spinner(text="Prawy łokieć", values=list(JOINTS.keys()), size_hint_x=0.4)
        self.btn = Button(text='Analizuj', size_hint_x=0.6)
        control_layout.add_widget(self.joint_spinner)
        control_layout.add_widget(self.btn)

        display_layout = BoxLayout(orientation='horizontal', size_hint_y=0.6)
        self.video_img = Image()
        self.plot_img = Image()
        display_layout.add_widget(self.video_img)
        display_layout.add_widget(self.plot_img)

        self.status = Label(text="Wybierz model i wideo.", size_hint_y=0.08)

        self.add_widget(file_select_layout)
        self.add_widget(control_layout)
        self.add_widget(display_layout)
        self.add_widget(self.status)

        self.btn.bind(on_press=self.start_analysis)
        self.joint_idx = JOINTS[self.joint_spinner.text]
        self.joint_spinner.bind(text=self.on_joint_select)
        self.fps = 1

    def on_joint_select(self, spinner, text):
        self.joint_idx = JOINTS[text]

    def start_analysis(self, instance):
        model = self.modelchooser.selection[0]
        video = self.videochooser.selection[0]

        threading.Thread(target=self.analyze_video, args=(model, video), daemon=True).start()

    def interpolate_angles(self, angles):
        angles = np.array(angles)
        nans = np.isnan(angles)
        not_nans = ~nans
        angles[nans] = np.interp(np.flatnonzero(nans), np.flatnonzero(not_nans), angles[not_nans])
        return angles

    def analyze_video(self, model_path, video_path):
        interpreter = Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        cap = cv2.VideoCapture(video_path)
        self.fps = cap.get(cv2.CAP_PROP_FPS) or 1
        angles = []
        frame_num = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            keypoints = detect_pose(interpreter, input_details, output_details, frame)
            pts = [(int(x * frame.shape[1]), int(y * frame.shape[0])) for x, y, c in keypoints]
            confs = [c for x, y, c in keypoints]

            if min([confs[i] for i in self.joint_idx]) > CONFIDENCE_THRESHOLD:
                angle = get_angle(*[pts[i] for i in self.joint_idx])
            else:
                angle = np.nan

            angles.append(angle)

            if frame_num % 3 == 0:
                Clock.schedule_once(lambda dt, f=frame.copy(): self.update_video_frame(f))

            if frame_num % 15 == 0:
                Clock.schedule_once(lambda dt, ang=np.array(angles): self.update_plot(ang))
                Clock.schedule_once(lambda dt, f=frame_num: self.update_status(f"Klatka: {f}"))

            frame_num += 1

        cap.release()
        Clock.schedule_once(lambda dt: self.update_status("Analiza zakończona"))

    def update_video_frame(self, frame):
        small_frame = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), (frame.shape[1] // 3, frame.shape[0] // 3))
        tex = Texture.create(size=(small_frame.shape[1], small_frame.shape[0]), colorfmt='rgb')
        tex.blit_buffer(small_frame.tobytes(), colorfmt='rgb', bufferfmt='ubyte')
        tex.flip_vertical()
        self.video_img.texture = tex

    def update_plot(self, angles):
        angles = self.interpolate_angles(angles)
        if len(angles) < 2:
            angular_velocity = np.array([])
        else:
            dt = 1 / self.fps
            angular_velocity = np.gradient(angles, dt)
        plt.figure(figsize=(4, 3))
        plt.plot(angular_velocity, color='cyan')
        plt.xlabel('Klatka')
        plt.ylabel('Prędkość kątowa [deg/s]')
        plt.grid(True)
        plt.tight_layout()

        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=100)
        buf.seek(0)
        plt.close()

        self.plot_img.texture = CoreImage(buf, ext='png').texture

    def update_status(self, txt):
        self.status.text = txt


class PoseAppRealtime(App):
    def build(self):
        return PoseApp()


if __name__ == '__main__':
    PoseAppRealtime().run()
