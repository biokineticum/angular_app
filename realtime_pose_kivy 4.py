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
from kivy.metrics import dp

from tensorflow.lite.python.interpreter import Interpreter

JOINTS = {
    "Prawy łokieć": [6, 8, 10],
    "Lewy łokieć": [5, 7, 9],
    "Prawe kolano": [12, 14, 16],
    "Lewe kolano": [11, 13, 15],
}
ANGLE_THRESHOLD = 0.2


def get_angle(a, b, c):
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

        file_select_layout = BoxLayout(
            orientation='horizontal',
            size_hint_y=0.3,
            spacing=dp(10),
            padding=dp(10),
        )

        model_layout = BoxLayout(orientation='vertical', size_hint_x=0.5, spacing=dp(5))
        model_layout.add_widget(Label(text='Model', size_hint_y=None, height=dp(24)))
        self.modelchooser = FileChooserIconView(filters=["*.tflite"])
        model_layout.add_widget(self.modelchooser)

        video_layout = BoxLayout(orientation='vertical', size_hint_x=0.5, spacing=dp(5))
        video_layout.add_widget(Label(text='Wideo', size_hint_y=None, height=dp(24)))
        self.videochooser = FileChooserIconView(filters=["*.mp4"])
        video_layout.add_widget(self.videochooser)

        file_select_layout.add_widget(model_layout)
        file_select_layout.add_widget(video_layout)

        control_layout = BoxLayout(
            size_hint_y=0.1,
            spacing=dp(10),
            padding=dp(10),
        )
        self.joint_spinner = Spinner(
            text="Prawy łokieć",
            values=list(JOINTS.keys()),
            size_hint_x=0.4,
            font_size='16sp',
        )
        self.btn = Button(text='Analizuj', size_hint_x=0.6, font_size='16sp')
        control_layout.add_widget(self.joint_spinner)
        control_layout.add_widget(self.btn)

        display_layout = BoxLayout(
            orientation='horizontal',
            size_hint_y=0.55,
            spacing=dp(10),
            padding=dp(10),
        )
        video_box = BoxLayout(orientation='vertical', size_hint_x=0.5, spacing=dp(5))
        video_box.add_widget(Label(text='Podgląd wideo', size_hint_y=None, height=dp(24)))
        self.video_img = Image()
        video_box.add_widget(self.video_img)

        plot_box = BoxLayout(orientation='vertical', size_hint_x=0.5, spacing=dp(5))
        plot_box.add_widget(Label(text='Wykres kąta', size_hint_y=None, height=dp(24)))
        self.plot_img = Image()
        plot_box.add_widget(self.plot_img)

        display_layout.add_widget(video_box)
        display_layout.add_widget(plot_box)

        self.status = Label(
            text="Wybierz model i wideo.",
            size_hint_y=0.05,
            halign='center',
            valign='middle',
            font_size='16sp',
        )
        self.status.bind(size=self.status.setter('text_size'))

        self.add_widget(file_select_layout)
        self.add_widget(control_layout)
        self.add_widget(display_layout)
        self.add_widget(self.status)

        self.btn.bind(on_press=self.start_analysis)
        self.joint_idx = JOINTS[self.joint_spinner.text]
        self.joint_spinner.bind(text=self.on_joint_select)

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
        angles = []
        frame_num = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            keypoints = detect_pose(interpreter, input_details, output_details, frame)
            pts = [(int(x * frame.shape[1]), int(y * frame.shape[0])) for x, y, c in keypoints]
            confs = [c for x, y, c in keypoints]

            if min([confs[i] for i in self.joint_idx]) > ANGLE_THRESHOLD:
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
        plt.figure(figsize=(4, 3))
        plt.plot(angles, color='cyan')
        plt.xlabel('Klatka')
        plt.ylabel('Kąt [deg]')
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
