import sys, json, time, urllib.request, webbrowser, os, subprocess, threading
from collections import deque
from pathlib import Path
import cv2, mediapipe as mp
cv2.setNumThreads(2)
import numpy as np
import pyautogui
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QImage, QPixmap, QAction, QKeySequence
from PySide6.QtWidgets import *

ROOT = Path(__file__).resolve().parent
MODEL = ROOT / "assets" / "hand_landmarker.task"
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
FACE_MODEL = ROOT / "assets" / "face_detector.tflite"
FACE_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite"
CONFIG = ROOT / "config.json"
GESTURES = {
    "THUMB_UP": "Великий палець угору", "PINKY": "Мізинець", "ONE": "Один палець",
    "PEACE": "Два пальці", "THREE": "Три пальці", "FOUR": "Чотири пальці",
    "OPEN_PALM": "Відкрита долоня", "FIST": "Кулак"
}
HANDS = {"LEFT": "Ліва рука", "RIGHT": "Права рука", "BOTH": "Обидві руки"}
DEFAULT = {
    "camera_index": 0, "gesture_hold_seconds": 0.35, "cooldown_seconds": 1.2,
    "night_vision": False, "gamma": 1.5, "brightness": 12, "contrast": 1.25,
    "camera_resolution": "1280x720", "mirror_camera": True, "face_presence_gate": False, "face_tracking_enabled": True, "hand_overlay_enabled": True, "virtual_camera_enabled": True, "inference_width": 640, "hand_overlay_hold_seconds": 0.5, "bindings": {}, "scripts": []
}

def load_config():
    if not CONFIG.exists():
        CONFIG.write_text(json.dumps(DEFAULT, indent=2, ensure_ascii=False), encoding="utf-8")
    try:
        data = json.loads(CONFIG.read_text(encoding="utf-8"))
        for k, v in DEFAULT.items(): data.setdefault(k, v)
        if not isinstance(data.get("bindings"), dict): data["bindings"] = {}
        if not isinstance(data.get("scripts"), list): data["scripts"] = []
        try: data["hand_overlay_hold_seconds"] = max(0.0, min(5.0, float(data.get("hand_overlay_hold_seconds", 0.5))))
        except (TypeError, ValueError): data["hand_overlay_hold_seconds"] = 0.5
        for binding in data["bindings"].values():
            if isinstance(binding, dict):
                try: binding["hold_seconds"] = max(0.05, min(5.0, float(binding.get("hold_seconds", 0.35))))
                except (TypeError, ValueError): binding["hold_seconds"] = 0.35
        return data
    except Exception:
        return DEFAULT.copy()

def _download_model(path: Path, url: str, min_size: int) -> Path:
    """Ensure a complete MediaPipe model exists; download it if missing/incomplete."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size >= min_size:
        return path

    # Download to a temporary file so an interrupted download cannot leave a
    # broken model that the next launch mistakenly treats as valid.
    temp_path = path.with_name(path.name + ".download")
    try:
        if temp_path.exists():
            temp_path.unlink()
        request = urllib.request.Request(
            url, headers={"User-Agent": "Mozilla/5.0"}
        )
        with urllib.request.urlopen(request, timeout=60) as response:
            data = response.read()
        if len(data) < min_size:
            raise RuntimeError(
                f"Завантажена модель занадто мала ({len(data)} байт). "
                "Перевір інтернет-з'єднання або доступ до storage.googleapis.com."
            )
        temp_path.write_bytes(data)
        temp_path.replace(path)
    except Exception as e:
        try:
            if temp_path.exists():
                temp_path.unlink()
        except OSError:
            pass
        raise RuntimeError(
            f"Не вдалося завантажити модель {path.name}. "
            f"Перевір інтернет-з'єднання. Деталі: {e}"
        ) from e
    return path

def get_model():
    return _download_model(MODEL, MODEL_URL, 100_000)

def get_face_model():
    return _download_model(FACE_MODEL, FACE_MODEL_URL, 50_000)

def _dist3(a, b):
    return ((a.x-b.x)**2 + (a.y-b.y)**2 + (a.z-b.z)**2) ** 0.5

def finger_up(lm, tip, pip):
    # Визначаємо розгинання відносно зап'ястя, а не вертикальної осі кадру.
    # Це працює і коли долоня повернута тильною стороною або нахилена.
    wrist = lm[0]
    return _dist3(wrist, lm[tip]) > _dist3(wrist, lm[pip]) * 1.13

def thumb_up(lm):
    # Визначаємо відведення великого пальця від долоні; працює для обох рук.
    tip, ip, index_mcp = lm[4], lm[3], lm[5]
    d_tip = ((tip.x-index_mcp.x)**2 + (tip.y-index_mcp.y)**2) ** 0.5
    d_ip = ((ip.x-index_mcp.x)**2 + (ip.y-index_mcp.y)**2) ** 0.5
    return d_tip > d_ip + 0.045

def detect_gesture(lm):
    t = thumb_up(lm)
    i, m, r, p = (finger_up(lm,8,6), finger_up(lm,12,10),
                  finger_up(lm,16,14), finger_up(lm,20,18))
    count = sum((i,m,r,p))
    if count == 4: return "OPEN_PALM"
    if t and count == 0: return "THUMB_UP"
    if p and not i and not m and not r: return "PINKY"
    if count == 1: return "ONE"
    if count == 2: return "PEACE"
    if count == 3: return "THREE"
    if count == 0: return "FIST"
    return "UNKNOWN"

LINKS = [(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),(5,9),(9,10),(10,11),(11,12),(9,13),(13,14),(14,15),(15,16),(13,17),(17,18),(18,19),(19,20),(0,17)]

SUPPORTED_SCRIPT_EXTENSIONS = {".ahk", ".exe", ".bat", ".cmd"}

class KeyCaptureEdit(QLineEdit):
    """Captures up to three key presses/chord parts instead of requiring manual key names."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self._captured = []
        self._capture_enabled = False
        self._before_capture = ""
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(850)
        self._timer.timeout.connect(self._finish_capture)
        self.setPlaceholderText("Натисни тут до 3 клавіш. Backspace очищає; Ctrl+Backspace призначає Backspace.")

    def set_capture_enabled(self, enabled):
        self._capture_enabled = bool(enabled)
        if not self._capture_enabled:
            self._timer.stop(); self._captured.clear()

    def focusInEvent(self, event):
        self._before_capture = self.text()
        super().focusInEvent(event)

    def keyPressEvent(self, event):
        if not self._capture_enabled:
            super().keyPressEvent(event)
            return
        key = event.key()
        modifiers = event.modifiers()
        # Backspace/Delete clear the field by default. Ctrl+Backspace explicitly
        # binds Backspace; Shift+Backspace and Alt+Backspace remain bindable too.
        if key == Qt.Key_Backspace and modifiers == Qt.NoModifier:
            self._timer.stop(); self._captured.clear(); self.clear(); event.accept(); return
        if key == Qt.Key_Delete and modifiers == Qt.NoModifier:
            self._timer.stop(); self._captured.clear(); self.clear(); event.accept(); return
        if key == Qt.Key_Escape:
            self._timer.stop(); self._captured.clear(); self.setText(self._before_capture); event.accept(); return
        if key in (Qt.Key_Return, Qt.Key_Enter, Qt.Key_Tab):
            if self._captured: self._finish_capture()
            else: super().keyPressEvent(event)
            return
        names = []
        if modifiers & Qt.ControlModifier: names.append("ctrl")
        if modifiers & Qt.AltModifier: names.append("alt")
        if modifiers & Qt.ShiftModifier: names.append("shift")
        if modifiers & Qt.MetaModifier: names.append("win")
        if key not in (Qt.Key_Control, Qt.Key_Alt, Qt.Key_Shift, Qt.Key_Meta):
            name = QKeySequence(key).toString().lower()
            aliases = {" ": "space", "esc": "esc", "return": "enter", "del": "delete", "backspace": "backspace", "pgup": "pageup", "pgdown": "pagedown"}
            name = aliases.get(name, name)
            if name and name not in names: names.append(name)
        for name in names:
            if name not in self._captured and len(self._captured) < 3: self._captured.append(name)
        if self._captured:
            self.setText("+".join(self._captured)); self._timer.start()
        event.accept()

    def _finish_capture(self):
        self._timer.stop()
        self._captured = []
        self._before_capture = self.text()

class ScriptDropList(QListWidget):
    def __init__(self, add_paths, parent=None):
        super().__init__(parent)
        self.add_paths = add_paths
        self.setAcceptDrops(True)
        self.setDragDropMode(QAbstractItemView.DropOnly)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls(): event.acceptProposedAction()
        else: event.ignore()

    def dragMoveEvent(self, event):
        if event.mimeData().hasUrls(): event.acceptProposedAction()
        else: event.ignore()

    def dropEvent(self, event):
        paths = [u.toLocalFile() for u in event.mimeData().urls() if u.isLocalFile()]
        self.add_paths(paths)
        event.acceptProposedAction()


class NoWheelComboBox(QComboBox):
    """Не змінює вибір від коліщатка, коли список не відкритий."""
    def wheelEvent(self, event):
        if self.view().isVisible():
            super().wheelEvent(event)
        else:
            event.ignore()


class Main(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Streamer Camera Control")
        self.resize(1020, 740)
        self.setMinimumSize(780, 560)
        self.cfg = load_config(); self.cap = None; self.tracker = None; self.virtual_cam = None; self.virtual_camera_size = None; self.virtual_frames_sent = 0
        self.active = False; self._shutdown_thread = None; self.current_key = "—"; self.since = 0; self.last_actions = {}; self.face_detector = None; self.last_timestamp_ms = 0
        self.allow_quit = False; self.binding_widgets = {}; self.gesture_since = {}; self.fired_gestures = set(); self.face_history = deque(maxlen=3); self.last_hand_overlay = None
        self.build(); self.apply_language(); self.setup_tray()
        self.timer = QTimer(self); self.timer.timeout.connect(self.frame)
        self.autosave_timer = QTimer(self); self.autosave_timer.setSingleShot(True); self.autosave_timer.setInterval(350); self.autosave_timer.timeout.connect(self.save_config_silent)
        self.connect_autosave_signals()

    def build(self):
        self.setStyleSheet("""
        QWidget,QMainWindow{background:#242629;color:#eeeeee;font-family:'Segoe UI';font-size:13px}
        QFrame#panel{background:#2d3033;border:1px solid #454a4e;border-radius:13px}
        QLabel#title{color:#d0d5d8;font-size:25px;font-weight:700}
        QLabel#metric{color:#d0d5d8;font-size:23px;font-weight:700}
        QPushButton{background:#3a3e42;border:1px solid #62686d;border-radius:9px;padding:9px 13px;color:#f0f0f0}
        QPushButton:hover{background:#4a5055;border:1px solid #858d93}
        QComboBox,QLineEdit{background:#202326;border:1px solid #50565b;border-radius:7px;padding:7px;color:#eeeeee;min-height:22px}
        QSlider::groove:horizontal{height:6px;background:#4a4f53;border-radius:3px}
        QSlider::sub-page:horizontal{background:#9ba3a8;border-radius:3px}
        QSlider::handle:horizontal{background:#d7dcdf;width:14px;margin:-5px 0;border-radius:7px}
        QCheckBox{spacing:9px;color:#eeeeee}
        QCheckBox::indicator{width:17px;height:17px}
        QCheckBox::indicator:unchecked{background:#303438;border:1px solid #686f74;border-radius:4px}
        QCheckBox::indicator:checked{background:#777f84;border:1px solid #c0c6ca;border-radius:4px}
        QScrollArea{border:0;background:transparent}
        """)
        root=QWidget(); outer=QVBoxLayout(root); outer.setContentsMargins(20,16,20,16)
        head=QHBoxLayout(); titlebox=QVBoxLayout(); t=QLabel("КЕРУВАННЯ ЖЕСТАМИ"); t.setObjectName("title")
        titlebox.addWidget(t); head.addLayout(titlebox); head.addStretch()
        self.status=QLabel("● НЕАКТИВНО"); self.status.setStyleSheet("color:#ff8094;font-weight:700"); head.addWidget(self.status); outer.addLayout(head)
        body=QHBoxLayout(); navpanel=QFrame(); navpanel.setObjectName("panel"); navpanel.setFixedWidth(150); nav=QVBoxLayout(navpanel)
        nav.addWidget(QLabel("РОЗДІЛИ")); self.nav=QListWidget(); self.nav.addItems(["Головна","Налаштування","Прив’язки команд","Мої скрипти"])
        nav.addWidget(self.nav); nav.addStretch(); body.addWidget(navpanel)
        self.pages=QStackedWidget(); self.pages.addWidget(self.dashboard()); self.pages.addWidget(self.settings()); self.pages.addWidget(self.bindings_page()); self.pages.addWidget(self.scripts_page())
        self.nav.currentRowChanged.connect(self.pages_index); self.nav.setCurrentRow(0)
        body.addWidget(self.pages,1); outer.addLayout(body,1)
        self.setCentralWidget(root)

    def pages_index(self,index): self.pages.setCurrentIndex(index)
    def panel(self,name):
        f=QFrame(); f.setObjectName("panel"); l=QVBoxLayout(f); h=QLabel(name); h.setFixedHeight(20); h.setStyleSheet("color:#d0d5d8;font-size:11px;font-weight:700;letter-spacing:1px"); l.addWidget(h); return f,l

    def dashboard(self):
        w=QWidget(); l=QVBoxLayout(w); row=QHBoxLayout(); f,fl=self.panel("ЗОБРАЖЕННЯ З КАМЕРИ")
        self.preview=QLabel("КАМЕРУ НЕ ЗАПУЩЕНО\n\nНатисни «Увімкнути камеру»"); self.preview.setAlignment(Qt.AlignCenter); self.preview.setMinimumSize(400,260); self.preview.setSizePolicy(QSizePolicy.Expanding,QSizePolicy.Expanding)
        self.preview.setStyleSheet("background:#1c1e20;border:1px dashed #62686d;border-radius:9px;color:#b9c0c5"); fl.addWidget(self.preview)
        row.addWidget(f,3)
        right=QVBoxLayout(); right.setSpacing(8)
        self.start=QPushButton("▶  УВІМКНУТИ КАМЕРУ"); self.start.clicked.connect(self.toggle); right.addWidget(self.start)

        # Праву колонку прокручуємо окремо: довгі назви та дані для двох рук
        # більше не стискають сусідні елементи й не накладаються одне на одне.
        details_scroll=QScrollArea(); details_scroll.setWidgetResizable(True)
        details_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        details_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        details_scroll.setMinimumWidth(220)
        details=QWidget(); details_layout=QVBoxLayout(details); details_layout.setContentsMargins(0,0,0,0); details_layout.setSpacing(8)

        st,sl=self.panel("РОЗПІЗНАВАННЯ")
        for caption in ("ПОТОЧНИЙ ЖЕСТ", "РУКА", "СТАН ВІДСТЕЖЕННЯ", "ОСТАННЯ ДІЯ"):
            label=QLabel(caption); label.setWordWrap(True); sl.addWidget(label)
            if caption == "ПОТОЧНИЙ ЖЕСТ":
                self.gval=QLabel("—"); self.gval.setObjectName("metric"); self.gval.setWordWrap(True); self.gval.setMinimumHeight(24); sl.addWidget(self.gval)
            elif caption == "РУКА":
                self.handval=QLabel("—"); self.handval.setWordWrap(True); sl.addWidget(self.handval)
            elif caption == "СТАН ВІДСТЕЖЕННЯ":
                self.track=QLabel("ОЧІКУВАННЯ"); self.track.setWordWrap(True); sl.addWidget(self.track)
            else:
                self.event=QLabel("Дій не виконано"); self.event.setWordWrap(True); sl.addWidget(self.event)
        details_layout.addWidget(st)

        info,il=self.panel("РОЗПІЗНАВАНІ ЖЕСТИ")
        gestures_label=QLabel("👍 Великий палець угору\n🤙 Мізинець\n☝ Один палець\n✌ Два пальці\n🤟 Три пальці\n🖖 Чотири пальці\n🖐 Відкрита долоня\n✊ Кулак")
        gestures_label.setWordWrap(True); gestures_label.setSizePolicy(QSizePolicy.Expanding,QSizePolicy.Preferred)
        il.addWidget(gestures_label); details_layout.addWidget(info); details_layout.addStretch()
        details_scroll.setWidget(details); right.addWidget(details_scroll,1)
        row.addLayout(right,2); l.addLayout(row); return w

    def settings(self):
        w=QWidget(); outer=QVBoxLayout(w); outer.setContentsMargins(0,0,0,0)
        scroll=QScrollArea(); scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        inner=QWidget(); l=QVBoxLayout(inner)
        l.setContentsMargins(8,8,8,8); l.setSpacing(5)
        f,fl=self.panel("КАМЕРА ТА ВИДИМІСТЬ")
        fl.addWidget(QLabel("Мова інтерфейсу")); self.language=NoWheelComboBox(); self.language.addItem("Українська","uk"); self.language.addItem("English","en"); self.language.addItem("Čeština","cs"); self.language.addItem("Русский","ru"); self.language.addItem("Português","pt"); self.language.setCurrentIndex(max(0,self.language.findData(self.cfg.get("language","uk")))); self.language.currentIndexChanged.connect(self.apply_language); fl.addWidget(self.language)
        self.autostart=QCheckBox("Запускати разом із Windows"); self.autostart.setChecked(self.autostart_enabled()); self.autostart.toggled.connect(self.set_autostart); fl.addWidget(self.autostart)
        fl.addWidget(QLabel("Пристрій камери")); self.cam=NoWheelComboBox(); self.cam.addItems(["Камера 0","Камера 1","Камера 2"]); self.cam.setCurrentIndex(min(int(self.cfg.get("camera_index",0)),2)); fl.addWidget(self.cam)
        fl.addWidget(QLabel("Роздільна здатність камери")); self.resolution=NoWheelComboBox()
        for label, value in [("1280 × 720 (рекомендовано)","1280x720"),("1920 × 1080 (Full HD)","1920x1080"),("640 × 480 (швидше)","640x480")]: self.resolution.addItem(label,value)
        self.resolution.setCurrentIndex(max(0,self.resolution.findData(self.cfg.get("camera_resolution","1280x720")))); fl.addWidget(self.resolution)
        self.mirror_camera=QCheckBox("Дзеркально відображати зображення камери")
        self.mirror_camera.setChecked(bool(self.cfg.get("mirror_camera", True))); fl.addWidget(self.mirror_camera)
        self.night=QCheckBox("Покращення видимості в темряві"); self.night.setChecked(bool(self.cfg.get("night_vision",False))); fl.addWidget(self.night)
        self.face_tracking=QCheckBox("Показувати рамки навколо виявлених облич")
        self.face_tracking.setChecked(bool(self.cfg.get("face_tracking_enabled", self.cfg.get("face_presence_gate", True))))
        fl.addWidget(self.face_tracking)
        self.face_gate=QCheckBox("Виконувати жести лише коли обличчя є в кадрі")
        self.face_gate.setChecked(bool(self.cfg.get("face_presence_gate",False))); fl.addWidget(self.face_gate)
        self.hand_overlay=QCheckBox("Показувати скелет рук у відео")
        self.hand_overlay.setChecked(bool(self.cfg.get("hand_overlay_enabled", True))); fl.addWidget(self.hand_overlay)
        self.virtual_camera_enabled=QCheckBox("Виводити оброблене відео у віртуальну камеру (OBS / Discord)")
        self.virtual_camera_enabled.setChecked(bool(self.cfg.get("virtual_camera_enabled", False))); fl.addWidget(self.virtual_camera_enabled)
        fl.addWidget(QLabel("Утримання зображення руки (секунди)")); self.hand_overlay_hold=QDoubleSpinBox(); self.hand_overlay_hold.setRange(0.0,5.0); self.hand_overlay_hold.setSingleStep(0.1); self.hand_overlay_hold.setDecimals(1); self.hand_overlay_hold.setValue(float(self.cfg.get("hand_overlay_hold_seconds",0.5))); fl.addWidget(self.hand_overlay_hold)
        fl.addWidget(QLabel("Гамма")); self.gamma=self.make_slider(5,30,int(float(self.cfg.get("gamma",1.5))*10),"{:.1f}",10); fl.addWidget(self.gamma[0])
        fl.addWidget(QLabel("Яскравість")); self.brightness=self.make_slider(-50,80,int(self.cfg.get("brightness",12)),"{}",1); fl.addWidget(self.brightness[0])
        fl.addWidget(QLabel("Контрастність")); self.contrast=self.make_slider(5,30,int(float(self.cfg.get("contrast",1.25))*10),"{:.1f}",10); fl.addWidget(self.contrast[0])
        hint=QLabel("Оброблене відео надсилається у віртуальну камеру Streamer Camera Control. У OBS та Discord вибери цей пристрій; фізичну камеру використовує ця програма."); hint.setWordWrap(True); hint.setStyleSheet("color:#c0c6ca;padding:8px;background:#202326;border-radius:8px"); fl.addWidget(hint)
        self.autosave_status=QLabel("Зміни зберігаються автоматично"); self.autosave_status.setStyleSheet("color:#8bd6ae;padding:4px"); fl.addWidget(self.autosave_status)
        copy_cfg=QPushButton("КОПІЮВАТИ КОНФІГ У БУФЕР ОБМІНУ"); copy_cfg.clicked.connect(self.copy_config); fl.addWidget(copy_cfg)
        trayhint=QLabel("✕ ховає програму в трей. Для виходу натисни правою кнопкою на іконку трея → «Вийти»."); trayhint.setWordWrap(True); trayhint.setStyleSheet("color:#c0c6ca;padding:8px"); fl.addWidget(trayhint)
        l.addWidget(f); l.addStretch()
        scroll.setWidget(inner); outer.addWidget(scroll)
        return w

    def bindings_page(self):
        w=QWidget(); outer=QVBoxLayout(w); scroll=QScrollArea(); scroll.setWidgetResizable(True); inner=QWidget(); lay=QVBoxLayout(inner)
        intro=QLabel("Налаштуй окрему команду для кожної руки. Для прив’язки «Обидві руки» жест має бути показаний одночасно двома руками."); intro.setWordWrap(True); intro.setStyleSheet("color:#c0c6ca;padding:8px"); lay.addWidget(intro)
        for hand,hand_name in HANDS.items():
            f,fl=self.panel(hand_name.upper()); grid=QGridLayout(); grid.addWidget(QLabel("ЖЕСТ"),0,0); grid.addWidget(QLabel("ДІЯ"),0,1); grid.addWidget(QLabel("КЛАВІША / ТЕКСТ / URL"),0,2); grid.addWidget(QLabel("УТРИМАННЯ (С)"),0,3)
            for ri,(gesture_key,gesture_name) in enumerate(GESTURES.items(),1):
                grid.addWidget(QLabel(gesture_name),ri,0)
                action=NoWheelComboBox(); action.addItem("Немає дії","none"); action.addItem("Натиснути клавішу","key"); action.addItem("Ввести текст","text"); action.addItem("Відкрити вебсторінку","url")
                target=KeyCaptureEdit(); target.setPlaceholderText("Клавіша / текст / URL або вибери скрипт у списку")
                hold=QDoubleSpinBox(); hold.setRange(0.05,5.0); hold.setSingleStep(0.05); hold.setDecimals(2); hold.setSuffix(" с"); hold.setValue(float(self.cfg.get("bindings",{}).get(f"{hand}:{gesture_key}",{}).get("hold_seconds",0.35)))
                saved=self.cfg.get("bindings",{}).get(f"{hand}:{gesture_key}",{})
                self._add_script_actions(action)
                saved_action=saved.get("action","none"); saved_target=saved.get("target","")
                saved_data=("script|"+saved_target) if saved_action=="script" else saved_action
                ix=action.findData(saved_data)
                if ix < 0 and saved_action=="script":
                    ix=action.findData("script")
                action.setCurrentIndex(max(0,ix)); target.setText(saved_target)
                target.setReadOnly(isinstance(action.currentData(),str) and action.currentData().startswith("script|"))
                target.set_capture_enabled(action.currentData() == "key")
                if action.currentData() == "key": target.setPlaceholderText("Натисни тут до 3 клавіш: Shift+Q+F")
                action.currentIndexChanged.connect(lambda _=None, cb=action, tb=target: self.binding_action_changed(cb,tb))
                grid.addWidget(action,ri,1); grid.addWidget(target,ri,2); grid.addWidget(hold,ri,3); self.binding_widgets[f"{hand}:{gesture_key}"]=(action,target,hold)
            fl.addLayout(grid); lay.addWidget(f)
        self.bindings_status=QLabel("Прив’язки зберігаються автоматично"); self.bindings_status.setStyleSheet("color:#8bd6ae;padding:4px"); lay.addWidget(self.bindings_status); lay.addStretch(); scroll.setWidget(inner); outer.addWidget(scroll); return w

    def _add_script_actions(self, combo):
        for item in self.cfg.get("scripts", []):
            path=str(item.get("path", "")).strip()
            if path:
                name=item.get("name") or Path(path).stem
                prefix = {"en":"Script: ", "cs":"Skript: ", "ru":"Скрипт: ", "pt":"Script: "}.get(self.language.currentData(), "Скрипт: ")
                combo.addItem(prefix+name, "script|"+path)

    def binding_action_changed(self, combo, target):
        data=combo.currentData()
        is_script=isinstance(data, str) and data.startswith("script|")
        target.setReadOnly(is_script)
        target.set_capture_enabled(data == "key")
        if data == "key": target.setPlaceholderText("Натисни тут до 3 клавіш: Shift+Q+F")
        else: target.setPlaceholderText("Клавіша / текст / URL або вибери скрипт у списку")
        if is_script:
            target.setText(data.split("|", 1)[1])
        self.schedule_autosave()

    def refresh_binding_script_options(self):
        for action, target, hold in self.binding_widgets.values():
            current=action.currentData()
            if isinstance(current, str) and current.startswith("script|"):
                current_path=current.split("|", 1)[1]
            else:
                current_path=target.text().strip() if current=="script" else ""
            while action.count():
                if isinstance(action.itemData(action.count()-1), str) and action.itemData(action.count()-1).startswith("script|"):
                    action.removeItem(action.count()-1)
                else:
                    break
            self._add_script_actions(action)
            if current_path:
                ix=action.findData("script|"+current_path)
                if ix>=0: action.setCurrentIndex(ix)
                else:
                    action.setCurrentIndex(0)
                    target.clear()

    def scripts_page(self):
        w=QWidget(); outer=QVBoxLayout(w); f,fl=self.panel("МОЇ СКРИПТИ · AHK / EXE / BAT")
        intro=QLabel("Додавай власні файли .ahk, .exe, .bat або .cmd. Додані файли зберігаються у конфігурації; їх можна запускати кнопкою або прив’язати до жесту на сторінці «Прив’язки команд».")
        intro.setWordWrap(True); fl.addWidget(intro)
        self.scripts_list=ScriptDropList(self.add_script_paths); self.scripts_list.setMinimumHeight(160); self.scripts_list.setToolTip("Перетягни сюди файли .ahk, .exe, .bat або .cmd"); fl.addWidget(self.scripts_list)
        row=QHBoxLayout(); add=QPushButton("＋ ДОДАТИ СКРИПТ"); add.clicked.connect(self.add_script); row.addWidget(add)
        run=QPushButton("▶ ЗАПУСТИТИ ВИБРАНИЙ"); run.clicked.connect(self.run_selected_script); row.addWidget(run)
        remove=QPushButton("ВИДАЛИТИ"); remove.clicked.connect(self.remove_selected_script); row.addWidget(remove); fl.addLayout(row)
        note=QLabel("Для AHK потрібен встановлений AutoHotkey або коректна асоціація .ahk у Windows. Скрипти запускаються від імені поточного користувача.")
        note.setWordWrap(True); note.setStyleSheet("color:#c0c6ca;padding:8px"); fl.addWidget(note)
        outer.addWidget(f); outer.addStretch(); self.refresh_scripts_list(); return w

    def connect_autosave_signals(self):
        for widget in (self.cam, self.resolution, self.language):
            widget.currentIndexChanged.connect(self.schedule_autosave)
        for widget in (self.night, self.mirror_camera, self.face_tracking, self.face_gate, self.hand_overlay, self.virtual_camera_enabled):
            widget.toggled.connect(self.schedule_autosave)
        self.face_tracking.toggled.connect(self.update_face_detector)
        self.face_gate.toggled.connect(self.update_face_detector)
        self.hand_overlay_hold.valueChanged.connect(self.schedule_autosave)
        for slider in (self.gamma[1], self.brightness[1], self.contrast[1]):
            slider.valueChanged.connect(self.schedule_autosave)
        for action, target, hold in self.binding_widgets.values():
            action.currentIndexChanged.connect(self.schedule_autosave)
            target.textChanged.connect(self.schedule_autosave)
            hold.valueChanged.connect(self.schedule_autosave)

    def schedule_autosave(self, *_):
        if hasattr(self, "autosave_timer"):
            self.autosave_timer.start()
            if hasattr(self, "autosave_status"): self.autosave_status.setText("Є незбережені зміни…")
            if hasattr(self, "bindings_status"): self.bindings_status.setText("Збереження…")

    def collect_config(self):
        self.cfg.update({"camera_index":self.cam.currentIndex(), "camera_resolution":self.resolution.currentData(),
            "night_vision":self.night.isChecked(), "mirror_camera":self.mirror_camera.isChecked(), "gamma":self.gamma[1].value()/10,
            "brightness":self.brightness[1].value(), "contrast":self.contrast[1].value()/10,
            "language":self.language.currentData(), "face_presence_gate":self.face_gate.isChecked(),
            "face_tracking_enabled":self.face_tracking.isChecked(), "hand_overlay_enabled":self.hand_overlay.isChecked(), "virtual_camera_enabled":self.virtual_camera_enabled.isChecked(), "hand_overlay_hold_seconds":self.hand_overlay_hold.value()})
        bindings={}
        for key,(action,target,hold) in self.binding_widgets.items():
            selected=action.currentData()
            if isinstance(selected,str) and selected.startswith("script|"):
                bindings[key]={"action":"script","target":selected.split("|",1)[1],"hold_seconds":hold.value()}
            else:
                bindings[key]={"action":selected,"target":target.text().strip(),"hold_seconds":hold.value()}
        self.cfg["bindings"]=bindings
        return self.cfg

    def write_config(self):
        CONFIG.parent.mkdir(parents=True, exist_ok=True)
        tmp=CONFIG.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(self.cfg,indent=2,ensure_ascii=False),encoding="utf-8")
        tmp.replace(CONFIG)

    def save_config_silent(self):
        try:
            self.collect_config(); self.write_config()
            if hasattr(self,"autosave_status"): self.autosave_status.setText("✓ Усі налаштування автоматично збережено")
            if hasattr(self,"bindings_status"): self.bindings_status.setText("✓ Прив’язки автоматично збережено")
        except Exception as e:
            if hasattr(self,"autosave_status"): self.autosave_status.setText("Помилка автозбереження: "+str(e))

    def refresh_scripts_list(self):
        if not hasattr(self,"scripts_list"): return
        self.scripts_list.clear()
        for item in self.cfg.get("scripts",[]):
            name=item.get("name") or Path(item.get("path","скрипт")).name
            self.scripts_list.addItem(f"{name}   ·   {item.get('path','')}")
        self.refresh_binding_script_options()

    def add_script(self):
        title = {"en":"Add script", "cs":"Přidat skript", "ru":"Добавить скрипт", "pt":"Adicionar script"}.get(self.language.currentData(), "Додати скрипт")
        paths,_=QFileDialog.getOpenFileNames(self,title,str(Path.home()),"Scripts (*.ahk *.exe *.bat *.cmd)")
        if paths: self.add_script_paths(paths)

    def add_script_paths(self, paths):
        added=0; rejected=[]
        for path in paths:
            file_path=Path(path)
            if not file_path.is_file() or file_path.suffix.lower() not in SUPPORTED_SCRIPT_EXTENSIONS:
                rejected.append(file_path.name); continue
            resolved=str(file_path.resolve())
            if any(str(x.get("path","")).lower()==resolved.lower() for x in self.cfg.get("scripts",[])):
                continue
            self.cfg.setdefault("scripts",[]).append({"name":file_path.stem,"path":resolved}); added+=1
        self.refresh_scripts_list()
        if added: self.save_config_silent()
        if rejected:
            if self.language.currentData() == "en":
                title, prefix, allowed = "Unsupported format", "Skipped files: ", "\nAllowed: AHK, EXE, BAT, CMD."
            elif self.language.currentData() == "cs":
                title, prefix, allowed = "Nepodporovaný formát", "Přeskočené soubory: ", "\nPovoleno: AHK, EXE, BAT, CMD."
            elif self.language.currentData() == "ru":
                title, prefix, allowed = "Неподдерживаемый формат", "Пропущенные файлы: ", "\nРазрешены: AHK, EXE, BAT, CMD."
            elif self.language.currentData() == "pt":
                title, prefix, allowed = "Formato não suportado", "Arquivos ignorados: ", "\nPermitidos: AHK, EXE, BAT, CMD."
            else:
                title, prefix, allowed = "Формат не підтримується", "Пропущено файли: ", "\nДозволені: AHK, EXE, BAT, CMD."
            QMessageBox.warning(self, title, prefix + ", ".join(rejected) + allowed)

    def remove_selected_script(self):
        ix=self.scripts_list.currentRow()
        if ix<0: return
        self.cfg["scripts"].pop(ix); self.refresh_scripts_list(); self.save_config_silent()

    def launch_script_path(self, path):
        # У прив'язці до жесту можна вказати повний шлях або назву скрипта зі списку «Мої скрипти».
        for registered in self.cfg.get("scripts", []):
            if str(path).strip().casefold() in (registered.get("name", "").casefold(), Path(registered.get("path", "")).name.casefold()):
                path = registered.get("path", path)
                break
        file=Path(path).expanduser()
        if not file.is_file(): raise FileNotFoundError(self.tr("Файл не знайдено: ")+str(file))
        ext=file.suffix.lower()
        if ext not in (".ahk",".exe",".bat",".cmd"):
            raise ValueError(self.tr("Дозволені лише .ahk, .exe, .bat та .cmd"))
        if sys.platform == "win32":
            if ext in (".bat",".cmd"):
                subprocess.Popen(["cmd.exe","/c","start","",str(file)],cwd=str(file.parent),shell=False)
            elif ext == ".exe":
                subprocess.Popen([str(file)],cwd=str(file.parent),shell=False)
            else:
                os.startfile(str(file))
        else:
            if ext == ".exe": subprocess.Popen([str(file)],cwd=str(file.parent))
            else: raise OSError("Запуск AHK/BAT доступний у Windows.")

    def run_selected_script(self):
        ix=self.scripts_list.currentRow()
        if ix<0:
            QMessageBox.information(self,self.tr("Мої скрипти"),self.tr("Спочатку вибери скрипт зі списку.")); return
        try:
            item=self.cfg["scripts"][ix]; self.launch_script_path(item["path"])
            self.event.setText(self.tr("Запущено скрипт: ")+item.get("name",Path(item["path"]).name))
        except Exception as e: QMessageBox.warning(self,self.tr("Не вдалося запустити"),str(e))

    def make_slider(self,minimum,maximum,value,fmt,divisor):
        row=QWidget(); lay=QHBoxLayout(row); lay.setContentsMargins(0,0,0,0); slider=QSlider(Qt.Horizontal); slider.setRange(minimum,maximum); slider.setValue(max(minimum,min(maximum,value)))
        label=QLabel(fmt.format(slider.value()/divisor)); label.setMinimumWidth(42); label.setAlignment(Qt.AlignRight|Qt.AlignVCenter)
        slider.valueChanged.connect(lambda v,lab=label,f=fmt,d=divisor:lab.setText(f.format(v/d))); lay.addWidget(slider,1); lay.addWidget(label); return row,slider

    def autostart_enabled(self):
        if sys.platform != "win32": return False
        try:
            import winreg
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, r"Software\Microsoft\Windows\CurrentVersion\Run") as key:
                winreg.QueryValueEx(key, "GestureControl")
            return True
        except Exception:
            return False

    def set_autostart(self, enabled):
        if sys.platform != "win32":
            return
        try:
            import winreg
            key_path = r"Software\Microsoft\Windows\CurrentVersion\Run"
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, key_path, 0, winreg.KEY_SET_VALUE) as key:
                if enabled:
                    pythonw = Path(sys.executable).with_name("pythonw.exe")
                    if not pythonw.exists(): pythonw = Path(sys.executable)
                    command = f'"{pythonw}" "{ROOT / "app.py"}"'
                    winreg.SetValueEx(key, "GestureControl", 0, winreg.REG_SZ, command)
                else:
                    try: winreg.DeleteValue(key, "GestureControl")
                    except FileNotFoundError: pass
        except Exception as e:
            QMessageBox.warning(self, self.tr("Автозапуск"), self.tr("Не вдалося змінити автозапуск: ")+str(e))
            self.autostart.blockSignals(True); self.autostart.setChecked(not enabled); self.autostart.blockSignals(False)

    def apply_language(self, *_):
        if not hasattr(self, "language"): return
        lang = self.language.currentData() or "uk"
        self.cfg["language"] = lang
        translations = {
          "en": {
            "Керування жестами":"Gesture Control", "Додати скрипт":"Add script", "Виявляти обличчя та показувати рамки":"Detect faces and show boxes", "Дозволені лише .ahk, .exe, .bat та .cmd":"Only .ahk, .exe, .bat and .cmd files are allowed", "Файл не знайдено: ":"File not found: ", "Запуск AHK/BAT доступний у Windows.":"Launching AHK/BAT files is available on Windows.", "Не вдалося відкрити посилання у браузері за замовчуванням.":"Could not open the link in the default browser.", "Не вдалося відкрити камеру. Перевір дозвіл Windows або вибери іншу камеру.":"Could not open the camera. Check Windows permissions or select another camera.", "Помилка запуску камери":"Camera startup error", "● ПОМИЛКА":"● ERROR", "Формат не підтримується":"Unsupported format", "Пропущено файли: ":"Skipped files: ", " байт). Перевір інтернет-з'єднання або доступ до storage.googleapis.com.":" bytes). Check your internet connection or access to storage.googleapis.com.", "Не вдалося завантажити модель ":"Could not download model ", ". Перевір інтернет-з'єднання. Деталі: ":". Check your internet connection. Details: ", "Помилка автозбереження: ":"Autosave error: ", "ПОМИЛКА ОБРОБКИ КАДРУ: ":"FRAME PROCESSING ERROR: ", " → виконано: ":" → executed: ", "ОБЛИЧЧЯ:":"FACES:", "Натисни тут до 3 клавіш. Backspace очищає; Ctrl+Backspace призначає Backspace.":"Click here and press up to 3 keys. Backspace clears; Ctrl+Backspace binds Backspace.",
            "👍 Великий палець угору\n🤙 Мізинець\n☝ Один палець\n✌ Два пальці\n🤟 Три пальці\n🖖 Чотири пальці\n🖐 Відкрита долоня\n✊ Кулак":"👍 Thumb up\n🤙 Pinky\n☝ One finger\n✌ Two fingers\n🤟 Three fingers\n🖖 Four fingers\n🖐 Open palm\n✊ Fist",
            "КЕРУВАННЯ ЖЕСТАМИ":"GESTURE CONTROL", "РОЗДІЛИ":"SECTIONS", "Головна":"Home", "Налаштування":"Settings", "Прив’язки команд":"Command bindings", "Мої скрипти":"My scripts",
            "ЗОБРАЖЕННЯ З КАМЕРИ":"CAMERA PREVIEW", "КАМЕРУ НЕ ЗАПУЩЕНО\n\nНатисни «Увімкнути камеру»":"CAMERA NOT STARTED\n\nPress “Start camera”",
            "▶  УВІМКНУТИ КАМЕРУ":"▶  START CAMERA", "■  ЗУПИНИТИ КАМЕРУ":"■  STOP CAMERA", "РОЗПІЗНАВАННЯ":"RECOGNITION", "ПОТОЧНИЙ ЖЕСТ":"CURRENT GESTURE", "РУКА":"HAND", "СТАН ВІДСТЕЖЕННЯ":"TRACKING STATUS", "ОСТАННЯ ДІЯ":"LAST ACTION", "Дій не виконано":"No actions performed", "РОЗПІЗНАВАНІ ЖЕСТИ":"RECOGNIZED GESTURES",
            "Виконувати жести лише коли обличчя є в кадрі":"Require a face in the camera view for gestures", "Показувати скелет рук у відео":"Show hand skeleton overlay", "Виводити оброблене відео у віртуальну камеру (OBS / Discord)":"Output processed video to virtual camera (OBS / Discord)", "Утримання зображення руки (секунди)":"Hand overlay hold time (seconds)", "Показувати рамки навколо виявлених облич":"Show boxes around detected faces", "КАМЕРА ТА ВИДИМІСТЬ":"CAMERA AND VISIBILITY", "Мова інтерфейсу":"Interface language", "Запускати разом із Windows":"Start with Windows", "Пристрій камери":"Camera device", "Роздільна здатність камери":"Camera resolution", "1280 × 720 (рекомендовано)":"1280 × 720 (recommended)", "1920 × 1080 (Full HD)":"1920 × 1080 (Full HD)", "640 × 480 (швидше)":"640 × 480 (faster)", "Камера 0":"Camera 0", "Камера 1":"Camera 1", "Камера 2":"Camera 2", "Покращення видимості в темряві":"Low-light enhancement", "Дзеркально відображати зображення камери":"Mirror camera image", "Гамма":"Gamma", "Яскравість":"Brightness", "Контрастність":"Contrast", "Зберегти налаштування":"Save settings", "ЗБЕРЕГТИ НАЛАШТУВАННЯ":"SAVE SETTINGS", "КОПІЮВАТИ КОНФІГ У БУФЕР ОБМІНУ":"COPY CONFIG TO CLIPBOARD", "✕ ховає програму в трей. Для виходу натисни правою кнопкою на іконку трея → «Вийти».":"✕ hides the app to the system tray. To exit, right-click the tray icon → Exit.",
            "ЖЕСТ":"GESTURE", "ДІЯ":"ACTION", "ЛІВА":"LEFT", "ПРАВА":"RIGHT", "Знайдено рук: ":"Hands found: ", "Прив’язки зберігаються автоматично":"Bindings are saved automatically",
            "ЛІВА РУКА":"LEFT HAND", "ПРАВА РУКА":"RIGHT HAND", "ОБИДВІ РУКИ":"BOTH HANDS", "РУКА":"HAND", "ДОДАТИ СКРИПТ":"ADD SCRIPT", "＋ ДОДАТИ СКРИПТ":"＋ ADD SCRIPT", "▶ ЗАПУСТИТИ ВИБРАНИЙ":"▶ RUN SELECTED", "ВИДАЛИТИ":"REMOVE",
            "Не вдалося запустити віртуальну камеру Streamer Camera Control. Запусти INSTALL.bat від імені адміністратора або перевстанови драйвер. Деталі: ":"Could not start the virtual camera. Install OBS Studio and start OBS Virtual Camera, or install Unity Capture. Details: ",
            "Мова інтерфейсу":"Interface language", "Запускати разом із Windows":"Start with Windows", "Оброблене відео надсилається у віртуальну камеру Streamer Camera Control. У OBS та Discord вибери цей пристрій; фізичну камеру використовує ця програма.":"Processed frames are sent to a compatible virtual-camera driver (OBS Virtual Camera or Unity Capture). Select that virtual device in OBS or Discord; this app owns the physical webcam stream.",
            "Зміни зберігаються автоматично":"Changes are saved automatically", "Є незбережені зміни…":"Unsaved changes…", "Збереження…":"Saving…", "✓ Усі налаштування автоматично збережено":"✓ All settings saved automatically", "✓ Прив’язки автоматично збережено":"✓ Bindings saved automatically",
            "Налаштуй окрему команду для кожної руки. Для прив’язки «Обидві руки» жест має бути показаний одночасно двома руками.":"Configure a separate action for each hand. For the Both hands binding, the gesture must be detected on both hands at the same time.", "УТРИМАННЯ (С)":"HOLD (S)",
            "Додавай власні файли .ahk, .exe, .bat або .cmd. Додані файли зберігаються у конфігурації; їх можна запускати кнопкою або прив’язати до жесту на сторінці «Прив’язки команд».":"Add your own .ahk, .exe, .bat or .cmd files. Added files are saved in the configuration; you can run them or assign them to a gesture on the Command bindings page.",
            "Для AHK потрібен встановлений AutoHotkey або коректна асоціація .ahk у Windows. Скрипти запускаються від імені поточного користувача.":"AutoHotkey or a valid .ahk file association in Windows is required for AHK scripts. Scripts run as the current user.",
            "Перетягни сюди файли .ahk, .exe, .bat або .cmd":"Drop .ahk, .exe, .bat or .cmd files here", "Клавіша / текст / URL або вибери скрипт у списку":"Key / text / URL, or select a script from the list", "Натисни тут до 3 клавіш: Shift+Q+F":"Click here and press up to 3 keys: Shift+Q+F", "Focus here, then press up to 3 keys":"Click here and press up to 3 keys. Backspace clears; Ctrl+Backspace binds Backspace.",
            "✕ ховає програму в трей. Для виходу натисни правою кнопкою на іконку трея → «Вийти».":"✕ hides the app to the system tray. To exit, right-click the tray icon and choose Exit.",
            "ЗОБРАЖЕННЯ З КАМЕРИ":"CAMERA PREVIEW", "МОЇ СКРИПТИ · AHK / EXE / BAT":"MY SCRIPTS · AHK / EXE / BAT", "Дій не виконано":"No actions performed", "ОСТАННЯ ДІЯ":"LAST ACTION", "РОЗПІЗНАВАНІ ЖЕСТИ":"RECOGNIZED GESTURES",
            "Обличчя не виявлено — очікую":"No face detected — waiting", "ОБЛИЧЧЯ НЕ ВИЯВЛЕНО — ОЧІКУЮ":"NO FACE DETECTED — WAITING", "ПОШУК РУКИ…":"SEARCHING FOR HAND…", "ЗНАЙДЕНО РУК: ":"HANDS FOUND: ", "НЕМАЄ КАДРУ З КАМЕРИ":"NO CAMERA FRAME", "ЗАВАНТАЖЕННЯ МОДЕЛІ…":"LOADING MODEL…", "ПОМИЛКА ОБРОБКИ КАДРУ:":"FRAME PROCESSING ERROR:", "Помилка виконання прив'язки: ":"Binding execution error: ", "Помилка налаштування детекції обличчя: ":"Face detection setup error: ",
            "Камеру зупинено через помилку, щоб уникнути зависання.":"Camera stopped after an error to prevent the app from freezing.", "Запущено скрипт: ":"Script started: ", "Не вдалося запустити":"Unable to launch", "Мої скрипти":"My scripts", "Спочатку вибери скрипт зі списку.":"Select a script from the list first.",
            "Зупинити камеру":"Stop camera", "Відкрити":"Open", "Згорнути в трей":"Minimize to tray", "Вийти":"Exit", "Конфіг":"Configuration", "Конфігурацію скопійовано в буфер обміну.":"Configuration copied to clipboard.", "Помилка":"Error", "Не вдалося скопіювати конфігурацію: ":"Could not copy configuration: ", "Автозапуск":"Startup", "Не вдалося змінити автозапуск: ":"Could not change startup setting: ",
            "КАМЕРУ ЗУПИНЕНО":"CAMERA STOPPED", "НЕАКТИВНО":"INACTIVE", "● НЕАКТИВНО":"● INACTIVE", "● ПРАЦЮЄ":"● RUNNING", "ПОМИЛКА ІНІЦІАЛІЗАЦІЇ":"INITIALIZATION ERROR", "ПОМИЛКА":"ERROR", "ЛІВА":"LEFT", "ПРАВА":"RIGHT", "Обличчя":"Face", "ЖЕСТИ АКТИВНІ":"GESTURES ACTIVE", " · ЖЕСТИ АКТИВНІ":" · GESTURES ACTIVE",
            "Налаштування":"Settings", "Прив’язки команд":"Command bindings", "Головна":"Home", "Мої скрипти":"My scripts", "Скрипт: ":"Script: ", "Підтримуються AHK, EXE, BAT, CMD":"Supported: AHK, EXE, BAT, CMD", "Завантаження моделі…":"Loading model…",
            "Клавіша / текст / URL":"Key / text / URL", "Утримання (с)":"Hold (s)", "УТРИМАННЯ (С)":"HOLD (S)", "ДІЯ":"ACTION", "ЖЕСТ":"GESTURE", "Налаштуй окрему команду для кожної руки. Для прив’язки «Обидві руки» жест має бути показаний одночасно двома руками.":"Configure a separate action for each hand. For the Both hands binding, the gesture must be detected on both hands at the same time.", "УТРИМАННЯ (С)":"HOLD (S)", "Утримання (с)":"Hold (s)", "КЛАВІША / ТЕКСТ / URL":"KEY / TEXT / URL", "Немає дії":"No action", "Натиснути клавішу":"Press key", "Ввести текст":"Type text", "Відкрити вебсторінку":"Open website", "Запустити скрипт (шлях до файлу)":"Run script (file path)", "ПОШУК РУКИ…":"SEARCHING FOR HAND…", "ЗНАЙДЕНО РУК: ":"HANDS FOUND: ", "ОЧІКУВАННЯ":"WAITING", "НЕАКТИВНО":"INACTIVE", "● ПРАЦЮЄ":"● RUNNING", "КАМЕРУ ЗУПИНЕНО":"CAMERA STOPPED", "Ліва рука":"Left hand", "Права рука":"Right hand", "Обидві руки":"Both hands",
            "Великий палець угору":"Thumb up", "Мізинець":"Pinky", "Один палець":"One finger", "Два пальці":"Two fingers", "Три пальці":"Three fingers", "Чотири пальці":"Four fingers", "Відкрита долоня":"Open palm", "Кулак":"Fist",
            "Завантаження моделі…":"Loading model…", "ПОМИЛКА ІНІЦІАЛІЗАЦІЇ":"INITIALIZATION ERROR", "КАМЕРУ ЗУПИНЕНО":"CAMERA STOPPED"
          },
          "cs": {
            "КЕРУВАННЯ ЖЕСТАМИ":"OVLÁDÁNÍ GESTY", "РОЗДІЛИ":"SEKCE", "Головна":"Domů", "Налаштування":"Nastavení", "Прив’язки команд":"Přiřazení příkazů", "Мої скрипти":"Moje skripty",
            "ЗОБРАЖЕННЯ З КАМЕРИ":"NÁHLED KAMERY", "КАМЕРУ НЕ ЗАПУЩЕНО\n\nНатисни «Увімкнути камеру»":"KAMERA NENÍ SPUŠTĚNA\n\nStiskni „Spustit kameru“",
            "▶  УВІМКНУТИ КАМЕРУ":"▶  SPUSTIT KAMERU", "■  ЗУПИНИТИ КАМЕРУ":"■  ZASTAVIT KAMERU", "РОЗПІЗНАВАННЯ":"ROZPOZNÁVÁNÍ", "ПОТОЧНИЙ ЖЕСТ":"AKTUÁLNÍ GESTO", "РУКА":"RUKA", "СТАН ВІДСТЕЖЕННЯ":"STAV SLEDOVÁNÍ", "ОСТАННЯ ДІЯ":"POSLEDNÍ AKCE", "Дій не виконано":"Žádné akce", "РОЗПІЗНАВАНІ ЖЕСТИ":"ROZPOZNANÁ GESTA",
            "Виконувати жести лише коли обличчя є в кадрі":"Provádět gesta pouze při detekci obličeje", "Оброблене відео надсилається у віртуальну камеру Streamer Camera Control. У OBS та Discord вибери цей пристрій; фізичну камеру використовує ця програма.":"Zpracované snímky se odesílají do kompatibilního ovladače virtuální kamery (OBS Virtual Camera nebo Unity Capture). V OBS či Discordu vyber toto virtuální zařízení; fyzickou kameru používá tato aplikace.", "Показувати скелет рук у відео":"Zobrazit kostru ruky ve videu", "Виводити оброблене відео у віртуальну камеру (OBS / Discord)":"Odesílat zpracované video do virtuální kamery (OBS / Discord)", "Утримання зображення руки (секунди)":"Doba zobrazení kostry ruky (sekundy)", "Виявляти обличчя та показувати рамки":"Detekovat obličeje a zobrazit rámečky", "КАМЕРА ТА ВИДИМІСТЬ":"KAMERA A VIDITELNOST", "Мова інтерфейсу":"Jazyk rozhraní", "Запускати разом із Windows":"Spouštět při startu Windows", "Пристрій камери":"Kamera", "Роздільна здатність камери":"Rozlišení kamery", "1280 × 720 (рекомендовано)":"1280 × 720 (doporučeno)", "1920 × 1080 (Full HD)":"1920 × 1080 (Full HD)", "640 × 480 (швидше)":"640 × 480 (rychlejší)", "Камера 0":"Kamera 0", "Камера 1":"Kamera 1", "Камера 2":"Kamera 2", "Покращення видимості в темряві":"Zlepšení viditelnosti ve tmě", "Дзеркально відображати зображення камери":"Zrcadlit obraz kamery", "Гамма":"Gama", "Яскравість":"Jas", "Контрастність":"Kontrast", "ЗБЕРЕГТИ НАЛАШТУВАННЯ":"ULOŽIT NASTAVENÍ", "КОПІЮВАТИ КОНФІГ У БУФЕР ОБМІНУ":"KOPÍROVAT KONFIGURACI", "ЖЕСТ":"GESTO", "ДІЯ":"AKCE", "КЛАВІША / ТЕКСТ / URL":"KLÁVESA / TEXT / URL", "Немає дії":"Žádná akce", "Натиснути клавішу":"Stisknout klávesu", "Ввести текст":"Napsat text", "Відкрити вебсторінку":"Otevřít web", "Запустити скрипт (шлях до файлу)":"Spustit skript (cesta k souboru)", "ПОШУК РУКИ…":"HLEDÁNÍ RUKY…", "ОЧІКУВАННЯ":"ČEKÁNÍ", "НЕАКТИВНО":"NEAKTIVNÍ", "● ПРАЦЮЄ":"● BĚŽÍ", "КАМЕРУ ЗУПИНЕНО":"KAMERA ZASTAVENA", "Ліва рука":"Levá ruka", "Права рука":"Pravá ruka", "Обидві руки":"Obě ruce",
            "Великий палець угору":"Palec nahoru", "Мізинець":"Malíček", "Один палець":"Jeden prst", "Два пальці":"Dva prsty", "Три пальці":"Tři prsty", "Чотири пальці":"Čtyři prsty", "Відкрита долоня":"Otevřená dlaň", "Кулак":"Pěst",
            "Завантаження моделі…":"Načítání modelu…", "ПОМИЛКА ІНІЦІАЛІЗАЦІЇ":"CHYBA INICIALIZACE"
          }
        }
        translations["ru"] = {
            "Керування жестами":"Управление жестами", "Додати скрипт":"Добавить скрипт", "КЕРУВАННЯ ЖЕСТАМИ":"УПРАВЛЕНИЕ ЖЕСТАМИ", "РОЗДІЛИ":"РАЗДЕЛЫ", "Головна":"Главная", "Налаштування":"Настройки", "Прив’язки команд":"Привязки команд", "Мої скрипти":"Мои скрипты",
            "ЗОБРАЖЕННЯ З КАМЕРИ":"ИЗОБРАЖЕНИЕ С КАМЕРЫ", "КАМЕРУ НЕ ЗАПУЩЕНО\n\nНатисни «Увімкнути камеру»":"КАМЕРА НЕ ЗАПУЩЕНА\n\nНажми «Включить камеру»", "▶  УВІМКНУТИ КАМЕРУ":"▶  ВКЛЮЧИТЬ КАМЕРУ", "■  ЗУПИНИТИ КАМЕРУ":"■  ОСТАНОВИТЬ КАМЕРУ", "РОЗПІЗНАВАННЯ":"РАСПОЗНАВАНИЕ", "ПОТОЧНИЙ ЖЕСТ":"ТЕКУЩИЙ ЖЕСТ", "РУКА":"РУКА", "СТАН ВІДСТЕЖЕННЯ":"СОСТОЯНИЕ ОТСЛЕЖИВАНИЯ", "ОСТАННЯ ДІЯ":"ПОСЛЕДНЕЕ ДЕЙСТВИЕ", "Дій не виконано":"Действий не выполнено", "РОЗПІЗНАВАНІ ЖЕСТИ":"РАСПОЗНАВАЕМЫЕ ЖЕСТЫ",
            "👍 Великий палець угору\n🤙 Мізинець\n☝ Один палець\n✌ Два пальці\n🤟 Три пальці\n🖖 Чотири пальці\n🖐 Відкрита долоня\n✊ Кулак":"👍 Большой палец вверх\n🤙 Мизинец\n☝ Один палец\n✌ Два пальца\n🤟 Три пальца\n🖖 Четыре пальца\n🖐 Открытая ладонь\n✊ Кулак",
            "КАМЕРА ТА ВИДИМІСТЬ":"КАМЕРА И ОТОБРАЖЕНИЕ", "Мова інтерфейсу":"Язык интерфейса", "Запускати разом із Windows":"Запускать вместе с Windows", "Пристрій камери":"Устройство камеры", "Роздільна здатність камери":"Разрешение камеры", "1280 × 720 (рекомендовано)":"1280 × 720 (рекомендуется)", "640 × 480 (швидше)":"640 × 480 (быстрее)", "Камера 0":"Камера 0", "Камера 1":"Камера 1", "Камера 2":"Камера 2", "Дзеркально відображати зображення камери":"Зеркально отображать изображение камеры", "Покращення видимості в темряві":"Улучшение видимости в темноте", "Показувати рамки навколо виявлених облич":"Показывать рамки вокруг обнаруженных лиц", "Виконувати жести лише коли обличчя є в кадрі":"Выполнять жесты, только если лицо находится в кадре", "Показувати скелет рук у відео":"Показывать скелет рук на видео", "Виводити оброблене відео у віртуальну камеру (OBS / Discord)":"Передавать обработанное видео в виртуальную камеру (OBS / Discord)", "Утримання зображення руки (секунди)":"Время отображения скелета руки (секунды)", "Гамма":"Гамма", "Яскравість":"Яркость", "Контрастність":"Контрастность", "Зміни зберігаються автоматично":"Изменения сохраняются автоматически", "✓ Усі налаштування автоматично збережено":"✓ Все настройки автоматически сохранены", "КОПІЮВАТИ КОНФІГ У БУФЕР ОБМІНУ":"КОПИРОВАТЬ КОНФИГУРАЦИЮ", "✕ ховає програму в трей. Для виходу натисни правою кнопкою на іконку трея → «Вийти».":"✕ сворачивает программу в трей. Для выхода нажми правой кнопкой по значку трея → «Выход».",
            "ЖЕСТ":"ЖЕСТ", "ДІЯ":"ДЕЙСТВИЕ", "КЛАВІША / ТЕКСТ / URL":"КЛАВИША / ТЕКСТ / URL", "УТРИМАННЯ (С)":"УДЕРЖАНИЕ (С)", "Немає дії":"Нет действия", "Натиснути клавішу":"Нажать клавишу", "Ввести текст":"Ввести текст", "Відкрити вебсторінку":"Открыть веб-страницу", "Запустити скрипт (шлях до файлу)":"Запустить скрипт (путь к файлу)", "Налаштуй окрему команду для кожної руки. Для прив’язки «Обидві руки» жест має бути показаний одночасно двома руками.":"Настрой отдельную команду для каждой руки. Для привязки «Обе руки» жест должен одновременно распознаваться обеими руками.", "Ліва рука":"Левая рука", "Права рука":"Правая рука", "Обидві руки":"Обе руки", "Великий палець угору":"Большой палец вверх", "Мізинець":"Мизинец", "Один палець":"Один палец", "Два пальці":"Два пальца", "Три пальці":"Три пальца", "Чотири пальці":"Четыре пальца", "Відкрита долоня":"Открытая ладонь", "Кулак":"Кулак",
            "＋ ДОДАТИ СКРИПТ":"＋ ДОБАВИТЬ СКРИПТ", "▶ ЗАПУСТИТИ ВИБРАНИЙ":"▶ ЗАПУСТИТЬ ВЫБРАННЫЙ", "ВИДАЛИТИ":"УДАЛИТЬ", "МОЇ СКРИПТИ · AHK / EXE / BAT":"МОИ СКРИПТЫ · AHK / EXE / BAT", "Перетягни сюди файли .ahk, .exe, .bat або .cmd":"Перетащи сюда файлы .ahk, .exe, .bat или .cmd", "Додавай власні файли .ahk, .exe, .bat або .cmd. Додані файли зберігаються у конфігурації; їх можна запускати кнопкою або прив’язати до жесту на сторінці «Прив’язки команд».":"Добавляй свои файлы .ahk, .exe, .bat или .cmd. Добавленные файлы сохраняются в конфигурации; их можно запускать кнопкой или привязать к жесту на странице «Привязки команд».", "Для AHK потрібен встановлений AutoHotkey або коректна асоціація .ahk у Windows. Скрипти запускаються від імені поточного користувача.":"Для AHK требуется установленный AutoHotkey или корректная ассоциация файлов .ahk в Windows. Скрипты запускаются от имени текущего пользователя.", "Клавіша / текст / URL або вибери скрипт у списку":"Клавиша / текст / URL или выбери скрипт из списка", "Натисни тут до 3 клавіш: Shift+Q+F":"Нажми здесь до 3 клавиш: Shift+Q+F", "Натисни тут до 3 клавіш. Backspace очищає; Ctrl+Backspace призначає Backspace.":"Нажми здесь до 3 клавиш. Backspace очищает; Ctrl+Backspace назначает Backspace.",
            "ПОШУК РУКИ…":"ПОИСК РУКИ…", "ЗНАЙДЕНО РУК: ":"НАЙДЕНО РУК: ", "ОЧІКУВАННЯ":"ОЖИДАНИЕ", "НЕАКТИВНО":"НЕАКТИВНО", "● НЕАКТИВНО":"● НЕАКТИВНО", "● ПРАЦЮЄ":"● РАБОТАЕТ", "КАМЕРУ ЗУПИНЕНО":"КАМЕРА ОСТАНОВЛЕНА", "ЗАВАНТАЖЕННЯ МОДЕЛІ…":"ЗАГРУЗКА МОДЕЛИ…", "ПОМИЛКА ІНІЦІАЛІЗАЦІЇ":"ОШИБКА ИНИЦИАЛИЗАЦИИ", "ПОМИЛКА ОБРОБКИ КАДРУ:":"ОШИБКА ОБРАБОТКИ КАДРА:", "ОБЛИЧЧЯ НЕ ВИЯВЛЕНО — ОЧІКУЮ":"ЛИЦО НЕ ОБНАРУЖЕНО — ОЖИДАНИЕ", "Обличчя не виявлено — очікую":"Лицо не обнаружено — ожидание", "ОБЛИЧЧЯ:":"ЛИЦА:", " · ЖЕСТИ АКТИВНІ":" · ЖЕСТЫ АКТИВНЫ", "ЛІВА":"ЛЕВАЯ", "ПРАВА":"ПРАВАЯ", "Запущено скрипт: ":"Запущен скрипт: ", "Помилка виконання прив'язки: ":"Ошибка выполнения привязки: ", "Помилка налаштування детекції обличчя: ":"Ошибка настройки распознавания лиц: ", "Зупинити камеру":"Остановить камеру", "Відкрити":"Открыть", "Згорнути в трей":"Свернуть в трей", "Вийти":"Выход", "Конфіг":"Конфигурация", "Помилка":"Ошибка", "Файл не знайдено: ":"Файл не найден: ", "Не вдалося відкрити камеру. Перевір дозвіл Windows або вибери іншу камеру.":"Не удалось открыть камеру. Проверь разрешения Windows или выбери другую камеру.", "Не вдалося запустити віртуальну камеру Streamer Camera Control. Запусти INSTALL.bat від імені адміністратора або перевстанови драйвер. Деталі: ":"Не удалось запустить виртуальную камеру Streamer Camera Control. Запусти INSTALL.bat от имени администратора или переустанови драйвер. Подробности: ", "Камеру зупинено":"Камера остановлена", "Завершення роботи камери…":"Завершение работы камеры…", "Зачекай завершення зупинки камери…":"Дождись завершения остановки камеры…"
        }
        translations["pt"] = {
            "Керування жестами":"Controle por gestos", "Додати скрипт":"Adicionar script", "КЕРУВАННЯ ЖЕСТАМИ":"CONTROLE POR GESTOS", "РОЗДІЛИ":"SEÇÕES", "Головна":"Início", "Налаштування":"Configurações", "Прив’язки команд":"Atribuições de comandos", "Мої скрипти":"Meus scripts",
            "ЗОБРАЖЕННЯ З КАМЕРИ":"PRÉVIA DA CÂMERA", "КАМЕРУ НЕ ЗАПУЩЕНО\n\nНатисни «Увімкнути камеру»":"CÂMERA NÃO INICIADA\n\nClique em “Iniciar câmera”", "▶  УВІМКНУТИ КАМЕРУ":"▶  INICIAR CÂMERA", "■  ЗУПИНИТИ КАМЕРУ":"■  PARAR CÂMERA", "РОЗПІЗНАВАННЯ":"RECONHECIMENTO", "ПОТОЧНИЙ ЖЕСТ":"GESTO ATUAL", "РУКА":"MÃO", "СТАН ВІДСТЕЖЕННЯ":"STATUS DO RASTREAMENTO", "ОСТАННЯ ДІЯ":"ÚLTIMA AÇÃO", "Дій не виконано":"Nenhuma ação executada", "РОЗПІЗНАВАНІ ЖЕСТИ":"GESTOS RECONHECIDOS",
            "👍 Великий палець угору\n🤙 Мізинець\n☝ Один палець\n✌ Два пальці\n🤟 Три пальці\n🖖 Чотири пальці\n🖐 Відкрита долоня\n✊ Кулак":"👍 Polegar para cima\n🤙 Dedo mínimo\n☝ Um dedo\n✌ Dois dedos\n🤟 Três dedos\n🖖 Quatro dedos\n🖐 Palma aberta\n✊ Punho",
            "КАМЕРА ТА ВИДИМІСТЬ":"CÂMERA E VISUALIZAÇÃO", "Мова інтерфейсу":"Idioma da interface", "Запускати разом із Windows":"Iniciar com o Windows", "Пристрій камери":"Dispositivo da câmera", "Роздільна здатність камери":"Resolução da câmera", "1280 × 720 (рекомендовано)":"1280 × 720 (recomendado)", "640 × 480 (швидше)":"640 × 480 (mais rápido)", "Камера 0":"Câmera 0", "Камера 1":"Câmera 1", "Камера 2":"Câmera 2", "Дзеркально відображати зображення камери":"Espelhar imagem da câmera", "Покращення видимості в темряві":"Melhorar visibilidade em pouca luz", "Показувати рамки навколо виявлених облич":"Mostrar molduras nos rostos detectados", "Виконувати жести лише коли обличчя є в кадрі":"Reconhecer gestos somente quando houver um rosto no enquadramento", "Показувати скелет рук у відео":"Mostrar o esqueleto das mãos no vídeo", "Виводити оброблене відео у віртуальну камеру (OBS / Discord)":"Enviar vídeo processado para a câmera virtual (OBS / Discord)", "Утримання зображення руки (секунди)":"Duração da exibição do esqueleto da mão (segundos)", "Гамма":"Gama", "Яскравість":"Brilho", "Контрастність":"Contraste", "Зміни зберігаються автоматично":"As alterações são salvas automaticamente", "✓ Усі налаштування автоматично збережено":"✓ Todas as configurações foram salvas automaticamente", "КОПІЮВАТИ КОНФІГ У БУФЕР ОБМІНУ":"COPIAR CONFIGURAÇÃO", "✕ ховає програму в трей. Для виходу натисни правою кнопкою на іконку трея → «Вийти».":"✕ minimiza o aplicativo para a bandeja. Para sair, clique com o botão direito no ícone da bandeja → “Sair”.",
            "ЖЕСТ":"GESTO", "ДІЯ":"AÇÃO", "КЛАВІША / ТЕКСТ / URL":"TECLA / TEXTO / URL", "УТРИМАННЯ (С)":"DURAÇÃO (S)", "Немає дії":"Nenhuma ação", "Натиснути клавішу":"Pressionar tecla", "Ввести текст":"Digitar texto", "Відкрити вебсторінку":"Abrir página da web", "Запустити скрипт (шлях до файлу)":"Executar script (caminho do arquivo)", "Налаштуй окрему команду для кожної руки. Для прив’язки «Обидві руки» жест має бути показаний одночасно двома руками.":"Configure uma ação para cada mão. Para a atribuição “Ambas as mãos”, o gesto deve ser detectado simultaneamente pelas duas mãos.", "Ліва рука":"Mão esquerda", "Права рука":"Mão direita", "Обидві руки":"Ambas as mãos", "Великий палець угору":"Polegar para cima", "Мізинець":"Dedo mínimo", "Один палець":"Um dedo", "Два пальці":"Dois dedos", "Три пальці":"Três dedos", "Чотири пальці":"Quatro dedos", "Відкрита долоня":"Palma aberta", "Кулак":"Punho",
            "＋ ДОДАТИ СКРИПТ":"＋ ADICIONAR SCRIPT", "▶ ЗАПУСТИТИ ВИБРАНИЙ":"▶ EXECUTAR SELECIONADO", "ВИДАЛИТИ":"REMOVER", "МОЇ СКРИПТИ · AHK / EXE / BAT":"MEUS SCRIPTS · AHK / EXE / BAT", "Перетягни сюди файли .ahk, .exe, .bat або .cmd":"Arraste arquivos .ahk, .exe, .bat ou .cmd para cá", "Додавай власні файли .ahk, .exe, .bat або .cmd. Додані файли зберігаються у конфігурації; їх можна запускати кнопкою або прив’язати до жесту на сторінці «Прив’язки команд».":"Adicione seus próprios arquivos .ahk, .exe, .bat ou .cmd. Os arquivos adicionados ficam salvos na configuração e podem ser executados pelo botão ou atribuídos a um gesto na página “Atribuições de comandos”.", "Для AHK потрібен встановлений AutoHotkey або коректна асоціація .ahk у Windows. Скрипти запускаються від імені поточного користувача.":"Para arquivos AHK, é necessário instalar o AutoHotkey ou ter uma associação .ahk válida no Windows. Os scripts são executados como o usuário atual.", "Клавіша / текст / URL або вибери скрипт у списку":"Tecla / texto / URL ou selecione um script na lista", "Натисни тут до 3 клавіш: Shift+Q+F":"Clique aqui e pressione até 3 teclas: Shift+Q+F", "Натисни тут до 3 клавіш. Backspace очищає; Ctrl+Backspace призначає Backspace.":"Clique aqui e pressione até 3 teclas. Backspace limpa; Ctrl+Backspace atribui Backspace.",
            "ПОШУК РУКИ…":"PROCURANDO MÃO…", "ЗНАЙДЕНО РУК: ":"MÃOS DETECTADAS: ", "ОЧІКУВАННЯ":"AGUARDANDO", "НЕАКТИВНО":"INATIVO", "● НЕАКТИВНО":"● INATIVO", "● ПРАЦЮЄ":"● EM EXECUÇÃO", "КАМЕРУ ЗУПИНЕНО":"CÂMERA PARADA", "ЗАВАНТАЖЕННЯ МОДЕЛІ…":"CARREGANDO MODELO…", "ПОМИЛКА ІНІЦІАЛІЗАЦІЇ":"ERRO DE INICIALIZAÇÃO", "ПОМИЛКА ОБРОБКИ КАДРУ:":"ERRO AO PROCESSAR QUADRO:", "ОБЛИЧЧЯ НЕ ВИЯВЛЕНО — ОЧІКУЮ":"ROSTO NÃO DETECTADO — AGUARDANDO", "Обличчя не виявлено — очікую":"Rosto não detectado — aguardando", "ОБЛИЧЧЯ:":"ROSTOS:", " · ЖЕСТИ АКТИВНІ":" · GESTOS ATIVOS", "ЛІВА":"ESQUERDA", "ПРАВА":"DIREITA", "Запущено скрипт: ":"Script iniciado: ", "Помилка виконання прив'язки: ":"Erro ao executar atribuição: ", "Помилка налаштування детекції обличчя: ":"Erro ao configurar a detecção facial: ", "Зупинити камеру":"Parar câmera", "Відкрити":"Abrir", "Згорнути в трей":"Minimizar para a bandeja", "Вийти":"Sair", "Конфіг":"Configuração", "Помилка":"Erro", "Файл не знайдено: ":"Arquivo não encontrado: ", "Не вдалося відкрити камеру. Перевір дозвіл Windows або вибери іншу камеру.":"Não foi possível abrir a câmera. Verifique as permissões do Windows ou selecione outra câmera.", "Не вдалося запустити віртуальну камеру Streamer Camera Control. Запусти INSTALL.bat від імені адміністратора або перевстанови драйвер. Деталі: ":"Não foi possível iniciar a câmera virtual Streamer Camera Control. Execute INSTALL.bat como administrador ou reinstale o driver. Detalhes: ", "Камеру зупинено":"Câmera parada", "Завершення роботи камери…":"Finalizando a câmera…", "Зачекай завершення зупинки камери…":"Aguarde a câmera terminar de parar…"
        }
        # Complete the Czech UI for controls and runtime status messages.
        translations["cs"].update({
            "Зміни зберігаються автоматично":"Změny se ukládají automaticky", "✓ Усі налаштування автоматично збережено":"✓ Všechna nastavení byla automaticky uložena", "МОЇ СКРИПТИ · AHK / EXE / BAT":"MOJE SKRIPTY · AHK / EXE / BAT", "＋ ДОДАТИ СКРИПТ":"＋ PŘIDAT SKRIPT", "▶ ЗАПУСТИТИ ВИБРАНИЙ":"▶ SPUSTIT VYBRANÝ", "ВИДАЛИТИ":"ODEBRAT", "Перетягни сюди файли .ahk, .exe, .bat або .cmd":"Sem přetáhněte soubory .ahk, .exe, .bat nebo .cmd", "Зміни зберігаються автоматично":"Změny se ukládají automaticky", "Дій не виконано":"Žádná akce nebyla provedena", "ЗНАЙДЕНО РУК: ":"NALEZENÉ RUCE: ", "ЗАВАНТАЖЕННЯ МОДЕЛІ…":"NAČÍTÁNÍ MODELU…", "ПОМИЛКА ОБРОБКИ КАДРУ:":"CHYBA ZPRACOVÁNÍ SNÍMKU:", "ПОШУК РУКИ…":"HLEDÁNÍ RUKY…", "ОБЛИЧЧЯ НЕ ВИЯВЛЕНО — ОЧІКУЮ":"TVÁŘ NENALEZENA — ČEKÁM", "ЛІВА":"LEVÁ", "ПРАВА":"PRAVÁ", "Зупинити камеру":"Zastavit kameru", "Відкрити":"Otevřít", "Згорнути в трей":"Minimalizovat do oznamovací oblasti", "Вийти":"Ukončit", "Конфіг":"Konfigurace", "Не вдалося відкрити камеру. Перевір дозвіл Windows або вибери іншу камеру.":"Kameru nelze otevřít. Zkontrolujte oprávnění Windows nebo vyberte jinou kameru.", "Камеру зупинено":"Kamera zastavena", "Завершення роботи камери…":"Ukončování kamery…", "Зачекай завершення зупинки камери…":"Počkejte na dokončení zastavování kamery…", "Помилка":"Chyba", "Файл не знайдено: ":"Soubor nebyl nalezen: ", "ЗБЕРЕГТИ НАЛАШТУВАННЯ":"ULOŽIT NASTAVENÍ", "Виявляти обличчя та показувати рамки":"Detekovat obličeje a zobrazit rámečky", "Показувати скелет рук у відео":"Zobrazit kostru rukou ve videu", "Виводити оброблене відео у віртуальну камеру (OBS / Discord)":"Odesílat zpracované video do virtuální kamery (OBS / Discord)", "Утримання зображення руки (секунди)":"Doba zobrazení kostry ruky (sekundy)", "Покращення видимості в темряві":"Zlepšení viditelnosti při slabém osvětlení", "Зміни зберігаються автоматично":"Změny se ukládají automaticky", "✓ Прив’язки автоматично збережено":"✓ Přiřazení byla automaticky uložena", "Прив’язки зберігаються автоматично":"Přiřazení se ukládají automaticky", "Підтримуються AHK, EXE, BAT, CMD":"Podporováno: AHK, EXE, BAT, CMD", "Натисни тут до 3 клавіш: Shift+Q+F":"Klikněte sem a stiskněte až 3 klávesy: Shift+Q+F", "Клавіша / текст / URL або вибери скрипт у списку":"Klávesa / text / URL nebo vyberte skript ze seznamu", "Оброблене відео надсилається у віртуальну камеру Streamer Camera Control. У OBS та Discord вибери цей пристрій; фізичну камеру використовує ця програма.":"Zpracované video se odesílá do virtuální kamery Streamer Camera Control. V OBS a Discordu vyberte toto zařízení; fyzickou kameru používá tato aplikace.", "Конфігурацію скопійовано в буфер обміну.":"Konfigurace byla zkopírována do schránky.", "Не вдалося скопіювати конфігурацію: ":"Konfiguraci se nepodařilo zkopírovat: ", "Не вдалося змінити автозапуск: ":"Nepodařilo se změnit automatické spouštění: "})

        mapping=translations.get(lang,{})
        self._active_mapping = mapping
        for spin in self.findChildren(QDoubleSpinBox):
            if spin.suffix() in (" с", " s"):
                spin.setSuffix(" s" if lang == "en" else (" s" if lang == "cs" else " с"))
        for widget in self.findChildren(QLabel)+self.findChildren(QPushButton)+self.findChildren(QCheckBox):
            original=widget.property("gc_original_text")
            if original is None:
                original=widget.text(); widget.setProperty("gc_original_text",original)
            widget.setText(mapping.get(original, original))
        for edit in self.findChildren(QLineEdit):
            original=edit.property("gc_original_placeholder")
            if original is None:
                original=edit.placeholderText(); edit.setProperty("gc_original_placeholder", original)
            edit.setPlaceholderText(mapping.get(original, original))
        for widget in self.findChildren(QListWidget):
            original=widget.property("gc_original_tooltip")
            if original is None:
                original=widget.toolTip(); widget.setProperty("gc_original_tooltip", original)
            widget.setToolTip(mapping.get(original, original))
        for combo in self.findChildren(QComboBox):
            if combo is self.language: continue
            for i in range(combo.count()):
                original=combo.itemData(i, Qt.UserRole+1)
                if original is None:
                    original=combo.itemText(i); combo.setItemData(i,original,Qt.UserRole+1)
                combo.setItemText(i, mapping.get(original, original))
        if hasattr(self,"nav"):
            for i in range(self.nav.count()):
                item=self.nav.item(i); original=item.data(Qt.UserRole+1)
                if original is None: original=item.text(); item.setData(Qt.UserRole+1,original)
                item.setText(mapping.get(original, original))
        if hasattr(self,"start"):
            self.start.setText(mapping.get("▶  УВІМКНУТИ КАМЕРУ" if not self.active else "■  ЗУПИНИТИ КАМЕРУ", "▶  УВІМКНУТИ КАМЕРУ" if not self.active else "■  ЗУПИНИТИ КАМЕРУ"))
        if hasattr(self, "tray"):
            self.tray.setToolTip(mapping.get("Керування жестами", "Керування жестами"))
        if hasattr(self, "tray_menu"):
            for action, original in ((self.tray_show,"Відкрити"),(self.tray_hide,"Згорнути в трей"),(self.tray_stop,"Зупинити камеру"),(self.tray_quit,"Вийти")):
                action.setText(mapping.get(original, original))

    def tr(self, text):
        return getattr(self, "_active_mapping", {}).get(text, text)

    def setup_tray(self):
        self.tray=QSystemTrayIcon(self); self.tray.setIcon(self.style().standardIcon(QStyle.SP_ComputerIcon)); self.tray.setToolTip("Керування жестами")
        self.tray_menu=QMenu(); self.tray_show=QAction("Відкрити",self); self.tray_show.triggered.connect(self.restore_window); self.tray_menu.addAction(self.tray_show)
        self.tray_hide=QAction("Згорнути в трей",self); self.tray_hide.triggered.connect(self.hide); self.tray_menu.addAction(self.tray_hide)
        self.tray_stop=QAction("Зупинити камеру",self); self.tray_stop.triggered.connect(self.stop_camera); self.tray_menu.addAction(self.tray_stop)
        self.tray_menu.addSeparator(); self.tray_quit=QAction("Вийти",self); self.tray_quit.triggered.connect(self.quit_app); self.tray_menu.addAction(self.tray_quit)
        self.tray.setContextMenu(self.tray_menu); self.tray.activated.connect(self.tray_activated); self.tray.show()
        self.apply_language()
    def tray_activated(self,reason):
        if reason in (QSystemTrayIcon.Trigger,QSystemTrayIcon.DoubleClick): self.restore_window()
    def restore_window(self): self.showNormal(); self.raise_(); self.activateWindow()
    def quit_app(self):
        self.allow_quit=True
        self.tray.hide()
        self.stop_camera()
        thread = self._shutdown_thread
        if thread is not None:
            thread.join(timeout=2.0)
        QApplication.quit()

    def save(self):
        self.save_config_silent()

    def copy_config(self):
        try:
            # Копіюємо актуальні значення, навіть якщо користувач ще не натиснув «Зберегти».
            current = dict(self.cfg)
            current.update({"camera_index":self.cam.currentIndex(), "camera_resolution":self.resolution.currentData(),
                            "night_vision":self.night.isChecked(),
                            "gamma":self.gamma[1].value()/10,
                            "brightness":self.brightness[1].value(),
                            "contrast":self.contrast[1].value()/10,"language":self.language.currentData(),"mirror_camera":self.mirror_camera.isChecked(),"face_presence_gate":self.face_gate.isChecked(),"face_tracking_enabled":self.face_tracking.isChecked(), "hand_overlay_enabled":self.hand_overlay.isChecked(), "virtual_camera_enabled":self.virtual_camera_enabled.isChecked(), "hand_overlay_hold_seconds":self.hand_overlay_hold.value()})
            current["bindings"] = {}
            for key,(action,target,hold) in self.binding_widgets.items():
                selected=action.currentData()
                if isinstance(selected,str) and selected.startswith("script|"):
                    current["bindings"][key]={"action":"script","target":selected.split("|",1)[1],"hold_seconds":hold.value()}
                else:
                    current["bindings"][key]={"action":selected,"target":target.text().strip(),"hold_seconds":hold.value()}
            QApplication.clipboard().setText(json.dumps(current, indent=2, ensure_ascii=False))
            QMessageBox.information(self, self.tr("Конфіг"), self.tr("Конфігурацію скопійовано в буфер обміну."))
        except Exception as e:
            QMessageBox.warning(self, self.tr("Помилка"), self.tr("Не вдалося скопіювати конфігурацію: ")+str(e))

    def save_bindings(self):
        self.save_config_silent()

    def update_face_detector(self, *_):
        if not self.active:
            return
        needed=self.face_tracking.isChecked() or self.face_gate.isChecked()
        try:
            if needed and self.face_detector is None:
                face_base=python.BaseOptions(model_asset_path=str(get_face_model()))
                face_opts=vision.FaceDetectorOptions(base_options=face_base, running_mode=vision.RunningMode.VIDEO, min_detection_confidence=0.78)
                self.face_detector=vision.FaceDetector.create_from_options(face_opts)
                self.face_history.clear()
            elif not needed and self.face_detector is not None:
                self.face_detector.close()
                self.face_detector=None
                self.face_history.clear()
        except Exception as e:
            self.event.setText(self.tr("Помилка налаштування детекції обличчя: ")+str(e))

    def stop_camera(self):
        """Stop the capture loop immediately; release native resources off the UI thread."""
        self.timer.stop()
        cap, virtual_cam = self.cap, self.virtual_cam
        tracker, face_detector = self.tracker, self.face_detector
        self.cap = None
        self.virtual_cam = None
        self.tracker = None
        self.face_detector = None
        self.virtual_camera_size = None
        self.active = False

        # DirectShow, Unity Capture and MediaPipe may block while releasing native
        # resources. Never close them synchronously in the Qt GUI thread.
        resources = (cap, virtual_cam, tracker, face_detector)
        def release_resources():
            for resource, method in zip(resources, ("release", "close", "close", "close")):
                if resource is not None:
                    try:
                        getattr(resource, method)()
                    except Exception:
                        pass
        self._shutdown_thread = threading.Thread(target=release_resources, name="CameraShutdown", daemon=True)
        self._shutdown_thread.start()
        if hasattr(self,"start"):
            self.start.setText(self.tr("▶  УВІМКНУТИ КАМЕРУ"))
            self.start.setEnabled(False)
        if hasattr(self,"status"):
            self.status.setText(self.tr("● НЕАКТИВНО")); self.status.setStyleSheet("color:#ff8094;font-weight:700")
        if hasattr(self,"event"):
            self.event.setText(self.tr("Завершення роботи камери…"))
        QTimer.singleShot(100, self._finish_camera_shutdown)

    def _finish_camera_shutdown(self):
        thread = self._shutdown_thread
        if thread is not None and thread.is_alive():
            QTimer.singleShot(100, self._finish_camera_shutdown)
            return
        self._shutdown_thread = None
        if hasattr(self, "start"):
            self.start.setEnabled(True)
        if hasattr(self, "event"):
            self.event.setText(self.tr("Камеру зупинено"))

    def toggle(self):
        if self.active:
            self.stop_camera()
            self.preview.setText(self.tr("КАМЕРУ ЗУПИНЕНО"))
            return
        if self._shutdown_thread is not None and self._shutdown_thread.is_alive():
            self.event.setText(self.tr("Зачекай завершення зупинки камери…"))
            return
        try:
            self.track.setText(self.tr("ЗАВАНТАЖЕННЯ МОДЕЛІ…")); base=python.BaseOptions(model_asset_path=str(get_model()))
            opts=vision.HandLandmarkerOptions(
                base_options=base, running_mode=vision.RunningMode.VIDEO, num_hands=2,
                min_hand_detection_confidence=0.42,
                min_hand_presence_confidence=0.42,
                min_tracking_confidence=0.42)
            self.tracker=vision.HandLandmarker.create_from_options(opts)
            self.face_detector = None
            if self.face_tracking.isChecked() or self.face_gate.isChecked():
                face_base = python.BaseOptions(model_asset_path=str(get_face_model()))
                face_opts = vision.FaceDetectorOptions(base_options=face_base, running_mode=vision.RunningMode.VIDEO, min_detection_confidence=0.78)
                self.face_detector = vision.FaceDetector.create_from_options(face_opts)
            self.cap=cv2.VideoCapture(self.cfg.get("camera_index",0),cv2.CAP_DSHOW)
            try: width, height = map(int, self.resolution.currentData().split("x"))
            except Exception: width, height = 1280, 720
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT,height)
            self.cap.set(cv2.CAP_PROP_FPS,30)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE,1)
            if not self.cap.isOpened(): raise RuntimeError(self.tr("Не вдалося відкрити камеру. Перевір дозвіл Windows або вибери іншу камеру."))
            if self.virtual_camera_enabled.isChecked():
                try:
                    import pyvirtualcam
                    self.virtual_camera_size = (width, height)
                    self.virtual_frames_sent = 0
                    self.virtual_cam = pyvirtualcam.Camera(width=width, height=height, fps=30, fmt=pyvirtualcam.PixelFormat.RGB, device="Streamer Camera Control", backend="unitycapture")
                    self.event.setText(f"Virtual camera ({self.virtual_cam.backend}): {self.virtual_cam.device} · {width}×{height} RGB")
                except Exception as vc_error:
                    raise RuntimeError(self.tr("Не вдалося запустити віртуальну камеру Streamer Camera Control. Запусти INSTALL.bat від імені адміністратора або перевстанови драйвер. Деталі: ") + str(vc_error))
            self.face_history.clear(); self.last_hand_overlay=None; self.active=True; self.start.setText(self.tr("■  ЗУПИНИТИ КАМЕРУ")); self.status.setText(self.tr("● ПРАЦЮЄ")); self.status.setStyleSheet("color:#63f2b2;font-weight:700"); self.timer.start(33 if self.virtual_cam is not None else 25)
        except Exception as e:
            if self.cap: self.cap.release()
            self.cap=None
            if self.virtual_cam is not None:
                try: self.virtual_cam.close()
                except Exception: pass
                self.virtual_cam=None
            self.virtual_camera_size=None
            QMessageBox.critical(self,self.tr("Помилка запуску камери"),str(e)); self.track.setText(self.tr("ПОМИЛКА ІНІЦІАЛІЗАЦІЇ"))

    def frame(self):
        try:
            self._frame_impl()
        except Exception as e:
            self.timer.stop()
            self.track.setText(self.tr("ПОМИЛКА ОБРОБКИ КАДРУ: ")+str(e))
            self.event.setText(self.tr("Камеру зупинено через помилку, щоб уникнути зависання."))
            self.stop_camera()
            self.start.setText(self.tr("▶  УВІМКНУТИ КАМЕРУ"))
            self.status.setText(self.tr("● ПОМИЛКА"))
            self.status.setStyleSheet("color:#ff8094;font-weight:700")

    def _frame_impl(self):
        if self.cap is None or self.tracker is None: return
        ok,frame=self.cap.read()
        if not ok: self.track.setText(self.tr("НЕМАЄ КАДРУ З КАМЕРИ")); return
        if self.cfg.get("mirror_camera", True):
            frame=cv2.flip(frame,1)
        if self.cfg.get("night_vision",False):
            gamma=max(.1,float(self.cfg.get("gamma",1.5))); lut=np.array([(i/255.0)**(1.0/gamma)*255 for i in range(256)],dtype="uint8")
            frame=cv2.LUT(frame,lut); frame=cv2.convertScaleAbs(frame,alpha=float(self.cfg.get("contrast",1.25)),beta=int(self.cfg.get("brightness",12)))
        # Модель обробляє зменшену копію: відео лишається у вибраній якості,
        # а нейромережа отримує 640 px по ширині для значно меншого навантаження.
        fh, fw = frame.shape[:2]
        infer_w = min(640, fw)
        infer_h = max(1, int(fh * infer_w / fw))
        small = cv2.resize(frame, (infer_w, infer_h), interpolation=cv2.INTER_AREA) if infer_w != fw else frame
        rgb=cv2.cvtColor(small,cv2.COLOR_BGR2RGB)
        timestamp_ms = max(int(time.monotonic()*1000), self.last_timestamp_ms + 1); self.last_timestamp_ms = timestamp_ms
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB,data=rgb)
        result=self.tracker.detect_for_video(mp_image,timestamp_ms)
        # Якщо gate увімкнений, відсутність доступного детектора не повинна
        # дозволяти жести за замовчуванням.
        face_present = not self.face_gate.isChecked()
        if self.face_detector is not None:
            face_result = self.face_detector.detect_for_video(mp_image, timestamp_ms)
            valid_faces = []
            for face in face_result.detections:
                bb = face.bounding_box
                score = face.categories[0].score if face.categories else 0.0
                # Відсікаємо слабкі збіги та занадто маленькі/витягнуті області,
                # які часто виникають на малюнках, руках, шиї та шумі в темряві.
                ratio = bb.width / max(1, bb.height)
                if score < 0.78 or min(bb.width, bb.height) < 46 or not (0.58 <= ratio <= 1.65):
                    continue
                valid_faces.append(face)
            self.face_history.append(bool(valid_faces))
            # Рамка з'являється лише після підтвердження у 2 з останніх 3 кадрів.
            face_present = sum(self.face_history) >= 2
            if self.face_tracking.isChecked() and face_present and valid_faces:
                for face_index, face in enumerate(valid_faces, 1):
                    bb = face.bounding_box
                    sx, sy = fw / infer_w, fh / infer_h
                    x1, y1 = max(0, int(bb.origin_x*sx)), max(0, int(bb.origin_y*sy))
                    x2, y2 = min(fw-1, int((bb.origin_x+bb.width)*sx)), min(fh-1, int((bb.origin_y+bb.height)*sy))
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (80,210,255), 3, cv2.LINE_AA)
                    cv2.putText(frame, f"FACE {face_index}", (x1, max(22,y1-8)), cv2.FONT_HERSHEY_SIMPLEX, .65, (80,210,255), 2, cv2.LINE_AA)
        detected={}; h,w=frame.shape[:2]
        overlay_items=[]
        if result.hand_landmarks:
            for idx,hand_lm in enumerate(result.hand_landmarks):
                score = 0.0
                if result.handedness and idx < len(result.handedness) and result.handedness[idx]:
                    score = float(result.handedness[idx][0].score)
                xs = [p.x for p in hand_lm]; ys = [p.y for p in hand_lm]
                box_w, box_h = max(xs)-min(xs), max(ys)-min(ys)
                side="RIGHT"
                if result.handedness and idx < len(result.handedness) and result.handedness[idx]:
                    side=result.handedness[idx][0].category_name.upper()
                    if side == "LEFT": side = "RIGHT"
                    elif side == "RIGHT": side = "LEFT"
                    else: side = "RIGHT"
                g=detect_gesture(hand_lm)
                pts=[(int(p.x*w),int(p.y*h)) for p in hand_lm]
                overlay_items.append((pts,side,g))
                # Візуалізація навмисно чутливіша за запуск команд:
                # показуємо знайдену модель руки навіть при частковому перекритті тілом.
                if score >= 0.48 and max(box_w, box_h) >= 0.08 and min(box_w, box_h) >= 0.018:
                    detected[side]=g
            if overlay_items:
                self.last_hand_overlay = (time.monotonic(), overlay_items)
        elif getattr(self, "last_hand_overlay", None):
            # Залишаємо скелет руки ще на 0,5 с, щоб трекінг не миготів
            # при коротких втратах руки за корпусом або під час руху.
            stamp, cached = self.last_hand_overlay
            if time.monotonic() - stamp <= float(self.cfg.get("hand_overlay_hold_seconds", 0.5)):
                overlay_items = cached
        if self.hand_overlay.isChecked():
          for pts,side,g in overlay_items:
            for a,b in LINKS:
                cv2.line(frame,pts[a],pts[b],(175,185,190),2,cv2.LINE_AA)
            for j,(x,y) in enumerate(pts):
                if j == 0:
                    cv2.circle(frame,(x,y),10,(40,220,255),-1,cv2.LINE_AA)
                    cv2.circle(frame,(x,y),13,(245,245,245),2,cv2.LINE_AA)
                elif j in (4,8,12,16,20):
                    cv2.circle(frame,(x,y),7,(70,235,130),-1,cv2.LINE_AA)
                    cv2.circle(frame,(x,y),9,(245,245,245),1,cv2.LINE_AA)
                else:
                    cv2.circle(frame,(x,y),3,(225,230,235),-1,cv2.LINE_AA)
            cv2.putText(frame,f"{self.tr('ЛІВА' if side=='LEFT' else 'ПРАВА')}: {self.tr(GESTURES.get(g,g))}",
                        (max(8,pts[0][0]-40),max(24,pts[0][1]-18)),
                        cv2.FONT_HERSHEY_SIMPLEX,.48,(80,230,255),1,cv2.LINE_AA)
        self.track.setText(self.tr("ЗНАЙДЕНО РУК: ")+str(len(result.hand_landmarks)) if result.hand_landmarks else self.tr("ПОШУК РУКИ…"))
        if self.face_gate.isChecked() and not face_present:
            detected = {}
            self.track.setText(self.tr("ОБЛИЧЧЯ НЕ ВИЯВЛЕНО — ОЧІКУЮ"))
        elif self.face_detector is not None and face_present:
            self.track.setText(self.tr("ОБЛИЧЧЯ:")+" "+str(len(face_result.detections))+self.tr(" · ЖЕСТИ АКТИВНІ"))
        self.process_detected(detected)
        if detected:
            self.handval.setText(" · ".join(self.tr(HANDS[k]) for k in detected)); self.gval.setText(" / ".join(self.tr(GESTURES.get(v,v)) for v in detected.values()))
        else: self.handval.setText("—"); self.gval.setText("—")
        # Важливо: показуємо RGB, а не BGR — це прибирає синій відтінок.
        display=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB); hh,ww,ch=display.shape
        if self.virtual_cam is not None:
            self._send_virtual_frame(display)
        q=QImage(display.data,ww,hh,ch*ww,QImage.Format_RGB888)
        if self.isVisible():
            self.preview.setPixmap(QPixmap.fromImage(q.copy()).scaled(self.preview.size(),Qt.KeepAspectRatio,Qt.FastTransformation))

    def _send_virtual_frame(self, rgb_frame):
        """Send one valid, contiguous RGB uint8 frame at the exact Unity Capture size."""
        if self.virtual_cam is None:
            return
        if self.virtual_camera_size is None:
            raise RuntimeError("Virtual camera output size is not initialized")
        out_w, out_h = self.virtual_camera_size
        if rgb_frame is None or rgb_frame.ndim != 3 or rgb_frame.shape[2] != 3:
            raise ValueError("Virtual camera expects an RGB image with 3 channels")
        if rgb_frame.dtype != np.uint8:
            rgb_frame = np.clip(rgb_frame, 0, 255).astype(np.uint8)
        if rgb_frame.shape[1] != out_w or rgb_frame.shape[0] != out_h:
            rgb_frame = cv2.resize(rgb_frame, (out_w, out_h), interpolation=cv2.INTER_AREA)
        # pyvirtualcam expects H×W×3 uint8 RGB; make the buffer contiguous explicitly.
        rgb_frame = np.ascontiguousarray(rgb_frame, dtype=np.uint8)
        # Unity Capture needs a paced frame producer. send() transfers the RGB
        # buffer; sleep_until_next_frame() keeps the stream alive at the device FPS
        # instead of letting the UI timer overrun the capture filter.
        self.virtual_cam.send(rgb_frame)
        self.virtual_cam.sleep_until_next_frame()
        self.virtual_frames_sent += 1
        if self.virtual_frames_sent == 1 or self.virtual_frames_sent % 30 == 0:
            self.event.setText(
                f"Virtual camera ({self.virtual_cam.backend}): "
                f"{self.virtual_cam.device} · {self.virtual_frames_sent} frames sent"
            )

    def process_detected(self,detected):
        """Виконує прив'язку лише після стабільного жесту; одна дія на одне утримання."""
        now=time.monotonic(); candidates=[]
        for side,g in detected.items():
            if g != "UNKNOWN": candidates.append((f"{side}:{g}",side,g))
        if len(detected)==2 and len(set(detected.values()))==1:
            g=next(iter(detected.values()))
            if g != "UNKNOWN": candidates.append((f"BOTH:{g}","BOTH",g))

        # Якщо для одночасного жесту обома руками задана дія, вона має пріоритет
        # над двома окремими прив'язками цього самого жесту.
        both_active={g for key,side,g in candidates
                     if side=="BOTH" and self.cfg.get("bindings",{}).get(f"BOTH:{g}",{}).get("action","none")!="none"}
        if both_active:
            candidates=[item for item in candidates if item[1]=="BOTH" or item[2] not in both_active]

        current_keys={key for key,_,_ in candidates}
        # Скидаємо таймер, коли жест зник або змінився.
        for key in list(self.gesture_since):
            if key not in current_keys:
                self.gesture_since.pop(key,None); self.fired_gestures.discard(key)

        cooldown=float(self.cfg.get("cooldown_seconds",1.2))
        for key,side,g in candidates:
            self.gesture_since.setdefault(key,now)
            binding=self.cfg.get("bindings",{}).get(f"{side}:{g}",{})
            try: hold=max(0.05, min(5.0, float(binding.get("hold_seconds",0.35))))
            except (TypeError, ValueError): hold=0.35
            if key in self.fired_gestures or now-self.gesture_since[key] < hold:
                continue
            action=binding.get("action","none"); target=binding.get("target","").strip()
            if action == "none" or not target:
                continue
            if now-self.last_actions.get(key,0) < cooldown:
                continue
            try:
                if action=="key":
                    keys=[x.strip().lower() for x in target.split("+") if x.strip()]
                    if not keys: continue
                    pyautogui.hotkey(*keys)
                elif action=="text":
                    pyautogui.write(target,interval=0.02)
                elif action=="url":
                    url = target.strip()
                    if not url.lower().startswith(("http://", "https://", "mailto:")):
                        url = "https://" + url
                    if sys.platform == "win32": os.startfile(url)
                    else:
                        opened = webbrowser.open(url, new=2)
                        if not opened: raise RuntimeError(self.tr("Не вдалося відкрити посилання у браузері за замовчуванням."))
                elif action=="script":
                    self.launch_script_path(target)
                else:
                    continue
                self.last_actions[key]=now
                self.fired_gestures.add(key)
                self.event.setText(f"{self.tr(HANDS[side])} · {self.tr(GESTURES.get(g,g))}{self.tr(' → виконано: ')}{action} ({target})")
            except Exception as e:
                self.fired_gestures.add(key)
                self.event.setText(self.tr("Помилка виконання прив'язки: ")+str(e))

    def closeEvent(self,event):
        if self.allow_quit:
            self.stop_camera()
            event.accept()
        else:
            event.ignore(); self.hide()

def log_unhandled(exc_type, exc_value, exc_tb):
    import traceback
    try:
        with (ROOT / "error.log").open("a", encoding="utf-8") as log:
            log.write("\n" + time.strftime("%Y-%m-%d %H:%M:%S") + "\n")
            traceback.print_exception(exc_type, exc_value, exc_tb, file=log)
    except Exception:
        pass
    sys.__excepthook__(exc_type, exc_value, exc_tb)

if __name__=="__main__":
    sys.excepthook = log_unhandled
    app=QApplication(sys.argv); app.setQuitOnLastWindowClosed(False); win=Main(); win.show(); sys.exit(app.exec())
