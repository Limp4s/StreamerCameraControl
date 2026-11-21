import sys, json, time, urllib.request, webbrowser, os, subprocess
from collections import deque
from pathlib import Path
import cv2, mediapipe as mp
cv2.setNumThreads(2)
import numpy as np
import pyautogui
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QImage, QPixmap, QAction
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
    "camera_resolution": "1280x720", "face_presence_gate": False, "face_tracking_enabled": True, "inference_width": 640, "hand_overlay_hold_seconds": 0.5, "bindings": {}, "scripts": []
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

def get_model():
    MODEL.parent.mkdir(exist_ok=True)
    if not MODEL.exists(): urllib.request.urlretrieve(MODEL_URL, MODEL)
    return MODEL

def get_face_model():
    FACE_MODEL.parent.mkdir(exist_ok=True)
    if not FACE_MODEL.exists(): urllib.request.urlretrieve(FACE_MODEL_URL, FACE_MODEL)
    return FACE_MODEL

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
        self.setWindowTitle("Керування жестами")
        self.resize(1020, 740)
        self.cfg = load_config(); self.cap = None; self.tracker = None
        self.active = False; self.current_key = "—"; self.since = 0; self.last_actions = {}; self.face_detector = None; self.last_timestamp_ms = 0
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
        body=QHBoxLayout(); navpanel=QFrame(); navpanel.setObjectName("panel"); navpanel.setFixedWidth(170); nav=QVBoxLayout(navpanel)
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
        self.preview=QLabel("КАМЕРУ НЕ ЗАПУЩЕНО\n\nНатисни «Увімкнути камеру»"); self.preview.setAlignment(Qt.AlignCenter); self.preview.setMinimumSize(500,320); self.preview.setSizePolicy(QSizePolicy.Expanding,QSizePolicy.Expanding)
        self.preview.setStyleSheet("background:#1c1e20;border:1px dashed #62686d;border-radius:9px;color:#b9c0c5"); fl.addWidget(self.preview)
        row.addWidget(f,3)
        right=QVBoxLayout(); self.start=QPushButton("▶  УВІМКНУТИ КАМЕРУ"); self.start.clicked.connect(self.toggle); right.addWidget(self.start)
        st,sl=self.panel("РОЗПІЗНАВАННЯ"); sl.addWidget(QLabel("ПОТОЧНИЙ ЖЕСТ")); self.gval=QLabel("—"); self.gval.setObjectName("metric"); sl.addWidget(self.gval)
        sl.addWidget(QLabel("РУКА")); self.handval=QLabel("—"); sl.addWidget(self.handval); sl.addWidget(QLabel("СТАН ВІДСТЕЖЕННЯ")); self.track=QLabel("ОЧІКУВАННЯ"); sl.addWidget(self.track)
        sl.addWidget(QLabel("ОСТАННЯ ДІЯ")); self.event=QLabel("Дій не виконано"); self.event.setWordWrap(True); sl.addWidget(self.event); right.addWidget(st)
        info,il=self.panel("РОЗПІЗНАВАНІ ЖЕСТИ")
        il.addWidget(QLabel("👍 Великий палець угору\n🤙 Мізинець\n☝ Один палець\n✌ Два пальці\n🤟 Три пальці\n🖖 Чотири пальці\n🖐 Відкрита долоня\n✊ Кулак")); right.addWidget(info); right.addStretch(); row.addLayout(right,2); l.addLayout(row); return w

    def settings(self):
        w=QWidget(); l=QVBoxLayout(w); f,fl=self.panel("КАМЕРА ТА ВИДИМІСТЬ")
        fl.addWidget(QLabel("Мова інтерфейсу")); self.language=NoWheelComboBox(); self.language.addItem("Українська","uk"); self.language.addItem("English","en"); self.language.addItem("Čeština","cs"); self.language.setCurrentIndex(max(0,self.language.findData(self.cfg.get("language","uk")))); self.language.currentIndexChanged.connect(self.apply_language); fl.addWidget(self.language)
        self.autostart=QCheckBox("Запускати разом із Windows"); self.autostart.setChecked(self.autostart_enabled()); self.autostart.toggled.connect(self.set_autostart); fl.addWidget(self.autostart)
        fl.addWidget(QLabel("Пристрій камери")); self.cam=NoWheelComboBox(); self.cam.addItems(["Камера 0","Камера 1","Камера 2"]); self.cam.setCurrentIndex(min(int(self.cfg.get("camera_index",0)),2)); fl.addWidget(self.cam)
        fl.addWidget(QLabel("Роздільна здатність камери")); self.resolution=NoWheelComboBox()
        for label, value in [("1280 × 720 (рекомендовано)","1280x720"),("1920 × 1080 (Full HD)","1920x1080"),("640 × 480 (швидше)","640x480")]: self.resolution.addItem(label,value)
        self.resolution.setCurrentIndex(max(0,self.resolution.findData(self.cfg.get("camera_resolution","1280x720")))); fl.addWidget(self.resolution)
        self.night=QCheckBox("Покращення видимості в темряві"); self.night.setChecked(bool(self.cfg.get("night_vision",False))); fl.addWidget(self.night)
        self.face_tracking=QCheckBox("Показувати рамки навколо виявлених облич")
        self.face_tracking.setChecked(bool(self.cfg.get("face_tracking_enabled", self.cfg.get("face_presence_gate", True))))
        fl.addWidget(self.face_tracking)
        self.face_gate=QCheckBox("Виконувати жести лише коли обличчя є в кадрі")
        self.face_gate.setChecked(bool(self.cfg.get("face_presence_gate",False))); fl.addWidget(self.face_gate)
        fl.addWidget(QLabel("Утримання зображення руки (секунди)")); self.hand_overlay_hold=QDoubleSpinBox(); self.hand_overlay_hold.setRange(0.0,5.0); self.hand_overlay_hold.setSingleStep(0.1); self.hand_overlay_hold.setDecimals(1); self.hand_overlay_hold.setValue(float(self.cfg.get("hand_overlay_hold_seconds",0.5))); fl.addWidget(self.hand_overlay_hold)
        fl.addWidget(QLabel("Гамма")); self.gamma=self.make_slider(5,30,int(float(self.cfg.get("gamma",1.5))*10),"{:.1f}",10); fl.addWidget(self.gamma[0])
        fl.addWidget(QLabel("Яскравість")); self.brightness=self.make_slider(-50,80,int(self.cfg.get("brightness",12)),"{}",1); fl.addWidget(self.brightness[0])
        fl.addWidget(QLabel("Контрастність")); self.contrast=self.make_slider(5,30,int(float(self.cfg.get("contrast",1.25))*10),"{:.1f}",10); fl.addWidget(self.contrast[0])
        hint=QLabel("Обробка впливає лише на зображення в програмі. У повній темряві потрібна ІЧ-камера або ІЧ-підсвічування."); hint.setWordWrap(True); hint.setStyleSheet("color:#c0c6ca;padding:8px;background:#202326;border-radius:8px"); fl.addWidget(hint)
        self.autosave_status=QLabel("Зміни зберігаються автоматично"); self.autosave_status.setStyleSheet("color:#8bd6ae;padding:4px"); fl.addWidget(self.autosave_status)
        copy_cfg=QPushButton("КОПІЮВАТИ КОНФІГ У БУФЕР ОБМІНУ"); copy_cfg.clicked.connect(self.copy_config); fl.addWidget(copy_cfg)
        trayhint=QLabel("✕ ховає програму в трей. Для виходу натисни правою кнопкою на іконку трея → «Вийти»."); trayhint.setWordWrap(True); trayhint.setStyleSheet("color:#c0c6ca;padding:8px"); fl.addWidget(trayhint)
        l.addWidget(f); l.addStretch(); return w

    def bindings_page(self):
        w=QWidget(); outer=QVBoxLayout(w); scroll=QScrollArea(); scroll.setWidgetResizable(True); inner=QWidget(); lay=QVBoxLayout(inner)
        intro=QLabel("Налаштуй окрему команду для кожної руки. Для прив’язки «Обидві руки» жест має бути показаний одночасно двома руками."); intro.setWordWrap(True); intro.setStyleSheet("color:#c0c6ca;padding:8px"); lay.addWidget(intro)
        for hand,hand_name in HANDS.items():
            f,fl=self.panel(hand_name.upper()); grid=QGridLayout(); grid.addWidget(QLabel("ЖЕСТ"),0,0); grid.addWidget(QLabel("ДІЯ"),0,1); grid.addWidget(QLabel("КЛАВІША / ТЕКСТ / URL"),0,2); grid.addWidget(QLabel("УТРИМАННЯ (С)"),0,3)
            for ri,(gesture_key,gesture_name) in enumerate(GESTURES.items(),1):
                grid.addWidget(QLabel(gesture_name),ri,0)
                action=NoWheelComboBox(); action.addItem("Немає дії","none"); action.addItem("Натиснути клавішу","key"); action.addItem("Ввести текст","text"); action.addItem("Відкрити вебсторінку","url")
                target=QLineEdit(); target.setPlaceholderText("Клавіша / текст / URL або вибери скрипт у списку")
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
                action.currentIndexChanged.connect(lambda _=None, cb=action, tb=target: self.binding_action_changed(cb,tb))
                grid.addWidget(action,ri,1); grid.addWidget(target,ri,2); grid.addWidget(hold,ri,3); self.binding_widgets[f"{hand}:{gesture_key}"]=(action,target,hold)
            fl.addLayout(grid); lay.addWidget(f)
        self.bindings_status=QLabel("Прив’язки зберігаються автоматично"); self.bindings_status.setStyleSheet("color:#8bd6ae;padding:4px"); lay.addWidget(self.bindings_status); lay.addStretch(); scroll.setWidget(inner); outer.addWidget(scroll); return w

    def _add_script_actions(self, combo):
        for item in self.cfg.get("scripts", []):
            path=str(item.get("path", "")).strip()
            if path:
                name=item.get("name") or Path(path).stem
                combo.addItem("Скрипт: "+name, "script|"+path)

    def binding_action_changed(self, combo, target):
        data=combo.currentData()
        is_script=isinstance(data, str) and data.startswith("script|")
        target.setReadOnly(is_script)
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
        self.scripts_list=QListWidget(); self.scripts_list.setMinimumHeight(220); fl.addWidget(self.scripts_list)
        row=QHBoxLayout(); add=QPushButton("＋ ДОДАТИ СКРИПТ"); add.clicked.connect(self.add_script); row.addWidget(add)
        run=QPushButton("▶ ЗАПУСТИТИ ВИБРАНИЙ"); run.clicked.connect(self.run_selected_script); row.addWidget(run)
        remove=QPushButton("ВИДАЛИТИ"); remove.clicked.connect(self.remove_selected_script); row.addWidget(remove); fl.addLayout(row)
        note=QLabel("Для AHK потрібен встановлений AutoHotkey або коректна асоціація .ahk у Windows. Скрипти запускаються від імені поточного користувача.")
        note.setWordWrap(True); note.setStyleSheet("color:#c0c6ca;padding:8px"); fl.addWidget(note)
        outer.addWidget(f); outer.addStretch(); self.refresh_scripts_list(); return w

    def connect_autosave_signals(self):
        for widget in (self.cam, self.resolution, self.language):
            widget.currentIndexChanged.connect(self.schedule_autosave)
        for widget in (self.night, self.face_tracking, self.face_gate):
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
            "night_vision":self.night.isChecked(), "gamma":self.gamma[1].value()/10,
            "brightness":self.brightness[1].value(), "contrast":self.contrast[1].value()/10,
            "language":self.language.currentData(), "face_presence_gate":self.face_gate.isChecked(),
            "face_tracking_enabled":self.face_tracking.isChecked(), "hand_overlay_hold_seconds":self.hand_overlay_hold.value()})
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
        path,_=QFileDialog.getOpenFileName(self,"Додати скрипт",str(Path.home()),"Скрипти (*.ahk *.exe *.bat *.cmd)")
        if not path: return
        ext=Path(path).suffix.lower()
        if ext not in (".ahk",".exe",".bat",".cmd"):
            QMessageBox.warning(self,"Формат не підтримується","Дозволені лише AHK, EXE, BAT та CMD."); return
        resolved=str(Path(path).resolve())
        if any(x.get("path","").lower()==resolved.lower() for x in self.cfg.get("scripts",[])):
            QMessageBox.information(self,"Скрипт уже доданий","Цей файл уже є у списку."); return
        self.cfg.setdefault("scripts",[]).append({"name":Path(path).stem,"path":resolved})
        self.refresh_scripts_list(); self.save_config_silent()

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
        if not file.is_file(): raise FileNotFoundError(f"Файл не знайдено: {file}")
        ext=file.suffix.lower()
        if ext not in (".ahk",".exe",".bat",".cmd"):
            raise ValueError("Дозволені лише .ahk, .exe, .bat та .cmd")
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
            QMessageBox.information(self,"Мої скрипти","Спочатку вибери скрипт зі списку."); return
        try:
            item=self.cfg["scripts"][ix]; self.launch_script_path(item["path"])
            self.event.setText("Запущено скрипт: "+item.get("name",Path(item["path"]).name))
        except Exception as e: QMessageBox.warning(self,"Не вдалося запустити",str(e))

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
            QMessageBox.warning(self, "Автозапуск", f"Не вдалося змінити автозапуск: {e}")
            self.autostart.blockSignals(True); self.autostart.setChecked(not enabled); self.autostart.blockSignals(False)

    def apply_language(self, *_):
        if not hasattr(self, "language"): return
        lang = self.language.currentData() or "uk"
        self.cfg["language"] = lang
        translations = {
          "en": {
            "КЕРУВАННЯ ЖЕСТАМИ":"GESTURE CONTROL", "РОЗДІЛИ":"SECTIONS", "Головна":"Home", "Налаштування":"Settings", "Прив’язки команд":"Command bindings", "Мої скрипти":"My scripts",
            "ЗОБРАЖЕННЯ З КАМЕРИ":"CAMERA PREVIEW", "КАМЕРУ НЕ ЗАПУЩЕНО\n\nНатисни «Увімкнути камеру»":"CAMERA NOT STARTED\n\nPress “Start camera”",
            "▶  УВІМКНУТИ КАМЕРУ":"▶  START CAMERA", "■  ЗУПИНИТИ КАМЕРУ":"■  STOP CAMERA", "РОЗПІЗНАВАННЯ":"RECOGNITION", "ПОТОЧНИЙ ЖЕСТ":"CURRENT GESTURE", "РУКА":"HAND", "СТАН ВІДСТЕЖЕННЯ":"TRACKING STATUS", "ОСТАННЯ ДІЯ":"LAST ACTION", "Дій не виконано":"No actions performed", "РОЗПІЗНАВАНІ ЖЕСТИ":"RECOGNIZED GESTURES",
            "Виконувати жести лише коли обличчя є в кадрі":"Require a face in the camera view for gestures", "Показувати рамки навколо виявлених облич":"Show boxes around detected faces", "КАМЕРА ТА ВИДИМІСТЬ":"CAMERA AND VISIBILITY", "Мова інтерфейсу":"Interface language", "Запускати разом із Windows":"Start with Windows", "Пристрій камери":"Camera device", "Роздільна здатність камери":"Camera resolution", "1280 × 720 (рекомендовано)":"1280 × 720 (recommended)", "1920 × 1080 (Full HD)":"1920 × 1080 (Full HD)", "640 × 480 (швидше)":"640 × 480 (faster)", "Камера 0":"Camera 0", "Камера 1":"Camera 1", "Камера 2":"Camera 2", "Покращення видимості в темряві":"Low-light enhancement", "Гамма":"Gamma", "Яскравість":"Brightness", "Контрастність":"Contrast", "Зберегти налаштування":"Save settings", "ЗБЕРЕГТИ НАЛАШТУВАННЯ":"SAVE SETTINGS", "КОПІЮВАТИ КОНФІГ У БУФЕР ОБМІНУ":"COPY CONFIG TO CLIPBOARD", "✕ ховає програму в трей. Для виходу натисни правою кнопкою на іконку трея → «Вийти».":"✕ hides the app to the system tray. To exit, right-click the tray icon → Exit.",
            "ЖЕСТ":"GESTURE", "ДІЯ":"ACTION", "КЛАВІША / ТЕКСТ / URL":"KEY / TEXT / URL", "Немає дії":"No action", "Натиснути клавішу":"Press key", "Ввести текст":"Type text", "Відкрити вебсторінку":"Open website", "Запустити скрипт (шлях до файлу)":"Run script (file path)", "ПОШУК РУКИ…":"SEARCHING FOR HAND…", "ЗНАЙДЕНО РУК: ":"HANDS FOUND: ", "ОЧІКУВАННЯ":"WAITING", "НЕАКТИВНО":"INACTIVE", "● ПРАЦЮЄ":"● RUNNING", "КАМЕРУ ЗУПИНЕНО":"CAMERA STOPPED", "Ліва рука":"Left hand", "Права рука":"Right hand", "Обидві руки":"Both hands",
            "Великий палець угору":"Thumb up", "Мізинець":"Pinky", "Один палець":"One finger", "Два пальці":"Two fingers", "Три пальці":"Three fingers", "Чотири пальці":"Four fingers", "Відкрита долоня":"Open palm", "Кулак":"Fist",
            "Завантаження моделі…":"Loading model…", "ПОМИЛКА ІНІЦІАЛІЗАЦІЇ":"INITIALIZATION ERROR", "КАМЕРУ ЗУПИНЕНО":"CAMERA STOPPED"
          },
          "cs": {
            "КЕРУВАННЯ ЖЕСТАМИ":"OVLÁDÁNÍ GESTY", "РОЗДІЛИ":"SEKCE", "Головна":"Domů", "Налаштування":"Nastavení", "Прив’язки команд":"Přiřazení příkazů", "Мої скрипти":"Moje skripty",
            "ЗОБРАЖЕННЯ З КАМЕРИ":"NÁHLED KAMERY", "КАМЕРУ НЕ ЗАПУЩЕНО\n\nНатисни «Увімкнути камеру»":"KAMERA NENÍ SPUŠTĚNA\n\nStiskni „Spustit kameru“",
            "▶  УВІМКНУТИ КАМЕРУ":"▶  SPUSTIT KAMERU", "■  ЗУПИНИТИ КАМЕРУ":"■  ZASTAVIT KAMERU", "РОЗПІЗНАВАННЯ":"ROZPOZNÁVÁNÍ", "ПОТОЧНИЙ ЖЕСТ":"AKTUÁLNÍ GESTO", "РУКА":"RUKA", "СТАН ВІДСТЕЖЕННЯ":"STAV SLEDOVÁNÍ", "ОСТАННЯ ДІЯ":"POSLEDNÍ AKCE", "Дій не виконано":"Žádné akce", "РОЗПІЗНАВАНІ ЖЕСТИ":"ROZPOZNANÁ GESTA",
            "Виконувати жести лише коли обличчя є в кадрі":"Provádět gesta pouze při detekci obličeje", "Виявляти обличчя та показувати рамки":"Detekovat obličeje a zobrazit rámečky", "КАМЕРА ТА ВИДИМІСТЬ":"KAMERA A VIDITELNOST", "Мова інтерфейсу":"Jazyk rozhraní", "Запускати разом із Windows":"Spouštět při startu Windows", "Пристрій камери":"Kamera", "Роздільна здатність камери":"Rozlišení kamery", "1280 × 720 (рекомендовано)":"1280 × 720 (doporučeno)", "1920 × 1080 (Full HD)":"1920 × 1080 (Full HD)", "640 × 480 (швидше)":"640 × 480 (rychlejší)", "Камера 0":"Kamera 0", "Камера 1":"Kamera 1", "Камера 2":"Kamera 2", "Покращення видимості в темряві":"Zlepšení viditelnosti ve tmě", "Гамма":"Gama", "Яскравість":"Jas", "Контрастність":"Kontrast", "ЗБЕРЕГТИ НАЛАШТУВАННЯ":"ULOŽIT NASTAVENÍ", "КОПІЮВАТИ КОНФІГ У БУФЕР ОБМІНУ":"KOPÍROVAT KONFIGURACI", "ЖЕСТ":"GESTO", "ДІЯ":"AKCE", "КЛАВІША / ТЕКСТ / URL":"KLÁVESA / TEXT / URL", "Немає дії":"Žádná akce", "Натиснути клавішу":"Stisknout klávesu", "Ввести текст":"Napsat text", "Відкрити вебсторінку":"Otevřít web", "Запустити скрипт (шлях до файлу)":"Spustit skript (cesta k souboru)", "ПОШУК РУКИ…":"HLEDÁNÍ RUKY…", "ОЧІКУВАННЯ":"ČEKÁNÍ", "НЕАКТИВНО":"NEAKTIVNÍ", "● ПРАЦЮЄ":"● BĚŽÍ", "КАМЕРУ ЗУПИНЕНО":"KAMERA ZASTAVENA", "Ліва рука":"Levá ruka", "Права рука":"Pravá ruka", "Обидві руки":"Obě ruce",
            "Великий палець угору":"Palec nahoru", "Мізинець":"Malíček", "Один палець":"Jeden prst", "Два пальці":"Dva prsty", "Три пальці":"Tři prsty", "Чотири пальці":"Čtyři prsty", "Відкрита долоня":"Otevřená dlaň", "Кулак":"Pěst",
            "Завантаження моделі…":"Načítání modelu…", "ПОМИЛКА ІНІЦІАЛІЗАЦІЇ":"CHYBA INICIALIZACE"
          }
        }
        mapping=translations.get(lang,{})
        for widget in self.findChildren(QLabel)+self.findChildren(QPushButton)+self.findChildren(QCheckBox):
            original=widget.property("gc_original_text")
            if original is None:
                original=widget.text(); widget.setProperty("gc_original_text",original)
            widget.setText(mapping.get(original, original))
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

    def setup_tray(self):
        self.tray=QSystemTrayIcon(self); self.tray.setIcon(self.style().standardIcon(QStyle.SP_ComputerIcon)); self.tray.setToolTip("Керування жестами")
        menu=QMenu(); show=QAction("Відкрити",self); show.triggered.connect(self.restore_window); menu.addAction(show)
        hide=QAction("Згорнути в трей",self); hide.triggered.connect(self.hide); menu.addAction(hide); menu.addSeparator()
        quit_action=QAction("Вийти",self); quit_action.triggered.connect(self.quit_app); menu.addAction(quit_action)
        self.tray.setContextMenu(menu); self.tray.activated.connect(self.tray_activated); self.tray.show()
    def tray_activated(self,reason):
        if reason in (QSystemTrayIcon.Trigger,QSystemTrayIcon.DoubleClick): self.restore_window()
    def restore_window(self): self.showNormal(); self.raise_(); self.activateWindow()
    def quit_app(self):
        self.allow_quit=True; self.tray.hide(); self.stop_camera()
        if self.tracker: self.tracker.close()
        if self.face_detector: self.face_detector.close()
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
                            "contrast":self.contrast[1].value()/10,"language":self.language.currentData(),"face_presence_gate":self.face_gate.isChecked(),"face_tracking_enabled":self.face_tracking.isChecked(), "hand_overlay_hold_seconds":self.hand_overlay_hold.value()})
            current["bindings"] = {}
            for key,(action,target,hold) in self.binding_widgets.items():
                selected=action.currentData()
                if isinstance(selected,str) and selected.startswith("script|"):
                    current["bindings"][key]={"action":"script","target":selected.split("|",1)[1],"hold_seconds":hold.value()}
                else:
                    current["bindings"][key]={"action":selected,"target":target.text().strip(),"hold_seconds":hold.value()}
            QApplication.clipboard().setText(json.dumps(current, indent=2, ensure_ascii=False))
            QMessageBox.information(self, "Конфіг", "Конфігурацію скопійовано в буфер обміну.")
        except Exception as e:
            QMessageBox.warning(self, "Помилка", f"Не вдалося скопіювати конфігурацію: {e}")

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
            self.event.setText("Помилка налаштування детекції обличчя: "+str(e))

    def stop_camera(self):
        self.timer.stop()
        if self.cap: self.cap.release()
        self.cap=None; self.active=False
        if self.tracker:
            self.tracker.close(); self.tracker=None
        if self.face_detector:
            self.face_detector.close(); self.face_detector=None
        if hasattr(self,"start"): self.start.setText("▶  УВІМКНУТИ КАМЕРУ")
        if hasattr(self,"status"): self.status.setText("● НЕАКТИВНО"); self.status.setStyleSheet("color:#ff8094;font-weight:700")
    def toggle(self):
        if self.active: self.stop_camera(); self.preview.setText("КАМЕРУ ЗУПИНЕНО"); return
        try:
            self.track.setText("ЗАВАНТАЖЕННЯ МОДЕЛІ…"); base=python.BaseOptions(model_asset_path=str(get_model()))
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
            if not self.cap.isOpened(): raise RuntimeError("Не вдалося відкрити камеру. Перевір дозвіл Windows або вибери іншу камеру.")
            self.face_history.clear(); self.last_hand_overlay=None; self.active=True; self.start.setText("■  ЗУПИНИТИ КАМЕРУ"); self.status.setText("● ПРАЦЮЄ"); self.status.setStyleSheet("color:#63f2b2;font-weight:700"); self.timer.start(25)
        except Exception as e:
            if self.cap: self.cap.release()
            self.cap=None; QMessageBox.critical(self,"Помилка запуску камери",str(e)); self.track.setText("ПОМИЛКА ІНІЦІАЛІЗАЦІЇ")

    def frame(self):
        try:
            self._frame_impl()
        except Exception as e:
            self.timer.stop()
            self.track.setText("ПОМИЛКА ОБРОБКИ КАДРУ: "+str(e))
            self.event.setText("Камеру зупинено через помилку, щоб уникнути зависання.")
            if self.cap: self.cap.release()
            self.cap=None; self.active=False
            if self.tracker: self.tracker.close(); self.tracker=None
            if self.face_detector: self.face_detector.close(); self.face_detector=None
            self.start.setText("▶  УВІМКНУТИ КАМЕРУ")
            self.status.setText("● ПОМИЛКА")
            self.status.setStyleSheet("color:#ff8094;font-weight:700")

    def _frame_impl(self):
        if self.cap is None or self.tracker is None: return
        ok,frame=self.cap.read()
        if not ok: self.track.setText("НЕМАЄ КАДРУ З КАМЕРИ"); return
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
            cv2.putText(frame,f"{'ЛІВА' if side=='LEFT' else 'ПРАВА'}: {GESTURES.get(g,g)}",
                        (max(8,pts[0][0]-40),max(24,pts[0][1]-18)),
                        cv2.FONT_HERSHEY_SIMPLEX,.48,(80,230,255),1,cv2.LINE_AA)
        self.track.setText("ЗНАЙДЕНО РУК: "+str(len(result.hand_landmarks)) if result.hand_landmarks else "ПОШУК РУКИ…")
        if self.face_gate.isChecked() and not face_present:
            detected = {}
            self.track.setText("ОБЛИЧЧЯ НЕ ВИЯВЛЕНО — ОЧІКУЮ")
        elif self.face_detector is not None and face_present:
            self.track.setText("ОБЛИЧЧЯ: "+str(len(face_result.detections))+" · ЖЕСТИ АКТИВНІ")
        self.process_detected(detected)
        if detected:
            self.handval.setText(" · ".join(HANDS[k] for k in detected)); self.gval.setText(" / ".join(GESTURES.get(v,v) for v in detected.values()))
        else: self.handval.setText("—"); self.gval.setText("—")
        # Важливо: показуємо RGB, а не BGR — це прибирає синій відтінок.
        display=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB); hh,ww,ch=display.shape
        q=QImage(display.data,ww,hh,ch*ww,QImage.Format_RGB888)
        self.preview.setPixmap(QPixmap.fromImage(q.copy()).scaled(self.preview.size(),Qt.KeepAspectRatio,Qt.FastTransformation))

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
                        if not opened: raise RuntimeError("Не вдалося відкрити посилання у браузері за замовчуванням.")
                elif action=="script":
                    self.launch_script_path(target)
                else:
                    continue
                self.last_actions[key]=now
                self.fired_gestures.add(key)
                self.event.setText(f"{HANDS[side]} · {GESTURES.get(g,g)} → виконано: {action} ({target})")
            except Exception as e:
                self.fired_gestures.add(key)
                self.event.setText("Помилка виконання прив'язки: "+str(e))

    def closeEvent(self,event):
        if self.allow_quit:
            self.stop_camera()
            if self.tracker: self.tracker.close()
            if self.face_detector: self.face_detector.close()
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
