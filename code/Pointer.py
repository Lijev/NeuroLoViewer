import json
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import os
from PIL import Image, ImageTk
import time
import bisect
import cv2
import numpy as np
from moviepy.editor import ImageSequenceClip
import threading
import sys
import configparser

class PointCreator:
    def __init__(self, root):
        self.root = root
        self.root.title("Point Coordinates Creator")
        
        # Configuration
        self.CONFIG_FILE = "config_pointer.ini"
        self.config = configparser.ConfigParser()
        self.load_config()
        
        # Загрузка переводов
        self.translations = self.load_default_translations()
        
        # Установка иконки
        try:
            icon_path = self.resource_path("lo.ico")
            if os.path.exists(icon_path):
                self.root.iconbitmap(icon_path)
        except:
            pass
        
        # Настройка полноэкранного режима
        self.is_fullscreen = False
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        window_width = int(screen_width * 0.95)
        window_height = int(screen_height * 0.8)
        x_pos = (screen_width - window_width) // 2
        y_pos = (screen_height - window_height) // 2
        self.root.geometry(f"{window_width}x{window_height}+{x_pos}+{y_pos}")
        
        # Стилизация
        self.setup_styles()
        
        # Загрузка списка сезонов
        self.seasons_list = self.load_seasons()
        
        # Переменные для хранения данных
        self.season_var = tk.StringVar(value=self.seasons_list[0] if self.seasons_list else self.translations.get("NEW_GENERATION", "Новое поколение"))
        self.episode_var = tk.StringVar(value="1")
        self.moment_var = tk.DoubleVar(value=50)
        
        # Переменные для времени
        self.hours_var = tk.StringVar(value="00")
        self.minutes_var = tk.StringVar(value="10")
        self.seconds_var = tk.StringVar(value="00")
        self.length_hours_var = tk.StringVar(value="00")
        self.length_minutes_var = tk.StringVar(value="20")
        self.length_seconds_var = tk.StringVar(value="00")
        
        # Переменные для анимации
        self.is_playing = False
        self.animation_start_time = 0
        self.start_moment = 0  # Момент начала воспроизведения
        self.episode_data = {"moments": [], "quartets": []}  # Новый формат для LVP
        self.current_points = [0.5, 0.5, 0.5, 0.5]  # Текущие позиции точек (x1, y1, x2, y2)
        
        # Переменные для создания MP4
        self.video_hours_var = tk.StringVar(value="0")
        self.video_minutes_var = tk.StringVar(value="2")
        self.video_seconds_var = tk.StringVar(value="0")
        self.video_path_var = tk.StringVar(value="")
        
        # Переменная для хранения загруженной капсулы
        self.loaded_capsule = None
        
        # Загрузка существующих данных
        try:
            with open(self.resource_path('IDS.json'), 'r') as f:
                self.data = json.load(f)
        except FileNotFoundError:
            self.data = {"X": [], "Y": []}
        
        self.create_widgets()
        self.load_background_images()
        
        # Проверка аргументов командной строки
        if len(sys.argv) > 1 and sys.argv[1].endswith('.lvp'):
            self.load_capsule(sys.argv[1])
    
    def resource_path(self, relative_path):
        """Get absolute path to resource, works for dev and for PyInstaller"""
        try:
            base_path = sys._MEIPASS
        except Exception:
            base_path = os.path.abspath("../data/")
        return os.path.join(base_path, relative_path)
    
    def load_config(self):
        """Load configuration from file"""
        if os.path.exists(self.CONFIG_FILE):
            self.config.read(self.CONFIG_FILE)
        else:
            self.config['Settings'] = {}
            self.config['Paths'] = {}
            self.save_config()
    
    def save_config(self):
        """Save configuration to file"""
        with open(self.CONFIG_FILE, 'w') as configfile:
            self.config.write(configfile)
    
    def load_default_translations(self):
        """Load default English translations"""
        return {
            # Основные заголовки
            "MODEL": "MODEL",
            "PREDICTION": "PREDICTION", 
            "DATASET": "DATASET",
            "LANGUAGE_SELECT": "LANGUAGE SELECT",
            
            # Кнопки
            "SAVE": "Save",
            "LOAD": "Load", 
            "TRAIN_BUTTON": "TRAIN",
            "PREDICT_POINT": "Predict Point", 
            "PREDICT_EPISODE": "Predict Episode",
            "CREATE_CAPS": "CREATE CAPSULE", 
            "FULLSCREEN": "Fullscreen",
            "LOAD_DATA": "Load Data", 
            "APPLY": "Apply",
            "LOAD_SEASON_NAMES": "Load Names",
            
            # Метки
            "MODEL_INFO": "Model Info",
            "MODEL_NAME": "Model Name:",
            "SUCCESS_RATE": "Success Rate", 
            "TRAINING": "Training",
            "EPOCHS": "Epochs:", 
            "BATCH_SIZE": "Batch Size:",
            "SEASON": "Season:", 
            "EPISODE": "Episode:",
            "TIMECODE": "Timecode (HH:MM:SS):", 
            "EPISODE_LENGTH": "Episode Length (HH:MM:SS):",
            "TIMECODE_PERCENT": "Timecode (%):", 
            "DATA_LOAD": "Data Loading",
            "DATA_FILE": "Data File:", 
            "CAPSULE": "Capsule Creation",
            "SAVE_PATH": "Save Path:", 
            "LOAD_PATH": "Load Path:",
            
            # Новые переводы для Pointer
            "TIMELINE": "Timeline",
            "EMOTION": "Emotion",
            "PLOT": "Plot",
            "PREVIEW": "PREVIEW",
            "PLAY": "PLAY",
            "CREATE_MP4": "Create MP4",
            "LOAD_CAPSULE": "Load Capsule",
            "HOURS": "Hours:",
            "MINUTES": "Minutes:",
            "SECONDS": "Seconds:",
            "SAVE_PATH_MP4": "Save Path:",
            "BROWSE": "Browse",
            "CREATE": "Create",
            "CANCEL": "Cancel",
            "NEW_GENERATION": "New Generation",
            "TIME_FORMAT_HH_MM_SS": "HH:MM:SS",
            "COORDINATES": "Coordinates",
            "X": "X",
            "Y": "Y",
            "MP4_CREATION": "MP4 Creation",
            "VIDEO_DURATION": "Video Duration:",
            "SUCCESS": "Success",
            "ERROR": "Error",
            "WARNING": "Warning",
            "INFO": "Information",
            "CONTROLS": "Controls"
        }
    
    def setup_styles(self):
        """Setup modern styles"""
        self.style = ttk.Style()
        self.style.theme_use('clam')
        
        # Цветовая палитра (как в LoView)
        BG_COLOR = "#2c3e50"
        FRAME_BG = "#34495e"
        BUTTON_BG = "#3498db"
        BUTTON_FG = "white"
        LABEL_BG = BG_COLOR
        LABEL_FG = "#ecf0f1"
        ENTRY_BG = "white"
        ENTRY_FG = "black"
        PROGRESS_BG = "#3498db"
        
        # Настройка стилей
        self.style.configure('TFrame', background=FRAME_BG)
        self.style.configure('TLabelframe', background=FRAME_BG, foreground=LABEL_FG)
        self.style.configure('TLabelframe.Label', background=FRAME_BG, foreground=LABEL_FG, font=('Arial', 10, 'bold'))
        self.style.configure('TLabel', background=LABEL_BG, foreground=LABEL_FG, font=('Arial', 10))
        self.style.configure('TButton', background=BUTTON_BG, foreground=BUTTON_FG, font=('Arial', 10, 'bold'), padding=8)
        self.style.map('TButton', background=[('active', '#2980b9')])
        self.style.configure('TScale', background=BG_COLOR)
        self.style.configure('TCombobox', fieldbackground=ENTRY_BG, background=ENTRY_BG)
        self.style.configure('TEntry', fieldbackground=ENTRY_BG, background=ENTRY_BG, font=('Arial', 10))
        
        self.root.configure(bg=BG_COLOR)
    
    def load_seasons(self):
        seasons_file = self.resource_path('seasons.txt')
        seasons = []
        try:
            if os.path.exists(seasons_file):
                with open(seasons_file, 'r', encoding='utf-8') as f:
                    seasons = [line.strip() for line in f if line.strip()]
            else:
                # Fallback список сезонов
                seasons = [
                    self.translations.get("NEW_GENERATION", "Новое поколение"),
                    "Игра бога",
                    "Идеальный мир",
                    "Голос времени",
                    "Тринадцать огней",
                    "Последняя реальность",
                    "Сердце вселенной",
                    "Точка невозврата",
                    "???"
                ]
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load seasons list: {str(e)}")
            seasons = [self.translations.get("NEW_GENERATION", "Новое поколение")]
        
        return seasons
    
    def load_background_images(self):
        # Загрузка фоновых изображений
        try:
            # Левое изображение - EmoPlain.png
            emo_image = Image.open(self.resource_path("EmoPlain.png"))
            emo_image = emo_image.resize((300, 300), Image.Resampling.LANCZOS)
            self.emo_bg = ImageTk.PhotoImage(emo_image)
            
            # Правое изображение - PlotPlain.png
            plot_image = Image.open(self.resource_path("PlotPlain.png"))
            plot_image = plot_image.resize((300, 300), Image.Resampling.LANCZOS)
            self.plot_bg = ImageTk.PhotoImage(plot_image)
            
            # Установка фоновых изображений
            self.canvas1.create_image(0, 0, anchor="nw", image=self.emo_bg)
            self.canvas2.create_image(0, 0, anchor="nw", image=self.plot_bg)
            
            # Перемещаем точки на передний план
            self.canvas1.tag_raise(self.point1)
            self.canvas2.tag_raise(self.point2)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load background images: {str(e)}")
    
    def create_widgets(self):
        # Main paned window for horizontal layout
        main_paned = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main_paned.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Left frame - Timeline controls (10%)
        left_frame = ttk.Frame(main_paned, width=500)
        main_paned.add(left_frame, weight=25)
        
        # Center frame - Points display (50%)
        center_frame = ttk.Frame(main_paned)
        main_paned.add(center_frame, weight=150)
        
        # Right frame - Buttons and controls (25%)
        right_frame = ttk.Frame(main_paned, width=50)
        main_paned.add(right_frame, weight=25)
        
        # ==================== LEFT FRAME - TIMELINE ====================
        timeline_title = ttk.Label(left_frame, text=self.translations.get("TIMELINE", "Хронология"), 
                                 font=('Arial', 12, 'bold'))
        timeline_title.pack(pady=10)
        
        timeline_frame = ttk.LabelFrame(left_frame, text=self.translations.get("TIMELINE", "Хронология"), padding=10)
        timeline_frame.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Season selection
        ttk.Label(timeline_frame, text=self.translations.get("SEASON", "Сезон:")).pack(anchor="w", padx=5, pady=5)
        season_combo = ttk.Combobox(timeline_frame, textvariable=self.season_var, 
                                   values=self.seasons_list, state="readonly")
        season_combo.pack(fill="x", padx=5, pady=5)
        
        # Episode selection
        ttk.Label(timeline_frame, text=self.translations.get("EPISODE", "Эпизод:")).pack(anchor="w", padx=5, pady=5)
        ttk.Entry(timeline_frame, textvariable=self.episode_var).pack(fill="x", padx=5, pady=5)
        
        # Timecode input
        ttk.Label(timeline_frame, text=self.translations.get("TIMECODE", "Timecode (HH:MM:SS):")).pack(anchor="w", padx=5, pady=5)
        
        timecode_frame = ttk.Frame(timeline_frame)
        timecode_frame.pack(fill="x", padx=5, pady=5)
        
        ttk.Entry(timecode_frame, textvariable=self.hours_var, width=3).pack(side="left")
        ttk.Label(timecode_frame, text=":").pack(side="left")
        ttk.Entry(timecode_frame, textvariable=self.minutes_var, width=3).pack(side="left")
        ttk.Label(timecode_frame, text=":").pack(side="left")
        ttk.Entry(timecode_frame, textvariable=self.seconds_var, width=3).pack(side="left")
        
        # Episode length input
        ttk.Label(timeline_frame, text=self.translations.get("EPISODE_LENGTH", "Episode Length (HH:MM:SS):")).pack(anchor="w", padx=5, pady=5)
        
        length_frame = ttk.Frame(timeline_frame)
        length_frame.pack(fill="x", padx=5, pady=5)
        
        ttk.Entry(length_frame, textvariable=self.length_hours_var, width=3).pack(side="left")
        ttk.Label(length_frame, text=":").pack(side="left")
        ttk.Entry(length_frame, textvariable=self.length_minutes_var, width=3).pack(side="left")
        ttk.Label(length_frame, text=":").pack(side="left")
        ttk.Entry(length_frame, textvariable=self.length_seconds_var, width=3).pack(side="left")
        
        # Moment percentage
        ttk.Label(timeline_frame, text=self.translations.get("TIMECODE_PERCENT", "Timecode (%):")).pack(anchor="w", padx=5, pady=5)
        self.moment_scale = ttk.Scale(timeline_frame, from_=0, to=100, variable=self.moment_var, 
                                     orient="horizontal")
        self.moment_scale.pack(fill="x", padx=5, pady=5)
        self.moment_label = ttk.Label(timeline_frame, text="50.0%")
        self.moment_label.pack(pady=5)
        
        # Bind scale movement to update function
        self.moment_scale.configure(command=self.update_timestamp_from_scale)
        
        # Bind timecode entries to update functions
        for var in [self.hours_var, self.minutes_var, self.seconds_var]:
            var.trace_add('write', lambda *args: self.update_timecode_from_entries())
        
        for var in [self.length_hours_var, self.length_minutes_var, self.length_seconds_var]:
            var.trace_add('write', lambda *args: self.update_length_from_entries())
        
        # ==================== CENTER FRAME - POINTS ====================
        points_title = ttk.Label(center_frame, text=self.translations.get("PREDICTION", "PREDICTION"), 
                               font=('Arial', 12, 'bold'))
        points_title.pack(pady=10)
        
        points_frame = ttk.Frame(center_frame)
        points_frame.pack(fill="both", expand=True, pady=5)
        
        # Point 1 (Emotion)
        point1_frame = ttk.LabelFrame(points_frame, text=self.translations.get("EMOTION", "Эмоция"), padding=10)
        point1_frame.pack(side="left", fill="both", expand=True, padx=10)
        
        self.canvas1 = tk.Canvas(point1_frame, width=300, height=300, bg="white", highlightthickness=1, highlightbackground="#34495e")
        self.canvas1.pack(pady=5)
        self.point1 = self.canvas1.create_oval(145, 145, 155, 155, fill="red", outline="red")
        
        # Detailed coordinates for Emotion point
        coord1_frame = ttk.Frame(point1_frame)
        coord1_frame.pack(pady=5)
        
        ttk.Label(coord1_frame, text=f"{self.translations.get('X', 'X')}:").pack(side="left")
        self.x1_var = tk.StringVar(value="0.500")
        ttk.Label(coord1_frame, textvariable=self.x1_var, width=6).pack(side="left", padx=(0, 10))
        
        ttk.Label(coord1_frame, text=f"{self.translations.get('Y', 'Y')}:").pack(side="left")
        self.y1_var = tk.StringVar(value="0.500")
        ttk.Label(coord1_frame, textvariable=self.y1_var, width=6).pack(side="left")
        
        # Point 2 (Plot)
        point2_frame = ttk.LabelFrame(points_frame, text=self.translations.get("PLOT", "Сюжет"), padding=10)
        point2_frame.pack(side="right", fill="both", expand=True, padx=10)
        
        self.canvas2 = tk.Canvas(point2_frame, width=300, height=300, bg="white", highlightthickness=1, highlightbackground="#34495e")
        self.canvas2.pack(pady=5)
        self.point2 = self.canvas2.create_oval(145, 145, 155, 155, fill="blue", outline="blue")
        
        # Detailed coordinates for Plot point
        coord2_frame = ttk.Frame(point2_frame)
        coord2_frame.pack(pady=5)
        
        ttk.Label(coord2_frame, text=f"{self.translations.get('X', 'X')}:").pack(side="left")
        self.x2_var = tk.StringVar(value="0.500")
        ttk.Label(coord2_frame, textvariable=self.x2_var, width=6).pack(side="left", padx=(0, 10))
        
        ttk.Label(coord2_frame, text=f"{self.translations.get('Y', 'Y')}:").pack(side="left")
        self.y2_var = tk.StringVar(value="0.500")
        ttk.Label(coord2_frame, textvariable=self.y2_var, width=6).pack(side="left")
        
        # Bind point movement events
        self.canvas1.bind("<B1-Motion>", lambda e: self.move_point(e, 1))
        self.canvas2.bind("<B1-Motion>", lambda e: self.move_point(e, 2))
        
        # ==================== RIGHT FRAME - CONTROLS ====================
        controls_title = ttk.Label(right_frame, text=self.translations.get("CONTROLS", "Controls"), 
                                 font=('Arial', 12, 'bold'))
        controls_title.pack(pady=10)
        
        controls_frame = ttk.LabelFrame(right_frame, text=self.translations.get("CONTROLS", "Controls"), padding=10)
        controls_frame.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Main action buttons
        ttk.Button(controls_frame, text=self.translations.get("PREVIEW", "PREVIEW"), 
                  command=self.preview, width=15).pack(pady=5, fill="x")
        ttk.Button(controls_frame, text=self.translations.get("PLAY", "PLAY"), 
                  command=self.toggle_play, width=15).pack(pady=5, fill="x")
        ttk.Button(controls_frame, text=self.translations.get("CREATE_MP4", "Create MP4"), 
                  command=self.open_mp4_dialog, width=15).pack(pady=5, fill="x")
        ttk.Button(controls_frame, text=self.translations.get("LOAD_CAPSULE", "Load Capsule"), 
                  command=self.load_capsule_dialog, width=15).pack(pady=5, fill="x")
        
        # Utility buttons
        ttk.Button(controls_frame, text=self.translations.get("LANGUAGE_SELECT", "LANGUAGE SELECT"), 
                  command=self.load_translations, width=15).pack(pady=5, fill="x")
        ttk.Button(controls_frame, text=self.translations.get("FULLSCREEN", "Fullscreen"), 
                  command=self.toggle_fullscreen, width=15).pack(pady=5, fill="x")
    
    def update_timestamp_from_scale(self, val):
        """Update timecode when scale changes"""
        percentage = float(val)
        self.moment_label.config(text=f"{percentage:.1f}%")
        
        # Update timecode based on episode length
        try:
            hours = int(self.length_hours_var.get())
            minutes = int(self.length_minutes_var.get())
            seconds = int(self.length_seconds_var.get())
            total_seconds = hours * 3600 + minutes * 60 + seconds
            
            if total_seconds > 0:
                current_seconds = (percentage / 100.0) * total_seconds
                hours = int(current_seconds // 3600)
                minutes = int((current_seconds % 3600) // 60)
                seconds = int(current_seconds % 60)
                
                self.hours_var.set(f"{hours:02d}")
                self.minutes_var.set(f"{minutes:02d}")
                self.seconds_var.set(f"{seconds:02d}")
        except ValueError:
            pass
    
    def update_timecode_from_entries(self):
        """Update scale when timecode entries change"""
        try:
            hours = int(self.hours_var.get())
            minutes = int(self.minutes_var.get())
            seconds = int(self.seconds_var.get())
            total_current = hours * 3600 + minutes * 60 + seconds
            
            total_hours = int(self.length_hours_var.get())
            total_minutes = int(self.length_minutes_var.get())
            total_seconds = int(self.length_seconds_var.get())
            total_length = total_hours * 3600 + total_minutes * 60 + total_seconds
            
            if total_length > 0:
                percentage = (total_current / total_length) * 100
                self.moment_var.set(min(max(percentage, 0), 100))
                self.moment_label.config(text=f"{percentage:.1f}%")
        except ValueError:
            pass
    
    def update_length_from_entries(self):
        """Update scale when length entries change"""
        self.update_timecode_from_entries()
    
    def load_translations(self):
        """Load translations from file (placeholder)"""
        # This would be implemented similarly to LoView's load_translations
        messagebox.showinfo("Info", "Translation loading would be implemented here")
    
    def toggle_fullscreen(self):
        """Toggle fullscreen mode"""
        self.is_fullscreen = not self.is_fullscreen
        self.root.attributes("-fullscreen", self.is_fullscreen)
    
    def move_point(self, event, point_num):
        canvas = self.canvas1 if point_num == 1 else self.canvas2
        point = self.point1 if point_num == 1 else self.point2
        
        # Ограничение перемещения в пределах холста
        x = max(0, min(event.x, 300))
        y = max(0, min(event.y, 300))
        
        canvas.coords(point, x-5, y-5, x+5, y+5)
        
        # Преобразование координат в [0,1]
        x_norm = x / 300
        y_norm = 1 - (y / 300)  # Инвертируем Y, так как в канвасе начало координат сверху
        
        # Обновляем текущие позиции
        if point_num == 1:
            self.current_points[0] = x_norm
            self.current_points[1] = y_norm
            self.x1_var.set(f"{x_norm:.3f}")
            self.y1_var.set(f"{y_norm:.3f}")
        else:
            self.current_points[2] = x_norm
            self.current_points[3] = y_norm
            self.x2_var.set(f"{x_norm:.3f}")
            self.y2_var.set(f"{y_norm:.3f}")
    
    def preview(self):
        # Создание окна предпросмотра
        preview_win = tk.Toplevel(self.root)
        preview_win.title("Preview")
        preview_win.geometry("400x200")
        preview_win.configure(bg='#2c3e50')
        
        # Установка иконки
        try:
            icon_path = self.resource_path("lo.ico")
            if os.path.exists(icon_path):
                preview_win.iconbitmap(icon_path)
        except:
            pass
        
        # Получение координат точек
        coords1 = self.canvas1.coords(self.point1)
        coords2 = self.canvas2.coords(self.point2)
        
        # Нормализация координат
        x1_norm = (coords1[0] + 5) / 300
        y1_norm = 1 - ((coords1[1] + 5) / 300)
        x2_norm = (coords2[0] + 5) / 300
        y2_norm = 1 - ((coords2[1] + 5) / 300)
        
        # Получение индекса выбранного сезона (начиная с 1)
        season_index = self.seasons_list.index(self.season_var.get()) + 1
        season_norm = season_index / 10
        
        # Нормализация входных данных
        episode_norm = float(self.episode_var.get()) / 1000
        moment_norm = self.moment_var.get() / 100
        
        # Текст предпросмотра
        preview_text = f"X: {{{season_index}, {self.episode_var.get()}, {self.moment_var.get():.1f}}}\n"
        preview_text += f"Y: {{{x1_norm:.3f}, {y1_norm:.3f}, {x2_norm:.3f}, {y2_norm:.3f}}}"
        
        ttk.Label(preview_win, text=preview_text, font=("Arial", 12)).pack(pady=20)
        
        # Кнопки
        btn_frame = ttk.Frame(preview_win)
        btn_frame.pack(pady=10)
        
        ttk.Button(btn_frame, text="ADD", 
                  command=lambda: self.add_data(season_norm, episode_norm, moment_norm, 
                                              x1_norm, y1_norm, x2_norm, y2_norm, preview_win)).pack(side="left", padx=10)
        ttk.Button(btn_frame, text="RETURN", command=preview_win.destroy).pack(side="right", padx=10)
        
    def add_data(self, season, episode, moment, x1, y1, x2, y2, window):
        # Добавление данных в структуру
        self.data["X"].append([season, episode, moment])
        self.data["Y"].append([x1, y1, x2, y2])
        
        # Сохранение в файл
        with open(self.resource_path('IDS.json'), 'w') as f:
            json.dump(self.data, f, indent=4)
        
        messagebox.showinfo("Успех", "Данные успешно добавлены!")
        window.destroy()
    
    def get_episode_data(self):
        """Получить данные только для выбранного сезона и эпизода (старый формат)"""
        season_index = self.seasons_list.index(self.season_var.get()) + 1
        season_norm = season_index / 10
        episode_norm = int(self.episode_var.get()) / 1000
        
        moments = []
        quartets = []
        
        for i in range(len(self.data["X"])):
            x_data = self.data["X"][i]
            y_data = self.data["Y"][i]
            
            # Проверяем, соответствует ли запись выбранному сезону и эпизоду
            if abs(x_data[0] - season_norm) < 0.001 and abs(x_data[1] - episode_norm) < 0.001:
                moments.append(x_data[2] * 100)  # Конвертируем в проценты (0-100)
                quartets.append(y_data)          # Координаты точек
        
        # Сортируем данные по моменту времени
        if moments:
            combined = list(zip(moments, quartets))
            combined.sort(key=lambda x: x[0])
            moments, quartets = zip(*combined)
            moments = list(moments)
            quartets = list(quartets)
        
        return {"moments": moments, "quartets": quartets}
    
    def toggle_play(self):
        if self.is_playing:
            self.is_playing = False
            self.moment_scale.config(state="normal")  # Возвращаем слайдер в нормальное состояние
            self.moment_label.config(foreground="#ecf0f1")
        else:
            # Получаем данные для выбранного эпизода
            if self.loaded_capsule:
                # Используем данные из загруженной капсулы
                season_index = self.seasons_list.index(self.season_var.get()) + 1
                if (self.loaded_capsule["season"] == season_index and 
                    self.loaded_capsule["episode"] == int(self.episode_var.get())):
                    self.episode_data = {
                        "moments": self.loaded_capsule["moments"],
                        "quartets": self.loaded_capsule["quartets"]
                    }
                else:
                    messagebox.showwarning("Предупреждение", "Загруженная капсула не соответствует выбранному сезону/эпизоду!")
                    return
            else:
                # Используем данные из IDS.json (старый формат)
                self.episode_data = self.get_episode_data()
            
            if not self.episode_data["moments"]:
                messagebox.showwarning("Предупреждение", "Нет данных для выбранного эпизода!")
                return
            
            # Запоминаем начальный момент времени
            self.start_moment = self.moment_var.get()  # В процентах (0-100)
            self.is_playing = True
            self.animation_start_time = time.time()
            self.moment_scale.config(state="disabled")  # Отключаем слайдер
            self.moment_label.config(foreground="red")
            self.play_animation()
    
    def get_interpolated_target(self, current_time):
        """Получить целевую позицию через интерполяцию между кадрами"""
        moments = self.episode_data["moments"]
        quartets = self.episode_data["quartets"]
        
        if not moments:
            return None
        
        # Если текущее время до первого кадра, используем первый кадр
        if current_time <= moments[0]:
            return quartets[0]
        
        # Если текущее время после последнего кадра, используем последний кадр
        if current_time >= moments[-1]:
            return quartets[-1]
        
        # Находим индекс первого кадра, который больше текущего времени
        idx = bisect.bisect_right(moments, current_time)
        
        # Интерполируем между двумя кадрами
        prev_time = moments[idx - 1]
        next_time = moments[idx]
        prev_points = quartets[idx - 1]
        next_points = quartets[idx]
        
        # Вычисляем коэффициент интерполяции
        t = (current_time - prev_time) / (next_time - prev_time)
        
        # Интерполируем каждую координату
        interpolated_points = []
        for i in range(4):  # x1, y1, x2, y2
            interpolated_value = prev_points[i] + (next_points[i] - prev_points[i]) * t
            interpolated_points.append(interpolated_value)
        
        return interpolated_points
    
    def play_animation(self):
        if not self.is_playing:
            return
            
        # Вычисляем прошедшее время с начала воспроизведения
        elapsed_time = time.time() - self.animation_start_time
        
        # Вычисляем текущее время с учетом начального момента (в процентах)
        current_time = self.start_moment + (elapsed_time * 0.8333)  # 0.8333% в секунду
        
        # Ограничиваем текущее время до 100%
        if current_time > 100.0:
            current_time = 100.0
            self.is_playing = False
            self.moment_scale.config(state="normal")
            self.moment_label.config(foreground="#ecf0f1")
        
        # Устанавливаем прогресс на слайдере
        self.moment_var.set(current_time)
        self.moment_label.config(text=f"{current_time:.1f}%")
        
        # Обновляем timecode
        try:
            hours = int(self.length_hours_var.get())
            minutes = int(self.length_minutes_var.get())
            seconds = int(self.length_seconds_var.get())
            total_seconds = hours * 3600 + minutes * 60 + seconds
            
            if total_seconds > 0:
                current_seconds = (current_time / 100.0) * total_seconds
                hours = int(current_seconds // 3600)
                minutes = int((current_seconds % 3600) // 60)
                seconds = int(current_seconds % 60)
                
                self.hours_var.set(f"{hours:02d}")
                self.minutes_var.set(f"{minutes:02d}")
                self.seconds_var.set(f"{seconds:02d}")
        except ValueError:
            pass
        
        # Получаем целевую позицию через интерполяцию
        target_points = self.get_interpolated_target(current_time)
        
        if target_points:
            # Плавно перемещаем точки к целевой позиции
            self.smooth_move_points(target_points)
        
        # Продолжаем анимацию, если не достигли конца
        if self.is_playing:
            self.root.after(50, self.play_animation)  # Обновляем каждые 50 мс
    
    def smooth_move_points(self, target_points):
        """Плавное перемещение точек к целевой позиции"""
        # Коэффициент плавности (чем больше, тем быстрее движение)
        smoothness = 0.2
        
        # Плавно перемещаем каждую точку
        for i in range(4):
            # Вычисляем новую позицию как взвешенное среднее
            self.current_points[i] = self.current_points[i] * (1 - smoothness) + target_points[i] * smoothness
        
        # Устанавливаем новые позиции
        self.set_point_position(1, self.current_points[0], self.current_points[1])
        self.set_point_position(2, self.current_points[2], self.current_points[3])
    
    def set_point_position(self, point_num, x_norm, y_norm):
        canvas = self.canvas1 if point_num == 1 else self.canvas2
        point = self.point1 if point_num == 1 else self.point2
        
        # Преобразование нормализованных координат в пиксели
        x = x_norm * 300
        y = (1 - y_norm) * 300  # Инвертируем Y
        
        # Установка позиции точки
        canvas.coords(point, x-5, y-5, x+5, y+5)
        
        # Обновление текстовых меток
        if point_num == 1:
            self.x1_var.set(f"{x_norm:.3f}")
            self.y1_var.set(f"{y_norm:.3f}")
        else:
            self.x2_var.set(f"{x_norm:.3f}")
            self.y2_var.set(f"{y_norm:.3f}")
    
    def load_capsule_dialog(self):
        """Открыть диалог для загрузки капсулы"""
        filepath = filedialog.askopenfilename(
            title="Выберите файл капсулы",
            filetypes=[("LVP files", "*.lvp"), ("All files", "*.*")]
        )
        
        if filepath:
            self.load_capsule(filepath)
    
    def load_capsule(self, filepath):
        """Загрузить капсулу из файла"""
        try:
            with open(filepath, 'r') as f:
                capsule_data = json.load(f)
            
            # Проверяем формат капсулы (новый формат LVP)
            if "season" in capsule_data and "episode" in capsule_data and "moments" in capsule_data and "quartets" in capsule_data:
                self.loaded_capsule = capsule_data
                
                # Устанавливаем сезон и эпизод из капсулы
                season_index = capsule_data["season"] - 1
                if 0 <= season_index < len(self.seasons_list):
                    self.season_var.set(self.seasons_list[season_index])
                self.episode_var.set(str(capsule_data["episode"]))
                
                # Обновляем данные эпизода для воспроизведения
                self.episode_data = {
                    "moments": capsule_data["moments"],
                    "quartets": capsule_data["quartets"]
                }
                
                messagebox.showinfo("Успех", f"Капсула загружена: Сезон {capsule_data['season']}, Эпизод {capsule_data['episode']}\n"
                                           f"Моментов: {len(capsule_data['moments'])}")
            else:
                messagebox.showerror("Ошибка", "Неверный формат капсулы")
                
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось загрузить капсулу: {str(e)}")
    
    def open_mp4_dialog(self):
        """Открыть диалог для создания MP4"""
        mp4_dialog = tk.Toplevel(self.root)
        mp4_dialog.title(self.translations.get("MP4_CREATION", "Создать MP4 видео"))
        mp4_dialog.geometry("400x300")
        mp4_dialog.resizable(False, False)
        mp4_dialog.configure(bg='#2c3e50')
        
        # Установка иконки
        try:
            icon_path = self.resource_path("lo.ico")
            if os.path.exists(icon_path):
                mp4_dialog.iconbitmap(icon_path)
        except:
            pass
        
        ttk.Label(mp4_dialog, text=self.translations.get("VIDEO_DURATION", "Длительность видео:"), font=("Arial", 12)).pack(pady=10)
        
        # Поля ввода времени
        time_frame = ttk.Frame(mp4_dialog)
        time_frame.pack(pady=10)
        
        ttk.Label(time_frame, text=self.translations.get("HOURS", "Часы:")).grid(row=0, column=0, padx=5)
        hours_entry = ttk.Entry(time_frame, textvariable=self.video_hours_var, width=5)
        hours_entry.grid(row=0, column=1, padx=5)
        
        ttk.Label(time_frame, text=self.translations.get("MINUTES", "Минуты:")).grid(row=0, column=2, padx=5)
        minutes_entry = ttk.Entry(time_frame, textvariable=self.video_minutes_var, width=5)
        minutes_entry.grid(row=0, column=3, padx=5)
        
        ttk.Label(time_frame, text=self.translations.get("SECONDS", "Секунды:")).grid(row=0, column=4, padx=5)
        seconds_entry = ttk.Entry(time_frame, textvariable=self.video_seconds_var, width=5)
        seconds_entry.grid(row=0, column=5, padx=5)
        
        # Выбор пути сохранения
        path_frame = ttk.Frame(mp4_dialog)
        path_frame.pack(pady=10, fill="x", padx=20)
        
        ttk.Label(path_frame, text=self.translations.get("SAVE_PATH_MP4", "Путь сохранения:")).pack(anchor="w")
        
        path_entry_frame = ttk.Frame(path_frame)
        path_entry_frame.pack(fill="x", pady=5)
        
        ttk.Entry(path_entry_frame, textvariable=self.video_path_var, width=30).pack(side="left", fill="x", expand=True, padx=(0, 5))
        ttk.Button(path_entry_frame, text=self.translations.get("BROWSE", "Обзор"), command=self.browse_save_path, width=8).pack(side="right")
        
        # Кнопки
        btn_frame = ttk.Frame(mp4_dialog)
        btn_frame.pack(pady=20)
        
        ttk.Button(btn_frame, text=self.translations.get("CREATE", "Создать"), 
                  command=lambda: self.start_mp4_creation(mp4_dialog)).pack(side="left", padx=10)
        ttk.Button(btn_frame, text=self.translations.get("CANCEL", "Отмена"), command=mp4_dialog.destroy).pack(side="right", padx=10)
        
    def browse_save_path(self):
        """Выбрать путь для сохранения видео"""
        # Создаем временное окно для удержания фокуса
        temp = tk.Toplevel(self.root)
        temp.withdraw()  # Скрываем временное окно
        
        file_path = filedialog.asksaveasfilename(
            defaultextension=".mp4",
            filetypes=[("MP4 files", "*.mp4"), ("All files", "*.*")],
            title="Сохранить видео как",
            parent=temp  # Указываем родительское окно
        )
    
        temp.destroy()  # Уничтожаем временное окно
    
        if file_path:
            self.video_path_var.set(file_path)
        
    def start_mp4_creation(self, dialog):
        """Начать создание MP4 в отдельном потоке"""
        try:
            hours = int(self.video_hours_var.get())
            minutes = int(self.video_minutes_var.get())
            seconds = int(self.video_seconds_var.get())
            
            total_seconds = hours * 3600 + minutes * 60 + seconds
            
            if total_seconds <= 0:
                messagebox.showerror("Ошибка", "Длительность должна быть больше 0 секунд!")
                return
                
            output_path = self.video_path_var.get()
            if not output_path:
                messagebox.showerror("Ошибка", "Пожалуйста, выберите путь для сохранения видео!")
                return
                
            dialog.destroy()
            
            # Запускаем создание MP4 в отдельном потоке
            threading.Thread(target=self.create_mp4_video, args=(total_seconds, output_path), daemon=True).start()
            
        except ValueError:
            messagebox.showerror("Ошибка", "Пожалуйста, введите корректные числовые значения!")
    
    def create_mp4_video(self, total_seconds, output_path):
        """Создать MP4 видео с анимации"""
        try:
            # Получаем данные эпизода
            if self.loaded_capsule:
                # Используем данные из загруженной капсулы
                self.episode_data = {
                    "moments": self.loaded_capsule["moments"],
                    "quartets": self.loaded_capsule["quartets"]
                }
            else:
                # Используем данные из IDS.json (старый формат)
                self.episode_data = self.get_episode_data()
            
            if not self.episode_data["moments"]:
                messagebox.showerror("Ошибка", "Нет данных для выбранного эпизода!")
                return
            
            # Создаем папку для кадров
            frames_dir = "../data/temp_frames"
            os.makedirs(frames_dir, exist_ok=True)
            
            # Вычисляем количество кадров (30 FPS)
            fps = 30
            total_frames = total_seconds * fps
            
            # Создаем прогресс-бар
            progress_window = tk.Toplevel(self.root)
            progress_window.title("Создание MP4")
            progress_window.geometry("300x100")
            progress_window.resizable(False, False)
            progress_window.configure(bg='#2c3e50')
            
            # Установка иконки
            try:
                icon_path = self.resource_path("lo.ico")
                if os.path.exists(icon_path):
                    progress_window.iconbitmap(icon_path)
            except:
                pass
            
            ttk.Label(progress_window, text="Создание видео...").pack(pady=5)
            progress_var = tk.DoubleVar()
            progress_bar = ttk.Progressbar(progress_window, variable=progress_var, maximum=total_frames)
            progress_bar.pack(pady=10, padx=20, fill="x")
            
            # Начинаем с центральных позиций
            current_video_points = [0.5, 0.5, 0.5, 0.5]  # x1, y1, x2, y2
            
            # Генерируем кадры
            frames = []
            for frame_num in range(total_frames):
                # Вычисляем текущее время (0-100%)
                current_time = (frame_num / total_frames) * 100.0
                
                # Получаем целевую позицию через интерполяцию
                target_points = self.get_interpolated_target(current_time)
                
                if target_points:
                    # Плавно перемещаем точки к целевой позиции
                    smoothness = 0.1  # Меньший коэффициент для более плавного видео
                    for i in range(4):
                        current_video_points[i] = current_video_points[i] * (1 - smoothness) + target_points[i] * smoothness
                    
                    # Создаем изображение кадра
                    frame = self.generate_frame(current_video_points)
                    frames.append(frame)
                
                # Обновляем прогресс
                progress_var.set(frame_num)
                progress_window.update()
            
            # Создаем видео из кадров
            clip = ImageSequenceClip(frames, fps=fps)
            clip.write_videofile(output_path, codec="libx264", audio=False)
            
            # Удаляем временные файлы
            for frame_file in os.listdir(frames_dir):
                os.remove(os.path.join(frames_dir, frame_file))
            os.rmdir(frames_dir)
            
            progress_window.destroy()
            messagebox.showinfo("Успех", f"Видео успешно создано: {output_path}")
            
        except Exception as e:
            messagebox.showerror("Ошибка", f"Ошибка при создании видео: {str(e)}")
    
    def generate_frame(self, points):
        """Сгенерировать кадр с точками на фоне"""
        # Создаем изображение размером 600x300 (объединяем два канваса)
        frame = np.ones((300, 600, 3), dtype=np.uint8) * 255  # Белый фон
        
        try:
            # Загружаем фоновые изображения
            emo_bg = cv2.imread(self.resource_path("EmoPlain.png"))
            plot_bg = cv2.imread(self.resource_path("PlotPlain.png"))
            
            # Изменяем размер изображений
            emo_bg = cv2.resize(emo_bg, (300, 300))
            plot_bg = cv2.resize(plot_bg, (300, 300))
            
            # Накладываем фоновые изображения
            frame[0:300, 0:300] = emo_bg
            frame[0:300, 300:600] = plot_bg
        except:
            # Если не удалось загрузить фоновые изображения, используем белый фон
            pass
        
        # Рисуем точки
        # Точка 1 (красная)
        x1 = int(points[0] * 300)
        y1 = int((1 - points[1]) * 300)  # Инвертируем Y
        cv2.circle(frame, (x1, y1), 5, (0, 0, 255), -1)  # Красный
        
        # Точка 2 (синяя)
        x2 = int(points[2] * 300) + 300  # Смещаем для правого канваса
        y2 = int((1 - points[3]) * 300)  # Инвертируем Y
        cv2.circle(frame, (x2, y2), 5, (255, 0, 0), -1)  # Синий
        
        # Конвертируем BGR в RGB для moviepy
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return frame_rgb

if __name__ == "__main__":
    root = tk.Tk()
    app = PointCreator(root)
    root.mainloop()