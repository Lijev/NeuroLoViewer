import json
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import os
import sys
import threading
import configparser
import time
import bisect
from PIL import Image, ImageTk
import cv2

# Global variables
X_train, X_test, y_train, y_test = None, None, None, None
current_model = None
training_history = None
test_loss = None
data_X = None
data_Y = None
is_fullscreen = True  # Start in fullscreen by default
seasons_list = []
is_predicting_episode = False
episode_prediction_thread = None
season_names = []
updating_time = False  # Flag to prevent recursive updates
current_language = "EN"  # Default language

# Translation dictionary
translations = {
    "EN": {
        "MODEL": "MODEL",
        "PREDICTION": "PREDICTION",
        "DATASET": "DATASET",
        "LANGUAGE_SELECTION": "LANGUAGE SELECTION",
        "SAVE": "Save",
        "LOAD": "Load",
        "TRAIN": "TRAIN",
        "PREDICT_POINT": "Predict Point",
        "PREDICT_EPISODE": "Predict Episode",
        "CREATE_CAPSULE": "CREATE CAPSULE",
        "FULLSCREEN": "Fullscreen",
        "LOAD_DATA": "Load Data",
        "APPLY": "Apply",
        "LOAD_NAMES": "Load Names",
        "MODEL_INFORMATION": "Model Information",
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
        "DATA_LOADING": "Data Loading",
        "DATA_FILE": "Data File:",
        "CAPSULE_CREATION": "Capsule Creation",
        "SAVE_PATH": "Save Path:",
        "LOAD_PATH": "Load Path:",
        "DATA_LOADED": "Data loaded successfully.",
        "MODEL_SAVED": "Model saved successfully.",
        "MODEL_LOADED": "Model loaded successfully.",
        "MODEL_TRAINED": "Model trained successfully.",
        "CAPSULE_SAVED": "Capsule saved successfully.",
        "SEASON_NAMES_LOADED": "Season names loaded successfully.",
        "TRANSLATION_LOADED": "Translation loaded successfully.",
        "ERROR_LOADING_DATA": "Error loading data.",
        "ERROR_SAVING_MODEL": "Error saving model.",
        "ERROR_LOADING_MODEL": "Error loading model.",
        "ERROR_TRAINING_MODEL": "Error training model.",
        "DATA_NOT_LOADED": "Data not loaded. Please load data first.",
        "MODEL_NOT_LOADED": "Model not loaded or trained. Please load or train a model first.",
        "INVALID_INPUT": "Invalid input. Please enter integer values for epochs and batch size.",
        "INVALID_EPISODE": "Episode number must be a natural number (1, 2, 3, ...).",
        "INVALID_TIMECODE": "Invalid timecode format. Use HH:MM:SS.",
        "INVALID_EPISODE_LENGTH": "Invalid episode length format. Use HH:MM:SS.",
        "ERROR_CREATING_CAPSULE": "Error creating capsule.",
        "ERROR_LOADING_SEASON_NAMES": "Error loading season names.",
        "ERROR_LOADING_TRANSLATION": "Error loading translation.",
        "TRANSLATION_FILE_SHORT": "Translation file must have at least 58 lines.",
        "WARNING": "Warning",
        "CONFIRMATION": "Confirmation",
        "INFORMATION": "Information",
        "TRAINING_PROGRESS": "Training Progress",
        "CAPSULE_CREATION_PROGRESS": "Capsule Creation Progress",
        "EPISODE_PREDICTION_PROGRESS": "Episode Prediction Progress",
        "TRAINING_ELIPSIS": "Training...",
        "PREDICTING_ELIPSIS": "Predicting...",
        "STOP": "Stop",
        "HHMMSS": "HH:MM:SS",
        "CURRENT_LENGTH": "Current/Length",
        "EPISODE_NUMBER": "Episode Number",
        "SEASON_NUMBER": "Season Number",
        "PERCENTAGE": "Percentage",
        "COORDINATES": "Coordinates",
        "TIMELINE": "Timeline",
        "EMOTION": "Emotion",
        "PLOT": "Plot",
        "PREVIEW": "PREVIEW",
        "PLAY": "PLAY",
        "CREATE_MP4": "CREATE MP4",
        "LOAD_CAPSULE": "LOAD CAPSULE",
        "HOURS": "Hours:",
        "MINUTES": "Minutes:",
        "SECONDS": "Seconds:",
        "BROWSE": "Browse",
        "CREATE": "Create",
        "CANCEL": "Cancel",
        "NEW_GENERATION": "New Generation",
        "MP4_CREATION": "MP4 Creation",
        "VIDEO_DURATION": "Video Duration:",
        "SUCCESS": "Success",
        "ERROR": "Error",
        "CONTROL": "Control",
        "CHANGE_LANGUAGE": "Change Language",
        "SELECT_TRANSLATION_FILE": "Select translation file"
    },
    "RU": {
        "MODEL": "МОДЕЛЬ",
        "PREDICTION": "ПРЕДСКАЗАНИЕ",
        "DATASET": "ДАННЫЕ",
        "LANGUAGE_SELECTION": "ВЫБОР ЯЗЫКА",
        "SAVE": "Сохранить",
        "LOAD": "Загрузить",
        "TRAIN": "ОБУЧИТЬ",
        "PREDICT_POINT": "Предсказать точку",
        "PREDICT_EPISODE": "Предсказать эпизод",
        "CREATE_CAPSULE": "СОЗДАТЬ КАПСУЛУ",
        "FULLSCREEN": "Полный экран",
        "LOAD_DATA": "Загрузить данные",
        "APPLY": "Применить",
        "LOAD_NAMES": "Загрузить названия",
        "MODEL_INFORMATION": "Информация о модели",
        "MODEL_NAME": "Название модели:",
        "SUCCESS_RATE": "Успешность",
        "TRAINING": "Обучение",
        "EPOCHS": "Эпохи:",
        "BATCH_SIZE": "Размер батча:",
        "SEASON": "Сезон:",
        "EPISODE": "Эпизод:",
        "TIMECODE": "Таймкод (ЧЧ:ММ:СС):",
        "EPISODE_LENGTH": "Длина эпизода (ЧЧ:ММ:СС):",
        "TIMECODE_PERCENT": "Таймкод (%):",
        "DATA_LOADING": "Загрузка данных",
        "DATA_FILE": "Файл данных:",
        "CAPSULE_CREATION": "Создание капсулы",
        "SAVE_PATH": "Путь сохранения:",
        "LOAD_PATH": "Путь загрузки:",
        "DATA_LOADED": "Данные успешно загружены.",
        "MODEL_SAVED": "Модель успешно сохранена.",
        "MODEL_LOADED": "Модель успешно загружена.",
        "MODEL_TRAINED": "Модель успешно обучена.",
        "CAPSULE_SAVED": "Капсула успешно сохранена.",
        "SEASON_NAMES_LOADED": "Названия сезонов успешно загружены.",
        "TRANSLATION_LOADED": "Перевод успешно загружен.",
        "ERROR_LOADING_DATA": "Ошибка загрузки данных.",
        "ERROR_SAVING_MODEL": "Ошибка сохранения модели.",
        "ERROR_LOADING_MODEL": "Ошибка загрузки модели.",
        "ERROR_TRAINING_MODEL": "Ошибка обучения модели.",
        "DATA_NOT_LOADED": "Данные не загружены. Сначала загрузите данные.",
        "MODEL_NOT_LOADED": "Модель не загружена или не обучена. Сначала загрузите или обучите модель.",
        "INVALID_INPUT": "Неверный ввод. Введите целые числа для эпох и размера батча.",
        "INVALID_EPISODE": "Номер эпизода должен быть натуральным числом (1, 2, 3, ...).",
        "INVALID_TIMECODE": "Неверный формат таймкода. Используйте ЧЧ:ММ:СС.",
        "INVALID_EPISODE_LENGTH": "Неверный формат длины эпизода. Используйте ЧЧ:ММ:СС.",
        "ERROR_CREATING_CAPSULE": "Ошибка создания капсулы.",
        "ERROR_LOADING_SEASON_NAMES": "Ошибка загрузки названий сезонов.",
        "ERROR_LOADING_TRANSLATION": "Ошибка загрузки перевода.",
        "TRANSLATION_FILE_SHORT": "Файл перевода должен содержать не менее 58 строк.",
        "WARNING": "Предупреждение",
        "CONFIRMATION": "Подтверждение",
        "INFORMATION": "Информация",
        "TRAINING_PROGRESS": "Прогресс обучения",
        "CAPSULE_CREATION_PROGRESS": "Прогресс создания капсулы",
        "EPISODE_PREDICTION_PROGRESS": "Прогресс предсказания эпизода",
        "TRAINING_ELIPSIS": "Обучение...",
        "PREDICTING_ELIPSIS": "Предсказание...",
        "STOP": "Стоп",
        "HHMMSS": "ЧЧ:ММ:СС",
        "CURRENT_LENGTH": "Текущее/Длина",
        "EPISODE_NUMBER": "Номер эпизода",
        "SEASON_NUMBER": "Номер сезона",
        "PERCENTAGE": "Процент",
        "COORDINATES": "Координаты",
        "TIMELINE": "Хронология",
        "EMOTION": "Эмоция",
        "PLOT": "Сюжет",
        "PREVIEW": "ПРЕДПРОСМОТР",
        "PLAY": "ВОСПРОИЗВЕСТИ",
        "CREATE_MP4": "СОЗДАТЬ MP4",
        "LOAD_CAPSULE": "ЗАГРУЗИТЬ КАПСУЛУ",
        "HOURS": "Часы:",
        "MINUTES": "Минуты:",
        "SECONDS": "Секунды:",
        "BROWSE": "Обзор",
        "CREATE": "Создать",
        "CANCEL": "Отмена",
        "NEW_GENERATION": "Новое поколение",
        "MP4_CREATION": "Создание MP4",
        "VIDEO_DURATION": "Длительность видео:",
        "SUCCESS": "Успех",
        "ERROR": "Ошибка",
        "CONTROL": "Управление",
        "CHANGE_LANGUAGE": "Сменить язык",
        "SELECT_TRANSLATION_FILE": "Выберите файл перевода"
    }
}

# Configuration file setup
CONFIG_FILE = "config.ini"
config = configparser.ConfigParser()
config.read(CONFIG_FILE)

def resource_path(relative_path):
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath("../data/")
    return os.path.join(base_path, relative_path)

def t(key):
    """Translation function"""
    return translations[current_language].get(key, key)

def update_ui_language():
    """Update all UI elements with current language"""
    # Left frame
    left_label.config(text=t("MODEL"))
    frame_model_info.config(text=t("MODEL_INFORMATION"))
    label_model_name.config(text=t("MODEL_NAME"))
    label_model_success.config(text=f"{t('SUCCESS_RATE')}: N/A")
    frame_training.config(text=t("TRAINING"))
    label_epochs.config(text=t("EPOCHS"))
    label_batch_size.config(text=t("BATCH_SIZE"))
    button_train.config(text=t("TRAIN"))
    label_save_path.config(text=t("SAVE_PATH"))
    label_load_path.config(text=t("LOAD_PATH"))
    button_save_model.config(text=t("SAVE"))
    button_load_model.config(text=t("LOAD"))
    
    # Center frame
    center_label.config(text=t("PREDICTION"))
    label_timecode.config(text=t("TIMECODE"))
    label_episode_length.config(text=t("EPISODE_LENGTH"))
    label_season.config(text=t("SEASON"))
    label_episode.config(text=t("EPISODE"))
    label_timecode_percent.config(text=t("TIMECODE_PERCENT"))
    button_predict_point.config(text=t("PREDICT_POINT"))
    button_predict_episode.config(text=t("PREDICT_EPISODE"))
    button_load_season_names.config(text=t("LOAD_NAMES"))
    
    # Right frame
    right_label.config(text=t("DATASET"))
    frame_dataset.config(text=t("DATA_LOADING"))
    label_data_file.config(text=t("DATA_FILE"))
    button_load_data.config(text=t("LOAD_DATA"))
    button_fullscreen.config(text=t("FULLSCREEN"))
    frame_capsule.config(text=t("CAPSULE_CREATION"))
    button_create_capsule.config(text=t("CREATE_CAPSULE"))
    button_change_language.config(text=t("CHANGE_LANGUAGE"))

def change_language():
    """Change application language by loading a translation file"""
    filepath = filedialog.askopenfilename(
        title=t("SELECT_TRANSLATION_FILE"),
        filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
    )
    
    if not filepath:
        return
        
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f.readlines()]
            
        if len(lines) < 58:
            messagebox.showerror(t("ERROR"), t("TRANSLATION_FILE_SHORT"))
            return
            
        # Extract language key from filename without extension
        language_key = os.path.splitext(os.path.basename(filepath))[0]
        
        # Create new translation dictionary
        en_keys = list(translations["EN"].keys())
        new_translations = {}
        for i, key in enumerate(en_keys):
            if i < len(lines):
                new_translations[key] = lines[i]
            else:
                new_translations[key] = key  # Fallback to key if not enough lines
                
        # Add or update the translation in the translations dictionary
        translations[language_key] = new_translations
        
        # Set current language
        global current_language
        current_language = language_key
        
        # Update UI
        update_ui_language()
        
        messagebox.showinfo(t("SUCCESS"), t("TRANSLATION_LOADED"))
        
    except Exception as e:
        messagebox.showerror(t("ERROR"), f"{t('ERROR_LOADING_TRANSLATION')}: {e}")

def load_data(filepath):
    global data_X, data_Y
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        data_X = np.array(data['X'])
        data_Y = np.array(data['Y'])
        return data_X, data_Y
    except Exception as e:
        messagebox.showerror(t("ERROR"), f"{t('ERROR_LOADING_DATA')}: {e}")
        return None, None

def create_and_train_model(X_train, y_train, X_test, y_test, epochs=100, batch_size=10, progress_callback=None):
    hidden_layer_size = 128
    model = keras.Sequential()
    model.add(layers.Dense(hidden_layer_size, activation='relu', input_shape=(3,)))
    model.add(layers.Dense(hidden_layer_size, activation='relu'))
    model.add(layers.Dense(hidden_layer_size // 2, activation='relu'))
    model.add(layers.Dense(4))
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    history = model.fit(
        X_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.2,
        verbose=0,
        callbacks=[keras.callbacks.LambdaCallback(
            on_epoch_end=lambda epoch, logs: progress_callback(epoch + 1, epochs) if progress_callback else None
        )]
    )
    
    test_loss = model.evaluate(X_test, y_test, verbose=0)
    
    # Calculate success rate
    success_rate = max(0, (1 - test_loss)) * 100
    
    return model, history, test_loss, success_rate

def save_model(model, model_name):
    """
    Save a Keras model to a file.

    Args:
        model (keras.Model): The Keras model to save.
        model_name (str): The path to save the model to.
    """
    try:
        model.save(model_name)
        messagebox.showinfo(t("SUCCESS"), f"{t('MODEL_SAVED')}: {model_name}")
        print(f'Model saved as {model_name}')

        # Save to config
        config['Paths']['save_path'] = model_name
        save_config()

    except Exception as e:
        messagebox.showerror(t("ERROR"), f"{t('ERROR_SAVING_MODEL')}: {e}")

def load_model(model_name):
    try:
        model = keras.models.load_model(model_name)
        messagebox.showinfo(t("SUCCESS"), f"{t('MODEL_LOADED')}: {model_name}")
        config['Paths']['load_path'] = model_name
        save_config()
        return model
    except Exception as e:
        messagebox.showerror(t("ERROR"), f"{t('ERROR_LOADING_MODEL')}: {e}")
        return None

def update_progress_bar(epoch, total_epochs):
    progress_value = int((epoch / total_epochs) * 100)
    progress_bar['value'] = progress_value
    root.update_idletasks()

def train_model_thread(X_train, y_train, X_test, y_test, epochs, batch_size):
    global current_model, training_history, test_loss
    try:
        current_model, training_history, test_loss, success_rate = create_and_train_model(
            X_train, y_train, X_test, y_test, epochs, batch_size,
            progress_callback=update_progress_bar
        )
        label_model_success.config(text=f'{t("SUCCESS_RATE")}: {success_rate:.2f}%')
        messagebox.showinfo(t("SUCCESS"), t("MODEL_TRAINED"))
    except Exception as e:
        messagebox.showerror(t("ERROR"), f"{t('ERROR_TRAINING_MODEL')}: {e}")
    finally:
        progress_bar['value'] = 0
        button_train['state'] = 'normal'

def train_model_command():
    global X_train, X_test, y_train, y_test
    if X_train is None or y_train is None or X_test is None or y_test is None:
        messagebox.showerror(t("ERROR"), t("DATA_NOT_LOADED"))
        return
    try:
        epochs = int(entry_epochs.get())
        batch_size = int(entry_batch_size.get())
    except ValueError:
        messagebox.showerror(t("ERROR"), t("INVALID_INPUT"))
        return
    button_train['state'] = 'disabled'
    threading.Thread(target=train_model_thread, args=(X_train, y_train, X_test, y_test, epochs, batch_size)).start()

def load_data_command():
    global X_train, X_test, y_train, y_test, data_X, data_Y
    filepath = filedialog.askopenfilename(title=t("SELECT_DATA_FILE"),
                                           filetypes=[("JSON files", "*.json"), ("All files", "*.*")])
    if not filepath:
        return
    
    # Use IDS.json as default if no file selected
    if not filepath:
        filepath = "IDS.json"
    
    entry_data_path.delete(0, tk.END)
    entry_data_path.insert(0, filepath)
    X, y = load_data(filepath)
    if X is not None and y is not None:
        X = np.array(X)
        y = np.array(y)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        messagebox.showinfo(t("SUCCESS"), t("DATA_LOADED"))
        config['Paths']['data_path'] = filepath
        save_config()
        
        # Update seasons list
        update_seasons_list()
    else:
        X_train, X_test, y_train, y_test = None, None, None, None
        data_X, data_Y = None, None

def update_seasons_list():
    global seasons_list
    # Always use loaded season names if available
    if season_names:
        seasons_list = season_names
    elif data_X is not None:
        # Extract unique seasons from data
        seasons = sorted(set(int(x[0] * 10) for x in data_X))
        seasons_list = [f"Season {s}" for s in seasons]
    else:
        seasons_list = []
        
    combo_season['values'] = seasons_list
    if seasons_list:
        combo_season.set(seasons_list[0])
        update_episodes_list()

def update_episodes_list():
    if data_X is not None and combo_season.get():
        try:
            # Get season number from name
            if season_names:
                season_num = season_names.index(combo_season.get()) + 1
            else:
                season_num = int(combo_season.get().split()[-1])
                
            # Extract episodes for selected season
            episodes = sorted(set(int(x[1] * 1000) for x in data_X if int(x[0] * 10) == season_num))
            # Update entry with available episodes
            entry_episode.delete(0, tk.END)
            if episodes:
                entry_episode.insert(0, episodes[0])
        except:
            pass

def save_model_command():
    global current_model
    if current_model is None:
        messagebox.showerror(t("ERROR"), t("MODEL_NOT_LOADED"))
        return
    
    # Suggest default filename
    default_filename = entry_model_name.get() + ".keras"
    
    filepath = filedialog.asksaveasfilename(
        title=t("SAVE_MODEL"),
        defaultextension=".keras",
        initialfile=default_filename,
        filetypes=[("Keras models", "*.keras"), ("All files", "*.*")]
    )
    
    if not filepath:
        return
    
    # Update the entry field
    entry_save_path.delete(0, tk.END)
    entry_save_path.insert(0, filepath)
    
    # Save the model
    if save_model(current_model, filepath):
        messagebox.showinfo(t("SUCCESS"), f"{t('MODEL_SAVED')}: {filepath}")

def load_model_command():
    global current_model
    filepath = filedialog.askopenfilename(
        title=t("SELECT_MODEL_FILE"),
        filetypes=[("Keras models", "*.keras"), ("All files", "*.*")]
    )
    
    if not filepath:
        return
    
    # Update the entry field
    entry_load_path.delete(0, tk.END)
    entry_load_path.insert(0, filepath)
    
    # Load the model
    current_model = load_model(filepath)
    
    # Update model name field with filename without extension
    if current_model:
        model_name = os.path.splitext(os.path.basename(filepath))[0]
        entry_model_name.delete(0, tk.END)
        entry_model_name.insert(0, model_name)

def validate_episode_number(episode_str):
    """Validate that episode number is a natural number"""
    try:
        episode = int(episode_str)
        if episode <= 0:
            raise ValueError("Episode number must be positive")
        return True, episode
    except ValueError:
        return False, None

def update_time_from_slider():
    """Update time entries based on slider percentage"""
    global updating_time
    if updating_time:
        return
        
    updating_time = True
    
    try:
        # Get percentage from slider
        percentage = scale_timestamp.get()
        
        # Calculate total seconds from length entries
        try:
            hours = int(length_hours_var.get())
            minutes = int(length_minutes_var.get())
            seconds = int(length_seconds_var.get())
            total_seconds = hours * 3600 + minutes * 60 + seconds
        except ValueError:
            total_seconds = 0
        
        if total_seconds > 0:
            # Calculate current time in seconds
            current_seconds = (percentage / 100.0) * total_seconds
            
            # Convert to hours, minutes, seconds
            hours = int(current_seconds // 3600)
            minutes = int((current_seconds % 3600) // 60)
            seconds = int(current_seconds % 60)
            
            # Update time entries
            current_hours_var.set(f"{hours:02d}")
            current_minutes_var.set(f"{minutes:02d}")
            current_seconds_var.set(f"{seconds:02d}")
    except Exception as e:
        print(f"Error updating time from slider: {e}")
    
    updating_time = False

def update_slider_from_time():
    """Update slider based on time entries"""
    global updating_time
    if updating_time:
        return
        
    updating_time = True
    
    try:
        # Get current time from entries
        try:
            current_hours = int(current_hours_var.get())
            current_minutes = int(current_minutes_var.get())
            current_seconds = int(current_seconds_var.get())
            current_total = current_hours * 3600 + current_minutes * 60 + current_seconds
        except ValueError:
            current_total = 0
        
        # Get total seconds from length entries
        try:
            length_hours = int(length_hours_var.get())
            length_minutes = int(length_minutes_var.get())
            length_seconds = int(length_seconds_var.get())
            total_seconds = length_hours * 3600 + length_minutes * 60 + length_seconds
        except ValueError:
            total_seconds = 1  # Avoid division by zero
        
        if total_seconds > 0:
            # Calculate percentage
            percentage = (current_total / total_seconds) * 100
            percentage = max(0, min(100, percentage))  # Clamp between 0-100
            
            # Update slider and label
            scale_timestamp.set(percentage)
            label_timestamp_value.config(text=f"{percentage:.1f}%")
    except Exception as e:
        print(f"Error updating slider from time: {e}")
    
    updating_time = False

def update_time_display(val):
    """Update timestamp label and time entries when slider changes"""
    # Update percentage label
    percentage = float(val)
    label_timestamp_value.config(text=f"{percentage:.1f}%")
    
    # Update time entries
    update_time_from_slider()

def validate_time_entry(P):
    """Validate time entry fields (only digits, max 2 digits)"""
    if P == "" or (P.isdigit() and len(P) <= 2):
        return True
    return False

def predict_point():
    global current_model
    if current_model is None:
        messagebox.showerror(t("ERROR"), t("MODEL_NOT_LOADED"))
        return
    
    try:
        # Get season number from name
        if season_names:
            season = season_names.index(combo_season.get()) + 1
        else:
            season = int(combo_season.get().split()[-1])
            
        # Validate episode number
        episode_str = entry_episode.get()
        is_valid, episode = validate_episode_number(episode_str)
        if not is_valid:
            messagebox.showerror(t("ERROR"), t("INVALID_EPISODE"))
            return
            
        moment = scale_timestamp.get() / 100.0
        
        # Normalize inputs
        season_norm = season / 10.0
        episode_norm = episode / 1000.0
        moment_norm = moment
        
        # Make prediction
        input_data = np.array([[season_norm, episode_norm, moment_norm]])
        predictions = current_model.predict(input_data, verbose=0)[0]
        
        # Update points on canvases
        update_canvas_points(predictions[0], predictions[1], predictions[2], predictions[3])
        
    except Exception as e:
        messagebox.showerror(t("ERROR"), f"{t('ERROR_TRAINING_MODEL')}: {e}")

def predict_episode():
    global is_predicting_episode, episode_prediction_thread
    if current_model is None:
        messagebox.showerror(t("ERROR"), t("MODEL_NOT_LOADED"))
        return
    
    try:
        # Validate episode number before starting prediction
        episode_str = entry_episode.get()
        is_valid, episode = validate_episode_number(episode_str)
        if not is_valid:
            messagebox.showerror(t("ERROR"), t("INVALID_EPISODE"))
            return
    except:
        messagebox.showerror(t("ERROR"), t("INVALID_EPISODE"))
        return
    
    if is_predicting_episode:
        # Stop prediction
        is_predicting_episode = False
        button_predict_episode.config(text=t("PREDICT_EPISODE"))
    else:
        # Start prediction
        is_predicting_episode = True
        button_predict_episode.config(text=t("STOP"))
        episode_prediction_thread = threading.Thread(target=predict_episode_thread)
        episode_prediction_thread.daemon = True
        episode_prediction_thread.start()

def predict_episode_thread():
    global is_predicting_episode
    try:
        # Get season number from name
        if season_names:
            season = season_names.index(combo_season.get()) + 1
        else:
            season = int(combo_season.get().split()[-1])
            
        episode = int(entry_episode.get())
        
        # Normalize inputs
        season_norm = season / 10.0
        episode_norm = episode / 1000.0
        
        # Predict for each moment (0% to 100% with 1% steps)
        start_time = time.time()
        for moment in range(0, 101):
            if not is_predicting_episode:
                break
                
            # Calculate elapsed time and adjust moment to complete in 10 seconds
            elapsed_time = time.time() - start_time
            target_time = moment * 0.1  # 10 seconds for 100% (0.1s per 1%)
            
            if elapsed_time < target_time:
                time.sleep(target_time - elapsed_time)
                
            moment_norm = moment / 100.0
            
            # Make prediction
            input_data = np.array([[season_norm, episode_norm, moment_norm]])
            predictions = current_model.predict(input_data, verbose=0)[0]
            
            # Update UI in main thread
            root.after(0, lambda m=moment, p=predictions: update_episode_prediction_ui(m, p))
            
    except Exception as e:
        root.after(0, lambda: messagebox.showerror(t("ERROR"), f"{t('ERROR_TRAINING_MODEL')}: {e}"))
    
    finally:
        is_predicting_episode = False
        root.after(0, lambda: button_predict_episode.config(text=t("PREDICT_EPISODE")))

def update_episode_prediction_ui(moment, predictions):
    scale_timestamp.set(moment)
    update_canvas_points(predictions[0], predictions[1], predictions[2], predictions[3])
    root.update_idletasks()

def update_canvas_points(x1_norm, y1_norm, x2_norm, y2_norm):
    # Convert normalized coordinates to canvas coordinates
    x1 = x1_norm * 300
    y1 = (1 - y1_norm) * 300  # Invert Y axis
    x2 = x2_norm * 300
    y2 = (1 - y2_norm) * 300  # Invert Y axis
    
    # Update point positions
    canvas_emo.coords(point_emo, x1-5, y1-5, x1+5, y1+5)
    canvas_plot.coords(point_plot, x2-5, y2-5, x2+5, y2+5)
    
    # Update coordinate labels
    x1_var.set(f"{x1_norm:.3f}")
    y1_var.set(f"{y1_norm:.3f}")
    x2_var.set(f"{x2_norm:.3f}")
    y2_var.set(f"{y2_norm:.3f}")

def create_capsule():
    try:
        # First ask for save location
        filepath = filedialog.asksaveasfilename(
            title=t("SAVE_CAPSULE"),
            defaultextension=".lvp",
            filetypes=[("LVP files", "*.lvp"), ("All files", "*.*")]
        )
        if not filepath:
            return
            
        # Get season number from name
        if season_names:
            season = season_names.index(combo_season.get()) + 1
        else:
            season = int(combo_season.get().split()[-1])
            
        # Validate episode number
        episode_str = entry_episode.get()
        is_valid, episode = validate_episode_number(episode_str)
        if not is_valid:
            messagebox.showerror(t("ERROR"), t("INVALID_EPISODE"))
            return
        
        # Create progress window with dark blue theme
        progress_window = tk.Toplevel(root)
        progress_window.title(t("CAPSULE_CREATION_PROGRESS"))
        progress_window.geometry("400x200")
        progress_window.resizable(False, False)
        progress_window.configure(bg='#2c3e50')
        
        # Set capsule window icon
        icon_path = resource_path("lo.ico")
        if os.path.exists(icon_path):
            progress_window.iconbitmap(icon_path)
        
        # Apply dark theme to labels
        style = ttk.Style()
        style.configure("Capsule.TLabel", background="#2c3e50", foreground="#ecf0f1")
        
        ttk.Label(progress_window, text=f"{t('CREATING_CAPSULE')} {season}, {t('EPISODE')} {episode}", 
                 style="Capsule.TLabel").pack(pady=10)
        
        # Create custom progress bar with turquoise color
        progress_frame = tk.Frame(progress_window, bg="#2c3e50")
        progress_frame.pack(pady=10)
        
        progress_bar = ttk.Progressbar(progress_frame, orient=tk.HORIZONTAL, length=350, mode='determinate')
        progress_bar.pack()
        
        # Configure progress bar style for turquoise color
        style.configure("Capsule.Horizontal.TProgressbar", 
                       background="#1abc9c",  # Turquoise color
                       troughcolor="#34495e")
        
        progress_bar.configure(style="Capsule.Horizontal.TProgressbar")
        
        progress_label = ttk.Label(progress_window, text="0.0%", style="Capsule.TLabel")
        progress_label.pack(pady=5)
        
        # Start capsule creation in a separate thread
        threading.Thread(target=create_capsule_thread, 
                        args=(season, episode, filepath, progress_bar, progress_label, progress_window)).start()
        
    except Exception as e:
        messagebox.showerror(t("ERROR"), f"{t('ERROR_CREATING_CAPSULE')}: {e}")

def create_capsule_thread(season, episode, filepath, progress_bar, progress_label, progress_window):
    try:
        # Normalize inputs
        season_norm = season / 10.0
        episode_norm = episode / 1000.0
        
        # Create capsule data structure with new format
        moments = []
        quartets = []
        
        # Predict for 1000 frames (0.0% to 100.0% with 0.1% steps)
        total_frames = 1000
        for i in range(total_frames):
            moment = i * 0.1  # 0.0% to 100.0% with 0.1% steps
            moment_norm = moment / 100.0
            
            # Make prediction
            input_data = np.array([[season_norm, episode_norm, moment_norm]])
            predictions = current_model.predict(input_data, verbose=0)[0]
            
            # Add to capsule data in new format
            moments.append(float(moment))
            quartets.append([
                float(predictions[0]),
                float(predictions[1]),
                float(predictions[2]),
                float(predictions[3])
            ])
            
            # Update progress
            progress_percent = (i + 1) / total_frames * 100
            progress_bar['value'] = progress_percent
            progress_label.config(text=f"{progress_percent:.1f}%")
            progress_window.update_idletasks()
        
        # Save to file with new format
        capsule_data = {
            "season": season,
            "episode": episode,
            "moments": moments,
            "quartets": quartets
        }
        
        with open(filepath, 'w') as f:
            json.dump(capsule_data, f, indent=4)
            
        # Show success message
        progress_window.after(0, lambda: messagebox.showinfo(t("SUCCESS"), f"{t('CAPSULE_SAVED')}: {filepath}"))
        progress_window.after(0, progress_window.destroy)
            
    except Exception as e:
        progress_window.after(0, lambda: messagebox.showerror(t("ERROR"), f"{t('ERROR_CREATING_CAPSULE')}: {e}"))
        progress_window.after(0, progress_window.destroy)

def load_season_names():
    global season_names
    filepath = filedialog.askopenfilename(title=t("SELECT_SEASON_NAMES_FILE"),
                                         filetypes=[("Text files", "*.txt"), ("All files", "*.*")])
    if filepath:
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                season_names = [line.strip() for line in f if line.strip()]
            messagebox.showinfo(t("SUCCESS"), t("SEASON_NAMES_LOADED"))
            # Update seasons list with all loaded names
            update_seasons_list()
        except Exception as e:
            messagebox.showerror(t("ERROR"), f"{t('ERROR_LOADING_SEASON_NAMES')}: {e}")

def toggle_fullscreen():
    global is_fullscreen
    is_fullscreen = not is_fullscreen
    root.attributes("-fullscreen", is_fullscreen)

def on_mousewheel(event):
    canvas.yview_scroll(int(-1 * (event.delta / 120) * 10), "units")

def save_config():
    with open(CONFIG_FILE, 'w') as configfile:
        config.write(configfile)

def load_config():
    global config
    if os.path.exists(CONFIG_FILE):
        config.read(CONFIG_FILE)
    else:
        if 'Paths' not in config:
            config['Paths'] = {}
        config['Paths']['data_path'] = 'IDS.json'
        config['Paths']['save_path'] = ''
        config['Paths']['load_path'] = ''
        save_config()

# Initialize the main Tkinter window
root = tk.Tk()
root.title("LoViewer")

# Load configuration
load_config()

# --- Color Palette (Dark theme like Pointer) ---
BG_COLOR = "#2c3e50"
FRAME_BG = "#34495e"
BUTTON_BG = "#3498db"
BUTTON_FG = "white"
LABEL_BG = BG_COLOR
LABEL_FG = "#ecf0f1"
ENTRY_BG = "white"
ENTRY_FG = "black"
PROGRESS_BG = "#1abc9c"  # Turquoise color

# Apply a Modern Theme
style = ttk.Style()
style.theme_use('clam')
style.configure("TLabel", font=('Arial', 10), background=LABEL_BG, foreground=LABEL_FG)
style.configure("TButton", font=('Arial', 10, 'bold'), padding=8, background=BUTTON_BG, foreground=BUTTON_FG)
style.configure("TEntry", font=('Arial', 10), background=ENTRY_BG, foreground=ENTRY_FG)
style.configure("TFrame", background=FRAME_BG)
style.configure("TLabelframe", background=FRAME_BG)
style.configure("TLabelframe.Label", font=('Arial', 10, 'bold'), background=FRAME_BG, foreground=LABEL_FG)
style.configure("Horizontal.TProgressbar", background=PROGRESS_BG, troughcolor=FRAME_BG)

# Configure window to start in fullscreen but reduced by 5%
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()
window_width = int(screen_width * 0.95)  # 5% reduction
window_height = int(screen_height * 0.95)  # 5% reduction
x_pos = (screen_width - window_width) // 2
y_pos = (screen_height - window_height) // 2
root.geometry(f"{window_width}x{window_height}+{x_pos}+{y_pos}")

# Set window icon
icon_path = resource_path("lo.ico")
if os.path.exists(icon_path):
    root.iconbitmap(icon_path)

root.configure(background=BG_COLOR)

# Create main frame with paned window for resizing
main_paned = ttk.PanedWindow(root, orient=tk.HORIZONTAL)
main_paned.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

# Left frame for model controls
left_frame = ttk.Frame(main_paned, width=window_width//3, relief=tk.RAISED, borderwidth=1)
main_paned.add(left_frame, weight=1)

# Center frame for prediction controls
center_frame = ttk.Frame(main_paned, width=window_width//3, relief=tk.RAISED, borderwidth=1)
main_paned.add(center_frame, weight=3)

# Right frame for dataset controls
right_frame = ttk.Frame(main_paned, width=window_width//4, relief=tk.RAISED, borderwidth=1)
main_paned.add(right_frame, weight=1)

# -------------------- Left Frame: Model Controls --------------------
left_label = ttk.Label(left_frame, text=t("MODEL"), font=('Arial', 12, 'bold'))
left_label.pack(pady=10)

frame_model_save_load = ttk.Frame(left_frame)
frame_model_save_load.pack(pady=5, fill=tk.X, padx=10)

button_save_model = ttk.Button(frame_model_save_load, text=t("SAVE"), command=save_model_command)
button_save_model.pack(side=tk.LEFT, padx=5)

button_load_model = ttk.Button(frame_model_save_load, text=t("LOAD"), command=load_model_command)
button_load_model.pack(side=tk.RIGHT, padx=5)

frame_model_info = ttk.LabelFrame(left_frame, text=t("MODEL_INFORMATION"))
frame_model_info.pack(pady=5, fill=tk.X, padx=10)

label_model_name = ttk.Label(frame_model_info, text=t("MODEL_NAME"))
label_model_name.grid(row=0, column=0, sticky="w", padx=5, pady=5)
entry_model_name = ttk.Entry(frame_model_info, width=20)
entry_model_name.grid(row=0, column=1, padx=5, pady=5)
entry_model_name.insert(0, "my_model")

label_model_success = ttk.Label(frame_model_info, text=f"{t('SUCCESS_RATE')}: N/A")
label_model_success.grid(row=1, column=0, columnspan=2, sticky="w", padx=5, pady=5)

frame_training = ttk.LabelFrame(left_frame, text=t("TRAINING"))
frame_training.pack(pady=5, fill=tk.X, padx=10)

label_epochs = ttk.Label(frame_training, text=t("EPOCHS"))
label_epochs.grid(row=0, column=0, sticky="w", padx=5, pady=5)
entry_epochs = ttk.Entry(frame_training, width=10)
entry_epochs.insert(0, "100")
entry_epochs.grid(row=0, column=1, padx=5, pady=5)

label_batch_size = ttk.Label(frame_training, text=t("BATCH_SIZE"))
label_batch_size.grid(row=1, column=0, sticky="w", padx=5, pady=5)
entry_batch_size = ttk.Entry(frame_training, width=10)
entry_batch_size.insert(0, "10")
entry_batch_size.grid(row=1, column=1, padx=5, pady=5)

button_train = ttk.Button(frame_training, text=t("TRAIN"), command=train_model_command)
button_train.grid(row=2, column=0, columnspan=2, pady=10)

progress_bar = ttk.Progressbar(frame_training, orient=tk.HORIZONTAL, length=200, mode='determinate', style="Horizontal.TProgressbar")
progress_bar.grid(row=3, column=0, columnspan=2, sticky="ew", padx=5, pady=5)

# Add entry fields for save/load paths
label_save_path = ttk.Label(frame_model_info, text=t("SAVE_PATH"))
label_save_path.grid(row=2, column=0, sticky="w", padx=5, pady=5)
entry_save_path = ttk.Entry(frame_model_info, width=20)
entry_save_path.grid(row=2, column=1, padx=5, pady=5)

label_load_path = ttk.Label(frame_model_info, text=t("LOAD_PATH"))
label_load_path.grid(row=3, column=0, sticky="w", padx=5, pady=5)
entry_load_path = ttk.Entry(frame_model_info, width=20)
entry_load_path.grid(row=3, column=1, padx=5, pady=5)

# -------------------- Center Frame: Prediction Controls --------------------
center_label = ttk.Label(center_frame, text=t("PREDICTION"), font=('Arial', 12, 'bold'))
center_label.pack(pady=10)

# Add time input fields like in Pointer
frame_time_input = ttk.Frame(center_frame)
frame_time_input.pack(pady=5, fill=tk.X, padx=10)

# Current time input
label_timecode = ttk.Label(frame_time_input, text=t("TIMECODE"))
label_timecode.grid(row=0, column=0, sticky="w", padx=5, pady=5)

# StringVars for time entries
current_hours_var = tk.StringVar(value="00")
current_minutes_var = tk.StringVar(value="00")
current_seconds_var = tk.StringVar(value="00")
length_hours_var = tk.StringVar(value="00")
length_minutes_var = tk.StringVar(value="20")
length_seconds_var = tk.StringVar(value="00")

# Validation function for time entries
validate_cmd = root.register(validate_time_entry)

# Current time entries
time_frame = ttk.Frame(frame_time_input)
time_frame.grid(row=0, column=1, columnspan=3, sticky="w", padx=5, pady=5)

current_hours_entry = ttk.Entry(time_frame, textvariable=current_hours_var, width=3, validate="key", validatecommand=(validate_cmd, '%P'))
current_hours_entry.pack(side=tk.LEFT)
ttk.Label(time_frame, text=":").pack(side=tk.LEFT)
current_minutes_entry = ttk.Entry(time_frame, textvariable=current_minutes_var, width=3, validate="key", validatecommand=(validate_cmd, '%P'))
current_minutes_entry.pack(side=tk.LEFT)
ttk.Label(time_frame, text=":").pack(side=tk.LEFT)
current_seconds_entry = ttk.Entry(time_frame, textvariable=current_seconds_var, width=3, validate="key", validatecommand=(validate_cmd, '%P'))
current_seconds_entry.pack(side=tk.LEFT)

# Episode length input
label_episode_length = ttk.Label(frame_time_input, text=t("EPISODE_LENGTH"))
label_episode_length.grid(row=1, column=0, sticky="w", padx=5, pady=5)

length_frame = ttk.Frame(frame_time_input)
length_frame.grid(row=1, column=1, columnspan=3, sticky="w", padx=5, pady=5)

length_hours_entry = ttk.Entry(length_frame, textvariable=length_hours_var, width=3, validate="key", validatecommand=(validate_cmd, '%P'))
length_hours_entry.pack(side=tk.LEFT)
ttk.Label(length_frame, text=":").pack(side=tk.LEFT)
length_minutes_entry = ttk.Entry(length_frame, textvariable=length_minutes_var, width=3, validate="key", validatecommand=(validate_cmd, '%P'))
length_minutes_entry.pack(side=tk.LEFT)
ttk.Label(length_frame, text=":").pack(side=tk.LEFT)
length_seconds_entry = ttk.Entry(length_frame, textvariable=length_seconds_var, width=3, validate="key", validatecommand=(validate_cmd, '%P'))
length_seconds_entry.pack(side=tk.LEFT)

# Set trace callbacks for time entries
current_hours_var.trace_add('write', lambda *args: update_slider_from_time())
current_minutes_var.trace_add('write', lambda *args: update_slider_from_time())
current_seconds_var.trace_add('write', lambda *args: update_slider_from_time())
length_hours_var.trace_add('write', lambda *args: update_time_from_slider())
length_minutes_var.trace_add('write', lambda *args: update_time_from_slider())
length_seconds_var.trace_add('write', lambda *args: update_time_from_slider())

frame_selection = ttk.Frame(center_frame)
frame_selection.pack(pady=5, fill=tk.X, padx=10)

label_season = ttk.Label(frame_selection, text=t("SEASON"))
label_season.grid(row=0, column=0, sticky="w", padx=5, pady=5)
combo_season = ttk.Combobox(frame_selection, width=15, state="readonly")
combo_season.grid(row=0, column=1, padx=5, pady=5)
combo_season.bind("<<ComboboxSelected>>", lambda e: update_episodes_list())

label_episode = ttk.Label(frame_selection, text=t("EPISODE"))
label_episode.grid(row=0, column=2, sticky="w", padx=5, pady=5)
entry_episode = ttk.Entry(frame_selection, width=15)
entry_episode.grid(row=0, column=3, padx=5, pady=5)

label_timecode_percent = ttk.Label(frame_selection, text=t("TIMECODE_PERCENT"))
label_timecode_percent.grid(row=1, column=0, sticky="w", padx=5, pady=5)
scale_timestamp = ttk.Scale(frame_selection, from_=0, to=100, orient=tk.HORIZONTAL, length=200)
scale_timestamp.set(0)
scale_timestamp.grid(row=1, column=1, columnspan=3, sticky="ew", padx=5, pady=5)

label_timestamp_value = ttk.Label(frame_selection, text="0.0%")
label_timestamp_value.grid(row=1, column=4, padx=5, pady=5)

# Update timestamp label and time entries when scale changes
scale_timestamp.configure(command=update_time_display)

# Canvases for EmoPlain and PlotPlain
frame_canvases = ttk.Frame(center_frame)
frame_canvases.pack(pady=10, fill=tk.BOTH, expand=True, padx=10)

# Load background images
emo_image = Image.open(resource_path("EmoPlain.png"))
emo_image = emo_image.resize((300, 300), Image.Resampling.LANCZOS)
emo_bg = ImageTk.PhotoImage(emo_image)

plot_image = Image.open(resource_path("PlotPlain.png"))
plot_image = plot_image.resize((300, 300), Image.Resampling.LANCZOS)
plot_bg = ImageTk.PhotoImage(plot_image)

# Create frames for each canvas with coordinates
frame_emo = ttk.Frame(frame_canvases)
frame_emo.pack(side=tk.LEFT, padx=10)

canvas_emo = tk.Canvas(frame_emo, width=300, height=300, bg="white")
canvas_emo.pack()
canvas_emo.create_image(0, 0, anchor="nw", image=emo_bg)
point_emo = canvas_emo.create_oval(145, 145, 155, 155, fill="red", outline="red")

# Coordinate labels for emotion point
emo_coord_frame = ttk.Frame(frame_emo)
emo_coord_frame.pack(pady=5)

ttk.Label(emo_coord_frame, text="X:").pack(side=tk.LEFT)
x1_var = tk.StringVar(value="0.500")
ttk.Label(emo_coord_frame, textvariable=x1_var, width=6).pack(side=tk.LEFT, padx=(0, 10))

ttk.Label(emo_coord_frame, text="Y:").pack(side=tk.LEFT)
y1_var = tk.StringVar(value="0.500")
ttk.Label(emo_coord_frame, textvariable=y1_var, width=6).pack(side=tk.LEFT)

frame_plot = ttk.Frame(frame_canvases)
frame_plot.pack(side=tk.RIGHT, padx=10)

canvas_plot = tk.Canvas(frame_plot, width=300, height=300, bg="white")
canvas_plot.pack()
canvas_plot.create_image(0, 0, anchor="nw", image=plot_bg)
point_plot = canvas_plot.create_oval(145, 145, 155, 155, fill="blue", outline="blue")

# Coordinate labels for plot point
plot_coord_frame = ttk.Frame(frame_plot)
plot_coord_frame.pack(pady=5)

ttk.Label(plot_coord_frame, text="X:").pack(side=tk.LEFT)
x2_var = tk.StringVar(value="0.500")
ttk.Label(plot_coord_frame, textvariable=x2_var, width=6).pack(side=tk.LEFT, padx=(0, 10))

ttk.Label(plot_coord_frame, text="Y:").pack(side=tk.LEFT)
y2_var = tk.StringVar(value="0.500")
ttk.Label(plot_coord_frame, textvariable=y2_var, width=6).pack(side=tk.LEFT)

frame_prediction_buttons = ttk.Frame(center_frame)
frame_prediction_buttons.pack(pady=10, fill=tk.X, padx=10)

button_predict_point = ttk.Button(frame_prediction_buttons, text=t("PREDICT_POINT"), command=predict_point)
button_predict_point.pack(side=tk.LEFT, padx=5)

button_predict_episode = ttk.Button(frame_prediction_buttons, text=t("PREDICT_EPISODE"), command=predict_episode)
button_predict_episode.pack(side=tk.RIGHT, padx=5)

# Button to load season names - moved to center frame near season selection
button_load_season_names = ttk.Button(frame_selection, text=t("LOAD_NAMES"), command=load_season_names, width=20)
button_load_season_names.grid(row=0, column=4, padx=5, pady=5)

# -------------------- Right Frame: Dataset Controls --------------------
right_label = ttk.Label(right_frame, text=t("DATASET"), font=('Arial', 12, 'bold'))
right_label.pack(pady=10)

frame_dataset = ttk.LabelFrame(right_frame, text=t("DATA_LOADING"))
frame_dataset.pack(pady=5, fill=tk.X, padx=10)

label_data_file = ttk.Label(frame_dataset, text=t("DATA_FILE"))
label_data_file.grid(row=0, column=0, sticky="w", padx=5, pady=5)
entry_data_path = ttk.Entry(frame_dataset, width=20)
entry_data_path.grid(row=0, column=1, padx=5, pady=5)
entry_data_path.insert(0, config['Paths'].get('data_path', 'IDS.json'))

button_load_data = ttk.Button(frame_dataset, text=t("LOAD_DATA"), command=load_data_command)
button_load_data.grid(row=1, column=0, columnspan=2, pady=10)

# Add fullscreen button
button_fullscreen = ttk.Button(frame_dataset, text=t("FULLSCREEN"), command=toggle_fullscreen)
button_fullscreen.grid(row=2, column=0, columnspan=2, pady=10)

# Add change language button
button_change_language = ttk.Button(frame_dataset, text=t("CHANGE_LANGUAGE"), command=change_language,width=20)
button_change_language.grid(row=3, column=0, columnspan=2, pady=10)

frame_capsule = ttk.LabelFrame(right_frame, text=t("CAPSULE_CREATION"))
frame_capsule.pack(pady=5, fill=tk.X, padx=10)

button_create_capsule = ttk.Button(frame_capsule, text=t("CREATE_CAPSULE"), command=create_capsule)
button_create_capsule.pack(pady=10)

# Initialize time display
update_time_from_slider()

# Start the Tkinter main loop
root.mainloop()
