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

def load_data(filepath):
    global data_X, data_Y
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        data_X = np.array(data['X'])
        data_Y = np.array(data['Y'])
        return data_X, data_Y
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred loading: {e}")
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
        messagebox.showinfo("Success", f"Model saved as {model_name}")
        print(f'Model saved as {model_name}')

        # Save to config
        config['Paths']['save_path'] = model_name
        save_config()

    except Exception as e:
        messagebox.showerror("Error", f"Error saving model: {e}")

def load_model(model_name):
    try:
        model = keras.models.load_model(model_name)
        messagebox.showinfo("Success", f"Model loaded from {model_name}")
        config['Paths']['load_path'] = model_name
        save_config()
        return model
    except Exception as e:
        messagebox.showerror("Error", f"Error loading model: {e}")
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
        label_model_success.config(text=f'Успешность: {success_rate:.2f}%')
        messagebox.showinfo("Успех", "Модель успешно обучена!")
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred during training: {e}")
    finally:
        progress_bar['value'] = 0
        button_train['state'] = 'normal'

def train_model_command():
    global X_train, X_test, y_train, y_test
    if X_train is None or y_train is None or X_test is None or y_test is None:
        messagebox.showerror("Error", "Data not loaded. Please load data first.")
        return
    try:
        epochs = int(entry_epochs.get())
        batch_size = int(entry_batch_size.get())
    except ValueError:
        messagebox.showerror("Error", "Invalid input. Please enter integer values for epochs and batch size.")
        return
    button_train['state'] = 'disabled'
    threading.Thread(target=train_model_thread, args=(X_train, y_train, X_test, y_test, epochs, batch_size)).start()

def load_data_command():
    global X_train, X_test, y_train, y_test, data_X, data_Y
    filepath = filedialog.askopenfilename(title="Select Data File",
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
        messagebox.showinfo("Success", "Data loaded successfully.")
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
        messagebox.showerror("Error", "No model to save. Please train or load a model first.")
        return
    
    # Suggest default filename
    default_filename = entry_model_name.get() + ".keras"
    
    filepath = filedialog.asksaveasfilename(
        title="Сохранить модель",
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
        messagebox.showinfo("Успех", f"Модель успешно сохранена как {filepath}")

def load_model_command():
    global current_model
    filepath = filedialog.askopenfilename(
        title="Выберите файл модели",
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

def predict_point():
    global current_model
    if current_model is None:
        messagebox.showerror("Error", "Model not loaded or trained. Please load or train a model first.")
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
            messagebox.showerror("Ошибка", "Номер эпизода должен быть натуральным числом (1, 2, 3, ...)")
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
        messagebox.showerror("Error", f"An error occurred during prediction: {e}")

def predict_episode():
    global is_predicting_episode, episode_prediction_thread
    if current_model is None:
        messagebox.showerror("Error", "Model not loaded or trained. Please load or train a model first.")
        return
    
    try:
        # Validate episode number before starting prediction
        episode_str = entry_episode.get()
        is_valid, episode = validate_episode_number(episode_str)
        if not is_valid:
            messagebox.showerror("Ошибка", "Номер эпизода должен быть натуральным числом (1, 2, 3, ...)")
            return
    except:
        messagebox.showerror("Ошибка", "Некорректный номер эпизода")
        return
    
    if is_predicting_episode:
        # Stop prediction
        is_predicting_episode = False
        button_predict_episode.config(text="Предсказание эпизода")
    else:
        # Start prediction
        is_predicting_episode = True
        button_predict_episode.config(text="Остановить")
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
        for moment in range(0, 101):
            if not is_predicting_episode:
                break
                
            moment_norm = moment / 100.0
            
            # Make prediction
            input_data = np.array([[season_norm, episode_norm, moment_norm]])
            predictions = current_model.predict(input_data, verbose=0)[0]
            
            # Update UI in main thread
            root.after(0, lambda m=moment, p=predictions: update_episode_prediction_ui(m, p))
            
            # Small delay for animation
            time.sleep(0.05)
            
    except Exception as e:
        root.after(0, lambda: messagebox.showerror("Error", f"An error occurred during episode prediction: {e}"))
    
    finally:
        is_predicting_episode = False
        root.after(0, lambda: button_predict_episode.config(text="Предсказание эпизода"))

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

def create_capsule():
    try:
        # First ask for save location
        filepath = filedialog.asksaveasfilename(
            title="Сохранить капсулу",
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
            messagebox.showerror("Ошибка", "Номер эпизода должен быть натуральным числом (1, 2, 3, ...)")
            return
        
        # Create progress window
        progress_window = tk.Toplevel(root)
        progress_window.title("Создание капсулы")
        progress_window.geometry("400x200")
        progress_window.resizable(False, False)
        
        # Set capsule window icon
        icon_path = resource_path("lo.ico")
        if os.path.exists(icon_path):
            progress_window.iconbitmap(icon_path)
        
        ttk.Label(progress_window, text=f"Создание капсулы для сезона {season}, эпизода {episode}").pack(pady=10)
        
        progress_bar = ttk.Progressbar(progress_window, orient=tk.HORIZONTAL, length=350, mode='determinate')
        progress_bar.pack(pady=10)
        
        progress_label = ttk.Label(progress_window, text="0.0%")
        progress_label.pack(pady=5)
        
        # Start capsule creation in a separate thread
        threading.Thread(target=create_capsule_thread, 
                        args=(season, episode, filepath, progress_bar, progress_label, progress_window)).start()
        
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred creating capsule: {e}")

def create_capsule_thread(season, episode, filepath, progress_bar, progress_label, progress_window):
    try:
        # Normalize inputs
        season_norm = season / 10.0
        episode_norm = episode / 1000.0
        
        # Create capsule data structure with new format
        moments = []
        quartets = []
        
        # Predict for 500 frames (0.0% to 100.0% with 0.2% steps)
        total_frames = 500
        for i in range(total_frames):
            moment = i * 0.2  # 0.0% to 100.0% with 0.2% steps
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
        progress_window.after(0, lambda: messagebox.showinfo("Успех", f"Капсула успешно сохранена как {filepath}"))
        progress_window.after(0, progress_window.destroy)
            
    except Exception as e:
        progress_window.after(0, lambda: messagebox.showerror("Error", f"An error occurred creating capsule: {e}"))
        progress_window.after(0, progress_window.destroy)

def load_season_names():
    global season_names
    filepath = filedialog.askopenfilename(title="Select Season Names File",
                                         filetypes=[("Text files", "*.txt"), ("All files", "*.*")])
    if filepath:
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                season_names = [line.strip() for line in f if line.strip()]
            messagebox.showinfo("Success", "Season names loaded successfully.")
            # Update seasons list with all loaded names
            update_seasons_list()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load season names: {e}")

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

# --- Color Palette ---
BG_COLOR = "#E6EE9C"
FRAME_BG = "#F0F4C3"
BUTTON_BG = "#AED581"
BUTTON_FG = "#33691E"
LABEL_BG = BG_COLOR
LABEL_FG = "#2E7D32"
ENTRY_BG = "#FFFFCC"
ENTRY_FG = "black"

# Apply a Modern Theme
style = ttk.Style()
style.theme_use('clam')
style.configure("TLabel", font=('Arial', 10), background=LABEL_BG, foreground=LABEL_FG)
style.configure("TButton", font=('Arial', 10, 'bold'), padding=8, background=BUTTON_BG, foreground=BUTTON_FG)
style.configure("TEntry", font=('Arial', 10), background=ENTRY_BG, foreground=ENTRY_FG)
style.configure("TFrame", background=FRAME_BG)
style.configure("TLabelframe", background=FRAME_BG)
style.configure("TLabelframe.Label", font=('Arial', 10, 'bold'))

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
main_paned.add(center_frame, weight=2)

# Right frame for dataset controls
right_frame = ttk.Frame(main_paned, width=window_width//3, relief=tk.RAISED, borderwidth=1)
main_paned.add(right_frame, weight=1)

# -------------------- Left Frame: Model Controls --------------------
ttk.Label(left_frame, text="МОДЕЛЬ", font=('Arial', 12, 'bold')).pack(pady=10)

frame_model_save_load = ttk.Frame(left_frame)
frame_model_save_load.pack(pady=5, fill=tk.X, padx=10)

button_save_model = ttk.Button(frame_model_save_load, text="Сохранить", command=save_model_command)
button_save_model.pack(side=tk.LEFT, padx=5)

button_load_model = ttk.Button(frame_model_save_load, text="Загрузить", command=load_model_command)
button_load_model.pack(side=tk.RIGHT, padx=5)

frame_model_info = ttk.LabelFrame(left_frame, text="Информация о модели")
frame_model_info.pack(pady=5, fill=tk.X, padx=10)

ttk.Label(frame_model_info, text="Имя модели:").grid(row=0, column=0, sticky="w", padx=5, pady=5)
entry_model_name = ttk.Entry(frame_model_info, width=20)
entry_model_name.grid(row=0, column=1, padx=5, pady=5)
entry_model_name.insert(0, "my_model")

label_model_success = ttk.Label(frame_model_info, text="Успешность: N/A")
label_model_success.grid(row=1, column=0, columnspan=2, sticky="w", padx=5, pady=5)

frame_training = ttk.LabelFrame(left_frame, text="Обучение")
frame_training.pack(pady=5, fill=tk.X, padx=10)

ttk.Label(frame_training, text="Эпохи:").grid(row=0, column=0, sticky="w", padx=5, pady=5)
entry_epochs = ttk.Entry(frame_training, width=10)
entry_epochs.insert(0, "100")
entry_epochs.grid(row=0, column=1, padx=5, pady=5)

ttk.Label(frame_training, text="Размер батча:").grid(row=1, column=0, sticky="w", padx=5, pady=5)
entry_batch_size = ttk.Entry(frame_training, width=10)
entry_batch_size.insert(0, "10")
entry_batch_size.grid(row=1, column=1, padx=5, pady=5)

button_train = ttk.Button(frame_training, text="ОБУЧЕНИЕ", command=train_model_command)
button_train.grid(row=2, column=0, columnspan=2, pady=10)

progress_bar = ttk.Progressbar(frame_training, orient=tk.HORIZONTAL, length=200, mode='determinate')
progress_bar.grid(row=3, column=0, columnspan=2, sticky="ew", padx=5, pady=5)

# Add entry fields for save/load paths
ttk.Label(frame_model_info, text="Путь сохранения:").grid(row=2, column=0, sticky="w", padx=5, pady=5)
entry_save_path = ttk.Entry(frame_model_info, width=20)
entry_save_path.grid(row=2, column=1, padx=5, pady=5)

ttk.Label(frame_model_info, text="Путь загрузки:").grid(row=3, column=0, sticky="w", padx=5, pady=5)
entry_load_path = ttk.Entry(frame_model_info, width=20)
entry_load_path.grid(row=3, column=1, padx=5, pady=5)

# -------------------- Center Frame: Prediction Controls --------------------
ttk.Label(center_frame, text="ПРЕДСКАЗАНИЕ", font=('Arial', 12, 'bold')).pack(pady=10)

frame_selection = ttk.Frame(center_frame)
frame_selection.pack(pady=5, fill=tk.X, padx=10)

ttk.Label(frame_selection, text="Сезон:").grid(row=0, column=0, sticky="w", padx=5, pady=5)
combo_season = ttk.Combobox(frame_selection, width=15, state="readonly")
combo_season.grid(row=0, column=1, padx=5, pady=5)
combo_season.bind("<<ComboboxSelected>>", lambda e: update_episodes_list())

ttk.Label(frame_selection, text="Эпизод:").grid(row=0, column=2, sticky="w", padx=5, pady=5)
entry_episode = ttk.Entry(frame_selection, width=15)
entry_episode.grid(row=0, column=3, padx=5, pady=5)

ttk.Label(frame_selection, text="Таймкод (%):").grid(row=1, column=0, sticky="w", padx=5, pady=5)
scale_timestamp = ttk.Scale(frame_selection, from_=0, to=100, orient=tk.HORIZONTAL, length=200)
scale_timestamp.set(50)
scale_timestamp.grid(row=1, column=1, columnspan=3, sticky="ew", padx=5, pady=5)

label_timestamp_value = ttk.Label(frame_selection, text="50.0%")
label_timestamp_value.grid(row=1, column=4, padx=5, pady=5)

# Update timestamp label when scale changes
def update_timestamp_label(val):
    label_timestamp_value.config(text=f"{float(val):.1f}%")
scale_timestamp.configure(command=update_timestamp_label)

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

canvas_emo = tk.Canvas(frame_canvases, width=300, height=300, bg="white")
canvas_emo.pack(side=tk.LEFT, padx=10)
canvas_emo.create_image(0, 0, anchor="nw", image=emo_bg)
point_emo = canvas_emo.create_oval(145, 145, 155, 155, fill="red", outline="red")

canvas_plot = tk.Canvas(frame_canvases, width=300, height=300, bg="white")
canvas_plot.pack(side=tk.RIGHT, padx=10)
canvas_plot.create_image(0, 0, anchor="nw", image=plot_bg)
point_plot = canvas_plot.create_oval(145, 145, 155, 155, fill="blue", outline="blue")

frame_prediction_buttons = ttk.Frame(center_frame)
frame_prediction_buttons.pack(pady=10, fill=tk.X, padx=10)

button_predict_point = ttk.Button(frame_prediction_buttons, text="Предсказание кусочка", command=predict_point)
button_predict_point.pack(side=tk.LEFT, padx=5)

button_predict_episode = ttk.Button(frame_prediction_buttons, text="Предсказание эпизода", command=predict_episode)
button_predict_episode.pack(side=tk.RIGHT, padx=5)

# Button to load season names - moved to center frame near season selection
button_load_season_names = ttk.Button(frame_selection, text="Загр. названия", command=load_season_names, width=12)
button_load_season_names.grid(row=0, column=4, padx=5, pady=5)

# -------------------- Right Frame: Dataset Controls --------------------
ttk.Label(right_frame, text="ДАТАСЕТ", font=('Arial', 12, 'bold')).pack(pady=10)

frame_dataset = ttk.LabelFrame(right_frame, text="Загрузка данных")
frame_dataset.pack(pady=5, fill=tk.X, padx=10)

ttk.Label(frame_dataset, text="Файл данных:").grid(row=0, column=0, sticky="w", padx=5, pady=5)
entry_data_path = ttk.Entry(frame_dataset, width=20)
entry_data_path.grid(row=0, column=1, padx=5, pady=5)
entry_data_path.insert(0, config['Paths'].get('data_path', 'IDS.json'))

button_load_data = ttk.Button(frame_dataset, text="Загрузить данные", command=load_data_command)
button_load_data.grid(row=1, column=0, columnspan=2, pady=10)

frame_capsule = ttk.LabelFrame(right_frame, text="Создание капсулы")
frame_capsule.pack(pady=5, fill=tk.X, padx=10)

button_create_capsule = ttk.Button(frame_capsule, text="СОЗДАТЬ КАПСУЛУ", command=create_capsule)
button_create_capsule.pack(pady=10)

# Start the Tkinter main loop
root.mainloop()
