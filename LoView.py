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

# --- Functions for creating executable ---
def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)


# Загрузка данных из файла IDS.json
def load_data(filepath):
    """Загружает данные из JSON файла."""
    global data_X, data_Y #Access the global X and Y
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        data_X = np.array(data['X'])
        data_Y = np.array(data['Y'])
        return data_X, data_Y
    except FileNotFoundError:
        messagebox.showerror("Error", f"File not found: {filepath}")
        return None, None
    except json.JSONDecodeError:
        messagebox.showerror("Error", f"Invalid JSON format in file: {filepath}")
        return None, None
    except KeyError as e:
        messagebox.showerror("Error", f"Missing key in JSON: {e}")
        return None, None
    except Exception as e:
         messagebox.showerror("Error", f"An unexpected error occurred loading {e}")
         return None, None

# Функция для создания и обучения модели
def create_and_train_model(X_train, y_train, X_test, y_test, epochs=100, batch_size=10, progress_callback=None): # Added progress_callback
    """Создает, компилирует, обучает и оценивает модель нейронной сети."""
    hidden_layer_size = 128  # Default value for hidden layer size
    model = keras.Sequential()
    model.add(layers.Dense(hidden_layer_size, activation='relu', input_shape=(3,)))
    model.add(layers.Dense(hidden_layer_size, activation='relu')),
    model.add(layers.Dense(hidden_layer_size // 2, activation='relu'))  # Новый скрытый слой
    model.add(layers.Dense(4))  # Выходной слой с 4 нейронами

    model.compile(optimizer='adam', loss='mean_squared_error')

    # Modify the training loop to use the callback
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.2, verbose=0,
                        callbacks=[keras.callbacks.LambdaCallback(on_epoch_end=lambda epoch, logs: progress_callback(epoch + 1, epochs))])

    test_loss = model.evaluate(X_test, y_test, verbose=0)
    print(f'Test loss: {test_loss}')

    return model, history, test_loss

# Функция для сохранения модели
def save_model(model, model_name):
    """Сохраняет модель в формате keras."""
    try:
        model.save(model_name)
        messagebox.showinfo("Success", f"Model saved as {model_name}")
        print(f'Model saved as {model_name}')
    except Exception as e:
        messagebox.showerror("Error", f"Error saving model: {e}")

# Функция для загрузки модели
def load_model(model_name):
    """Загружает модель из файла keras."""
    try:
        model = keras.models.load_model(model_name)
        messagebox.showinfo("Success", f"Model loaded from {model_name}")
        print(f'Model loaded from {model_name}')
        return model
    except FileNotFoundError:
        messagebox.showerror("Error", f"Model file not found: {model_name}")
        return None
    except Exception as e:
        messagebox.showerror("Error", f"Error loading model: {e}")
        return None

# --- UI Functions ---
def predict_data():
    """Получает данные из полей ввода, делает предсказание и выводит результат."""
    try:
        x1 = float(entry_x1.get())
        x2 = float(entry_x2.get())
        x3_input = entry_x3.get().lower()

        if x3_input == "all":
            x3_values = list(range(1, 11))  # Список от 1 до 10 включительно
        else:
            try:
                x3_values = [float(x3_input)]  # Попытка преобразовать введенное значение в число
            except ValueError:
                messagebox.showerror("Error", "Invalid input for X3. Please enter a number or 'all'.")
                return

        if current_model is None:
            messagebox.showerror("Error", "Model not loaded or trained. Please load or train a model first.")
            return

        all_predictions = []
        for x3 in x3_values:
            input_data = np.array([[x1, x2, x3]])
            predictions = current_model.predict(input_data, verbose=0)
            all_predictions.append([round(pred, 2) for pred in predictions[0]]) # Rounded to 2 digits

        # Store predictions in the GUI's dictionary.
        root.gui_predictions = all_predictions

        update_predictions_answers_display()

    except ValueError:
        messagebox.showerror("Error", "Invalid input. Please enter numeric values for X1 and X2.")
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred during prediction: {e}")

def show_answer():
    """Получает данные из полей ввода и показывает соответствующий "правильный ответ" из датасета."""
    global data_X, data_Y

    if data_X is None or data_Y is None:
        messagebox.showerror("Error", "Data not loaded. Please load data first.")
        return

    try:
        x1 = float(entry_x1.get())
        x2 = float(entry_x2.get())
        x3_input = entry_x3.get().lower()

        if x3_input == "all":
            x3_values = list(range(1, 11))  # Список от 1 до 10 включительно
        else:
            try:
                x3_values = [float(x3_input)]  # Попытка преобразовать введенное значение в число
            except ValueError:
                messagebox.showerror("Error", "Invalid input for X3. Please enter a number or 'all'.")
                return

        all_answers = []
        for x3 in x3_values:
            input_data = np.array([x1, x2, x3])  # Convert to numpy array

            # Find the matching data point in the dataset
            match_index = np.where(np.all(data_X == input_data, axis=1))

            if match_index[0].size > 0:  # Found a match
                answer = data_Y[match_index[0][0]]  # Get the corresponding answer
                all_answers.append([round(ans, 2) for ans in answer.tolist()]) # Rounded to 2 digits
            else:
                all_answers.append("idk lol")

        # Store answers in the GUI's dictionary.
        root.gui_answers = all_answers

        update_predictions_answers_display()

    except ValueError:
        messagebox.showerror("Error", "Invalid input. Please enter numeric values for X1 and X2.")
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {e}")

def update_predictions_answers_display():
    """Displays predictions and answers side by side."""
    predictions = getattr(root, "gui_predictions", [])
    answers = getattr(root, "gui_answers", [])

    output_text = ""
    max_lines = max(len(predictions), len(answers))

    for i in range(max_lines):
        prediction_str = str(predictions[i]) if i < len(predictions) else ""
        answer_str = str(answers[i]) if i < len(answers) else ""
        output_text += f"Prediction: {prediction_str:<30} Answer: {answer_str}\n"

    label_prediction.config(text=output_text)
    print(output_text)


def update_progress_bar(epoch, total_epochs):
    """Updates the progress bar."""
    progress_value = int((epoch / total_epochs) * 100)
    progress_bar['value'] = progress_value
    root.update_idletasks()  # Force UI update

def train_model_thread(X_train, y_train, X_test, y_test, epochs, batch_size):
    """Trains the model in a separate thread."""
    global current_model, training_history, test_loss  # Declare globals at the beginning

    try:
        current_model, training_history, test_loss = create_and_train_model(
            X_train, y_train, X_test, y_test, epochs, batch_size,
            progress_callback=update_progress_bar  # Pass the callback
        )

        label_test_loss.config(text=f'Test loss: {test_loss}')
        messagebox.showinfo("Success", "Model trained successfully.")
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred during training: {e}")
    finally:
        progress_bar['value'] = 0  # Reset the progress bar
        button_train['state'] = 'normal'  # Re-enable the Train button
        button_show_history['state'] = 'normal'  # Enable the history button after training


def train_model_command():
    """Обработчик нажатия кнопки "Train Model"."""
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

    # Disable the Train button to prevent multiple training threads
    button_train['state'] = 'disabled'
    button_show_history['state'] = 'disabled'  # Disable the history button during training

    # Start the training process in a separate thread
    threading.Thread(target=train_model_thread, args=(X_train, y_train, X_test, y_test, epochs, batch_size)).start()


def load_data_command():
    """Обработчик нажатия кнопки "Load Data"."""
    global X_train, X_test, y_train, y_test, data_X, data_Y
    filepath = filedialog.askopenfilename(title="Select Data File", filetypes=[("JSON files", "*.json"), ("All files", "*.*")])  # File dialog
    if not filepath:  # User cancelled the dialog
        return

    entry_data_path.delete(0, tk.END)  # Clear the entry field
    entry_data_path.insert(0, filepath) # Set the entry field

    X, y = load_data(filepath)

    if X is not None and y is not None:
        X = np.array(X)
        y = np.array(y)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        messagebox.showinfo("Success", "Data loaded successfully.")
    else:
        X_train, X_test, y_train, y_test = None, None, None, None
        data_X, data_Y = None, None

def save_model_command():
    """Обработчик нажатия кнопки "Save Model"."""
    global current_model

    if current_model is None:
        messagebox.showerror("Error", "No model to save. Please train or load a model first.")
        return

    filepath = filedialog.asksaveasfilename(title="Save Model", defaultextension=".keras", filetypes=[("Keras models", "*.keras"), ("All files", "*.*")])
    if not filepath:
        return

    entry_save_path.delete(0, tk.END)
    entry_save_path.insert(0, filepath)

    model_name = filepath
    save_model(current_model, model_name)

def load_model_command():
    """Обработчик нажатия кнопки "Load Model"."""
    global current_model

    filepath = filedialog.askopenfilename(title="Select Model File", filetypes=[("Keras models", "*.keras"), ("All files", "*.*")])
    if not filepath:
        return

    entry_load_path.delete(0, tk.END)
    entry_load_path.insert(0, filepath)

    model_name = filepath
    current_model = load_model(model_name)

def plot_training_history(history):
    """Отображает график истории обучения (loss) в отдельном окне Tkinter."""

    history_window = tk.Toplevel(root)
    history_window.title("Training History")

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(history.history['loss'], label='loss')
    ax.plot(history.history['val_loss'], label='val_loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training History')
    ax.legend()
    fig.tight_layout()

    canvas = FigureCanvasTkAgg(fig, master=history_window)
    canvas_widget = canvas.get_tk_widget()
    canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    toolbar = NavigationToolbar2Tk(canvas, history_window)
    toolbar.update()
    canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    canvas.draw()

def plot_predictions_answers():
    """Plots the predictions and answers as a function of X3."""
    global data_X, data_Y

    if data_X is None or data_Y is None:
        messagebox.showerror("Error", "Data not loaded. Please load data first.")
        return

    try:
        x1_val = float(entry_x1.get())
        x2_val = float(entry_x2.get())

        # Use extended range for plotting only
        x3_values = np.arange(0, 10.01, 0.01).tolist()

        if current_model is None:
            messagebox.showerror("Error", "Model not loaded or trained. Please load or train a model first.")
            return

        # Create a dictionary to store X3, predictions, and answers
        plot_data = {}
        for x3 in x3_values:
            # Create a single X value.
            key = (x1_val, x2_val, x3)

            # Append prediction to the dictionary.
            input_data = np.array([[x1_val, x2_val, x3]])
            predictions = current_model.predict(input_data, verbose=0)[0].tolist()
            plot_data[key] = {"prediction": predictions}

            # Add the answers to the dictionary.
            input_data = np.array([x1_val, x2_val, x3])  # Convert to numpy array
            match_index = np.where(np.all(data_X == input_data, axis=1))
            if match_index[0].size > 0:  # Found a match
                plot_data[key]["answer"] = data_Y[match_index[0][0]].tolist()
            else:
                plot_data[key]["answer"] = None

        create_plots(plot_data, x1_val, x2_val)  # Pass x1_val and x2_val

    except ValueError:
        messagebox.showerror("Error", "Invalid input. Please enter numeric values for X1 and X2.")
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {e}")


def create_plots(plot_data, x1_val, x2_val):
    """Create two new windows and plot the prediction and answers data on them."""

    # Create the answers window and display the answers.
    answers_window = tk.Toplevel(root)
    answers_window.title(f"Answers of Se{x1_val}Ep{x2_val}") # Set the title of the window

    # Create the predictions window and display the predictions.
    predictions_window = tk.Toplevel(root)
    predictions_window.title(f"Prediction of Se{x1_val}Ep{x2_val}") # Set the title of the window

    x_values = list(plot_data.keys())
    answers_figure, answers_axes = plt.subplots(figsize=(8, 6))
    predictions_figure, predictions_axes = plt.subplots(figsize=(8, 6))

    # Plot answers for each of the y-values.
    for i in range(4):
        y_values = []
        for x in x_values:
            if plot_data[x]["answer"] is not None:
                y_values.append(plot_data[x]["answer"][i])
            else:
                y_values.append(None)
        # Filter out the x-values that don't have a valid y-value
        valid_x_values = [x[2] for i, x in enumerate(x_values) if y_values[i] is not None]
        valid_y_values = [y for y in y_values if y is not None]

        answers_axes.plot(valid_x_values, valid_y_values, label=f"Y_{i+1}")

    # Plot predictions for each of the y-values.
    for i in range(4):
        y_values = []
        for x in x_values:
            y_values.append(plot_data[x]["prediction"][i])
        predictions_axes.plot([x[2] for x in x_values], y_values, label=f"Y_{i + 1}")

    # Add some plot fluff.
    answers_axes.set_xlabel("X3")
    answers_axes.set_ylabel("Answer Values")
    answers_axes.set_title("Answers vs. X3")
    answers_axes.legend()
    answers_axes.grid(True)

    # Add some plot fluff.
    predictions_axes.set_xlabel("X3")
    predictions_axes.set_ylabel("Predicted Values")
    predictions_axes.set_title("Predictions vs. X3")
    predictions_axes.legend()
    predictions_axes.grid(True)

    # --- Add Matplotlib toolbar to the Answers Window ---
    answers_canvas = FigureCanvasTkAgg(answers_figure, master=answers_window)
    answers_canvas_widget = answers_canvas.get_tk_widget()
    answers_canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    answers_toolbar = NavigationToolbar2Tk(answers_canvas, answers_window)
    answers_toolbar.update()
    answers_canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
    answers_canvas.draw()

    # --- Add Matplotlib toolbar to the Predictions Window ---
    predictions_canvas = FigureCanvasTkAgg(predictions_figure, master=predictions_window)
    predictions_canvas_widget = predictions_canvas.get_tk_widget()
    predictions_canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    predictions_toolbar = NavigationToolbar2Tk(predictions_canvas, predictions_window)
    predictions_toolbar.update()
    predictions_canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
    predictions_canvas.draw()


def open_graphs():
    """Opens the predictions/answers plots."""
    plot_predictions_answers()

def open_training_history():
    """Opens the training history plot."""
    global training_history

    if training_history is None:
        messagebox.showerror("Error", "No training history available. Please train a model first.")
        return

    plot_training_history(training_history)

def toggle_fullscreen():
    """Toggles fullscreen mode."""
    global is_fullscreen
    is_fullscreen = not is_fullscreen  # Toggle the boolean
    root.attributes("-fullscreen", is_fullscreen)

def on_mousewheel(event):
    """Handles mousewheel scrolling with increased delta."""
    canvas.yview_scroll(int(-1*(event.delta/120)*10), "units") # Increase scroll speed by factor of 10

# --- Global Variables ---
X_train, X_test, y_train, y_test = None, None, None, None
current_model = None
training_history = None
test_loss = None
data_X = None
data_Y = None
is_fullscreen = False #Variable for fullscreen

# --- UI Setup ---
root = tk.Tk()
root.title("LoViewer")

# Get screen width and height
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()

# Calculate 1.5 times the screen width for a wider window
window_width = int(screen_width * 1.5)
# Use a max width to limit the window size
max_window_width = 1200
window_width = min(window_width, max_window_width)

# Reduce the height by 30%
window_height = int(screen_height * 0.7)  # Reduce by 30%

# Set window position
x = (screen_width - window_width) // 2
y = (screen_height - window_height) // 2

root.geometry(f"{window_width}x{window_height}+{x}+{y}")

icon_path = resource_path("lo.ico")  # Replace with your icon file
root.iconbitmap(icon_path) # Set the icon
# Remove window resizing ability
root.resizable(False, False)

# --- Main Frame with Scrollbar ---
main_frame = ttk.Frame(root, padding=10)
main_frame.pack(fill=tk.BOTH, expand=True)

canvas = tk.Canvas(main_frame)
canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

scrollbar = ttk.Scrollbar(main_frame, orient=tk.VERTICAL, command=canvas.yview)
scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

canvas.configure(yscrollcommand=scrollbar.set)
canvas.bind('<Configure>', lambda e: canvas.configure(scrollregion=canvas.bbox("all")))

inner_frame = ttk.Frame(canvas, padding=10)
canvas.create_window((0, 0), window=inner_frame, anchor="nw")

# Bind mousewheel scrolling to the canvas
canvas.bind_all("<MouseWheel>", on_mousewheel)

# --- Data Loading Frame ---
frame_data = ttk.LabelFrame(inner_frame, text="Data Loading", padding=10)
frame_data.pack(fill=tk.X, padx=10, pady=10)

label_data_path = ttk.Label(frame_data, text="Data File Path:")
label_data_path.pack(side=tk.LEFT, padx=5)
entry_data_path = ttk.Entry(frame_data, width=50)
entry_data_path.insert(0, 'IDS.json') # Default value
entry_data_path.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=5)
button_load_data = ttk.Button(frame_data, text="Load Data", command=load_data_command)
button_load_data.pack(side=tk.LEFT, padx=5)

# --- Model Training Frame ---
frame_training = ttk.LabelFrame(inner_frame, text="Model Training", padding=10)
frame_training.pack(fill=tk.X, padx=10, pady=10)

label_epochs = ttk.Label(frame_training, text="Epochs:")
label_epochs.pack(side=tk.LEFT, padx=5)
entry_epochs = ttk.Entry(frame_training, width=10)
entry_epochs.insert(0, "100")  # Default value
entry_epochs.pack(side=tk.LEFT, padx=5)

label_batch_size = ttk.Label(frame_training, text="Batch Size:")
label_batch_size.pack(side=tk.LEFT, padx=5)
entry_batch_size = ttk.Entry(frame_training, width=10)
entry_batch_size.insert(0, "10")  # Default value
entry_batch_size.pack(side=tk.LEFT, padx=5)

button_train = ttk.Button(frame_training, text="Train Model", command=train_model_command)
button_train.pack(side=tk.LEFT, padx=5)

# Progress Bar
progress_bar = ttk.Progressbar(frame_training, orient=tk.HORIZONTAL, length=200, mode='determinate')
progress_bar.pack(side=tk.LEFT, padx=5)

# Button to show training history (initially disabled)
button_show_history = ttk.Button(frame_training, text="Show History", command=open_training_history, state='disabled')
button_show_history.pack(side=tk.LEFT, padx=5)

label_test_loss = ttk.Label(frame_training, text="Test Loss: N/A")
label_test_loss.pack(side=tk.LEFT, padx=5)

# --- Prediction Frame ---
frame_prediction = ttk.LabelFrame(inner_frame, text="Prediction", padding=10)
frame_prediction.pack(fill=tk.X, padx=10, pady=10)

label_x1 = ttk.Label(frame_prediction, text="X1:")
label_x1.pack(side=tk.LEFT, padx=5)
entry_x1 = ttk.Entry(frame_prediction, width=10)
entry_x1.pack(side=tk.LEFT, padx=5)

label_x2 = ttk.Label(frame_prediction, text="X2:")
label_x2.pack(side=tk.LEFT, padx=5)
entry_x2 = ttk.Entry(frame_prediction, width=10)
entry_x2.pack(side=tk.LEFT, padx=5)

label_x3 = ttk.Label(frame_prediction, text="X3 (Number or 'all'):")
label_x3.pack(side=tk.LEFT, padx=5)
entry_x3 = ttk.Entry(frame_prediction, width=20)
entry_x3.pack(side=tk.LEFT, padx=5)

button_predict = ttk.Button(frame_prediction, text="Predict", command=predict_data)
button_predict.pack(side=tk.LEFT, padx=5)

button_show = ttk.Button(frame_prediction, text="Show Answer", command=show_answer)
button_show.pack(side=tk.LEFT, padx=5)

#Label for prections
label_prediction = ttk.Label(inner_frame, text="", wraplength=window_width - 50)  # Adjust wraplength as needed
label_prediction.pack(fill=tk.X, padx=10, pady=5)


# --- Model Save/Load Frame ---
frame_model = ttk.LabelFrame(inner_frame, text="Model Save/Load", padding=10)
frame_model.pack(fill=tk.X, padx=10, pady=10)

label_save_path = ttk.Label(frame_model, text="Save Model Path:")
label_save_path.pack(side=tk.LEFT, padx=5)
entry_save_path = ttk.Entry(frame_model, width=40)
entry_save_path.insert(0, "my_model.keras")  # Default value
entry_save_path.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=5)
button_save_model = ttk.Button(frame_model, text="Save Model", command=save_model_command)
button_save_model.pack(side=tk.LEFT, padx=5)

label_load_path = ttk.Label(frame_model, text="Load Model Path:")
label_load_path.pack(side=tk.LEFT, padx=5)
entry_load_path = ttk.Entry(frame_model, width=40)
entry_load_path.insert(0, "my_model.keras")  # Default value
entry_load_path.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=5)
button_load_model = ttk.Button(frame_model, text="Load Model", command=load_model_command)
button_load_model.pack(side=tk.LEFT, padx=5)

# --- Graphs Button ---
button_graphs = ttk.Button(inner_frame, text="Graphs", command=open_graphs)
button_graphs.pack(side=tk.LEFT, padx=5)

# --- Fullscreen Button ---
button_fullscreen = ttk.Button(inner_frame, text="Fullscreen", command=toggle_fullscreen)
button_fullscreen.pack(side=tk.LEFT, padx=5)

root.mainloop()
