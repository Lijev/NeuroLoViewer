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
import configparser  # For saving settings

# Global variables (initialized to None)
X_train, X_test, y_train, y_test = None, None, None, None
current_model = None
training_history = None
test_loss = None
data_X = None
data_Y = None
is_fullscreen = False

# Configuration file setup
CONFIG_FILE = "config.ini"
config = configparser.ConfigParser()
config.read(CONFIG_FILE)

def resource_path(relative_path):
    """
    Get the absolute path to a resource, handling cases for both development and PyInstaller.
    This function helps locate files bundled with the application, regardless of how it's run.

    Args:
        relative_path (str): The path to the resource relative to the application's root.

    Returns:
        str: The absolute path to the resource.
    """
    try:
        # PyInstaller creates a _MEIPASS attribute
        base_path = sys._MEIPASS
    except Exception:
        # If running as a script
        base_path = os.path.abspath("../data/")

    return os.path.join(base_path, relative_path)

def load_data(filepath):
    """
    Load data from a JSON file.

    Args:
        filepath (str): The path to the JSON file containing the data.

    Returns:
        tuple: A tuple containing the data_X and data_Y as NumPy arrays, or (None, None) if an error occurs.
    """
    global data_X, data_Y
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
        messagebox.showerror("Error", f"An unexpected error occurred loading: {e}")
        return None, None


def create_and_train_model(X_train, y_train, X_test, y_test, epochs=100, batch_size=10, progress_callback=None):
    """
    Create, compile, and train a Keras Sequential model.

    Args:
        X_train (np.ndarray): Training data features.
        y_train (np.ndarray): Training data labels.
        X_test (np.ndarray): Testing data features.
        y_test (np.ndarray): Testing data labels.
        epochs (int): Number of training epochs. Defaults to 100.
        batch_size (int): Batch size for training. Defaults to 10.
        progress_callback (callable, optional): A function to call after each epoch
            with the current epoch and total epochs as arguments. Defaults to None.

    Returns:
        tuple: A tuple containing the trained model, the training history, and the test loss.
    """

    hidden_layer_size = 128

    # Create a Sequential model
    model = keras.Sequential()
    model.add(layers.Dense(hidden_layer_size, activation='relu', input_shape=(3,)))
    model.add(layers.Dense(hidden_layer_size, activation='relu'))
    model.add(layers.Dense(hidden_layer_size // 2, activation='relu'))
    model.add(layers.Dense(4))  # Output layer with 4 nodes

    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Train the model with progress updates
    history = model.fit(
        X_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.2,
        verbose=0,  # Suppress training output
        callbacks=[keras.callbacks.LambdaCallback(
            on_epoch_end=lambda epoch, logs: progress_callback(epoch + 1, epochs) if progress_callback else None
        )]
    )

    # Evaluate the model on the test set
    test_loss = model.evaluate(X_test, y_test, verbose=0)
    print(f'Test loss: {test_loss}')

    return model, history, test_loss


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
    """
    Load a Keras model from a file.

    Args:
        model_name (str): The path to the model file.

    Returns:
        keras.Model: The loaded Keras model, or None if an error occurs.
    """
    try:
        model = keras.models.load_model(model_name)
        messagebox.showinfo("Success", f"Model loaded from {model_name}")
        print(f'Model loaded from {model_name}')

        # Save to config
        config['Paths']['load_path'] = model_name
        save_config()

        return model
    except FileNotFoundError:
        messagebox.showerror("Error", f"Model file not found: {model_name}")
        return None
    except Exception as e:
        messagebox.showerror("Error", f"Error loading model: {e}")
        return None


def predict_data():
    """
    Predict data using the loaded model based on user inputs.
    Handles cases where X3 is a single value or "all" (a range from 1 to 10).
    """
    try:
        # Get input values from the GUI
        x1 = float(entry_x1.get())
        x2 = float(entry_x2.get())
        x3_input = entry_x3.get().lower()

        # Determine the values for X3 based on user input
        if x3_input == "all":
            x3_values = list(range(1, 11))  # X3 will range from 1 to 10
        else:
            try:
                x3_values = [float(x3_input)]  # X3 is a single value
            except ValueError:
                messagebox.showerror("Error", "Invalid input for X3. Please enter a number or 'all'.")
                return

        # Check if a model has been loaded or trained
        if current_model is None:
            messagebox.showerror("Error", "Model not loaded or trained. Please load or train a model first.")
            return

        # Make predictions for each value of X3
        all_predictions = []
        for x3 in x3_values:
            input_data = np.array([[x1, x2, x3]])  # Create a NumPy array for prediction
            predictions = current_model.predict(input_data, verbose=0)  # Make prediction
            all_predictions.append([round(pred, 2) for pred in predictions[0]])  # Round to 2 digits

        # Store predictions in the GUI for later display
        root.gui_predictions = all_predictions
        update_predictions_answers_display()  # Update the display with predictions

    except ValueError:
        messagebox.showerror("Error", "Invalid input. Please enter numeric values for X1 and X2.")
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred during prediction: {e}")


def show_answer():
    """
    Display the ground truth answer from the loaded dataset corresponding to the user inputs.
    Handles cases where X3 is a single value or "all" (a range from 1 to 10).
    """
    global data_X, data_Y

    # Check if data has been loaded
    if data_X is None or data_Y is None:
        messagebox.showerror("Error", "Data not loaded. Please load data first.")
        return

    try:
        # Get input values from the GUI
        x1 = float(entry_x1.get())
        x2 = float(entry_x2.get())
        x3_input = entry_x3.get().lower()

        # Determine the values for X3 based on user input
        if x3_input == "all":
            x3_values = list(range(1, 11))  # X3 will range from 1 to 10
        else:
            try:
                x3_values = [float(x3_input)]  # X3 is a single value
            except ValueError:
                messagebox.showerror("Error", "Invalid input for X3. Please enter a number or 'all'.")
                return

        # Find the answers for each value of X3 in the dataset
        all_answers = []
        for x3 in x3_values:
            input_data = np.array([x1, x2, x3])  # Create a NumPy array for comparison
            match_index = np.where(np.all(data_X == input_data, axis=1))  # Find matching indices

            if match_index[0].size > 0:
                answer = data_Y[match_index[0][0]]  # Get the answer from the dataset
                all_answers.append([round(ans, 2) for ans in answer.tolist()])  # Round to 2 digits
            else:
                all_answers.append("idk lol")  # No matching data found

        # Store answers in the GUI for later display
        root.gui_answers = all_answers
        update_predictions_answers_display()  # Update the display with answers

    except ValueError:
        messagebox.showerror("Error", "Invalid input. Please enter numeric values for X1 and X2.")
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {e}")


def update_predictions_answers_display():
    """
    Displays predictions and answers side by side in the GUI.
    Retrieves predictions and answers from the GUI's attributes, formats them, and displays them in a label.
    """
    # Retrieve predictions and answers from GUI attributes
    predictions = getattr(root, "gui_predictions", [])
    answers = getattr(root, "gui_answers", [])

    # Build the output text by aligning predictions and answers
    output_text = ""
    max_lines = max(len(predictions), len(answers))  # Find the maximum number of lines to display

    for i in range(max_lines):
        prediction_str = str(predictions[i]) if i < len(predictions) else ""  # Get prediction string
        answer_str = str(answers[i]) if i < len(answers) else ""  # Get answer string
        output_text += f"Prediction: {prediction_str:<30} Answer: {answer_str}\n"  # Align the output

    # Update the GUI label with the output text
    label_prediction.config(text=output_text)
    print(output_text)


def update_progress_bar(epoch, total_epochs):
    """
    Updates the progress bar in the GUI.

    Args:
        epoch (int): The current epoch number.
        total_epochs (int): The total number of epochs.
    """
    progress_value = int((epoch / total_epochs) * 100)  # Calculate progress percentage
    progress_bar['value'] = progress_value  # Update progress bar value
    root.update_idletasks()  # Update the GUI to reflect the changes


def train_model_thread(X_train, y_train, X_test, y_test, epochs, batch_size):
    """
    Trains the model in a separate thread to prevent the GUI from freezing.

    Args:
        X_train (np.ndarray): Training data features.
        y_train (np.ndarray): Training data labels.
        X_test (np.ndarray): Testing data features.
        y_test (np.ndarray): Testing data labels.
        epochs (int): Number of training epochs.
        batch_size (int): Batch size for training.
    """
    global current_model, training_history, test_loss

    try:
        # Train the model using the provided data and parameters
        current_model, training_history, test_loss = create_and_train_model(
            X_train, y_train, X_test, y_test, epochs, batch_size,
            progress_callback=update_progress_bar  # Provide the progress callback
        )

        # Update the test loss label in the GUI
        label_test_loss.config(text=f'Test loss: {test_loss:.4f}')
        messagebox.showinfo("Success", "Model trained successfully.")

    except Exception as e:
        messagebox.showerror("Error", f"An error occurred during training: {e}")

    finally:
        # Reset the progress bar and enable the train button after training
        progress_bar['value'] = 0
        button_train['state'] = 'normal'
        button_show_history['state'] = 'normal'


def train_model_command():
    """
    Command to initiate model training.  Validates input, disables the train button,
    and starts a new thread for training to prevent the GUI from freezing.
    """
    global X_train, X_test, y_train, y_test

    # Check if data has been loaded
    if X_train is None or y_train is None or X_test is None or y_test is None:
        messagebox.showerror("Error", "Data not loaded. Please load data first.")
        return

    try:
        # Get epochs and batch size from the GUI
        epochs = int(entry_epochs.get())
        batch_size = int(entry_batch_size.get())

    except ValueError:
        messagebox.showerror("Error", "Invalid input. Please enter integer values for epochs and batch size.")
        return

    # Disable the train button to prevent multiple training processes
    button_train['state'] = 'disabled'
    button_show_history['state'] = 'disabled'

    # Create and start a new thread for training
    threading.Thread(target=train_model_thread, args=(X_train, y_train, X_test, y_test, epochs, batch_size)).start()


def load_data_command():
    """
    Command to load data from a JSON file. Opens a file dialog, loads the data,
    and splits it into training and testing sets.
    """
    global X_train, X_test, y_train, y_test, data_X, data_Y

    # Open a file dialog to select a data file
    filepath = filedialog.askopenfilename(title="Select Data File",
                                           filetypes=[("JSON files", "*.json"), ("All files", "*.*")])

    # If no file is selected, return
    if not filepath:
        return

    # Update the data path entry with the selected file path
    entry_data_path.delete(0, tk.END)
    entry_data_path.insert(0, filepath)

    # Load the data from the selected file
    X, y = load_data(filepath)

    # If data is loaded successfully, split it into training and testing sets
    if X is not None and y is not None:
        X = np.array(X)
        y = np.array(y)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        messagebox.showinfo("Success", "Data loaded successfully.")

        # Save to config
        config['Paths']['data_path'] = filepath
        save_config()
    else:
        # If data loading fails, reset the training and testing sets
        X_train, X_test, y_train, y_test = None, None, None, None
        data_X, data_Y = None, None


def save_model_command():
    """
    Command to save the current model. Opens a file dialog to select a save location
    and then saves the model to the specified path.
    """
    global current_model

    # Check if a model has been trained or loaded
    if current_model is None:
        messagebox.showerror("Error", "No model to save. Please train or load a model first.")
        return

    # Open a file dialog to select a save location
    filepath = filedialog.asksaveasfilename(title="Save Model", defaultextension=".keras",
                                             filetypes=[("Keras models", "*.keras"), ("All files", "*.*")])

    # If no file is selected, return
    if not filepath:
        return

    # Update the save path entry with the selected file path
    entry_save_path.delete(0, tk.END)
    entry_save_path.insert(0, filepath)

    # Save the model to the specified path
    model_name = filepath
    save_model(current_model, model_name)


def load_model_command():
    """
    Command to load a model from a file. Opens a file dialog to select a model file
    and then loads the model from the specified path.
    """
    global current_model

    # Open a file dialog to select a model file
    filepath = filedialog.askopenfilename(title="Select Model File",
                                             filetypes=[("Keras models", "*.keras"), ("All files", "*.*")])

    # If no file is selected, return
    if not filepath:
        return

    # Update the load path entry with the selected file path
    entry_load_path.delete(0, tk.END)
    entry_load_path.insert(0, filepath)

    # Load the model from the specified path
    model_name = filepath
    current_model = load_model(model_name)


def plot_training_history(history):
    """
    Plots the training history (loss and validation loss) using Matplotlib and displays it in a Tkinter window.

    Args:
        history: The training history object returned by model.fit().
    """
    # Create a new top-level window for the training history plot
    history_window = tk.Toplevel(root)
    history_window.title("Training History")

    # Create a Matplotlib figure and axes
    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot the training loss and validation loss
    ax.plot(history.history['loss'], label='loss')
    ax.plot(history.history['val_loss'], label='val_loss')

    # Set labels and title for the plot
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training History')

    # Add a legend to the plot
    ax.legend()

    # Ensure tight layout to prevent labels from overlapping
    fig.tight_layout()

    # Embed the Matplotlib figure in the Tkinter window
    canvas = FigureCanvasTkAgg(fig, master=history_window)
    canvas_widget = canvas.get_tk_widget()
    canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    # Add a navigation toolbar to the Tkinter window
    toolbar = NavigationToolbar2Tk(canvas, history_window)
    toolbar.update()
    canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    # Draw the Matplotlib figure on the canvas
    canvas.draw()


def plot_predictions_answers():
    """
    Plots the predicted and ground truth answers against X3 values using Matplotlib and displays them in separate Tkinter windows.
    """
    global data_X, data_Y

    # Check if data has been loaded
    if data_X is None or data_Y is None:
        messagebox.showerror("Error", "Data not loaded. Please load data first.")
        return

    try:
        # Get the values of X1 and X2 from the GUI
        x1_val = float(entry_x1.get())
        x2_val = float(entry_x2.get())

        # Generate a range of X3 values
        x3_values = np.arange(0, 10.01, 0.01).tolist()

        # Check if a model has been loaded or trained
        if current_model is None:
            messagebox.showerror("Error", "Model not loaded or trained. Please load or train a model first.")
            return

        # Create a dictionary to store plot data
        plot_data = {}

        # Iterate over the range of X3 values
        for x3 in x3_values:
            key = (x1_val, x2_val, x3)

            # Make a prediction for the current X3 value
            input_data = np.array([[x1_val, x2_val, x3]])
            predictions = current_model.predict(input_data, verbose=0)[0].tolist()
            plot_data[key] = {"prediction": predictions}

            # Find the corresponding answer in the dataset
            input_data = np.array([x1_val, x2_val, x3])
            match_index = np.where(np.all(data_X == input_data, axis=1))
            if match_index[0].size > 0:
                plot_data[key]["answer"] = data_Y[match_index[0][0]].tolist()
            else:
                plot_data[key]["answer"] = None

        # Create the plots
        create_plots(plot_data, x1_val, x2_val)

    except ValueError:
        messagebox.showerror("Error", "Invalid input. Please enter numeric values for X1 and X2.")
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {e}")


def create_plots(plot_data, x1_val, x2_val):
    """
    Creates and displays plots of answers and predictions against X3 values in separate Tkinter windows.

    Args:
        plot_data (dict): A dictionary containing the data to plot.
        x1_val (float): The value of X1.
        x2_val (float): The value of X2.
    """
    # Create new top-level windows for answers and predictions plots
    answers_window = tk.Toplevel(root)
    answers_window.title(f"Answers of Se{x1_val}Ep{x2_val}")

    predictions_window = tk.Toplevel(root)
    predictions_window.title(f"Prediction of Se{x1_val}Ep{x2_val}")

    # Get X3 values from plot data
    x_values = list(plot_data.keys())

    # Create Matplotlib figures and axes for answers and predictions
    answers_figure, answers_axes = plt.subplots(figsize=(8, 6))
    predictions_figure, predictions_axes = plt.subplots(figsize=(8, 6))

    # Iterate over the four Y values
    for i in range(4):
        # Extract answer values for the current Y
        y_values = []
        for x in x_values:
            if plot_data[x]["answer"] is not None:
                y_values.append(plot_data[x]["answer"][i])
            else:
                y_values.append(None)

        # Filter out invalid (None) data points
        valid_x_values = [x[2] for i, x in enumerate(x_values) if y_values[i] is not None]
        valid_y_values = [y for y in y_values if y is not None]

        # Plot the valid answer values
        answers_axes.plot(valid_x_values, valid_y_values, label=f"Y_{i + 1}")

    # Iterate over the four Y values
    for i in range(4):
        # Extract predicted values for the current Y
        y_values = []
        for x in x_values:
            y_values.append(plot_data[x]["prediction"][i])

        # Plot the predicted values
        predictions_axes.plot([x[2] for x in x_values], y_values, label=f"Y_{i + 1}")

    # Configure the answers plot
    answers_axes.set_xlabel("X3")
    answers_axes.set_ylabel("Answer Values")
    answers_axes.set_title("Answers vs. X3")
    answers_axes.legend()
    answers_axes.grid(True)

    # Configure the predictions plot
    predictions_axes.set_xlabel("X3")
    predictions_axes.set_ylabel("Predicted Values")
    predictions_axes.set_title("Predictions vs. X3")
    predictions_axes.legend()
    predictions_axes.grid(True)

    # Embed the answers plot in the Tkinter window
    answers_canvas = FigureCanvasTkAgg(answers_figure, master=answers_window)
    answers_canvas_widget = answers_canvas.get_tk_widget()
    answers_canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    answers_toolbar = NavigationToolbar2Tk(answers_canvas, answers_window)
    answers_toolbar.update()
    answers_canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
    answers_canvas.draw()

    # Embed the predictions plot in the Tkinter window
    predictions_canvas = FigureCanvasTkAgg(predictions_figure, master=predictions_window)
    predictions_canvas_widget = predictions_canvas.get_tk_widget()
    predictions_canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    predictions_toolbar = NavigationToolbar2Tk(predictions_canvas, predictions_window)
    predictions_toolbar.update()
    predictions_canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
    predictions_canvas.draw()


def open_graphs():
    """
    Opens the graphs plotting predictions and answers.
    """
    plot_predictions_answers()


def open_training_history():
    """
    Opens a window displaying the training history plot.
    """
    global training_history

    # Check if training history is available
    if training_history is None:
        messagebox.showerror("Error", "No training history available. Please train a model first.")
        return

    # Plot and display the training history
    plot_training_history(training_history)


def toggle_fullscreen():
    """
    Toggles fullscreen mode for the application window.
    """
    global is_fullscreen

    # Toggle the fullscreen flag
    is_fullscreen = not is_fullscreen

    # Set the fullscreen attribute of the root window
    root.attributes("-fullscreen", is_fullscreen)


def on_mousewheel(event):
    """
    Handles mousewheel scrolling with increased delta.
    """
    canvas.yview_scroll(int(-1 * (event.delta / 120) * 10), "units")

def save_config():
    """Saves the configuration to the config file."""
    with open(CONFIG_FILE, 'w') as configfile:
        config.write(configfile)

def load_config():
    """Loads the configuration from the config file."""
    global config
    if os.path.exists(CONFIG_FILE):
        config.read(CONFIG_FILE)
    else:
        # Create the 'Paths' section if it doesn't exist
        if 'Paths' not in config:
            config['Paths'] = {}
        # Initialize the paths in the config
        config['Paths']['data_path'] = ''
        config['Paths']['save_path'] = ''
        config['Paths']['load_path'] = ''
        save_config()  # Save the initialized config to the file

class ToolTip:
    def __init__(self, widget, text=''):
        self.widget = widget
        self.text = text
        self.tooltip_window = None

        self.widget.bind("<Enter>", self.show_tooltip)
        self.widget.bind("<Leave>", self.hide_tooltip)

    def show_tooltip(self, event=None):
        """Show the tooltip immediately."""
        self._show()

    def _show(self):
        """Actually create and show the tooltip window."""
        x, y, _, _ = self.widget.bbox("insert")
        x += self.widget.winfo_rootx() + 25
        y += self.widget.winfo_rooty() + 25

        # creates a toplevel window
        self.tooltip_window = tk.Toplevel(self.widget)

        # Leaves only the label and removes the app window
        self.tooltip_window.wm_overrideredirect(True)
        self.tooltip_window.wm_geometry(f"+{x}+{y}")

        label = tk.Label(self.tooltip_window, text=self.text, background="#ffffe0",
                           relief="solid", borderwidth=1, font=("arial", "10", "normal"))
        label.pack()

    def hide_tooltip(self, event=None):
        """Hide the tooltip."""
        if self.tooltip_window:
            self.tooltip_window.destroy()
        self.tooltip_window = None

# Initialize the main Tkinter window
root = tk.Tk()
root.title("LoViewer")

# Load configuration
load_config()

# --- Color Palette ---
BG_COLOR = "#E6EE9C"  # Light Yellow-Green
FRAME_BG = "#F0F4C3"   # Lighter Yellow-Green
BUTTON_BG = "#AED581"  # Medium Green
BUTTON_FG = "#33691E"  # Dark Green
LABEL_BG = BG_COLOR
LABEL_FG = "#2E7D32"  # Darker Green
ENTRY_BG = "#FFFFCC"
ENTRY_FG = "black"

# Apply a Modern Theme
style = ttk.Style()
style.theme_use('clam')  # ('clam', 'alt', 'default', 'classic', 'vista', 'xp')

# Configure some basic style elements - Customize these!
style.configure("TLabel", font=('Arial', 10), background=LABEL_BG, foreground=LABEL_FG)
style.configure("TButton", font=('Arial', 10, 'bold'), padding=8, background=BUTTON_BG, foreground=BUTTON_FG)
style.configure("TEntry", font=('Arial', 10), background=ENTRY_BG, foreground=ENTRY_FG)
style.configure("TFrame", background=FRAME_BG) # Light background
style.configure("TLabelframe", background=FRAME_BG)
style.configure("TLabelframe.Label", font=('Arial', 10, 'bold'))

# Configure window size and position
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()
window_width = int(screen_width * 0.45)  # Reduced width
window_height = int(screen_height * 0.7)
x = (screen_width - window_width) // 2
y = (screen_height - window_height) // 2
root.geometry(f"{window_width}x{window_height}+{x}+{y}")

# Set window icon
icon_path = resource_path("lo.ico")
root.iconbitmap(icon_path)

# Disable window resizing
root.resizable(False, False)
root.configure(background=BG_COLOR)

# Create main frame
main_frame = ttk.Frame(root, padding=10)
main_frame.pack(fill=tk.BOTH, expand=True)

# Create canvas and scrollbar for scrollable content
canvas = tk.Canvas(main_frame, bd=0, highlightthickness=0, background=BG_COLOR)  # Remove border
canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

scrollbar = ttk.Scrollbar(main_frame, orient=tk.VERTICAL, command=canvas.yview)
scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

canvas.configure(yscrollcommand=scrollbar.set)
canvas.bind('<Configure>', lambda e: canvas.configure(scrollregion=canvas.bbox("all")))

# Create inner frame to hold content
inner_frame = ttk.Frame(canvas, padding=10)
canvas.create_window((0, 0), window=inner_frame, anchor="nw")

# Bind mousewheel scrolling
canvas.bind_all("<MouseWheel>", on_mousewheel)

# -------------------- Data Loading Section --------------------
frame_data = ttk.LabelFrame(inner_frame, text="Data Loading", padding=10)
frame_data.grid(row=0, column=0, columnspan=2, sticky="ew", padx=10, pady=10)

label_data_path = ttk.Label(frame_data, text="Data File Path:")
label_data_path.grid(row=0, column=0, sticky="w", padx=5, pady=5)

entry_data_path = ttk.Entry(frame_data, width=50)
entry_data_path.grid(row=0, column=1, sticky="ew", padx=5, pady=5)
entry_data_path.insert(0, config['Paths'].get('data_path', 'IDS.json'))  # Load from config
frame_data.columnconfigure(1, weight=1) # Make entry stretch

button_load_data = ttk.Button(frame_data, text="Load Data", command=load_data_command)
button_load_data.grid(row=0, column=2, sticky="e", padx=5, pady=5)
ToolTip(button_load_data, text="Load data from JSON file.\nHotkey: Ctrl+L")

# -------------------- Model Training Section --------------------
frame_training = ttk.LabelFrame(inner_frame, text="Model Training", padding=10)
frame_training.grid(row=1, column=0, columnspan=2, sticky="ew", padx=10, pady=10)

label_epochs = ttk.Label(frame_training, text="Epochs:")
label_epochs.grid(row=0, column=0, sticky="w", padx=5, pady=5)

entry_epochs = ttk.Entry(frame_training, width=10)
entry_epochs.insert(0, "100")
entry_epochs.grid(row=0, column=1, sticky="w", padx=5, pady=5)

label_batch_size = ttk.Label(frame_training, text="Batch Size:")
label_batch_size.grid(row=0, column=2, sticky="w", padx=5, pady=5)

entry_batch_size = ttk.Entry(frame_training, width=10)
entry_batch_size.insert(0, "10")
entry_batch_size.grid(row=0, column=3, sticky="w", padx=5, pady=5)

button_train = ttk.Button(frame_training, text="Train Model", command=train_model_command)
button_train.grid(row=0, column=4, sticky="e", padx=5, pady=5)
ToolTip(button_train, text="Train the neural network model.\nHotkey: Ctrl+T")

progress_bar = ttk.Progressbar(frame_training, orient=tk.HORIZONTAL, length=200, mode='determinate')
progress_bar.grid(row=1, column=0, columnspan=5, sticky="ew", padx=5, pady=5)

button_show_history = ttk.Button(frame_training, text="Show History", command=open_training_history, state='disabled')
button_show_history.grid(row=2, column=0, columnspan=5, sticky="ew", padx=5, pady=5)
ToolTip(button_show_history, text="Show the training history plot.\nHotkey: Ctrl+H")

label_test_loss = ttk.Label(frame_training, text="Test Loss: N/A")
label_test_loss.grid(row=3, column=0, columnspan=5, sticky="w", padx=5, pady=5)

# -------------------- Prediction Section --------------------
frame_prediction = ttk.LabelFrame(inner_frame, text="Prediction", padding=10)
frame_prediction.grid(row=2, column=0, columnspan=2, sticky="ew", padx=10, pady=10)

label_x1 = ttk.Label(frame_prediction, text="X1:")
label_x1.grid(row=0, column=0, sticky="w", padx=5, pady=5)

entry_x1 = ttk.Entry(frame_prediction, width=10)
entry_x1.grid(row=0, column=1, sticky="w", padx=5, pady=5)

label_x2 = ttk.Label(frame_prediction, text="X2:")
label_x2.grid(row=0, column=2, sticky="w", padx=5, pady=5)

entry_x2 = ttk.Entry(frame_prediction, width=10)
entry_x2.grid(row=0, column=3, sticky="w", padx=5, pady=5)

label_x3 = ttk.Label(frame_prediction, text="X3 (Number or 'all'):")
label_x3.grid(row=1, column=0, sticky="w", padx=5, pady=5)

entry_x3 = ttk.Entry(frame_prediction, width=20)
entry_x3.grid(row=1, column=1, columnspan=3, sticky="ew", padx=5, pady=5)

button_predict = ttk.Button(frame_prediction, text="Predict", command=predict_data)
button_predict.grid(row=2, column=0, columnspan=2, sticky="ew", padx=5, pady=5)
ToolTip(button_predict, text="Predict data using the loaded model.\nHotkey: Ctrl+P")

button_show = ttk.Button(frame_prediction, text="Show Answer", command=show_answer)
button_show.grid(row=2, column=2, columnspan=2, sticky="ew", padx=5, pady=5)
ToolTip(button_show, text="Show the ground truth answer from the loaded dataset.\nHotkey: Ctrl+S")

label_prediction = ttk.Label(inner_frame, text="", wraplength=window_width - 50)
label_prediction.grid(row=3, column=0, columnspan=2, sticky="ew", padx=10, pady=5)

# -------------------- Model Save/Load Section --------------------
frame_model = ttk.LabelFrame(inner_frame, text="Model Save/Load", padding=10)
frame_model.grid(row=4, column=0, columnspan=2, sticky="ew", padx=10, pady=10)

label_save_path = ttk.Label(frame_model, text="Save Model Path:")
label_save_path.grid(row=0, column=0, sticky="w", padx=5, pady=5)

entry_save_path = ttk.Entry(frame_model, width=40)
entry_save_path.grid(row=0, column=1, sticky="ew", padx=5, pady=5)
entry_save_path.insert(0, config['Paths'].get('save_path', 'my_model.keras')) # Load from config

button_save_model = ttk.Button(frame_model, text="Save Model", command=save_model_command)
button_save_model.grid(row=0, column=2, sticky="e", padx=5, pady=5)
ToolTip(button_save_model, text="Save the current model to a file.\nHotkey: Ctrl+Shift+S")

label_load_path = ttk.Label(frame_model, text="Load Model Path:")
label_load_path.grid(row=1, column=0, sticky="w", padx=5, pady=5)

entry_load_path = ttk.Entry(frame_model, width=40)
entry_load_path.grid(row=1, column=1, sticky="ew", padx=5, pady=5)
entry_load_path.insert(0, config['Paths'].get('load_path', 'my_model.keras')) # Load from config

button_load_model = ttk.Button(frame_model, text="Load Model", command=load_model_command)
button_load_model.grid(row=1, column=2, sticky="e", padx=5, pady=5)
ToolTip(button_load_model, text="Load a model from a file.\nHotkey: Ctrl+O")

# -------------------- Buttons for graphs and fullscreen --------------------
frame_buttons = ttk.Frame(inner_frame, padding=10)
frame_buttons.grid(row=5, column=0, columnspan=2, sticky="ew", padx=10, pady=10)

button_graphs = ttk.Button(frame_buttons, text="Graphs", command=open_graphs)
button_graphs.pack(side=tk.LEFT, padx=5)
ToolTip(button_graphs, text="Open graphs plotting predictions and answers.\nHotkey: Ctrl+G")

button_fullscreen = ttk.Button(frame_buttons, text="Fullscreen", command=toggle_fullscreen)
button_fullscreen.pack(side=tk.LEFT, padx=5)
ToolTip(button_fullscreen, text="Toggle fullscreen mode.\nHotkey: F11")

# Make the inner frame resize with the canvas
inner_frame.bind("<Configure>", lambda event: canvas.configure(scrollregion=canvas.bbox("all")))

# Initial resize to calculate scroll region after all widgets are created
root.update()
canvas.configure(scrollregion=canvas.bbox("all"))

# --- Keyboard Shortcuts ---
root.bind("<Control-l>", lambda event: load_data_command())
root.bind("<Control-t>", lambda event: train_model_command())
root.bind("<Control-h>", lambda event: open_training_history())
root.bind("<Control-p>", lambda event: predict_data())
root.bind("<Control-s>", lambda event: show_answer())
root.bind("<Control-Shift-S>", lambda event: save_model_command())
root.bind("<Control-o>", lambda event: load_model_command())
root.bind("<Control-g>", lambda event: open_graphs())
root.bind("<F11>", lambda event: toggle_fullscreen())

# Start the Tkinter main loop
root.mainloop()
