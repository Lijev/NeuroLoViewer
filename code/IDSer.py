import json
import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog  # Import simpledialog
import ast
import os  # For checking file existence
import configparser  # For remembering window size
import sys  # For resource path

class DataManagerGUI:
    def __init__(self, master):
        self.master = master
        master.title("Info Data Set Tool")

        # --- Resource Path for Icon ---
        self.icon_path = self.resource_path("../data/lo.ico")
        if os.path.exists(self.icon_path):
            master.iconbitmap(self.icon_path)
        else:
            print("Icon file 'lo.ico' not found.")

        # --- Configuration ---
        self.config = configparser.ConfigParser()
        self.config.read('config.ini')
        if 'Window' not in self.config:
            self.config['Window'] = {'width': '800', 'height': '600'}  # Default values
        self.window_width = int(self.config['Window']['width'])
        self.window_height = int(self.config['Window']['height'])
        master.geometry(f"{self.window_width}x{self.window_height}")

        master.protocol("WM_DELETE_WINDOW", self.on_closing) # Save configuration

        # --- Styling ---
        self.style = ttk.Style()
        self.configure_styles()  # Centralized styling

        # --- Data ---
        self.filepath = '../data/IDS.json'
        self.X, self.Y = [], []

        # --- UI Elements ---
        self.create_widgets()  # Create widgets BEFORE loading data
        self.load_data()  # Load data after widgets are created

    def resource_path(self, relative_path):
        """ Get absolute path to resource, works for dev and for PyInstaller """
        try:
            # PyInstaller creates a temp folder and stores path in _MEIPASS
            base_path = sys._MEIPASS
        except Exception:
            base_path = os.path.abspath(".")

        return os.path.join(base_path, relative_path)

    def configure_styles(self):
        """Configures ttk styles for a modern look."""
        self.style.theme_use('clam')  # A clean, modern theme

        # Main Colors
        bg_color = "#ECEFF1"  # Light Gray-Blue
        button_color = "#7986CB" # Indigo
        label_color = "#37474F" # Dark Gray-Blue
        entry_color = "#CFD8DC"  # Light Gray-Blue

        # Button Style
        self.style.configure('TButton', padding=(10, 8), relief="raised", background=button_color, foreground="white", font=('Arial', 10, 'bold'))
        self.style.map('TButton', background=[('active', '#5C6BC0')])  # Darker Indigo on hover

        # Label Style
        self.style.configure('TLabel', padding=5, background=bg_color, foreground=label_color, font=('Arial', 10))

        # Entry Style
        self.style.configure('TEntry', padding=5, fieldbackground=entry_color, font=('Arial', 10))

        # Frame Style
        self.style.configure('TFrame', background=bg_color)

        # Menu style (separate because it's not ttk)
        self.master.option_add('*tearOff', False)  # Remove tear-off option for menus
        self.master.config(bg=bg_color)  # Apply main background color

    def load_data(self):
        """Loads data from the JSON file and updates the GUI."""
        try:
            if not os.path.exists(self.filepath):
                messagebox.showinfo("Info", "IDS.json not found. Creating an empty one.")
                with open(self.filepath, 'w') as f:
                    json.dump({'X': [], 'Y': []}, f, indent=4) # Initialize with empty data
            with open(self.filepath, 'r') as f:
                data = json.load(f)
            self.X = data['X']
            self.Y = data['Y']
            self.update_data_display()
        except FileNotFoundError:
            messagebox.showerror("Error", f"File not found: {self.filepath}")
        except json.JSONDecodeError:
            messagebox.showerror("Error", f"Invalid JSON format in file: {self.filepath}")
        except KeyError as e:
            messagebox.showerror("Error", f"Missing key in JSON: {e}")
        except Exception as e:
            messagebox.showerror("Error", f"An unexpected error occurred: {e}")


    def save_data(self):
        """Saves the current data to the JSON file."""
        try:
            data = {'X': self.X, 'Y': self.Y}
            with open(self.filepath, 'w') as f:
                json.dump(data, f, indent=4)
            messagebox.showinfo("Success", f"Data saved to {self.filepath}")
        except Exception as e:
            messagebox.showerror("Error", f"Error saving: {e}")

    def add_data(self, new_x, new_y):
        """Adds new data point to the existing X and Y lists."""
        self.X.append(new_x)
        self.Y.append(new_y)
        self.update_data_display()
        messagebox.showinfo("Success", f"Added: X={new_x}, Y={new_y}")

    def edit_data(self, edit_x, new_y):
        """Edits the Y value for a given X in the dataset."""
        try:
            edit_x_np = np.array(edit_x)
            X_np = np.array(self.X)

            indices = np.where(np.all(X_np == edit_x_np, axis=1))[0]

            if indices.size == 0:
                self.data_text.insert(tk.END, f"Error: X={edit_x} not found in the dataset.\n")  # Display in Text Widget
                return

            index = indices[0]
            self.Y[index] = new_y
            self.update_data_display()
            self.data_text.insert(tk.END, f"Edited: X={edit_x}, Y={new_y}\n")  # Display in Text Widget

        except Exception as e:
            self.data_text.insert(tk.END, f"An error occurred during editing: {e}\n")  # Display in Text Widget


    def remove_data(self, remove_x):
        """Removes a data point from the dataset."""
        try:
            remove_x_np = np.array(remove_x)
            X_np = np.array(self.X)
            indices = np.where(np.all(X_np == remove_x_np, axis=1))[0]

            if indices.size == 0:
                self.data_text.insert(tk.END, f"Error: X={remove_x} not found in the dataset.\n")  # Display in Text Widget
                return

            index = indices[0]
            del self.X[index]
            del self.Y[index]
            self.update_data_display()
            self.data_text.insert(tk.END, f"Removed: X={remove_x}\n")  # Display in Text Widget
        except Exception as e:
            self.data_text.insert(tk.END, f"An error occurred during removal: {e}\n")  # Display in Text Widget


    def show_data(self, show_x_values):
        """Displays the Y value for each provided X value in the data text widget."""
        # Clear the data text widget before showing new data.
        self.data_text.delete("1.0", tk.END)
        for show_x in show_x_values:
            try:
                show_x_np = np.array(show_x)
                X_np = np.array(self.X)
                indices = np.where(np.all(X_np == show_x_np, axis=1))[0]

                if indices.size == 0:
                    self.data_text.insert(tk.END, f"Error: X={show_x} not found in the dataset.\n")  # Display in Text Widget
                    continue

                index = indices[0]
                self.data_text.insert(tk.END, f"X={show_x}, Y={self.Y[index]}\n")  # Display in Text Widget

            except Exception as e:
                self.data_text.insert(tk.END, f"An error occurred during showing: {e}\n")  # Display in Text Widget


    def get_x_input(self, editing=False):
        """Prompts the user for X1, X2, and X3 input, handling the 'all' option for X3 using entry fields."""
        try:
            x1 = float(self.x1_entry.get())
            x2 = float(self.x2_entry.get())
            x3_input = self.x3_entry.get().lower()

            if x3_input == "all":
                x3_values = list(range(1, 11))  # List from 1 to 10 inclusive
                if editing:
                    # For editing, we want to return a single X value but iterate through all of X3 later
                    x_values = [[x1, x2, x3] for x3 in x3_values]  # List comprehension for clarity
                    return x_values
                else:
                    x_values = [[x1, x2, x3] for x3 in x3_values]  # List comprehension for clarity
                    return x_values # Return all x values
            else:
                try:
                    x3_values = [float(x3_input)]
                    x_values = [[x1, x2, x3] for x3 in x3_values]
                    return x_values
                except ValueError:
                    messagebox.showerror("Error", "Invalid input for X3. Please enter a number or 'all'.")
                    return None

        except ValueError:
            messagebox.showerror("Error", "Invalid input for X1, X2, or X3. Please enter a number.")
            return None


    def open_file_dialog(self):
        """Opens a file dialog to select a JSON file."""
        filepath = filedialog.askopenfilename(filetypes=[("JSON Files", "*.json")])
        if filepath:
            self.filepath = filepath
            self.load_data()

    def create_widgets(self):
        """Creates all the UI elements."""

        # --- Menu Bar ---
        menubar = tk.Menu(self.master)
        filemenu = tk.Menu(menubar, tearoff=0)
        filemenu.add_command(label="Open", command=self.open_file_dialog)
        filemenu.add_command(label="Save", command=self.save_data)
        filemenu.add_separator()
        filemenu.add_command(label="Exit", command=self.master.quit)
        menubar.add_cascade(label="File", menu=filemenu)
        self.master.config(menu=menubar)

        # --- Main Frame ---
        main_frame = ttk.Frame(self.master, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # --- Input Frame ---
        input_frame = ttk.Frame(main_frame, padding=10)
        input_frame.pack(pady=10)

        # X1 Input
        ttk.Label(input_frame, text="X1:").grid(row=0, column=0, sticky="w", padx=5, pady=5)
        self.x1_entry = ttk.Entry(input_frame)
        self.x1_entry.grid(row=0, column=1, sticky="ew", padx=5, pady=5)

        # X2 Input
        ttk.Label(input_frame, text="X2:").grid(row=1, column=0, sticky="w", padx=5, pady=5)
        self.x2_entry = ttk.Entry(input_frame)
        self.x2_entry.grid(row=1, column=1, sticky="ew", padx=5, pady=5)

        # X3 Input
        ttk.Label(input_frame, text="X3 (number or 'all'):").grid(row=2, column=0, sticky="w", padx=5, pady=5)
        self.x3_entry = ttk.Entry(input_frame)
        self.x3_entry.grid(row=2, column=1, sticky="ew", padx=5, pady=5)
        input_frame.columnconfigure(1, weight=1) # Make X3 entry stretch

        # --- Button Frame ---
        button_frame = ttk.Frame(main_frame, padding=10)
        button_frame.pack(pady=10)

        # Add Button
        add_button = ttk.Button(button_frame, text="Add Data", command=self.add_data_gui)
        add_button.grid(row=0, column=0, padx=5, pady=5)

        # Edit Button
        edit_button = ttk.Button(button_frame, text="Edit Data", command=self.edit_data_gui)
        edit_button.grid(row=0, column=1, padx=5, pady=5)

        # Remove Button
        remove_button = ttk.Button(button_frame, text="Remove Data", command=self.remove_data_gui)
        remove_button.grid(row=0, column=2, padx=5, pady=5)

        # Show Button
        show_button = ttk.Button(button_frame, text="Show Data", command=self.show_data_gui)
        show_button.grid(row=0, column=3, padx=5, pady=5)

        # Save Button
        save_button = ttk.Button(button_frame, text="Save Data", command=self.save_data)
        save_button.grid(row=0, column=4, padx=5, pady=5)

        # --- Data Display ---
        self.data_label = ttk.Label(main_frame, text="Data:")
        self.data_label.pack()

        self.data_text = tk.Text(main_frame, height=15, width=70)
        self.data_text.pack(padx=10, pady=5)

    def update_data_display(self):
        """Updates the data display in the Text widget."""
        self.data_text.delete("1.0", tk.END)  # Clear existing text
        for i in range(len(self.X)):
            self.data_text.insert(tk.END, f"X: {self.X[i]}, Y: {self.Y[i]}\n")

    def add_data_gui(self):
        """Handles adding data through the GUI, including getting Y input."""
        x_values = self.get_x_input()
        if x_values is None:
            return

        for x in x_values:
            # Get Y input using a simple dialog or entry fields
            y_input_str = simpledialog.askstring("Input", f"Enter Y for X={x} (as a list, e.g., [1.0, 2.0, 3.0, 4.0]):")  # Corrected Name
            if not y_input_str:  # User cancelled
                continue

            try:
                y_input = ast.literal_eval(y_input_str)  # Use ast.literal_eval for safety
                if not isinstance(y_input, list) or len(y_input) != 4:
                    messagebox.showerror("Error", "Invalid input for Y. Please enter a list of 4 numbers.")
                    continue
                self.add_data(x, y_input)
            except (ValueError, SyntaxError):
                messagebox.showerror("Error", "Invalid input for Y. Please enter a valid list.")


    def edit_data_gui(self):
        """Handles editing data through the GUI, allowing 'all' for X3."""
        x_values = self.get_x_input(editing=True)  # Pass editing flag
        if x_values is None:
            return

        # If 'all' was specified for X3, x_values will contain multiple X values to edit.
        # Otherwise, it will contain only one X value to edit.

        for x in x_values:
            y_input_str = simpledialog.askstring("Input", f"Enter new Y for X={x} (as a list, e.g., [1.0, 2.0, 3.0, 4.0]):")
            if not y_input_str:
                continue  # User cancelled

            try:
                y_input = ast.literal_eval(y_input_str)
                if not isinstance(y_input, list) or len(y_input) != 4:
                    messagebox.showerror("Error", "Invalid input for Y. Please enter a list of 4 numbers.")
                    continue

                self.edit_data(x, y_input)  # Call edit function.

            except (ValueError, SyntaxError):
                messagebox.showerror("Error", "Invalid input for Y. Please enter a valid list.")


    def remove_data_gui(self):
        """Handles removing data through the GUI."""
        x_values = self.get_x_input()
        if x_values is None:
            return

        for x in x_values:
            self.remove_data(x)

    def show_data_gui(self):
        """Handles showing data through the GUI."""
        x_values = self.get_x_input()
        if x_values is None:
            return

        self.show_data(x_values)  # Use the new show_data function.

    def on_closing(self):
        """Handles window closing event."""
        self.config['Window']['width'] = str(self.master.winfo_width())
        self.config['Window']['height'] = str(self.master.winfo_height())
        with open('config.ini', 'w') as configfile:
            self.config.write(configfile)
        self.master.destroy()


def main():
    root = tk.Tk()
    gui = DataManagerGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
