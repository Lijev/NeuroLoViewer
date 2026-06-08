import json
import tkinter as tk
from tkinter import ttk, messagebox, filedialog, simpledialog
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

os.chdir(os.path.dirname(os.path.abspath(__file__)))

class Anchor:
    """Класс для хранения данных якоря"""
    def __init__(self, name, coordinates):
        self.name = name
        self.coordinates = coordinates.copy()  # [x1, y1, x2, y2]
        self.created_at = time.time()
    
    def to_dict(self):
        return {
            "name": self.name,
            "coordinates": self.coordinates,
            "created_at": self.created_at
        }
    
    @classmethod
    def from_dict(cls, data):
        anchor = cls(data["name"], data["coordinates"])
        anchor.created_at = data.get("created_at", time.time())
        return anchor

class AnchorMenu:
    """Окно управления якорями"""
    def __init__(self, parent, point_creator):
        self.parent = parent
        self.point_creator = point_creator
        self.window = tk.Toplevel(parent)
        self.window.title("Anchor Manager")
        self.window.geometry("650x550")
        self.window.configure(bg='#2c3e50')
        
        # Set icon
        try:
            icon_path = point_creator.resource_path("lo.ico")
            if os.path.exists(icon_path):
                self.window.iconbitmap(icon_path)
        except:
            pass
        
        # Make window not modal
        self.window.transient(parent)
        
        # Bind close event to cleanup
        self.window.protocol("WM_DELETE_WINDOW", self.on_close)
        
        self.create_widgets()
        self.refresh_anchor_list()
    
    def on_close(self):
        """Handle window close"""
        self.window.destroy()
    
    def create_widgets(self):
        # Title
        title_label = ttk.Label(self.window, text="ANCHOR MANAGER", font=('Arial', 14, 'bold'))
        title_label.pack(pady=10)
        
        # Import/Export buttons frame
        io_frame = ttk.Frame(self.window)
        io_frame.pack(pady=5)
        
        # Кнопки: ↓AnchorBook, ❌ Clear All, ↑AnchorBook
        ttk.Button(io_frame, text="↓AnchorBook", command=self.import_anchors, width=12).pack(side="left", padx=5)
        ttk.Button(io_frame, text="❌ Clear All", command=self.clear_all_anchors, width=12).pack(side="left", padx=5)
        ttk.Button(io_frame, text="↑AnchorBook", command=self.export_anchors, width=12).pack(side="left", padx=5)
        
        # List frame
        list_frame = ttk.LabelFrame(self.window, text="Anchors", padding=10)
        list_frame.pack(fill="both", expand=True, padx=10, pady=5)
        
        # Create Treeview for anchors
        columns = ("Name", "X1", "Y1", "X2", "Y2")
        self.tree = ttk.Treeview(list_frame, columns=columns, show="headings", height=12)
        
        # Define headings
        self.tree.heading("Name", text="Anchor Name")
        self.tree.heading("X1", text="Emotion X")
        self.tree.heading("Y1", text="Emotion Y")
        self.tree.heading("X2", text="Plot X")
        self.tree.heading("Y2", text="Plot Y")
        
        # Set column widths
        self.tree.column("Name", width=150)
        self.tree.column("X1", width=80)
        self.tree.column("Y1", width=80)
        self.tree.column("X2", width=80)
        self.tree.column("Y2", width=80)
        
        # Add scrollbar
        scrollbar = ttk.Scrollbar(list_frame, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=scrollbar.set)
        
        self.tree.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Bind selection event
        self.tree.bind('<<TreeviewSelect>>', self.on_anchor_selected)
        
        # Buttons frame
        buttons_frame = ttk.Frame(self.window)
        buttons_frame.pack(pady=15)
        
        # 5 buttons in a grid
        btn_style = {"width": 12, "padding": 5}
        
        ttk.Button(buttons_frame, text="Clone", command=self.clone_anchor, **btn_style).grid(row=0, column=0, padx=5, pady=5)
        ttk.Button(buttons_frame, text="Rename", command=self.rename_anchor, **btn_style).grid(row=0, column=1, padx=5, pady=5)
        ttk.Button(buttons_frame, text="Delete", command=self.delete_anchor, **btn_style).grid(row=0, column=2, padx=5, pady=5)
        ttk.Button(buttons_frame, text="Move", command=self.move_anchor, **btn_style).grid(row=1, column=0, padx=5, pady=5)
        ttk.Button(buttons_frame, text="Use", command=self.use_anchor, **btn_style).grid(row=1, column=1, padx=5, pady=5)
        ttk.Button(buttons_frame, text="Close", command=self.on_close, **btn_style).grid(row=1, column=2, padx=5, pady=5)
        
        self.selected_anchor = None
    
    def clear_all_anchors(self):
        """Удалить все якоря"""
        if not self.point_creator.anchors:
            messagebox.showinfo("Info", "No anchors to clear!", parent=self.window)
            return
        
        # Confirm deletion
        if messagebox.askyesno("Confirm Clear All", 
                               f"Are you sure you want to delete ALL {len(self.point_creator.anchors)} anchors?\n\nThis action cannot be undone!",
                               parent=self.window):
            self.point_creator.anchors.clear()
            self.point_creator.save_anchors_to_config()
            self.selected_anchor = None
            self.refresh_anchor_list()
            self.point_creator.update_nearest_anchor_display()
            messagebox.showinfo("Success", "All anchors have been cleared!", parent=self.window)
    
    def refresh_anchor_list(self):
        """Обновить список якорей"""
        # Clear existing items
        for item in self.tree.get_children():
            self.tree.delete(item)
        
        # Add anchors to tree
        for anchor in self.point_creator.anchors:
            coords = anchor.coordinates
            self.tree.insert("", "end", values=(
                anchor.name,
                f"{coords[0]:.3f}",
                f"{coords[1]:.3f}",
                f"{coords[2]:.3f}",
                f"{coords[3]:.3f}"
            ), tags=(anchor.name,))
    
    def on_anchor_selected(self, event):
        """Обработка выбора якоря"""
        selection = self.tree.selection()
        if selection:
            item = self.tree.item(selection[0])
            anchor_name = item['values'][0]
            # Find anchor by name
            for anchor in self.point_creator.anchors:
                if anchor.name == anchor_name:
                    self.selected_anchor = anchor
                    break
    
    def import_anchors(self):
        """Import anchors from AnchorBook file"""
        filepath = filedialog.askopenfilename(
            title="Import AnchorBook",
            filetypes=[("AnchorBook files", "*.json"), ("All files", "*.*")],
            parent=self.window
        )
        
        if filepath:
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    anchorbook_data = json.load(f)
                
                # Validate format
                if "anchors" in anchorbook_data and isinstance(anchorbook_data["anchors"], list):
                    imported_count = 0
                    for anchor_data in anchorbook_data["anchors"]:
                        # Check if anchor with same name exists
                        existing = [a for a in self.point_creator.anchors if a.name == anchor_data["name"]]
                        if existing:
                            if messagebox.askyesno("Overwrite", 
                                f"Anchor '{anchor_data['name']}' already exists. Overwrite?",
                                parent=self.window):
                                self.point_creator.anchors.remove(existing[0])
                                new_anchor = Anchor.from_dict(anchor_data)
                                self.point_creator.anchors.append(new_anchor)
                                imported_count += 1
                        else:
                            new_anchor = Anchor.from_dict(anchor_data)
                            self.point_creator.anchors.append(new_anchor)
                            imported_count += 1
                    
                    self.point_creator.save_anchors_to_config()
                    self.refresh_anchor_list()
                    messagebox.showinfo("Success", f"Imported {imported_count} anchors from AnchorBook!", parent=self.window)
                else:
                    messagebox.showerror("Error", "Invalid AnchorBook format!", parent=self.window)
                    
            except Exception as e:
                messagebox.showerror("Error", f"Failed to import AnchorBook: {str(e)}", parent=self.window)
    
    def export_anchors(self):
        """Export anchors to AnchorBook file"""
        if not self.point_creator.anchors:
            messagebox.showwarning("Warning", "No anchors to export!", parent=self.window)
            return
        
        filepath = filedialog.asksaveasfilename(
            title="Export AnchorBook",
            defaultextension=".json",
            filetypes=[("AnchorBook files", "*.json"), ("All files", "*.*")],
            parent=self.window
        )
        
        if filepath:
            try:
                anchorbook_data = {
                    "version": "1.0",
                    "created": time.time(),
                    "anchor_count": len(self.point_creator.anchors),
                    "anchors": [anchor.to_dict() for anchor in self.point_creator.anchors]
                }
                
                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(anchorbook_data, f, indent=4, ensure_ascii=False)
                
                messagebox.showinfo("Success", f"Exported {len(self.point_creator.anchors)} anchors to AnchorBook!", parent=self.window)
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to export AnchorBook: {str(e)}", parent=self.window)
    
    def clone_anchor(self):
        """Клонировать выбранный якорь"""
        if not self.selected_anchor:
            messagebox.showwarning("Warning", "Please select an anchor first!", parent=self.window)
            return
        
        # Ask for new name
        new_name = simpledialog.askstring("Clone Anchor", f"Enter name for cloned anchor:", parent=self.window)
        if new_name:
            # Check if name already exists
            if any(anchor.name == new_name for anchor in self.point_creator.anchors):
                messagebox.showerror("Error", f"Anchor with name '{new_name}' already exists!", parent=self.window)
                return
            
            # Create new anchor with same coordinates
            new_anchor = Anchor(new_name, self.selected_anchor.coordinates)
            self.point_creator.anchors.append(new_anchor)
            self.point_creator.save_anchors_to_config()
            self.refresh_anchor_list()
            messagebox.showinfo("Success", f"Anchor '{new_name}' cloned successfully!", parent=self.window)
    
    def rename_anchor(self):
        """Переименовать выбранный якорь"""
        if not self.selected_anchor:
            messagebox.showwarning("Warning", "Please select an anchor first!", parent=self.window)
            return
        
        new_name = simpledialog.askstring("Rename Anchor", f"Enter new name for '{self.selected_anchor.name}':", parent=self.window)
        if new_name:
            # Check if name already exists
            if any(anchor.name == new_name for anchor in self.point_creator.anchors):
                messagebox.showerror("Error", f"Anchor with name '{new_name}' already exists!", parent=self.window)
                return
            
            old_name = self.selected_anchor.name
            self.selected_anchor.name = new_name
            self.point_creator.save_anchors_to_config()
            self.refresh_anchor_list()
            messagebox.showinfo("Success", f"Anchor renamed from '{old_name}' to '{new_name}'!", parent=self.window)
    
    def delete_anchor(self):
        """Удалить выбранный якорь"""
        if not self.selected_anchor:
            messagebox.showwarning("Warning", "Please select an anchor first!", parent=self.window)
            return
        
        # Confirm deletion
        if messagebox.askyesno("Confirm Delete", f"Are you sure you want to delete anchor '{self.selected_anchor.name}'?", parent=self.window):
            self.point_creator.anchors.remove(self.selected_anchor)
            self.point_creator.save_anchors_to_config()
            self.selected_anchor = None
            self.refresh_anchor_list()
            self.point_creator.update_nearest_anchor_display()
            messagebox.showinfo("Success", "Anchor deleted successfully!", parent=self.window)
    
    def move_anchor(self):
        """Изменить координаты якоря через ввод чисел"""
        if not self.selected_anchor:
            messagebox.showwarning("Warning", "Please select an anchor first!", parent=self.window)
            return
        
        # Create move dialog
        move_dialog = tk.Toplevel(self.window)
        move_dialog.title("Move Anchor")
        move_dialog.geometry("300x250")
        move_dialog.configure(bg='#2c3e50')
        move_dialog.transient(self.window)
        move_dialog.grab_set()
        
        ttk.Label(move_dialog, text=f"Moving Anchor: {self.selected_anchor.name}", font=('Arial', 10, 'bold')).pack(pady=10)
        
        # Create input fields
        fields = [
            ("Emotion X (0-1):", self.selected_anchor.coordinates[0]),
            ("Emotion Y (0-1):", self.selected_anchor.coordinates[1]),
            ("Plot X (0-1):", self.selected_anchor.coordinates[2]),
            ("Plot Y (0-1):", self.selected_anchor.coordinates[3])
        ]
        
        entries = []
        for label_text, default_value in fields:
            frame = ttk.Frame(move_dialog)
            frame.pack(pady=5, padx=20, fill="x")
            
            ttk.Label(frame, text=label_text).pack(side="left")
            entry = ttk.Entry(frame)
            entry.insert(0, f"{default_value:.3f}")
            entry.pack(side="right", expand=True, fill="x", padx=(10, 0))
            entries.append(entry)
        
        def apply_move():
            try:
                new_coords = []
                for entry in entries:
                    value = float(entry.get())
                    if 0 <= value <= 1:
                        new_coords.append(value)
                    else:
                        messagebox.showerror("Error", "Coordinates must be between 0 and 1!", parent=move_dialog)
                        return
                
                self.selected_anchor.coordinates = new_coords
                self.point_creator.save_anchors_to_config()
                self.refresh_anchor_list()
                move_dialog.destroy()
                messagebox.showinfo("Success", "Anchor moved successfully!", parent=self.window)
                
                # Update current points display if this anchor is being used
                self.point_creator.update_nearest_anchor_display()
                
            except ValueError:
                messagebox.showerror("Error", "Please enter valid numbers!", parent=move_dialog)
        
        ttk.Button(move_dialog, text="Apply", command=apply_move).pack(pady=15)
    
    def use_anchor(self):
        """Переместить текущие точки на координаты якоря"""
        if not self.selected_anchor:
            messagebox.showwarning("Warning", "Please select an anchor first!", parent=self.window)
            return
        
        # Move points to anchor coordinates
        self.point_creator.set_point_position(1, self.selected_anchor.coordinates[0], self.selected_anchor.coordinates[1])
        self.point_creator.set_point_position(2, self.selected_anchor.coordinates[2], self.selected_anchor.coordinates[3])
        
        # Update current points
        self.point_creator.current_points = self.selected_anchor.coordinates.copy()
        
        # Update nearest anchor display
        self.point_creator.update_nearest_anchor_display()
        
        messagebox.showinfo("Success", f"Points moved to anchor '{self.selected_anchor.name}'!", parent=self.window)
        # Don't close the window automatically


class PointCreator:
    def __init__(self, root):
        self.root = root
        self.root.title("Point Coordinates Creator")
        
        # Configuration
        self.CONFIG_FILE = "config_pointer.ini"
        self.config = configparser.ConfigParser()
        self.load_config()
        
        # Anchors storage
        self.anchors = []  # List of Anchor objects
        self.load_anchors_from_config()
        
        # Episode playback variables
        self.is_episode_playing = False
        self.episode_animation_id = None
        
        # Set icon
        try:
            icon_path = self.resource_path("lo.ico")
            if os.path.exists(icon_path):
                self.root.iconbitmap(icon_path)
        except:
            pass
        
        # Fullscreen setup
        self.is_fullscreen = False
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        window_width = int(screen_width * 0.95)
        window_height = int(screen_height * 0.8)
        x_pos = (screen_width - window_width) // 2
        y_pos = (screen_height - window_height) // 2
        self.root.geometry(f"{window_width}x{window_height}+{x_pos}+{y_pos}")
        
        # Styling
        self.setup_styles()
        
        # Load seasons list
        self.seasons_list = self.load_seasons()
        
        # Data storage variables
        self.season_var = tk.StringVar(value=self.seasons_list[0] if self.seasons_list else "New Generation")
        self.episode_var = tk.StringVar(value="1")
        self.moment_var = tk.DoubleVar(value=50)
        
        # Time variables
        self.hours_var = tk.StringVar(value="00")
        self.minutes_var = tk.StringVar(value="10")
        self.seconds_var = tk.StringVar(value="00")
        self.length_hours_var = tk.StringVar(value="00")
        self.length_minutes_var = tk.StringVar(value="20")
        self.length_seconds_var = tk.StringVar(value="00")
        
        # Animation variables
        self.is_playing = False
        self.animation_start_time = 0
        self.start_moment = 0
        self.episode_data = {"moments": [], "quartets": []}
        self.current_points = [0.5, 0.5, 0.5, 0.5]
        
        # MP4 creation variables
        self.video_hours_var = tk.StringVar(value="0")
        self.video_minutes_var = tk.StringVar(value="2")
        self.video_seconds_var = tk.StringVar(value="0")
        self.video_path_var = tk.StringVar(value="")
        
        # Variable for loaded capsule
        self.loaded_capsule = None
        
        # Initialize empty dataset
        self.data = {"X": [], "Y": []}
        
        # Label for nearest anchor display
        self.nearest_anchor_label = None
        self.play_episode_button = None
        
        self.create_widgets()
        self.load_background_images()
        
        # Check command line arguments
        if len(sys.argv) > 1 and sys.argv[1].endswith('.lvp'):
            self.load_capsule(sys.argv[1])
    
    def load_anchors_from_config(self):
        """Load anchors from configuration file"""
        self.anchors = []
        if self.config.has_section('Anchors'):
            for key in self.config.options('Anchors'):
                if key.startswith('anchor_'):
                    try:
                        anchor_data = json.loads(self.config.get('Anchors', key))
                        anchor = Anchor.from_dict(anchor_data)
                        self.anchors.append(anchor)
                    except Exception as e:
                        print(f"Error loading anchor {key}: {e}")
        
        # Add default anchors if none exist
        if not self.anchors:
            default_anchors = [
                Anchor("Idle", [0.5, 0.5, 0.5, 0.5])
            ]
            self.anchors = default_anchors
            self.save_anchors_to_config()
    
    def save_anchors_to_config(self):
        """Save anchors to configuration file"""
        if not self.config.has_section('Anchors'):
            self.config.add_section('Anchors')
        
        # Remove old anchor entries
        for key in self.config.options('Anchors'):
            self.config.remove_option('Anchors', key)
        
        # Save current anchors
        for i, anchor in enumerate(self.anchors):
            self.config.set('Anchors', f'anchor_{i}', json.dumps(anchor.to_dict()))
        
        self.save_config()
    
    def add_anchor_from_current_points(self):
        """Create a new anchor from current point positions"""
        # Ask for anchor name
        name = simpledialog.askstring("Add Anchor", "Enter name for this anchor:", parent=self.root)
        if not name:
            return
        
        # Check if name already exists
        if any(anchor.name == name for anchor in self.anchors):
            if not messagebox.askyesno("Warning", f"Anchor '{name}' already exists. Overwrite?"):
                return
            # Remove existing anchor with same name
            self.anchors = [a for a in self.anchors if a.name != name]
        
        # Create new anchor with current points
        new_anchor = Anchor(name, self.current_points)
        self.anchors.append(new_anchor)
        self.save_anchors_to_config()
        
        messagebox.showinfo("Success", f"Anchor '{name}' added successfully!")
        self.update_nearest_anchor_display()
    
    def calculate_distance(self, points1, points2):
        """Calculate Euclidean distance between two 4D points"""
        return sum((p1 - p2) ** 2 for p1, p2 in zip(points1, points2)) ** 0.5
    
    def find_nearest_anchor(self):
        """Find the nearest anchor to current points"""
        if not self.anchors:
            return None, None
        
        min_distance = float('inf')
        nearest_anchor = None
        
        for anchor in self.anchors:
            distance = self.calculate_distance(self.current_points, anchor.coordinates)
            if distance < min_distance:
                min_distance = distance
                nearest_anchor = anchor
        
        return nearest_anchor, min_distance
    
    def update_nearest_anchor_display(self):
        """Update the display showing nearest anchor"""
        if not self.nearest_anchor_label:
            return
        
        nearest_anchor, distance = self.find_nearest_anchor()
        
        if nearest_anchor:
            # Calculate max possible distance (distance from (0,0,0,0) to (1,1,1,1))
            max_distance = self.calculate_distance([0,0,0,0], [1,1,1,1])
            # Calculate percentage (inverted: closer = smaller distance = smaller percentage)
            # We want: closer = higher percentage
            percentage = (1 - (distance / max_distance)) * 100
            
            self.nearest_anchor_label.config(
                text=f"📍 Nearest Anchor: {nearest_anchor.name} (Match: {percentage:.1f}%)",
                foreground="#40E0D0"
            )
        else:
            self.nearest_anchor_label.config(text="📍 No anchors available", foreground="#ecf0f1")
    
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
    
    def setup_styles(self):
        """Setup modern styles"""
        self.style = ttk.Style()
        self.style.theme_use('clam')
        
        # Color palette
        BG_COLOR = "#2c3e50"
        FRAME_BG = "#34495e"
        BUTTON_BG = "#3498db"
        BUTTON_FG = "white"
        LABEL_BG = BG_COLOR
        LABEL_FG = "#ecf0f1"
        ENTRY_BG = "white"
        ENTRY_FG = "black"
        PROGRESS_BG = "#40E0D0"
        
        # Configure styles
        self.style.configure('TFrame', background=FRAME_BG)
        self.style.configure('TLabelframe', background=FRAME_BG, foreground=LABEL_FG)
        self.style.configure('TLabelframe.Label', background=FRAME_BG, foreground=LABEL_FG, font=('Arial', 10, 'bold'))
        self.style.configure('TLabel', background=LABEL_BG, foreground=LABEL_FG, font=('Arial', 10))
        self.style.configure('TButton', background=BUTTON_BG, foreground=BUTTON_FG, font=('Arial', 10, 'bold'), padding=8)
        self.style.map('TButton', background=[('active', '#2980b9')])
        self.style.configure('TScale', background=BG_COLOR)
        self.style.configure('TCombobox', fieldbackground=ENTRY_BG, background=ENTRY_BG)
        self.style.configure('TEntry', fieldbackground=ENTRY_BG, background=ENTRY_BG, font=('Arial', 10))
        self.style.configure('Horizontal.TProgressbar', troughcolor=FRAME_BG, background=PROGRESS_BG)
        
        self.root.configure(bg=BG_COLOR)
    
    def load_seasons(self):
        """Load seasons list from file or use default"""
        seasons_file = self.resource_path('seasons.txt')
        seasons = []
        try:
            if os.path.exists(seasons_file):
                with open(seasons_file, 'r', encoding='utf-8') as f:
                    seasons = [line.strip() for line in f if line.strip()]
            else:
                # Fallback seasons list
                seasons = [
                    "New Generation",
                    "Game of God",
                    "Perfect World",
                    "Voice of Time",
                    "Thirteen Lights",
                    "Final Reality",
                    "Heart of the Universe",
                    "Point of No Return",
                    "Workshop [47]"
                    "???"
                ]
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load seasons list: {str(e)}")
            seasons = ["New Generation"]
        
        return seasons
    
    def load_background_images(self):
        """Load background images for the point canvases"""
        try:
            # Left image - EmoPlain.png
            emo_image = Image.open(self.resource_path("EmoPlain.png"))
            emo_image = emo_image.resize((300, 300), Image.Resampling.LANCZOS)
            self.emo_bg = ImageTk.PhotoImage(emo_image)
            
            # Right image - PlotPlain.png
            plot_image = Image.open(self.resource_path("PlotPlain.png"))
            plot_image = plot_image.resize((300, 300), Image.Resampling.LANCZOS)
            self.plot_bg = ImageTk.PhotoImage(plot_image)
            
            # Set background images
            self.canvas1.create_image(0, 0, anchor="nw", image=self.emo_bg)
            self.canvas2.create_image(0, 0, anchor="nw", image=self.plot_bg)
            
            # Bring points to front
            self.canvas1.tag_raise(self.point1)
            self.canvas2.tag_raise(self.point2)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load background images: {str(e)}")
    
    def create_widgets(self):
        """Create all GUI widgets"""
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
        timeline_title = ttk.Label(left_frame, text="TIMELINE", font=('Arial', 12, 'bold'))
        timeline_title.pack(pady=10)
        
        timeline_frame = ttk.LabelFrame(left_frame, text="Timeline", padding=10)
        timeline_frame.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Season selection
        ttk.Label(timeline_frame, text="Season:").pack(anchor="w", padx=5, pady=5)
        season_combo = ttk.Combobox(timeline_frame, textvariable=self.season_var, 
                                   values=self.seasons_list, state="readonly")
        season_combo.pack(fill="x", padx=5, pady=5)
        
        # Episode selection
        ttk.Label(timeline_frame, text="Episode:").pack(anchor="w", padx=5, pady=5)
        ttk.Entry(timeline_frame, textvariable=self.episode_var).pack(fill="x", padx=5, pady=5)
        
        # Timecode input
        ttk.Label(timeline_frame, text="Timecode (HH:MM:SS):").pack(anchor="w", padx=5, pady=5)
        
        timecode_frame = ttk.Frame(timeline_frame)
        timecode_frame.pack(fill="x", padx=5, pady=5)
        
        ttk.Entry(timecode_frame, textvariable=self.hours_var, width=3).pack(side="left")
        ttk.Label(timecode_frame, text=":").pack(side="left")
        ttk.Entry(timecode_frame, textvariable=self.minutes_var, width=3).pack(side="left")
        ttk.Label(timecode_frame, text=":").pack(side="left")
        ttk.Entry(timecode_frame, textvariable=self.seconds_var, width=3).pack(side="left")
        
        # Episode length input
        ttk.Label(timeline_frame, text="Episode Length (HH:MM:SS):").pack(anchor="w", padx=5, pady=5)
        
        length_frame = ttk.Frame(timeline_frame)
        length_frame.pack(fill="x", padx=5, pady=5)
        
        ttk.Entry(length_frame, textvariable=self.length_hours_var, width=3).pack(side="left")
        ttk.Label(length_frame, text=":").pack(side="left")
        ttk.Entry(length_frame, textvariable=self.length_minutes_var, width=3).pack(side="left")
        ttk.Label(length_frame, text=":").pack(side="left")
        ttk.Entry(length_frame, textvariable=self.length_seconds_var, width=3).pack(side="left")
        
        # Moment percentage
        ttk.Label(timeline_frame, text="Timecode (%):").pack(anchor="w", padx=5, pady=5)
        self.moment_scale = ttk.Scale(timeline_frame, from_=0, to=100, variable=self.moment_var, 
                                     orient="horizontal")
        self.moment_scale.pack(fill="x", padx=5, pady=5)
        self.moment_label = ttk.Label(timeline_frame, text="50.0%")
        self.moment_label.pack(pady=5)
        
        # Play Episode button under timeline
        self.play_episode_button = ttk.Button(timeline_frame, text="▶ Play Episode", command=self.toggle_episode_playback, width=20)
        self.play_episode_button.pack(pady=10)
        
        # Bind scale movement to update function
        self.moment_scale.configure(command=self.update_timestamp_from_scale)
        
        # Bind timecode entries to update functions
        for var in [self.hours_var, self.minutes_var, self.seconds_var]:
            var.trace_add('write', lambda *args: self.update_timecode_from_entries())
        
        for var in [self.length_hours_var, self.length_minutes_var, self.length_seconds_var]:
            var.trace_add('write', lambda *args: self.update_length_from_entries())
        
        # ==================== CENTER FRAME - POINTS ====================
        points_title = ttk.Label(center_frame, text="DESCRIPTION", font=('Arial', 12, 'bold'))
        points_title.pack(pady=10)
        
        points_frame = ttk.Frame(center_frame)
        points_frame.pack(fill="both", expand=True, pady=5)
        
        # Point 1 (Emotion)
        point1_frame = ttk.LabelFrame(points_frame, text="Emotion", padding=10)
        point1_frame.pack(side="left", fill="both", expand=True, padx=10)
        
        self.canvas1 = tk.Canvas(point1_frame, width=300, height=300, bg="white", highlightthickness=1, highlightbackground="#34495e")
        self.canvas1.pack(pady=5)
        self.point1 = self.canvas1.create_oval(145, 145, 155, 155, fill="red", outline="red")
        
        # Detailed coordinates for Emotion point
        coord1_frame = ttk.Frame(point1_frame)
        coord1_frame.pack(pady=5)
        
        ttk.Label(coord1_frame, text="X:").pack(side="left")
        self.x1_var = tk.StringVar(value="0.500")
        ttk.Label(coord1_frame, textvariable=self.x1_var, width=6).pack(side="left", padx=(0, 10))
        
        ttk.Label(coord1_frame, text="Y:").pack(side="left")
        self.y1_var = tk.StringVar(value="0.500")
        ttk.Label(coord1_frame, textvariable=self.y1_var, width=6).pack(side="left")
        
        # Point 2 (Plot)
        point2_frame = ttk.LabelFrame(points_frame, text="Plot", padding=10)
        point2_frame.pack(side="right", fill="both", expand=True, padx=10)
        
        self.canvas2 = tk.Canvas(point2_frame, width=300, height=300, bg="white", highlightthickness=1, highlightbackground="#34495e")
        self.canvas2.pack(pady=5)
        self.point2 = self.canvas2.create_oval(145, 145, 155, 155, fill="blue", outline="blue")
        
        # Detailed coordinates for Plot point
        coord2_frame = ttk.Frame(point2_frame)
        coord2_frame.pack(pady=5)
        
        ttk.Label(coord2_frame, text="X:").pack(side="left")
        self.x2_var = tk.StringVar(value="0.500")
        ttk.Label(coord2_frame, textvariable=self.x2_var, width=6).pack(side="left", padx=(0, 10))
        
        ttk.Label(coord2_frame, text="Y:").pack(side="left")
        self.y2_var = tk.StringVar(value="0.500")
        ttk.Label(coord2_frame, textvariable=self.y2_var, width=6).pack(side="left")
        
        # Buttons frame under squares
        buttons_bottom_frame = ttk.Frame(center_frame)
        buttons_bottom_frame.pack(pady=10)
        
        add_button = ttk.Button(buttons_bottom_frame, text="ADD", command=self.add_current_data, width=10)
        add_button.pack(side="left", padx=5)
        
        add_anchor_button = ttk.Button(buttons_bottom_frame, text="Add Anchor", command=self.add_anchor_from_current_points, width=12)
        add_anchor_button.pack(side="left", padx=5)
        
        anchor_menu_button = ttk.Button(buttons_bottom_frame, text="Anchor Menu", command=self.open_anchor_menu, width=12)
        anchor_menu_button.pack(side="left", padx=5)
        
        # Nearest anchor display label
        self.nearest_anchor_label = ttk.Label(center_frame, text="📍 Nearest Anchor: None", font=('Arial', 10, 'bold'))
        self.nearest_anchor_label.pack(pady=5)
        self.update_nearest_anchor_display()
        
        # Bind point movement events
        self.canvas1.bind("<B1-Motion>", lambda e: self.move_point(e, 1))
        self.canvas2.bind("<B1-Motion>", lambda e: self.move_point(e, 2))
        
        # ==================== RIGHT FRAME - CONTROLS ====================
        controls_title = ttk.Label(right_frame, text="Controls", font=('Arial', 12, 'bold'))
        controls_title.pack(pady=10)
        
        controls_frame = ttk.LabelFrame(right_frame, text="Controls", padding=10)
        controls_frame.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Dataset operation buttons
        ttk.Button(controls_frame, text="Load data", command=self.load_dataset, width=15).pack(pady=5, fill="x")
        ttk.Button(controls_frame, text="Save data", command=self.save_dataset_as, width=15).pack(pady=5, fill="x")
        
        # Playback and export buttons
        ttk.Button(controls_frame, text="Create MP4", command=self.open_mp4_dialog, width=15).pack(pady=5, fill="x")
        ttk.Button(controls_frame, text="Load Capsule", command=self.load_capsule_dialog, width=15).pack(pady=5, fill="x")
        
        # Utility buttons
        ttk.Button(controls_frame, text="Fullscreen", command=self.toggle_fullscreen, width=15).pack(pady=5, fill="x")
    
    def open_anchor_menu(self):
        """Open the anchor management window"""
        AnchorMenu(self.root, self)
    
    def toggle_episode_playback(self):
        """Toggle episode playback with speed 10% per second"""
        if self.is_episode_playing:
            # Stop playback
            self.is_episode_playing = False
            if self.episode_animation_id:
                self.root.after_cancel(self.episode_animation_id)
                self.episode_animation_id = None
            self.play_episode_button.config(text="▶ Play Episode")
            self.moment_scale.config(state="normal")
        else:
            # Start playback
            # Get data for selected episode
            if self.loaded_capsule:
                season_index = self.seasons_list.index(self.season_var.get()) + 1
                if (self.loaded_capsule["season"] == season_index and 
                    self.loaded_capsule["episode"] == int(self.episode_var.get())):
                    self.episode_data = {
                        "moments": self.loaded_capsule["moments"],
                        "quartets": self.loaded_capsule["quartets"]
                    }
                else:
                    messagebox.showwarning("Warning", "Loaded capsule doesn't match selected season/episode!")
                    return
            else:
                self.episode_data = self.get_episode_data()
            
            if not self.episode_data["moments"]:
                messagebox.showwarning("Warning", "No data for selected episode!")
                return
            
            self.is_episode_playing = True
            self.play_episode_button.config(text="⏹ Stop Episode")
            self.moment_scale.config(state="disabled")
            self.play_episode_animation()
    
    def play_episode_animation(self):
        """Play episode animation at 10% per second"""
        if not self.is_episode_playing:
            return
        
        # Get current moment
        current_moment = self.moment_var.get()
        
        # Increase by 10% per second (10% per 1000ms, so 1% per 100ms)
        # Update every 50ms for smooth animation
        increment_per_frame = 0.5  # 0.5% per 50ms = 10% per second
        
        new_moment = current_moment + increment_per_frame
        
        if new_moment >= 100.0:
            # Reached the end
            new_moment = 100.0
            self.moment_var.set(new_moment)
            self.moment_label.config(text=f"{new_moment:.1f}%")
            self.update_timestamp_from_scale(new_moment)
            # IMPORTANT: Update points when reaching the end
            self.update_points_from_scale()
            self.is_episode_playing = False
            self.play_episode_button.config(text="▶ Play Episode")
            self.moment_scale.config(state="normal")
            return
        
        # Update moment
        self.moment_var.set(new_moment)
        self.moment_label.config(text=f"{new_moment:.1f}%")
        self.update_timestamp_from_scale(new_moment)
        # IMPORTANT: Update points on each frame
        self.update_points_from_scale()
        
        # Schedule next update
        self.episode_animation_id = self.root.after(50, self.play_episode_animation)
    
    def add_current_data(self):
        """Add current data to dataset"""
        # Get point coordinates
        coords1 = self.canvas1.coords(self.point1)
        coords2 = self.canvas2.coords(self.point2)
        
        # Normalize coordinates
        x1_norm = (coords1[0] + 5) / 300
        y1_norm = 1 - ((coords1[1] + 5) / 300)
        x2_norm = (coords2[0] + 5) / 300
        y2_norm = 1 - ((coords2[1] + 5) / 300)
        
        # Get selected season index (starting from 1)
        season_index = self.seasons_list.index(self.season_var.get()) + 1
        season_norm = season_index / 10
        
        # Normalize input data
        try:
            episode_norm = float(self.episode_var.get()) / 1000
            moment_norm = self.moment_var.get() / 100
            
            # Add data to structure
            self.data["X"].append([season_norm, episode_norm, moment_norm])
            self.data["Y"].append([x1_norm, y1_norm, x2_norm, y2_norm])
            
            messagebox.showinfo("Success", "Data successfully added to dataset!")
        except ValueError:
            messagebox.showerror("Error", "Invalid episode number!")
    
    def load_dataset(self):
        """Load dataset from file"""
        filepath = filedialog.askopenfilename(
            title="Select dataset file",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        if filepath:
            try:
                with open(filepath, 'r') as f:
                    self.data = json.load(f)
                messagebox.showinfo("Success", "Dataset loaded!")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load dataset: {str(e)}")
    
    def save_dataset_as(self):
        """Save dataset to selected file"""
        filepath = filedialog.asksaveasfilename(
            title="Save dataset as",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        if filepath:
            try:
                with open(filepath, 'w') as f:
                    json.dump(self.data, f, indent=4)
                messagebox.showinfo("Success", "Dataset saved!")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save dataset: {str(e)}")
    
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
        
        # Automatically update point positions when slider moves
        self.update_points_from_scale()
    
    def update_points_from_scale(self):
        """Update point positions when scale moves"""
        # If episode playback is active, still update points (we want them to move)
        # Remove the check for is_episode_playing to allow updates during playback
            
        # Get data for selected episode
        if self.loaded_capsule:
            season_index = self.seasons_list.index(self.season_var.get()) + 1
            if (self.loaded_capsule["season"] == season_index and 
                self.loaded_capsule["episode"] == int(self.episode_var.get())):
                self.episode_data = {
                    "moments": self.loaded_capsule["moments"],
                    "quartets": self.loaded_capsule["quartets"]
                }
            else:
                # Capsule doesn't match selected episode
                return
        else:
            # Use data from IDS.json (old format)
            self.episode_data = self.get_episode_data()
        
        if not self.episode_data["moments"]:
            return
        
        # Get target position for current moment
        current_time = self.moment_var.get()
        target_points = self.get_interpolated_target(current_time)
        
        if target_points:
            # Set points to target position
            self.set_point_position(1, target_points[0], target_points[1])
            self.set_point_position(2, target_points[2], target_points[3])
            # Update current points
            self.current_points = target_points.copy()
            # Update nearest anchor display
            self.update_nearest_anchor_display()
    
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
    
    def toggle_fullscreen(self):
        """Toggle fullscreen mode"""
        self.is_fullscreen = not self.is_fullscreen
        self.root.attributes("-fullscreen", self.is_fullscreen)
    
    def move_point(self, event, point_num):
        """Move point on canvas when dragged"""
        canvas = self.canvas1 if point_num == 1 else self.canvas2
        point = self.point1 if point_num == 1 else self.point2
        
        # Limit movement within canvas
        x = max(0, min(event.x, 300))
        y = max(0, min(event.y, 300))
        
        canvas.coords(point, x-5, y-5, x+5, y+5)
        
        # Convert coordinates to [0,1]
        x_norm = x / 300
        y_norm = 1 - (y / 300)
        
        # Update current positions
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
        
        # Update nearest anchor display
        self.update_nearest_anchor_display()
    
    def get_episode_data(self):
        """Get data only for selected season and episode (old format)"""
        season_index = self.seasons_list.index(self.season_var.get()) + 1
        season_norm = season_index / 10
        episode_norm = int(self.episode_var.get()) / 1000
        
        moments = []
        quartets = []
        
        for i in range(len(self.data["X"])):
            x_data = self.data["X"][i]
            y_data = self.data["Y"][i]
            
            # Check if record matches selected season and episode
            if abs(x_data[0] - season_norm) < 0.001 and abs(x_data[1] - episode_norm) < 0.001:
                moments.append(x_data[2] * 100)
                quartets.append(y_data)
        
        # Sort data by time moment
        if moments:
            combined = list(zip(moments, quartets))
            combined.sort(key=lambda x: x[0])
            moments, quartets = zip(*combined)
            moments = list(moments)
            quartets = list(quartets)
        
        return {"moments": moments, "quartets": quartets}
    
    def get_interpolated_target(self, current_time):
        """Get target position through interpolation between frames"""
        moments = self.episode_data["moments"]
        quartets = self.episode_data["quartets"]
        
        if not moments:
            return None
        
        # If current time is before first frame, use first frame
        if current_time <= moments[0]:
            return quartets[0]
        
        # If current time is after last frame, use last frame
        if current_time >= moments[-1]:
            return quartets[-1]
        
        # Find index of first frame that's greater than current time
        idx = bisect.bisect_right(moments, current_time)
        
        # Interpolate between two frames
        prev_time = moments[idx - 1]
        next_time = moments[idx]
        prev_points = quartets[idx - 1]
        next_points = quartets[idx]
        
        # Calculate interpolation coefficient
        t = (current_time - prev_time) / (next_time - prev_time)
        
        # Interpolate each coordinate
        interpolated_points = []
        for i in range(4):
            interpolated_value = prev_points[i] + (next_points[i] - prev_points[i]) * t
            interpolated_points.append(interpolated_value)
        
        return interpolated_points
    
    def set_point_position(self, point_num, x_norm, y_norm):
        """Set point position from normalized coordinates"""
        canvas = self.canvas1 if point_num == 1 else self.canvas2
        point = self.point1 if point_num == 1 else self.point2
        
        # Convert normalized coordinates to pixels
        x = x_norm * 300
        y = (1 - y_norm) * 300
        
        # Set point position
        canvas.coords(point, x-5, y-5, x+5, y+5)
        
        # Update text labels
        if point_num == 1:
            self.x1_var.set(f"{x_norm:.3f}")
            self.y1_var.set(f"{y_norm:.3f}")
        else:
            self.x2_var.set(f"{x_norm:.3f}")
            self.y2_var.set(f"{y_norm:.3f}")
    
    def load_capsule_dialog(self):
        """Open dialog to load capsule"""
        filepath = filedialog.askopenfilename(
            title="Select capsule file",
            filetypes=[("LVP files", "*.lvp"), ("All files", "*.*")]
        )
        
        if filepath:
            self.load_capsule(filepath)
    
    def load_capsule(self, filepath):
        """Load capsule from file"""
        try:
            with open(filepath, 'r') as f:
                capsule_data = json.load(f)
            
            # Check capsule format (new LVP format)
            if "season" in capsule_data and "episode" in capsule_data and "moments" in capsule_data and "quartets" in capsule_data:
                self.loaded_capsule = capsule_data
                
                # Set season and episode from capsule
                season_index = capsule_data["season"] - 1
                if 0 <= season_index < len(self.seasons_list):
                    self.season_var.set(self.seasons_list[season_index])
                self.episode_var.set(str(capsule_data["episode"]))
                
                # Update episode data for playback
                self.episode_data = {
                    "moments": capsule_data["moments"],
                    "quartets": capsule_data["quartets"]
                }
                
                messagebox.showinfo("Success", f"Capsule loaded: Season {capsule_data['season']}, Episode {capsule_data['episode']}\n"
                                           f"Moments: {len(capsule_data['moments'])}")
            else:
                messagebox.showerror("Error", "Invalid capsule format")
                
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load capsule: {str(e)}")
    
    def open_mp4_dialog(self):
        """Open dialog for MP4 creation"""
        mp4_dialog = tk.Toplevel(self.root)
        mp4_dialog.title("Create MP4 Video")
        mp4_dialog.geometry("400x300")
        mp4_dialog.resizable(False, False)
        mp4_dialog.configure(bg='#2c3e50')
        
        # Set icon
        try:
            icon_path = self.resource_path("lo.ico")
            if os.path.exists(icon_path):
                mp4_dialog.iconbitmap(icon_path)
        except:
            pass
        
        ttk.Label(mp4_dialog, text="Video Duration:", font=("Arial", 12)).pack(pady=10)
        
        # Time input fields
        time_frame = ttk.Frame(mp4_dialog)
        time_frame.pack(pady=10)
        
        ttk.Label(time_frame, text="Hours:").grid(row=0, column=0, padx=5)
        hours_entry = ttk.Entry(time_frame, textvariable=self.video_hours_var, width=5)
        hours_entry.grid(row=0, column=1, padx=5)
        
        ttk.Label(time_frame, text="Minutes:").grid(row=0, column=2, padx=5)
        minutes_entry = ttk.Entry(time_frame, textvariable=self.video_minutes_var, width=5)
        minutes_entry.grid(row=0, column=3, padx=5)
        
        ttk.Label(time_frame, text="Seconds:").grid(row=0, column=4, padx=5)
        seconds_entry = ttk.Entry(time_frame, textvariable=self.video_seconds_var, width=5)
        seconds_entry.grid(row=0, column=5, padx=5)
        
        # Save path selection
        path_frame = ttk.Frame(mp4_dialog)
        path_frame.pack(pady=10, fill="x", padx=20)
        
        ttk.Label(path_frame, text="Save Path:").pack(anchor="w")
        
        path_entry_frame = ttk.Frame(path_frame)
        path_entry_frame.pack(fill="x", pady=5)
        
        ttk.Entry(path_entry_frame, textvariable=self.video_path_var, width=30).pack(side="left", fill="x", expand=True, padx=(0, 5))
        ttk.Button(path_entry_frame, text="Browse", command=self.browse_save_path, width=8).pack(side="right")
        
        # Buttons
        btn_frame = ttk.Frame(mp4_dialog)
        btn_frame.pack(pady=20)
        
        ttk.Button(btn_frame, text="Create", 
                  command=lambda: self.start_mp4_creation(mp4_dialog)).pack(side="left", padx=10)
        ttk.Button(btn_frame, text="Cancel", command=mp4_dialog.destroy).pack(side="right", padx=10)
    
    def browse_save_path(self):
        """Select path for saving video"""
        temp = tk.Toplevel(self.root)
        temp.withdraw()
        
        file_path = filedialog.asksaveasfilename(
            defaultextension=".mp4",
            filetypes=[("MP4 files", "*.mp4"), ("All files", "*.*")],
            title="Save video as",
            parent=temp
        )
    
        temp.destroy()
    
        if file_path:
            self.video_path_var.set(file_path)
    
    def start_mp4_creation(self, dialog):
        """Start MP4 creation in separate thread"""
        try:
            hours = int(self.video_hours_var.get())
            minutes = int(self.video_minutes_var.get())
            seconds = int(self.video_seconds_var.get())
            
            total_seconds = hours * 3600 + minutes * 60 + seconds
            
            if total_seconds <= 0:
                messagebox.showerror("Error", "Duration must be greater than 0 seconds!")
                return
                
            output_path = self.video_path_var.get()
            if not output_path:
                messagebox.showerror("Error", "Please select a save path for the video!")
                return
                
            dialog.destroy()
            
            # Start MP4 creation in separate thread
            threading.Thread(target=self.create_mp4_video, args=(total_seconds, output_path), daemon=True).start()
            
        except ValueError:
            messagebox.showerror("Error", "Please enter valid numeric values!")
    
    def create_mp4_video(self, total_seconds, output_path):
        """Create MP4 video with animation"""
        try:
            # Get episode data
            if self.loaded_capsule:
                # Use data from loaded capsule
                self.episode_data = {
                    "moments": self.loaded_capsule["moments"],
                    "quartets": self.loaded_capsule["quartets"]
                }
            else:
                # Use data from IDS.json (old format)
                self.episode_data = self.get_episode_data()
            
            if not self.episode_data["moments"]:
                messagebox.showerror("Error", "No data for selected episode!")
                return
            
            # Create folder for frames
            frames_dir = "../data/temp_frames"
            os.makedirs(frames_dir, exist_ok=True)
            
            # Calculate number of frames (30 FPS)
            fps = 30
            total_frames = total_seconds * fps
            
            # Create progress bar
            progress_window = tk.Toplevel(self.root)
            progress_window.title("Creating MP4")
            progress_window.geometry("300x100")
            progress_window.resizable(False, False)
            progress_window.configure(bg='#2c3e50')
            
            # Set icon
            try:
                icon_path = self.resource_path("lo.ico")
                if os.path.exists(icon_path):
                    progress_window.iconbitmap(icon_path)
            except:
                pass
            
            ttk.Label(progress_window, text="Creating video...").pack(pady=5)
            progress_var = tk.DoubleVar()
            progress_bar = ttk.Progressbar(progress_window, variable=progress_var, maximum=total_frames, style="Horizontal.TProgressbar")
            progress_bar.pack(pady=10, padx=20, fill="x")
            
            # Start from center positions
            current_video_points = [0.5, 0.5, 0.5, 0.5]
            
            # Generate frames
            frames = []
            for frame_num in range(total_frames):
                # Calculate current time (0-100%)
                current_time = (frame_num / total_frames) * 100.0
                
                # Get target position through interpolation
                target_points = self.get_interpolated_target(current_time)
                
                if target_points:
                    # Smoothly move points to target position
                    smoothness = 0.1
                    for i in range(4):
                        current_video_points[i] = current_video_points[i] * (1 - smoothness) + target_points[i] * smoothness
                    
                    # Create frame image
                    frame = self.generate_frame(current_video_points)
                    frames.append(frame)
                
                # Update progress
                progress_var.set(frame_num)
                progress_window.update()
            
            # Create video from frames
            clip = ImageSequenceClip(frames, fps=fps)
            clip.write_videofile(output_path, codec="libx264", audio=False)
            
            # Delete temporary files
            for frame_file in os.listdir(frames_dir):
                os.remove(os.path.join(frames_dir, frame_file))
            os.rmdir(frames_dir)
            
            progress_window.destroy()
            messagebox.showinfo("Success", f"Video successfully created: {output_path}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Error creating video: {str(e)}")
    
    def generate_frame(self, points):
        """Generate frame with points on background"""
        # Create image 600x300 (combine two canvases)
        frame = np.ones((300, 600, 3), dtype=np.uint8) * 255
        
        try:
            # Load background images
            emo_bg = cv2.imread(self.resource_path("EmoPlain.png"))
            plot_bg = cv2.imread(self.resource_path("PlotPlain.png"))
            
            # Resize images
            emo_bg = cv2.resize(emo_bg, (300, 300))
            plot_bg = cv2.resize(plot_bg, (300, 300))
            
            # Overlay background images
            frame[0:300, 0:300] = emo_bg
            frame[0:300, 300:600] = plot_bg
        except:
            # If failed to load background images, use white background
            pass
        
        # Draw points
        # Point 1 (red)
        x1 = int(points[0] * 300)
        y1 = int((1 - points[1]) * 300)
        cv2.circle(frame, (x1, y1), 5, (0, 0, 255), -1)
        
        # Point 2 (blue)
        x2 = int(points[2] * 300) + 300
        y2 = int((1 - points[3]) * 300)
        cv2.circle(frame, (x2, y2), 5, (255, 0, 0), -1)
        
        # Convert BGR to RGB for moviepy
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return frame_rgb


if __name__ == "__main__":
    root = tk.Tk()
    app = PointCreator(root)
    root.mainloop()
