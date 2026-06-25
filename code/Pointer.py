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
import math
import shutil
import re
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

os.chdir(os.path.dirname(os.path.abspath(__file__)))

class Anchor:
    """Класс для хранения данных якоря"""
    def __init__(self, name, coordinates):
        self.name = name
        self.coordinates = coordinates.copy()
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
        
        try:
            icon_path = point_creator.resource_path("lo.ico")
            if os.path.exists(icon_path):
                self.window.iconbitmap(icon_path)
        except:
            pass
        
        self.window.transient(parent)
        self.window.protocol("WM_DELETE_WINDOW", self.on_close)
        
        self.create_widgets()
        self.refresh_anchor_list()
    
    def on_close(self):
        self.window.destroy()
    
    def create_widgets(self):
        title_label = ttk.Label(self.window, text="ANCHOR MANAGER", font=('Arial', 14, 'bold'))
        title_label.pack(pady=10)
        
        io_frame = ttk.Frame(self.window)
        io_frame.pack(pady=5)
        
        ttk.Button(io_frame, text="↓AnchorBook", command=self.import_anchors, width=12).pack(side="left", padx=5)
        ttk.Button(io_frame, text="❌ Clear All", command=self.clear_all_anchors, width=12).pack(side="left", padx=5)
        ttk.Button(io_frame, text="↑AnchorBook", command=self.export_anchors, width=12).pack(side="left", padx=5)
        
        list_frame = ttk.LabelFrame(self.window, text="Anchors", padding=10)
        list_frame.pack(fill="both", expand=True, padx=10, pady=5)
        
        columns = ("Name", "X1", "Y1", "X2", "Y2")
        self.tree = ttk.Treeview(list_frame, columns=columns, show="headings", height=12)
        
        self.tree.heading("Name", text="Anchor Name")
        self.tree.heading("X1", text="Emotion X")
        self.tree.heading("Y1", text="Emotion Y")
        self.tree.heading("X2", text="Plot X")
        self.tree.heading("Y2", text="Plot Y")
        
        self.tree.column("Name", width=150)
        self.tree.column("X1", width=80)
        self.tree.column("Y1", width=80)
        self.tree.column("X2", width=80)
        self.tree.column("Y2", width=80)
        
        scrollbar = ttk.Scrollbar(list_frame, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=scrollbar.set)
        
        self.tree.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        self.tree.bind('<<TreeviewSelect>>', self.on_anchor_selected)
        
        buttons_frame = ttk.Frame(self.window)
        buttons_frame.pack(pady=15)
        
        btn_style = {"width": 12, "padding": 5}
        
        ttk.Button(buttons_frame, text="Clone", command=self.clone_anchor, **btn_style).grid(row=0, column=0, padx=5, pady=5)
        ttk.Button(buttons_frame, text="Rename", command=self.rename_anchor, **btn_style).grid(row=0, column=1, padx=5, pady=5)
        ttk.Button(buttons_frame, text="Delete", command=self.delete_anchor, **btn_style).grid(row=0, column=2, padx=5, pady=5)
        ttk.Button(buttons_frame, text="Move", command=self.move_anchor, **btn_style).grid(row=1, column=0, padx=5, pady=5)
        ttk.Button(buttons_frame, text="Use", command=self.use_anchor, **btn_style).grid(row=1, column=1, padx=5, pady=5)
        ttk.Button(buttons_frame, text="Close", command=self.on_close, **btn_style).grid(row=1, column=2, padx=5, pady=5)
        
        self.selected_anchor = None
    
    def clear_all_anchors(self):
        if not self.point_creator.anchors:
            messagebox.showinfo("Info", "No anchors to clear!", parent=self.window)
            return
        
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
        for item in self.tree.get_children():
            self.tree.delete(item)
        
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
        selection = self.tree.selection()
        if selection:
            item = self.tree.item(selection[0])
            anchor_name = item['values'][0]
            for anchor in self.point_creator.anchors:
                if anchor.name == anchor_name:
                    self.selected_anchor = anchor
                    self.point_creator.start_anchor_follow_mode(anchor)
                    break
    
    def import_anchors(self):
        filepath = filedialog.askopenfilename(
            title="Import AnchorBook",
            filetypes=[("AnchorBook files", "*.json"), ("All files", "*.*")],
            parent=self.window
        )
        
        if filepath:
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    anchorbook_data = json.load(f)
                
                if "anchors" in anchorbook_data and isinstance(anchorbook_data["anchors"], list):
                    imported_count = 0
                    for anchor_data in anchorbook_data["anchors"]:
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
        if not self.selected_anchor:
            messagebox.showwarning("Warning", "Please select an anchor first!", parent=self.window)
            return
        
        new_name = simpledialog.askstring("Clone Anchor", f"Enter name for cloned anchor:", parent=self.window)
        if new_name:
            if any(anchor.name == new_name for anchor in self.point_creator.anchors):
                messagebox.showerror("Error", f"Anchor with name '{new_name}' already exists!", parent=self.window)
                return
            
            new_anchor = Anchor(new_name, self.selected_anchor.coordinates)
            self.point_creator.anchors.append(new_anchor)
            self.point_creator.save_anchors_to_config()
            self.refresh_anchor_list()
            messagebox.showinfo("Success", f"Anchor '{new_name}' cloned successfully!", parent=self.window)
    
    def rename_anchor(self):
        if not self.selected_anchor:
            messagebox.showwarning("Warning", "Please select an anchor first!", parent=self.window)
            return
        
        new_name = simpledialog.askstring("Rename Anchor", f"Enter new name for '{self.selected_anchor.name}':", parent=self.window)
        if new_name:
            if any(anchor.name == new_name for anchor in self.point_creator.anchors):
                messagebox.showerror("Error", f"Anchor with name '{new_name}' already exists!", parent=self.window)
                return
            
            old_name = self.selected_anchor.name
            self.selected_anchor.name = new_name
            self.point_creator.save_anchors_to_config()
            self.refresh_anchor_list()
            messagebox.showinfo("Success", f"Anchor renamed from '{old_name}' to '{new_name}'!", parent=self.window)
    
    def delete_anchor(self):
        if not self.selected_anchor:
            messagebox.showwarning("Warning", "Please select an anchor first!", parent=self.window)
            return
        
        if messagebox.askyesno("Confirm Delete", f"Are you sure you want to delete anchor '{self.selected_anchor.name}'?", parent=self.window):
            self.point_creator.anchors.remove(self.selected_anchor)
            self.point_creator.save_anchors_to_config()
            self.selected_anchor = None
            self.refresh_anchor_list()
            self.point_creator.update_nearest_anchor_display()
            self.point_creator.stop_anchor_follow_mode()
            messagebox.showinfo("Success", "Anchor deleted successfully!", parent=self.window)
    
    def move_anchor(self):
        if not self.selected_anchor:
            messagebox.showwarning("Warning", "Please select an anchor first!", parent=self.window)
            return
        
        move_dialog = tk.Toplevel(self.window)
        move_dialog.title("Move Anchor")
        move_dialog.geometry("300x250")
        move_dialog.configure(bg='#2c3e50')
        move_dialog.transient(self.window)
        move_dialog.grab_set()
        
        ttk.Label(move_dialog, text=f"Moving Anchor: {self.selected_anchor.name}", font=('Arial', 10, 'bold')).pack(pady=10)
        
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
                self.point_creator.update_nearest_anchor_display()
                
            except ValueError:
                messagebox.showerror("Error", "Please enter valid numbers!", parent=move_dialog)
        
        ttk.Button(move_dialog, text="Apply", command=apply_move).pack(pady=15)
    
    def use_anchor(self):
        if not self.selected_anchor:
            messagebox.showwarning("Warning", "Please select an anchor first!", parent=self.window)
            return
        
        self.point_creator.set_point_position(1, self.selected_anchor.coordinates[0], self.selected_anchor.coordinates[1])
        self.point_creator.set_point_position(2, self.selected_anchor.coordinates[2], self.selected_anchor.coordinates[3])
        self.point_creator.current_points = self.selected_anchor.coordinates.copy()
        self.point_creator.update_nearest_anchor_display()
        
        messagebox.showinfo("Success", f"Points moved to anchor '{self.selected_anchor.name}'!", parent=self.window)


class DatasetManager:
    """Менеджер датасетов"""
    def __init__(self, point_creator):
        self.point_creator = point_creator
        self.datasets = {}
        self.current_dataset = None
    
    def load_datasets_list(self):
        # Не загружаем автоматически
        pass
    
    def save_dataset(self, name, data):
        """Сохранить датасет через проводник"""
        filepath = filedialog.asksaveasfilename(
            title=f"Save Dataset '{name}'",
            defaultextension=".json",
            initialfile=f"{name}.json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            parent=self.point_creator.root
        )
        if filepath:
            try:
                with open(filepath, 'w') as f:
                    json.dump(data, f, indent=4)
                self.datasets[name] = data
                return True
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save: {str(e)}")
                return False
        return False
    
    def delete_dataset(self, name):
        if name in self.datasets:
            del self.datasets[name]
    
    def get_dataset_names(self):
        return list(self.datasets.keys())
    
    def get_episodes(self, dataset_name):
        """Вернуть список эпизодов с количеством точек"""
        if dataset_name not in self.datasets:
            return []
        
        data = self.datasets[dataset_name]
        episodes_dict = {}
        
        for x in data["X"]:
            season = int(round(x[0] * 10))
            episode = int(round(x[1] * 1000))
            key = (season, episode)
            if key not in episodes_dict:
                episodes_dict[key] = {
                    "season": season,
                    "episode": episode,
                    "count": 0
                }
            episodes_dict[key]["count"] += 1
        
        result = list(episodes_dict.values())
        return sorted(result, key=lambda x: (x["season"], x["episode"]))
    
    def get_episode_data(self, dataset_name, season, episode):
        """Вернуть данные для конкретного эпизода"""
        if dataset_name not in self.datasets:
            return None
        
        data = self.datasets[dataset_name]
        moments = []
        quartets = []
        
        for i in range(len(data["X"])):
            x = data["X"][i]
            if int(round(x[0] * 10)) == season and int(round(x[1] * 1000)) == episode:
                moments.append(x[2] * 100)
                quartets.append(data["Y"][i])
        
        if moments:
            combined = list(zip(moments, quartets))
            combined.sort(key=lambda x: x[0])
            moments, quartets = zip(*combined)
            moments = list(moments)
            quartets = list(quartets)
        
        return {"moments": moments, "quartets": quartets}

class DatasetManagerWindow:
    """Главное окно менеджера датасетов"""
    def __init__(self, parent, point_creator):
        self.parent = parent
        self.point_creator = point_creator
        self.manager = point_creator.dataset_manager
        self.window = tk.Toplevel(parent)
        self.window.title("Dataset Manager")
        self.window.geometry("500x450")
        self.window.configure(bg='#2c3e50')
        
        try:
            icon_path = point_creator.resource_path("lo.ico")
            if os.path.exists(icon_path):
                self.window.iconbitmap(icon_path)
        except:
            pass
        
        self.window.transient(parent)
        self.window.protocol("WM_DELETE_WINDOW", self.on_close)
        
        self.create_widgets()
        self.refresh_datasets()
    
    def on_close(self):
        self.window.destroy()
    
    def create_widgets(self):
        title_label = ttk.Label(self.window, text="DATASET MANAGER", font=('Arial', 14, 'bold'))
        title_label.pack(pady=10)
        
        list_frame = ttk.LabelFrame(self.window, text="Datasets", padding=10)
        list_frame.pack(fill="both", expand=True, padx=10, pady=5)
        
        self.dataset_list = tk.Listbox(list_frame, height=10, bg='#34495e', fg='white', 
                                       selectbackground='#3498db', font=('Arial', 10))
        self.dataset_list.pack(fill="both", expand=True, pady=5)
        self.dataset_list.bind('<<ListboxSelect>>', self.on_dataset_selected)
        self.dataset_list.bind('<Double-Button-1>', lambda e: self.use_dataset())
        
        btn_frame = ttk.Frame(self.window)
        btn_frame.pack(pady=10)
        
        ttk.Button(btn_frame, text="Use", command=self.use_dataset, width=8).pack(side="left", padx=5)
        ttk.Button(btn_frame, text="Remove", command=self.remove_dataset, width=8).pack(side="left", padx=5)
        ttk.Button(btn_frame, text="Save", command=self.save_dataset, width=8).pack(side="left", padx=5)
        ttk.Button(btn_frame, text="Load", command=self.load_dataset, width=8).pack(side="left", padx=5)
        ttk.Button(btn_frame, text="Show", command=self.show_episodes, width=8).pack(side="left", padx=5)
        ttk.Button(btn_frame, text="Close", command=self.on_close, width=8).pack(side="left", padx=5)
        
        self.selected_dataset = None
    
    def refresh_datasets(self):
        self.dataset_list.delete(0, tk.END)
        for name in self.manager.get_dataset_names():
            self.dataset_list.insert(tk.END, name)
    
    def on_dataset_selected(self, event):
        selection = self.dataset_list.curselection()
        if selection:
            self.selected_dataset = self.dataset_list.get(selection[0])
    
    def use_dataset(self):
        if not self.selected_dataset:
            messagebox.showwarning("Warning", "Please select a dataset first!")
            return
        
        data = self.manager.datasets[self.selected_dataset]
        self.point_creator.data = data.copy()
        messagebox.showinfo("Success", f"Dataset '{self.selected_dataset}' loaded into memory!")
    
    def remove_dataset(self):
        if not self.selected_dataset:
            messagebox.showwarning("Warning", "Please select a dataset first!")
            return
        
        if messagebox.askyesno("Confirm", f"Delete dataset '{self.selected_dataset}' from memory?"):
            self.manager.delete_dataset(self.selected_dataset)
            self.selected_dataset = None
            self.refresh_datasets()
            messagebox.showinfo("Success", "Dataset removed from memory!")
    
    def save_dataset(self):
        if not self.point_creator.data["X"]:
            messagebox.showwarning("Warning", "No data to save!")
            return
        
        name = simpledialog.askstring("Save Dataset", "Enter dataset name:", parent=self.window)
        if name:
            if name in self.manager.datasets:
                if not messagebox.askyesno("Overwrite", f"Dataset '{name}' already exists in memory. Overwrite?"):
                    return
            
            if self.manager.save_dataset(name, self.point_creator.data):
                self.refresh_datasets()
                messagebox.showinfo("Success", f"Dataset '{name}' saved!")
    
    def load_dataset(self):
        filepath = filedialog.askopenfilename(
            title="Load dataset",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            parent=self.window
        )
        
        if filepath:
            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)
                
                name = os.path.basename(filepath)[:-5]
                self.manager.datasets[name] = data
                self.refresh_datasets()
                messagebox.showinfo("Success", f"Dataset '{name}' loaded from:\n{filepath}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load: {str(e)}")
    
    def show_episodes(self):
        if not self.selected_dataset:
            messagebox.showwarning("Warning", "Please select a dataset first!")
            return
        
        episodes = self.manager.get_episodes(self.selected_dataset)
        if not episodes:
            messagebox.showinfo("Info", "No episodes found in this dataset!")
            return
        
        EpisodeListWindow(self.window, self.point_creator, self.manager, 
                         self.selected_dataset, episodes)


class EpisodeListWindow:
    """Окно со списком эпизодов датасета"""
    def __init__(self, parent, point_creator, manager, dataset_name, episodes):
        self.parent = parent
        self.point_creator = point_creator
        self.manager = manager
        self.dataset_name = dataset_name
        self.episodes = episodes
        self.selected_episodes = []
        self.sort_mode = "chronological"
        self.sort_reverse = False
        
        self.window = tk.Toplevel(parent)
        self.window.title(f"Episodes - {dataset_name}")
        self.window.geometry("650x500")
        self.window.configure(bg='#2c3e50')
        
        try:
            icon_path = point_creator.resource_path("lo.ico")
            if os.path.exists(icon_path):
                self.window.iconbitmap(icon_path)
        except:
            pass
        
        self.window.transient(parent)
        self.create_widgets()
        self.refresh_episodes()
        
        # Горячие клавиши
        self.window.bind('<Control-a>', lambda e: self.select_all())
        self.window.bind('<Control-A>', lambda e: self.select_all())
        self.window.bind('<Control-d>', lambda e: self.deselect_all())
        self.window.bind('<Control-D>', lambda e: self.deselect_all())
    
    def create_widgets(self):
        title_label = ttk.Label(self.window, text=f"EPISODES: {self.dataset_name}", 
                               font=('Arial', 12, 'bold'))
        title_label.pack(pady=10)
        
        sort_frame = ttk.Frame(self.window)
        sort_frame.pack(pady=5)
        
        ttk.Button(sort_frame, text="📅 Chronological", 
                  command=lambda: self.set_sort("chronological")).pack(side="left", padx=5)
        ttk.Button(sort_frame, text="📊 Points count", 
                  command=lambda: self.set_sort("points")).pack(side="left", padx=5)
        ttk.Button(sort_frame, text="↕ Reverse", 
                  command=self.toggle_reverse).pack(side="left", padx=5)
        
        list_frame = ttk.LabelFrame(self.window, text="Episodes", padding=10)
        list_frame.pack(fill="both", expand=True, padx=10, pady=5)
        
        list_container = ttk.Frame(list_frame)
        list_container.pack(fill="both", expand=True)
        
        columns = ("Season", "Episode", "Points")
        self.episode_tree = ttk.Treeview(list_container, columns=columns, 
                                         show="headings", height=15,
                                         selectmode="extended")
        
        self.episode_tree.heading("Season", text="Season")
        self.episode_tree.heading("Episode", text="Episode")
        self.episode_tree.heading("Points", text="Points")
        
        self.episode_tree.column("Season", width=250)
        self.episode_tree.column("Episode", width=100)
        self.episode_tree.column("Points", width=80)
        
        scrollbar = ttk.Scrollbar(list_container, orient="vertical", 
                                 command=self.episode_tree.yview)
        self.episode_tree.configure(yscrollcommand=scrollbar.set)
        
        self.episode_tree.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Обработка клика - переключение выделения
        self.episode_tree.bind('<Button-1>', self.on_tree_click)
        self.episode_tree.bind('<<TreeviewSelect>>', self.on_episode_selected)
        self.episode_tree.bind('<Double-Button-1>', lambda e: self.use_episodes())
        
        # Select/Deselect frame
        select_frame = ttk.Frame(list_frame)
        select_frame.pack(pady=5)
        
        ttk.Button(select_frame, text="Select All (Ctrl+A)", command=self.select_all, width=15).pack(side="left", padx=5)
        ttk.Button(select_frame, text="Deselect All (Ctrl+D)", command=self.deselect_all, width=15).pack(side="left", padx=5)
        
        btn_frame = ttk.Frame(self.window)
        btn_frame.pack(pady=10)
        
        ttk.Button(btn_frame, text="Use", command=self.use_episodes, width=8).pack(side="left", padx=5)
        ttk.Button(btn_frame, text="Remove", command=self.remove_episodes, width=8).pack(side="left", padx=5)
        ttk.Button(btn_frame, text="Save as", command=self.save_episodes, width=8).pack(side="left", padx=5)
        ttk.Button(btn_frame, text="Load to", command=self.load_to_dataset, width=8).pack(side="left", padx=5)
        ttk.Button(btn_frame, text="Close", command=self.window.destroy, width=8).pack(side="left", padx=5)
    
    def on_tree_click(self, event):
        """Обработка клика по дереву - переключение выделения"""
        item = self.episode_tree.identify_row(event.y)
        if not item:
            return
        
        if item in self.episode_tree.selection():
            self.episode_tree.selection_remove(item)
        else:
            self.episode_tree.selection_add(item)
        
        return "break"
    
    def set_sort(self, mode):
        self.sort_mode = mode
        self.refresh_episodes()
    
    def toggle_reverse(self):
        self.sort_reverse = not self.sort_reverse
        self.refresh_episodes()
    
    def refresh_episodes(self):
        selected_values = []
        for item in self.episode_tree.selection():
            values = self.episode_tree.item(item)['values']
            if values:
                selected_values.append((values[0], values[1]))
        
        for item in self.episode_tree.get_children():
            self.episode_tree.delete(item)
        
        if self.sort_mode == "chronological":
            sorted_episodes = sorted(self.episodes, key=lambda x: (x["season"], x["episode"]), 
                                    reverse=self.sort_reverse)
        else:
            sorted_episodes = sorted(self.episodes, key=lambda x: x["count"], 
                                    reverse=not self.sort_reverse)
        
        for ep in sorted_episodes:
            season_name = self.point_creator.get_season_name(ep["season"])
            display_season = f"{season_name} ({ep['season']:02d})"
            item_id = self.episode_tree.insert("", "end", values=(
                display_season,
                f"{ep['episode']:03d}",
                f"{ep['count']}"
            ), tags=(str(ep["season"]), str(ep["episode"])))
            
            for sel_season, sel_episode in selected_values:
                if sel_season == display_season and sel_episode == f"{ep['episode']:03d}":
                    self.episode_tree.selection_add(item_id)
                    break
    
    def on_episode_selected(self, event):
        pass
    
    def get_selected_episodes(self):
        selected = []
        for item in self.episode_tree.selection():
            values = self.episode_tree.item(item)['values']
            if not values:
                continue
            season_str = values[0]
            match = re.search(r'\((\d+)\)', season_str)
            if match:
                season = int(match.group(1))
                episode = int(values[1])
                for ep in self.episodes:
                    if ep["season"] == season and ep["episode"] == episode:
                        selected.append(ep)
                        break
        return selected
    
    def select_all(self):
        for item in self.episode_tree.get_children():
            self.episode_tree.selection_add(item)
    
    def deselect_all(self):
        self.episode_tree.selection_remove(*self.episode_tree.selection())
    
    def use_episodes(self):
        selected_episodes = self.get_selected_episodes()
        if not selected_episodes:
            messagebox.showwarning("Warning", "Please select at least one episode!")
            return
        
        selected_data = {"X": [], "Y": []}
        for ep in selected_episodes:
            ep_data = self.manager.get_episode_data(self.dataset_name, ep["season"], ep["episode"])
            if ep_data:
                for i in range(len(ep_data["moments"])):
                    moment = ep_data["moments"][i] / 100
                    quartet = ep_data["quartets"][i]
                    selected_data["X"].append([ep["season"]/10, ep["episode"]/1000, moment])
                    selected_data["Y"].append(quartet)
        
        self.point_creator.data = selected_data
        
        first_ep = selected_episodes[0]
        season_name = self.point_creator.get_season_name(first_ep["season"])
        if season_name in self.point_creator.seasons_list:
            self.point_creator.season_var.set(season_name)
        self.point_creator.episode_var.set(str(first_ep["episode"]))
        
        messagebox.showinfo("Success", f"Loaded {len(selected_data['X'])} points from {len(selected_episodes)} episodes!")
    
    def remove_episodes(self):
        selected_episodes = self.get_selected_episodes()
        if not selected_episodes:
            messagebox.showwarning("Warning", "Please select at least one episode!")
            return
        
        if not messagebox.askyesno("Confirm", f"Remove {len(selected_episodes)} episodes from '{self.dataset_name}'?"):
            return
        
        full_data = self.manager.datasets[self.dataset_name]
        
        to_remove = set()
        for ep in selected_episodes:
            to_remove.add((ep["season"], ep["episode"]))
        
        new_X = []
        new_Y = []
        for i in range(len(full_data["X"])):
            x_data = full_data["X"][i]
            season = int(round(x_data[0] * 10))
            episode = int(round(x_data[1] * 1000))
            if (season, episode) not in to_remove:
                new_X.append(x_data)
                new_Y.append(full_data["Y"][i])
        
        new_data = {"X": new_X, "Y": new_Y}
        self.manager.save_dataset(self.dataset_name, new_data)
        
        self.episodes = self.manager.get_episodes(self.dataset_name)
        self.refresh_episodes()
        messagebox.showinfo("Success", f"Removed {len(selected_episodes)} episodes!")
    
    def save_episodes(self):
        selected_episodes = self.get_selected_episodes()
        if not selected_episodes:
            messagebox.showwarning("Warning", "Please select at least one episode!")
            return
        
        name = simpledialog.askstring("Save Episodes", "Enter new dataset name:", parent=self.window)
        if not name:
            return
        
        if name in self.manager.datasets:
            if not messagebox.askyesno("Overwrite", f"Dataset '{name}' already exists. Overwrite?"):
                return
        
        new_data = {"X": [], "Y": []}
        for ep in selected_episodes:
            ep_data = self.manager.get_episode_data(self.dataset_name, ep["season"], ep["episode"])
            if ep_data:
                for i in range(len(ep_data["moments"])):
                    moment = ep_data["moments"][i] / 100
                    quartet = ep_data["quartets"][i]
                    new_data["X"].append([ep["season"]/10, ep["episode"]/1000, moment])
                    new_data["Y"].append(quartet)
        
        if new_data["X"]:
            # Сохраняем через проводник
            filepath = filedialog.asksaveasfilename(
                title=f"Save Episodes as '{name}'",
                defaultextension=".json",
                initialfile=f"{name}.json",
                filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
                parent=self.window
            )
            if filepath:
                try:
                    with open(filepath, 'w') as f:
                        json.dump(new_data, f, indent=4)
                    self.manager.datasets[name] = new_data
                    messagebox.showinfo("Success", f"Saved {len(new_data['X'])} points as '{name}'!")
                except Exception as e:
                    messagebox.showerror("Error", f"Failed to save: {str(e)}")
    
    def load_to_dataset(self):
        selected_episodes = self.get_selected_episodes()
        if not selected_episodes:
            messagebox.showwarning("Warning", "Please select at least one episode!")
            return
        
        target_name = simpledialog.askstring("Load To", 
            "Enter target dataset name (will be created if doesn't exist):", 
            parent=self.window)
        if not target_name:
            return
        
        source_data = {"X": [], "Y": []}
        for ep in selected_episodes:
            ep_data = self.manager.get_episode_data(self.dataset_name, ep["season"], ep["episode"])
            if ep_data:
                for i in range(len(ep_data["moments"])):
                    moment = ep_data["moments"][i] / 100
                    quartet = ep_data["quartets"][i]
                    source_data["X"].append([ep["season"]/10, ep["episode"]/1000, moment])
                    source_data["Y"].append(quartet)
        
        if not source_data["X"]:
            messagebox.showerror("Error", "No data to load!")
            return
        
        target_data = self.manager.datasets.get(target_name, {"X": [], "Y": []})
        
        source_episodes = set()
        for x in source_data["X"]:
            source_episodes.add((int(round(x[0]*10)), int(round(x[1]*1000))))
        
        target_episodes = set()
        for x in target_data["X"]:
            target_episodes.add((int(round(x[0]*10)), int(round(x[1]*1000))))
        
        conflicts = source_episodes & target_episodes
        
        if conflicts:
            conflict_window = tk.Toplevel(self.window)
            conflict_window.title("Conflict Resolution")
            conflict_window.geometry("500x400")
            conflict_window.configure(bg='#2c3e50')
            conflict_window.transient(self.window)
            conflict_window.grab_set()
            
            ttk.Label(conflict_window, text="Conflict Resolution", font=('Arial', 12, 'bold')).pack(pady=10)
            
            list_frame = ttk.Frame(conflict_window)
            list_frame.pack(fill="both", expand=True, padx=10, pady=5)
            
            conflict_list = tk.Listbox(list_frame, height=10, bg='#34495e', fg='white',
                                      selectbackground='#3498db', font=('Arial', 10))
            conflict_list.pack(fill="both", expand=True)
            
            for season, episode in sorted(conflicts):
                season_name = self.point_creator.get_season_name(season)
                conflict_list.insert(tk.END, f"{season_name} ({season:02d}) - Episode {episode:03d}")
            
            action_var = tk.StringVar(value="keep_source")
            action_frame = ttk.LabelFrame(conflict_window, text="Action for selected", padding=10)
            action_frame.pack(fill="x", padx=10, pady=5)
            
            ttk.Radiobutton(action_frame, text="Keep from source (A)", 
                          variable=action_var, value="keep_source").pack(anchor="w")
            ttk.Radiobutton(action_frame, text="Keep from target (B)", 
                          variable=action_var, value="keep_target").pack(anchor="w")
            ttk.Radiobutton(action_frame, text="Skip", 
                          variable=action_var, value="skip").pack(anchor="w")
            
            apply_all_var = tk.BooleanVar(value=False)
            ttk.Checkbutton(conflict_window, text="Apply to all", 
                          variable=apply_all_var).pack(pady=5)
            
            result = {"action": None, "apply_all": False}
            
            def apply_action():
                result["action"] = action_var.get()
                result["apply_all"] = apply_all_var.get()
                conflict_window.destroy()
            
            ttk.Button(conflict_window, text="Apply", command=apply_action).pack(pady=10)
            
            self.window.wait_window(conflict_window)
            
            if result["action"] is None:
                return
            
            if result["apply_all"]:
                if result["action"] == "keep_target" or result["action"] == "skip":
                    source_data["X"] = [x for x in source_data["X"] 
                                      if (int(round(x[0]*10)), int(round(x[1]*1000))) not in conflicts]
                    source_data["Y"] = source_data["Y"][:len(source_data["X"])]
            else:
                if result["action"] == "keep_target" or result["action"] == "skip":
                    source_data["X"] = [x for x in source_data["X"] 
                                      if (int(round(x[0]*10)), int(round(x[1]*1000))) not in conflicts]
                    source_data["Y"] = source_data["Y"][:len(source_data["X"])]
        
        combined_data = {"X": target_data["X"] + source_data["X"], 
                        "Y": target_data["Y"] + source_data["Y"]}
        
        self.manager.datasets[target_name] = combined_data
        messagebox.showinfo("Success", 
            f"Loaded {len(source_data['X'])} points into '{target_name}'!")


class GraphWindow:
    """Окно для отображения графика Match vs Moment"""
    def __init__(self, parent, point_creator):
        self.parent = parent
        self.point_creator = point_creator
        self.fig = None
        self.canvas = None
        self.anchor_vars = {}  # имя якоря -> BooleanVar
        
        self.window = tk.Toplevel(parent)
        self.window.title("Match Graph")
        self.window.geometry("1100x700")
        self.window.configure(bg='#2c3e50')
        
        try:
            icon_path = point_creator.resource_path("lo.ico")
            if os.path.exists(icon_path):
                self.window.iconbitmap(icon_path)
        except:
            pass
        
        self.window.transient(parent)
        self.create_widgets()
        self.plot_graph()
    
    def create_widgets(self):
        # Основной фрейм
        main_frame = ttk.Frame(self.window)
        main_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Левая панель - список якорей
        left_panel = ttk.Frame(main_frame, width=200)
        left_panel.pack(side="left", fill="y", padx=(0, 10))
        
        # Правая панель - график
        right_panel = ttk.Frame(main_frame)
        right_panel.pack(side="right", fill="both", expand=True)
        
        # ========== ЛЕВАЯ ПАНЕЛЬ ==========
        title_label = ttk.Label(left_panel, text="ANCHORS", font=('Arial', 12, 'bold'))
        title_label.pack(pady=5)
        
        # Фрейм со списком якорей
        anchors_frame = ttk.LabelFrame(left_panel, text="Show/Hide", padding=5)
        anchors_frame.pack(fill="both", expand=True, pady=5)
        
        # Canvas для скролла
        canvas_container = tk.Canvas(anchors_frame, bg='#34495e', highlightthickness=0)
        scrollbar = ttk.Scrollbar(anchors_frame, orient="vertical", command=canvas_container.yview)
        scrollable_frame = ttk.Frame(canvas_container)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas_container.configure(scrollregion=canvas_container.bbox("all"))
        )
        
        canvas_container.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas_container.configure(yscrollcommand=scrollbar.set)
        
        canvas_container.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Чекбоксы для якорей
        self.anchor_checkboxes = []
        for anchor in self.point_creator.anchors:
            var = tk.BooleanVar(value=True)
            self.anchor_vars[anchor.name] = var
            cb = ttk.Checkbutton(scrollable_frame, text=anchor.name, variable=var,
                                command=self.update_graph)
            cb.pack(anchor="w", pady=2, padx=5)
            self.anchor_checkboxes.append(cb)
        
        # Кнопки управления - ВСЕ КНОПКИ СЛЕВА ВНИЗУ
        btn_frame = ttk.Frame(left_panel)
        btn_frame.pack(fill="x", pady=10)
        
        ttk.Button(btn_frame, text="Select All", command=self.select_all_anchors, width=10).pack(side="left", padx=2)
        ttk.Button(btn_frame, text="Deselect All", command=self.deselect_all_anchors, width=10).pack(side="left", padx=2)
        
        # Разделитель
        ttk.Separator(btn_frame, orient='horizontal').pack(fill="x", pady=5)
        
        ttk.Button(btn_frame, text="💾 Save PNG", command=self.export_graph, width=10).pack(side="left", padx=2)
        ttk.Button(btn_frame, text="Close", command=self.window.destroy, width=10).pack(side="left", padx=2)
        
        # ========== ПРАВАЯ ПАНЕЛЬ - ГРАФИК ==========
        self.graph_frame = ttk.Frame(right_panel)
        self.graph_frame.pack(fill="both", expand=True)
    
    def select_all_anchors(self):
        for var in self.anchor_vars.values():
            var.set(True)
        self.update_graph()
    
    def deselect_all_anchors(self):
        for var in self.anchor_vars.values():
            var.set(False)
        self.update_graph()
    
    def get_hsv_colors(self, n):
        """Получить n цветов, равноудаленных по тону HSV"""
        colors = []
        for i in range(n):
            hue = i / n
            r, g, b = self.hsv_to_rgb(hue, 0.9, 0.9)
            colors.append((r/255, g/255, b/255))
        return colors
    
    def hsv_to_rgb(self, h, s, v):
        """Конвертировать HSV в RGB"""
        if s == 0.0:
            return (v*255, v*255, v*255)
        i = int(h * 6.0)
        f = (h * 6.0) - i
        p = v * (1.0 - s)
        q = v * (1.0 - s * f)
        t = v * (1.0 - s * (1.0 - f))
        i = i % 6
        if i == 0:
            return (v*255, t*255, p*255)
        elif i == 1:
            return (q*255, v*255, p*255)
        elif i == 2:
            return (p*255, v*255, t*255)
        elif i == 3:
            return (p*255, q*255, v*255)
        elif i == 4:
            return (t*255, p*255, v*255)
        else:
            return (v*255, p*255, q*255)
    
    def update_graph(self):
        """Обновить график при изменении чекбоксов"""
        if self.fig:
            plt.close(self.fig)
        self.plot_graph()
    
    def plot_graph(self):
        """Построить график"""
        episode_data = self.point_creator.get_episode_data()
        
        if not episode_data["moments"]:
            messagebox.showerror("Error", "No data for current episode!")
            self.window.destroy()
            return
        
        moments = episode_data["moments"]
        quartets = episode_data["quartets"]
        anchors = self.point_creator.anchors
        
        if not anchors:
            messagebox.showerror("Error", "No anchors available!")
            self.window.destroy()
            return
        
        # Создаем фигуру
        self.fig, ax = plt.subplots(figsize=(10, 7), dpi=100)
        self.fig.patch.set_facecolor('#2c3e50')
        ax.set_facecolor('#34495e')
        
        # Получаем только активные якоря
        active_anchors = [a for a in anchors if self.anchor_vars.get(a.name, tk.BooleanVar(value=True)).get()]
        
        if not active_anchors:
            ax.text(0.5, 0.5, "No anchors selected", 
                   ha='center', va='center', color='white', fontsize=16)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_xticks([])
            ax.set_yticks([])
        else:
            colors = self.get_hsv_colors(len(active_anchors))
            
            # Для каждого активного якоря вычисляем Match
            for idx, anchor in enumerate(active_anchors):
                match_values = []
                anchor_pos = anchor.coordinates
                
                for quartet in quartets:
                    distance = self.point_creator.calculate_distance(quartet, anchor_pos)
                    max_distance = self.point_creator.calculate_distance([0,0,0,0], [1,1,1,1])
                    match = (1 - (distance / max_distance)) * 100
                    match_values.append(match)
                
                color = colors[idx]
                ax.plot(moments, match_values, 
                       label=anchor.name, 
                       color=color, 
                       linewidth=2.5,
                       alpha=0.9)
                
                ax.scatter(moments, match_values, 
                          color=color, 
                          s=30, 
                          alpha=0.7,
                          zorder=5)
            
            # Настройка графика
            ax.set_xlabel('Moment (%)', fontsize=14, color='white', fontweight='bold')
            ax.set_ylabel('Match (%)', fontsize=14, color='white', fontweight='bold')
            
            season_name = self.point_creator.get_season_name(
                self.point_creator.seasons_list.index(self.point_creator.season_var.get()) + 1
            )
            episode_num = self.point_creator.episode_var.get()
            ax.set_title(f'{season_name} - Episode {episode_num}', 
                        fontsize=16, color='white', pad=20, fontweight='bold')
            
            ax.grid(True, alpha=0.3, color='gray', linestyle='--')
            ax.set_xlim(0, 100)
            ax.set_ylim(0, 100)
            
            # Легенда
            ax.legend(loc='best', facecolor='#2c3e50', edgecolor='white', 
                     labelcolor='white', fontsize=11)
            
            # Цвета осей и тиков
            ax.tick_params(colors='white', labelsize=11)
            for spine in ax.spines.values():
                spine.set_color('white')
                spine.set_linewidth(1.5)
        
        # Очищаем предыдущий график
        for widget in self.graph_frame.winfo_children():
            widget.destroy()
        
        # Встраиваем новый график
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.graph_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
    
    def export_graph(self):
        """Экспортировать график в PNG"""
        if not self.fig:
            messagebox.showerror("Error", "No graph to export!")
            return
        
        season_name = self.point_creator.get_season_name(
            self.point_creator.seasons_list.index(self.point_creator.season_var.get()) + 1
        )
        episode_num = self.point_creator.episode_var.get()
        default_name = f"Match_Graph_{season_name}_Ep{episode_num}.png"
        
        filepath = filedialog.asksaveasfilename(
            title="Export Graph",
            defaultextension=".png",
            initialfile=default_name,
            filetypes=[("PNG files", "*.png"), ("All files", "*.*")],
            parent=self.window
        )
        
        if filepath:
            try:
                self.fig.savefig(filepath, dpi=300, bbox_inches='tight', 
                                facecolor='#2c3e50', edgecolor='none')
                messagebox.showinfo("Success", f"Graph exported to:\n{filepath}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to export: {str(e)}")


class PointCreator:
    def __init__(self, root):
        self.root = root
        self.root.title("Point Coordinates Creator")
        
        self.CONFIG_FILE = "config_pointer.ini"
        self.config = configparser.ConfigParser()
        self.load_config()
        
        self.anchors = []
        self.load_anchors_from_config()
        
        self.dataset_manager = DatasetManager(self)
        
        self.is_episode_playing = False
        self.episode_animation_id = None
        
        self.follow_mode_active = False
        self.follow_anchor = None
        self.follow_start_points = None
        self.follow_slider = None
        self.follow_label = None
        self.follow_frame = None
        self.follow_percentage = 0
        
        try:
            icon_path = self.resource_path("lo.ico")
            if os.path.exists(icon_path):
                self.root.iconbitmap(icon_path)
        except:
            pass
        
        self.is_fullscreen = False
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        window_width = int(screen_width * 0.95)
        window_height = int(screen_height * 0.8)
        x_pos = (screen_width - window_width) // 2
        y_pos = (screen_height - window_height) // 2
        self.root.geometry(f"{window_width}x{window_height}+{x_pos}+{y_pos}")
        
        self.setup_styles()
        self.seasons_list = self.load_seasons()
        
        self.season_var = tk.StringVar(value=self.seasons_list[0] if self.seasons_list else "New Generation")
        self.episode_var = tk.StringVar(value="1")
        self.moment_var = tk.DoubleVar(value=50)
        
        self.hours_var = tk.StringVar(value="00")
        self.minutes_var = tk.StringVar(value="10")
        self.seconds_var = tk.StringVar(value="00")
        self.length_hours_var = tk.StringVar(value="00")
        self.length_minutes_var = tk.StringVar(value="20")
        self.length_seconds_var = tk.StringVar(value="00")
        
        self.is_playing = False
        self.animation_start_time = 0
        self.start_moment = 0
        self.episode_data = {"moments": [], "quartets": []}
        self.current_points = [0.5, 0.5, 0.5, 0.5]
        
        self.video_hours_var = tk.StringVar(value="0")
        self.video_minutes_var = tk.StringVar(value="2")
        self.video_seconds_var = tk.StringVar(value="0")
        self.video_path_var = tk.StringVar(value="")
        
        self.loaded_capsule = None
        self.data = {"X": [], "Y": []}
        
        self.nearest_anchor_label = None
        self.play_episode_button = None
        
        self.create_widgets()
        self.load_background_images()
        
        if len(sys.argv) > 1 and sys.argv[1].endswith('.lvp'):
            self.load_capsule(sys.argv[1])
    
    def get_season_name(self, season_index):
        if 1 <= season_index <= len(self.seasons_list):
            return self.seasons_list[season_index - 1]
        return f"Season {season_index:02d}"
    
    def load_anchors_from_config(self):
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
        
        if not self.anchors:
            default_anchors = [
                Anchor("Idle", [0.5, 0.5, 0.5, 0.5])
            ]
            self.anchors = default_anchors
            self.save_anchors_to_config()
    
    def save_anchors_to_config(self):
        if not self.config.has_section('Anchors'):
            self.config.add_section('Anchors')
        
        for key in self.config.options('Anchors'):
            self.config.remove_option('Anchors', key)
        
        for i, anchor in enumerate(self.anchors):
            self.config.set('Anchors', f'anchor_{i}', json.dumps(anchor.to_dict()))
        
        self.save_config()
    
    def add_anchor_from_current_points(self):
        name = simpledialog.askstring("Add Anchor", "Enter name for this anchor:", parent=self.root)
        if not name:
            return
        
        if any(anchor.name == name for anchor in self.anchors):
            if not messagebox.askyesno("Warning", f"Anchor '{name}' already exists. Overwrite?"):
                return
            self.anchors = [a for a in self.anchors if a.name != name]
        
        new_anchor = Anchor(name, self.current_points)
        self.anchors.append(new_anchor)
        self.save_anchors_to_config()
        
        messagebox.showinfo("Success", f"Anchor '{name}' added successfully!")
        self.update_nearest_anchor_display()
    
    def calculate_distance(self, points1, points2):
        return sum((p1 - p2) ** 2 for p1, p2 in zip(points1, points2)) ** 0.5
    
    def find_nearest_anchor(self):
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
        if not self.nearest_anchor_label:
            return
        
        if self.follow_mode_active and self.follow_anchor:
            self.nearest_anchor_label.config(
                text=f"📍 Following: {self.follow_anchor.name} ({self.follow_percentage:.1f}%)",
                foreground="#FFD700"
            )
            return
        
        nearest_anchor, distance = self.find_nearest_anchor()
        
        if nearest_anchor:
            max_distance = self.calculate_distance([0,0,0,0], [1,1,1,1])
            percentage = (1 - (distance / max_distance)) * 100
            
            self.nearest_anchor_label.config(
                text=f"📍 Nearest Anchor: {nearest_anchor.name} (Match: {percentage:.1f}%)",
                foreground="#40E0D0"
            )
        else:
            self.nearest_anchor_label.config(text="📍 No anchors available", foreground="#ecf0f1")
    
    def resource_path(self, relative_path):
        try:
            base_path = sys._MEIPASS
        except Exception:
            base_path = os.path.abspath("../data/")
        return os.path.join(base_path, relative_path)
    
    def load_config(self):
        if os.path.exists(self.CONFIG_FILE):
            self.config.read(self.CONFIG_FILE)
        else:
            self.config['Settings'] = {}
            self.config['Paths'] = {}
            self.save_config()
    
    def save_config(self):
        with open(self.CONFIG_FILE, 'w') as configfile:
            self.config.write(configfile)
    
    def setup_styles(self):
        self.style = ttk.Style()
        self.style.theme_use('clam')
        
        BG_COLOR = "#2c3e50"
        FRAME_BG = "#34495e"
        BUTTON_BG = "#3498db"
        BUTTON_FG = "white"
        LABEL_BG = BG_COLOR
        LABEL_FG = "#ecf0f1"
        ENTRY_BG = "white"
        ENTRY_FG = "black"
        PROGRESS_BG = "#40E0D0"
        
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
        seasons_file = self.resource_path('seasons.txt')
        seasons = []
        try:
            if os.path.exists(seasons_file):
                with open(seasons_file, 'r', encoding='utf-8') as f:
                    seasons = [line.strip() for line in f if line.strip()]
            else:
                seasons = [
                    "New Generation",
                    "Game of God",
                    "Perfect World",
                    "Voice of Time",
                    "Thirteen Lights",
                    "Final Reality",
                    "Heart of the Universe",
                    "Point of No Return",
                    "Workshop [47]",
                    "???"
                ]
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load seasons list: {str(e)}")
            seasons = ["New Generation"]
        
        return seasons
    
    def load_background_images(self):
        try:
            emo_image = Image.open(self.resource_path("EmoPlain.png"))
            emo_image = emo_image.resize((300, 300), Image.Resampling.LANCZOS)
            self.emo_bg = ImageTk.PhotoImage(emo_image)
            
            plot_image = Image.open(self.resource_path("PlotPlain.png"))
            plot_image = plot_image.resize((300, 300), Image.Resampling.LANCZOS)
            self.plot_bg = ImageTk.PhotoImage(plot_image)
            
            self.canvas1.create_image(0, 0, anchor="nw", image=self.emo_bg)
            self.canvas2.create_image(0, 0, anchor="nw", image=self.plot_bg)
            
            self.canvas1.tag_raise(self.point1)
            self.canvas2.tag_raise(self.point2)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load background images: {str(e)}")
    
    def create_widgets(self):
        main_paned = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main_paned.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        left_frame = ttk.Frame(main_paned, width=500)
        main_paned.add(left_frame, weight=25)
        
        center_frame = ttk.Frame(main_paned)
        main_paned.add(center_frame, weight=150)
        
        right_frame = ttk.Frame(main_paned, width=50)
        main_paned.add(right_frame, weight=25)
        
        # ==================== LEFT FRAME - TIMELINE ====================
        timeline_title = ttk.Label(left_frame, text="TIMELINE", font=('Arial', 12, 'bold'))
        timeline_title.pack(pady=10)
        
        timeline_frame = ttk.LabelFrame(left_frame, text="Timeline", padding=10)
        timeline_frame.pack(fill="both", expand=True, padx=5, pady=5)
        
        ttk.Label(timeline_frame, text="Season:").pack(anchor="w", padx=5, pady=5)
        season_combo = ttk.Combobox(timeline_frame, textvariable=self.season_var, 
                                   values=self.seasons_list, state="readonly")
        season_combo.pack(fill="x", padx=5, pady=5)
        
        ttk.Label(timeline_frame, text="Episode:").pack(anchor="w", padx=5, pady=5)
        ttk.Entry(timeline_frame, textvariable=self.episode_var).pack(fill="x", padx=5, pady=5)
        
        ttk.Label(timeline_frame, text="Timecode (HH:MM:SS):").pack(anchor="w", padx=5, pady=5)
        
        timecode_frame = ttk.Frame(timeline_frame)
        timecode_frame.pack(fill="x", padx=5, pady=5)
        
        ttk.Entry(timecode_frame, textvariable=self.hours_var, width=3).pack(side="left")
        ttk.Label(timecode_frame, text=":").pack(side="left")
        ttk.Entry(timecode_frame, textvariable=self.minutes_var, width=3).pack(side="left")
        ttk.Label(timecode_frame, text=":").pack(side="left")
        ttk.Entry(timecode_frame, textvariable=self.seconds_var, width=3).pack(side="left")
        
        ttk.Label(timeline_frame, text="Episode Length (HH:MM:SS):").pack(anchor="w", padx=5, pady=5)
        
        length_frame = ttk.Frame(timeline_frame)
        length_frame.pack(fill="x", padx=5, pady=5)
        
        ttk.Entry(length_frame, textvariable=self.length_hours_var, width=3).pack(side="left")
        ttk.Label(length_frame, text=":").pack(side="left")
        ttk.Entry(length_frame, textvariable=self.length_minutes_var, width=3).pack(side="left")
        ttk.Label(length_frame, text=":").pack(side="left")
        ttk.Entry(length_frame, textvariable=self.length_seconds_var, width=3).pack(side="left")
        
        ttk.Label(timeline_frame, text="Timecode (%):").pack(anchor="w", padx=5, pady=5)
        self.moment_scale = ttk.Scale(timeline_frame, from_=0, to=100, variable=self.moment_var, 
                                     orient="horizontal")
        self.moment_scale.pack(fill="x", padx=5, pady=5)
        self.moment_label = ttk.Label(timeline_frame, text="50.0%")
        self.moment_label.pack(pady=5)
        
        self.play_episode_button = ttk.Button(timeline_frame, text="▶ Play Episode", command=self.toggle_episode_playback, width=20)
        self.play_episode_button.pack(pady=10)
        
        self.moment_scale.configure(command=self.update_timestamp_from_scale)
        
        for var in [self.hours_var, self.minutes_var, self.seconds_var]:
            var.trace_add('write', lambda *args: self.update_timecode_from_entries())
        
        for var in [self.length_hours_var, self.length_minutes_var, self.length_seconds_var]:
            var.trace_add('write', lambda *args: self.update_length_from_entries())
        
        # ==================== CENTER FRAME - POINTS ====================
        points_title = ttk.Label(center_frame, text="DESCRIPTION", font=('Arial', 12, 'bold'))
        points_title.pack(pady=10)
        
        points_frame = ttk.Frame(center_frame)
        points_frame.pack(fill="both", expand=True, pady=5)
        
        point1_frame = ttk.LabelFrame(points_frame, text="Emotion", padding=10)
        point1_frame.pack(side="left", fill="both", expand=True, padx=10)
        
        self.canvas1 = tk.Canvas(point1_frame, width=300, height=300, bg="white", highlightthickness=1, highlightbackground="#34495e")
        self.canvas1.pack(pady=5)
        self.point1 = self.canvas1.create_oval(145, 145, 155, 155, fill="red", outline="red")
        
        coord1_frame = ttk.Frame(point1_frame)
        coord1_frame.pack(pady=5)
        
        ttk.Label(coord1_frame, text="X:").pack(side="left")
        self.x1_var = tk.StringVar(value="0.500")
        ttk.Label(coord1_frame, textvariable=self.x1_var, width=6).pack(side="left", padx=(0, 10))
        
        ttk.Label(coord1_frame, text="Y:").pack(side="left")
        self.y1_var = tk.StringVar(value="0.500")
        ttk.Label(coord1_frame, textvariable=self.y1_var, width=6).pack(side="left")
        
        point2_frame = ttk.LabelFrame(points_frame, text="Plot", padding=10)
        point2_frame.pack(side="right", fill="both", expand=True, padx=10)
        
        self.canvas2 = tk.Canvas(point2_frame, width=300, height=300, bg="white", highlightthickness=1, highlightbackground="#34495e")
        self.canvas2.pack(pady=5)
        self.point2 = self.canvas2.create_oval(145, 145, 155, 155, fill="blue", outline="blue")
        
        coord2_frame = ttk.Frame(point2_frame)
        coord2_frame.pack(pady=5)
        
        ttk.Label(coord2_frame, text="X:").pack(side="left")
        self.x2_var = tk.StringVar(value="0.500")
        ttk.Label(coord2_frame, textvariable=self.x2_var, width=6).pack(side="left", padx=(0, 10))
        
        ttk.Label(coord2_frame, text="Y:").pack(side="left")
        self.y2_var = tk.StringVar(value="0.500")
        ttk.Label(coord2_frame, textvariable=self.y2_var, width=6).pack(side="left")
        
        buttons_bottom_frame = ttk.Frame(center_frame)
        buttons_bottom_frame.pack(pady=10)
        
        add_button = ttk.Button(buttons_bottom_frame, text="ADD", command=self.add_current_data, width=10)
        add_button.pack(side="left", padx=5)
        
        add_anchor_button = ttk.Button(buttons_bottom_frame, text="Add Anchor", command=self.add_anchor_from_current_points, width=12)
        add_anchor_button.pack(side="left", padx=5)
        
        anchor_menu_button = ttk.Button(buttons_bottom_frame, text="Anchor Menu", command=self.open_anchor_menu, width=12)
        anchor_menu_button.pack(side="left", padx=5)
        
        self.nearest_anchor_label = ttk.Label(center_frame, text="📍 Nearest Anchor: None", font=('Arial', 10, 'bold'))
        self.nearest_anchor_label.pack(pady=5)
        self.update_nearest_anchor_display()
        
        # ==================== ANCHOR FOLLOW FRAME ====================
        self.follow_frame = ttk.LabelFrame(center_frame, text="Anchor Follow", padding=10)
        self.follow_frame.pack(fill="x", padx=10, pady=5)
        
        follow_controls = ttk.Frame(self.follow_frame)
        follow_controls.pack(fill="x", pady=5)
        
        self.follow_label = ttk.Label(follow_controls, text="0%", width=8)
        self.follow_label.pack(side="left", padx=5)
        
        self.follow_slider = ttk.Scale(follow_controls, from_=0, to=100, orient="horizontal")
        self.follow_slider.pack(side="left", fill="x", expand=True, padx=5)
        self.follow_slider.set(0)
        
        stop_follow_btn = ttk.Button(follow_controls, text="✕ Stop Follow", command=self.stop_anchor_follow_mode, width=12)
        stop_follow_btn.pack(side="right", padx=5)
        
        self.follow_frame.pack_forget()
        
        self.follow_slider.configure(command=self.update_follow_position)
        
        self.canvas1.bind("<B1-Motion>", lambda e: self.move_point(e, 1))
        self.canvas2.bind("<B1-Motion>", lambda e: self.move_point(e, 2))
        
        # ==================== RIGHT FRAME - CONTROLS ====================
        controls_title = ttk.Label(right_frame, text="Controls", font=('Arial', 12, 'bold'))
        controls_title.pack(pady=10)
        
        controls_frame = ttk.LabelFrame(right_frame, text="Controls", padding=10)
        controls_frame.pack(fill="both", expand=True, padx=5, pady=5)
        
        ttk.Button(controls_frame, text="📊 Data", command=self.open_dataset_manager, width=15).pack(pady=5, fill="x")
        ttk.Button(controls_frame, text="📈 Create Graph", command=self.create_graph, width=15).pack(pady=5, fill="x")
        ttk.Button(controls_frame, text="Create MP4", command=self.open_mp4_dialog, width=15).pack(pady=5, fill="x")
        ttk.Button(controls_frame, text="Load Capsule", command=self.load_capsule_dialog, width=15).pack(pady=5, fill="x")
        ttk.Button(controls_frame, text="Fullscreen", command=self.toggle_fullscreen, width=15).pack(pady=5, fill="x")
    
    def open_dataset_manager(self):
        DatasetManagerWindow(self.root, self)
    
    def create_graph(self):
        """Создать график Match vs Moment"""
        GraphWindow(self.root, self)
    
    def start_anchor_follow_mode(self, anchor):
        if not anchor:
            return
        
        self.follow_start_points = self.current_points.copy()
        self.follow_anchor = anchor
        self.follow_mode_active = True
        self.follow_percentage = 0
        
        self.follow_frame.pack(fill="x", padx=10, pady=5, after=self.nearest_anchor_label)
        self.follow_slider.set(0)
        self.follow_label.config(text="0%")
        
        self.update_follow_info()
        self.update_nearest_anchor_display()
    
    def stop_anchor_follow_mode(self):
        self.follow_mode_active = False
        self.follow_anchor = None
        self.follow_start_points = None
        self.follow_percentage = 0
        self.follow_frame.pack_forget()
        self.update_nearest_anchor_display()
    
    def update_follow_position(self, value):
        if not self.follow_mode_active or self.follow_anchor is None or self.follow_start_points is None:
            return
        
        progress = float(value) / 100.0
        self.follow_percentage = float(value)
        
        self.follow_label.config(text=f"{float(value):.0f}%")
        
        anchor_pos = self.follow_anchor.coordinates
        start_pos = self.follow_start_points
        
        new_points = []
        for i in range(4):
            new_points.append(start_pos[i] + (anchor_pos[i] - start_pos[i]) * progress)
        
        self.set_point_position(1, new_points[0], new_points[1])
        self.set_point_position(2, new_points[2], new_points[3])
        
        self.current_points = new_points.copy()
        self.update_nearest_anchor_display()
    
    def update_follow_info(self):
        if self.follow_anchor:
            self.follow_frame.config(text=f"Anchor Follow: {self.follow_anchor.name}")
    
    def open_anchor_menu(self):
        AnchorMenu(self.root, self)
    
    def toggle_episode_playback(self):
        if self.is_episode_playing:
            self.is_episode_playing = False
            if self.episode_animation_id:
                self.root.after_cancel(self.episode_animation_id)
                self.episode_animation_id = None
            self.play_episode_button.config(text="▶ Play Episode")
            self.moment_scale.config(state="normal")
        else:
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
        if not self.is_episode_playing:
            return
        
        current_moment = self.moment_var.get()
        increment_per_frame = 0.5
        
        new_moment = current_moment + increment_per_frame
        
        if new_moment >= 100.0:
            new_moment = 100.0
            self.moment_var.set(new_moment)
            self.moment_label.config(text=f"{new_moment:.1f}%")
            self.update_timestamp_from_scale(new_moment)
            self.update_points_from_scale()
            self.is_episode_playing = False
            self.play_episode_button.config(text="▶ Play Episode")
            self.moment_scale.config(state="normal")
            return
        
        self.moment_var.set(new_moment)
        self.moment_label.config(text=f"{new_moment:.1f}%")
        self.update_timestamp_from_scale(new_moment)
        self.update_points_from_scale()
        
        self.episode_animation_id = self.root.after(50, self.play_episode_animation)
    
    def add_current_data(self):
        coords1 = self.canvas1.coords(self.point1)
        coords2 = self.canvas2.coords(self.point2)
        
        x1_norm = (coords1[0] + 5) / 300
        y1_norm = 1 - ((coords1[1] + 5) / 300)
        x2_norm = (coords2[0] + 5) / 300
        y2_norm = 1 - ((coords2[1] + 5) / 300)
        
        season_index = self.seasons_list.index(self.season_var.get()) + 1
        season_norm = season_index / 10
        
        try:
            episode_norm = float(self.episode_var.get()) / 1000
            moment_norm = self.moment_var.get() / 100
            
            self.data["X"].append([season_norm, episode_norm, moment_norm])
            self.data["Y"].append([x1_norm, y1_norm, x2_norm, y2_norm])
            
            messagebox.showinfo("Success", "Data successfully added to dataset!")
        except ValueError:
            messagebox.showerror("Error", "Invalid episode number!")
    
    def update_timestamp_from_scale(self, val):
        percentage = float(val)
        self.moment_label.config(text=f"{percentage:.1f}%")
        
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
        
        self.update_points_from_scale()
    
    def update_points_from_scale(self):
        if self.loaded_capsule:
            season_index = self.seasons_list.index(self.season_var.get()) + 1
            if (self.loaded_capsule["season"] == season_index and 
                self.loaded_capsule["episode"] == int(self.episode_var.get())):
                self.episode_data = {
                    "moments": self.loaded_capsule["moments"],
                    "quartets": self.loaded_capsule["quartets"]
                }
            else:
                return
        else:
            self.episode_data = self.get_episode_data()
        
        if not self.episode_data["moments"]:
            return
        
        current_time = self.moment_var.get()
        target_points = self.get_interpolated_target(current_time)
        
        if target_points:
            self.set_point_position(1, target_points[0], target_points[1])
            self.set_point_position(2, target_points[2], target_points[3])
            self.current_points = target_points.copy()
            self.update_nearest_anchor_display()
    
    def update_timecode_from_entries(self):
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
        self.update_timecode_from_entries()
    
    def toggle_fullscreen(self):
        self.is_fullscreen = not self.is_fullscreen
        self.root.attributes("-fullscreen", self.is_fullscreen)
    
    def move_point(self, event, point_num):
        if self.follow_mode_active:
            self.stop_anchor_follow_mode()
        
        canvas = self.canvas1 if point_num == 1 else self.canvas2
        point = self.point1 if point_num == 1 else self.point2
        
        x = max(0, min(event.x, 300))
        y = max(0, min(event.y, 300))
        
        canvas.coords(point, x-5, y-5, x+5, y+5)
        
        x_norm = x / 300
        y_norm = 1 - (y / 300)
        
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
        
        self.update_nearest_anchor_display()
    
    def get_episode_data(self):
        season_index = self.seasons_list.index(self.season_var.get()) + 1
        episode_num = int(self.episode_var.get())
        
        moments = []
        quartets = []
        
        for i in range(len(self.data["X"])):
            x_data = self.data["X"][i]
            y_data = self.data["Y"][i]
            
            if int(round(x_data[0] * 10)) == season_index and int(round(x_data[1] * 1000)) == episode_num:
                moments.append(x_data[2] * 100)
                quartets.append(y_data)
        
        if moments:
            combined = list(zip(moments, quartets))
            combined.sort(key=lambda x: x[0])
            moments, quartets = zip(*combined)
            moments = list(moments)
            quartets = list(quartets)
        
        return {"moments": moments, "quartets": quartets}
    
    def get_interpolated_target(self, current_time):
        moments = self.episode_data["moments"]
        quartets = self.episode_data["quartets"]
        
        if not moments:
            return None
        
        if current_time <= moments[0]:
            return quartets[0]
        
        if current_time >= moments[-1]:
            return quartets[-1]
        
        idx = bisect.bisect_right(moments, current_time)
        
        prev_time = moments[idx - 1]
        next_time = moments[idx]
        prev_points = quartets[idx - 1]
        next_points = quartets[idx]
        
        t = (current_time - prev_time) / (next_time - prev_time)
        
        interpolated_points = []
        for i in range(4):
            interpolated_value = prev_points[i] + (next_points[i] - prev_points[i]) * t
            interpolated_points.append(interpolated_value)
        
        return interpolated_points
    
    def set_point_position(self, point_num, x_norm, y_norm):
        canvas = self.canvas1 if point_num == 1 else self.canvas2
        point = self.point1 if point_num == 1 else self.point2
        
        x = x_norm * 300
        y = (1 - y_norm) * 300
        
        canvas.coords(point, x-5, y-5, x+5, y+5)
        
        if point_num == 1:
            self.x1_var.set(f"{x_norm:.3f}")
            self.y1_var.set(f"{y_norm:.3f}")
        else:
            self.x2_var.set(f"{x_norm:.3f}")
            self.y2_var.set(f"{y_norm:.3f}")
    
    def load_capsule_dialog(self):
        filepath = filedialog.askopenfilename(
            title="Select capsule file",
            filetypes=[("LVP files", "*.lvp"), ("All files", "*.*")]
        )
        
        if filepath:
            self.load_capsule(filepath)
    
    def load_capsule(self, filepath):
        try:
            with open(filepath, 'r') as f:
                capsule_data = json.load(f)
            
            if "season" in capsule_data and "episode" in capsule_data and "moments" in capsule_data and "quartets" in capsule_data:
                self.loaded_capsule = capsule_data
                
                season_index = capsule_data["season"] - 1
                if 0 <= season_index < len(self.seasons_list):
                    self.season_var.set(self.seasons_list[season_index])
                self.episode_var.set(str(capsule_data["episode"]))
                
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
        mp4_dialog = tk.Toplevel(self.root)
        mp4_dialog.title("Create MP4 Video")
        mp4_dialog.geometry("400x300")
        mp4_dialog.resizable(False, False)
        mp4_dialog.configure(bg='#2c3e50')
        
        try:
            icon_path = self.resource_path("lo.ico")
            if os.path.exists(icon_path):
                mp4_dialog.iconbitmap(icon_path)
        except:
            pass
        
        ttk.Label(mp4_dialog, text="Video Duration:", font=("Arial", 12)).pack(pady=10)
        
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
        
        path_frame = ttk.Frame(mp4_dialog)
        path_frame.pack(pady=10, fill="x", padx=20)
        
        ttk.Label(path_frame, text="Save Path:").pack(anchor="w")
        
        path_entry_frame = ttk.Frame(path_frame)
        path_entry_frame.pack(fill="x", pady=5)
        
        ttk.Entry(path_entry_frame, textvariable=self.video_path_var, width=30).pack(side="left", fill="x", expand=True, padx=(0, 5))
        ttk.Button(path_entry_frame, text="Browse", command=self.browse_save_path, width=8).pack(side="right")
        
        btn_frame = ttk.Frame(mp4_dialog)
        btn_frame.pack(pady=20)
        
        ttk.Button(btn_frame, text="Create", 
                  command=lambda: self.start_mp4_creation(mp4_dialog)).pack(side="left", padx=10)
        ttk.Button(btn_frame, text="Cancel", command=mp4_dialog.destroy).pack(side="right", padx=10)
    
    def browse_save_path(self):
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
            
            threading.Thread(target=self.create_mp4_video, args=(total_seconds, output_path), daemon=True).start()
            
        except ValueError:
            messagebox.showerror("Error", "Please enter valid numeric values!")
    
    def create_mp4_video(self, total_seconds, output_path):
        try:
            if self.loaded_capsule:
                self.episode_data = {
                    "moments": self.loaded_capsule["moments"],
                    "quartets": self.loaded_capsule["quartets"]
                }
            else:
                self.episode_data = self.get_episode_data()
            
            if not self.episode_data["moments"]:
                messagebox.showerror("Error", "No data for selected episode!")
                return
            
            frames_dir = "../data/temp_frames"
            os.makedirs(frames_dir, exist_ok=True)
            
            fps = 30
            total_frames = total_seconds * fps
            
            progress_window = tk.Toplevel(self.root)
            progress_window.title("Creating MP4")
            progress_window.geometry("300x100")
            progress_window.resizable(False, False)
            progress_window.configure(bg='#2c3e50')
            
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
            
            current_video_points = [0.5, 0.5, 0.5, 0.5]
            
            frames = []
            for frame_num in range(total_frames):
                current_time = (frame_num / total_frames) * 100.0
                
                target_points = self.get_interpolated_target(current_time)
                
                if target_points:
                    smoothness = 0.1
                    for i in range(4):
                        current_video_points[i] = current_video_points[i] * (1 - smoothness) + target_points[i] * smoothness
                    
                    frame = self.generate_frame(current_video_points)
                    frames.append(frame)
                
                progress_var.set(frame_num)
                progress_window.update()
            
            clip = ImageSequenceClip(frames, fps=fps)
            clip.write_videofile(output_path, codec="libx264", audio=False)
            
            for frame_file in os.listdir(frames_dir):
                os.remove(os.path.join(frames_dir, frame_file))
            os.rmdir(frames_dir)
            
            progress_window.destroy()
            messagebox.showinfo("Success", f"Video successfully created: {output_path}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Error creating video: {str(e)}")
    
    def generate_frame(self, points):
        frame = np.ones((300, 600, 3), dtype=np.uint8) * 255
        
        try:
            emo_bg = cv2.imread(self.resource_path("EmoPlain.png"))
            plot_bg = cv2.imread(self.resource_path("PlotPlain.png"))
            
            emo_bg = cv2.resize(emo_bg, (300, 300))
            plot_bg = cv2.resize(plot_bg, (300, 300))
            
            frame[0:300, 0:300] = emo_bg
            frame[0:300, 300:600] = plot_bg
        except:
            pass
        
        x1 = int(points[0] * 300)
        y1 = int((1 - points[1]) * 300)
        cv2.circle(frame, (x1, y1), 5, (0, 0, 255), -1)
        
        x2 = int(points[2] * 300) + 300
        y2 = int((1 - points[3]) * 300)
        cv2.circle(frame, (x2, y2), 5, (255, 0, 0), -1)
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return frame_rgb


if __name__ == "__main__":
    root = tk.Tk()
    app = PointCreator(root)
    root.mainloop()
