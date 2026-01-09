import tkinter as tk
from tkinter import messagebox
import threading
import cv2
from PIL import Image, ImageTk
import time
from ultralytics import YOLO

class SoccerTrainingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Soccer Training System")
        self.root.geometry("800x480")
        self.root.config(cursor="none")
        self.root.configure(bg="#4c4c4c")
        self.root.overrideredirect(True)
        #self.root.iconbitmap("kickeen.ico")
        #self.root.attributes("-fullscreen", True)
        backgroundphoto = Image.open("data/bg.png")
        bg = ImageTk.PhotoImage(backgroundphoto)
        canvas = tk.Canvas(root, highlightthickness=0)
        canvas.pack(fill="both", expand=True)
        # Place the image on the canvas and stretch it to fit the window
        canvas.create_image(0, 0, image=bg, anchor="nw")
        # Keep a reference to the image
        canvas.image = bg
        canvas.config(bg="#444444", width=root.winfo_width(), height=root.winfo_height())
        

        # Buttons
        self.start_button = tk.Button(self.root, text="START", width=50, height=2, command=self.start_detection_window,
                                      font=("Arial", 10, "bold"), bg="#444444", fg="#dadada", activebackground="#555555", 
                                      activeforeground="white", relief="flat")
        self.start_button.place(relx=0.5, rely=0.4, anchor="center")

        self.test_button = tk.Button(self.root, text="SIMULATION", width=50, height=2, command=self.start_testing_window,
                                      font=("Arial", 10, "bold"), bg="#444444", fg="#dadada", activebackground="#555555", 
                                      activeforeground="white", relief="flat")
        self.test_button.place(relx=0.5, rely=0.5, anchor="center")

        self.statistics_button = tk.Button(self.root, text="STATISTICS", width=50, height=2, command=self.show_statistics,
                                           font=("Arial", 10, "bold"), bg="#444444", fg="#dadada", activebackground="#555555", 
                                           activeforeground="white", relief="flat")
        self.statistics_button.place(relx=0.5, rely=0.6, anchor="center")

        self.exit_button = tk.Button(self.root, text="EXIT", width=50, height=2, command=self.exit_app,
                                     font=("Arial", 10, "bold"), bg="#444444", fg="#dadada", activebackground="#555555", 
                                     activeforeground="white", relief="flat")
        self.exit_button.place(relx=0.5, rely=0.7, anchor="center")

        self.distance_categories = [
            "<4m", "4-5m", "5-6m", "6-7m", "7-8m", "8-9m", "9-10m", "10-11m",
            "11-12m", "12-13m", "13-14m", "14-15m", "15-16m", "16-17m",
            "17-18m", "18-19m", ">19m"
        ]

        self.goal_counts = [0 for _ in self.distance_categories]
        self.percentages = [0 for _ in self.distance_categories]
        self.load_statistics_from_file()

        self.last_ball_distance = None
        self.goalpost_distance = None
        self.player_distance = None
        self.goal_on_cooldown = False
        self.goal_lock = threading.Lock()
        self.last_goal_timestamp = 0 
        self.last_logged_goal_signature = None
        self.stop_event = threading.Event()

        self.last_ball_bbox = None # (x1, y1, x2, y2)
        self.goalpost_bbox = None # (x1, y1, x2, y2)

        self.root.mainloop()

    def start_detection_window(self):
        self.detect_window = tk.Toplevel(self.root)
        self.detect_window.title("YOLOv8 Detection")
        self.detect_window.geometry("800x480")
        self.detect_window.configure(bg="#4c4c4c")
        self.detect_window.overrideredirect(True)
        self.detect_window.config(cursor="none")
        self.start_label = tk.Label(self.detect_window, text="Start Training", font=("Arial", 12), anchor="center", bg="#4c4c4c", fg="white")
        self.start_label.pack(pady=10)

        self.video_label = tk.Label(self.detect_window, bg="#4c4c4c", width=640, height=360)
        self.video_label.pack(pady=10)

        back_button = tk.Button(self.detect_window, text="BACK", command=self.stop_detection_thread,
                                bg="#444444", fg="white", activebackground="#555555", activeforeground="white", relief="flat", font=("Arial", 10, "bold"))
        back_button.pack(pady=10)

        self.stop_event.clear()
        self.detection_thread = threading.Thread(target=self.run_detection, daemon=True)
        self.detection_thread.start()
    
    def start_testing_window(self):
        self.detect_window = tk.Toplevel(self.root)
        self.detect_window.title("YOLOv8 Detection")
        self.detect_window.geometry("800x480")
        self.detect_window.overrideredirect(True)
        self.detect_window.configure(bg="#4c4c4c")
        self.detect_window.config(cursor="none")
        self.test_label = tk.Label(self.detect_window, text="Simulation Testing Mode", font=("Arial", 12), anchor="center", bg="#4c4c4c", fg="white")
        self.test_label.pack(pady=10)

        self.video_label = tk.Label(self.detect_window, bg="#4c4c4c", width=640, height=360)
        self.video_label.pack(pady=10)

        back_button = tk.Button(self.detect_window, text="BACK", command=self.stop_detection_thread,
                                bg="#444444", fg="white", activebackground="#555555", activeforeground="white", relief="flat", font=("Arial", 10, "bold"))
        back_button.pack(pady=10)

        self.stop_event.clear()
        self.test_detection_thread = threading.Thread(target=self.run_test_detection, daemon=True)
        self.test_detection_thread.start()

    def stop_detection_thread(self):
        self.stop_event.set()
        if hasattr(self, 'detect_window') and self.detect_window.winfo_exists():
            self.detect_window.destroy()

    def run_detection(self):
        model = YOLO("data/best2.pt")
        cap = cv2.VideoCapture(2)

        if not cap.isOpened():
            print("Error: Cannot open video.")
            self.detect_window.destroy()
            return

        class_names = ["ball", "goalpost", "player"]
        start_distances = [0.1, 0.5, 0.5]
        scale_factors = [50, 170, 140]

        self.last_ball_bbox = None
        self.goalpost_bbox = None

        while not self.stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                break

            results = model(frame, imgsz=240)
            annotated_frame = results[0].orig_img.copy()
            boxes = results[0].boxes

            detected_ball_distance = None
            detected_player_distance = None
            detected_goalpost_distance = None

            current_frame_ball_bbox = None
            current_frame_goalpost_bbox = None

            if boxes is not None:
                for box, cls_id in zip(boxes.xyxy, boxes.cls):
                    x1, y1, x2, y2 = map(int, box)
                    class_id = int(cls_id)
                    if 0 <= class_id < len(class_names):
                        label_name = class_names[class_id]
                        box_height = y2 - y1
                        distance_m = round(start_distances[class_id] + (scale_factors[class_id] / box_height), 2) if box_height > 0 else None

                        if label_name == "ball":
                            detected_ball_distance = distance_m
                            current_frame_ball_bbox = (x1, y1, x2, y2)
                        elif label_name == "player":
                            detected_player_distance = distance_m
                        elif label_name == "goalpost":
                            detected_goalpost_distance = distance_m
                            current_frame_goalpost_bbox = (x1, y1, x2, y2)

                        label = f"{label_name}: {distance_m:.1f} m" if distance_m else f"{label_name}: N/A"
                        color = (0, 255, 0) if class_id == 0 else (255, 0, 0)
                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(annotated_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_DUPLEX, 0.8, color, 2)

            current_time = time.time()

            self.last_ball_distance = detected_ball_distance if detected_ball_distance else self.last_ball_distance
            self.goalpost_distance = detected_goalpost_distance if detected_goalpost_distance else self.goalpost_distance
            self.player_distance = detected_player_distance if detected_player_distance else self.player_distance

            if current_frame_ball_bbox:
                self.last_ball_bbox = current_frame_ball_bbox
            if current_frame_goalpost_bbox:
                self.goalpost_bbox = current_frame_goalpost_bbox

            # Check for goal using BBOX intersection
            is_goal = False
            if self.last_ball_bbox and self.goalpost_bbox:
                if self._check_bbox_intersection(self.last_ball_bbox, self.goalpost_bbox):
                    is_goal = True

            if is_goal:
                if self.last_ball_distance and self.goalpost_distance and self.player_distance:
                    with self.goal_lock:
                        if not self.goal_on_cooldown and (current_time - self.last_goal_timestamp) > 10:
                            self.goal_on_cooldown = True
                            self.last_goal_timestamp = current_time
                            goal_distance = round(self.goalpost_distance - self.player_distance, 2)
                            print(f"Recording goal (BBOX method) at {goal_distance}m at {current_time}")
                            self.record_goal(goal_distance)
                            self.root.after(10000, self.reset_goal_cooldown)

            resized_frame = cv2.resize(annotated_frame, (640, 360))
            frame_rgb = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
            frame_pil = Image.fromarray(frame_rgb)
            frame_tk = ImageTk.PhotoImage(image=frame_pil)
            self.root.after(0, self.update_video_feed, frame_tk)
            
            time.sleep(0.03)

        cap.release()
    
    def run_test_detection(self):
        model = YOLO("data/best.pt")
        cap = cv2.VideoCapture("data/simulation.mp4")

        if not cap.isOpened():
            print("Error: Cannot open video.")
            self.detect_window.destroy()
            return

        class_names = ["Ball", "Goalkeeper", "Goalpost", "Player"]
        start_distances = [1.0, 15.0, 15.0, 1.0]
        scale_factors = [700, 300, 700, 1000]

        while not self.stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                break

            results = model(frame, imgsz=360)
            annotated_frame = results[0].orig_img.copy()
            boxes = results[0].boxes

            detected_ball_distance = None
            detected_player_distance = None
            detected_goalpost_distance = None

            if boxes is not None:
                for box, cls_id in zip(boxes.xyxy, boxes.cls):
                    x1, y1, x2, y2 = map(int, box)
                    class_id = int(cls_id)
                    if 0 <= class_id < len(class_names):
                        label_name = class_names[class_id]
                        box_height = y2 - y1
                        distance_m = round(start_distances[class_id] + (scale_factors[class_id] / box_height), 2) if box_height > 0 else None

                        if label_name == "Ball":
                            detected_ball_distance = distance_m
                        elif label_name == "Player":
                            detected_player_distance = distance_m
                        elif label_name == "Goalpost":
                            detected_goalpost_distance = distance_m

                        label = f"{label_name}: {distance_m:.1f} m" if distance_m else f"{label_name}: N/A"
                        color = (0, 255, 0) if class_id == 0 else (255, 0, 0)
                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(annotated_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_DUPLEX, 0.8, color, 2)

            current_time = time.time()

            self.last_ball_distance = detected_ball_distance if detected_ball_distance else self.last_ball_distance
            self.goalpost_distance = detected_goalpost_distance if detected_goalpost_distance else self.goalpost_distance
            self.player_distance = detected_player_distance if detected_player_distance else self.player_distance

            if self.last_ball_distance and self.goalpost_distance and self.player_distance:
                if self.goalpost_distance - 0.1 <= self.last_ball_distance <= self.goalpost_distance:
                    with self.goal_lock:
                        if not self.goal_on_cooldown and (current_time - self.last_goal_timestamp) > 2:
                            self.goal_on_cooldown = True
                            self.last_goal_timestamp = current_time
                            goal_distance = round(self.goalpost_distance - self.player_distance, 2)
                            print(f"Recording goal at {goal_distance}m at {current_time}")
                            self.record_goal(goal_distance)
                            self.root.after(2000, self.reset_goal_cooldown)

            resized_frame = cv2.resize(annotated_frame, (640, 360))
            frame_rgb = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
            frame_pil = Image.fromarray(frame_rgb)
            frame_tk = ImageTk.PhotoImage(image=frame_pil)
            self.root.after(0, self.update_video_feed, frame_tk)
            
            time.sleep(0.03)

        cap.release()
    
    def _check_bbox_intersection(self, bbox_a, bbox_b):
        """
        Checks if two bounding boxes intersect.
        BBox format: (x1, y1, x2, y2)
        """
        Ax1, Ay1, Ax2, Ay2 = bbox_a
        Bx1, By1, Bx2, By2 = bbox_b
        
        # Check for non-intersection first
        # If A is to the right of B, or A is to the left of B,
        # or A is below B, or A is above B, they don't intersect.
        if Ax1 > Bx2 or Ax2 < Bx1 or Ay1 > By2 or Ay2 < By1:
            return False
        
        # If they don't meet any of the non-intersection criteria, they must intersect.
        return True
    
    def reset_goal_cooldown(self):
        print("Cooldown reset")
        self.goal_on_cooldown = False

    def update_video_feed(self, img_tk):
        if hasattr(self, "video_label") and self.video_label.winfo_exists():
            self.video_label.configure(image=img_tk)
            self.video_label.image = img_tk

    def record_goal(self, goal_distance):
        """Update statistics for the captured goal."""
        # Convert distance categories to numerical ranges for comparison
        distance_ranges = [
            (0, 4), (4, 5), (5, 6), (6, 7), (7, 8), (8, 9), (9, 10), (10, 11),
            (11, 12), (12, 13), (13, 14), (14, 15), (15, 16), (16, 17), (17, 18),
            (18, 19), (19, 100)  # ">19m" assumes anything above 19
        ]

        # Find the correct category index
        for i, (low, high) in enumerate(distance_ranges):
            if low <= goal_distance < high:
                closest_distance_index = i
                break

        # Increase goal count
        self.goal_counts[closest_distance_index] += 1

        # Update percentages dynamically
        total_goals = sum(self.goal_counts)
        self.percentages = [(g / total_goals) * 100 if total_goals > 0 else 0 for g in self.goal_counts]
        
        if hasattr(self, "stats_window") and self.stats_window.winfo_exists():
            self.update_statistics_display()

    def show_statistics(self):
        self.stats_window = tk.Toplevel(self.root)
        self.stats_window.title("Player Statistics")
        self.stats_window.geometry("800x480")
        self.stats_window.configure(bg="#4c4c4c")
        self.stats_window.overrideredirect(True)
        self.stats_window.config(cursor="none")

        title_label = tk.Label(self.stats_window, text="Statistics", font=("Arial", 24, "bold"),
                               bg="#4c4c4c", fg="white")
        title_label.pack(pady=10)

        headers = ["Distance", "Goals", "Percentage"]
        for i, header in enumerate(headers):
            tk.Label(self.stats_window, text=header, font=("Arial", 11, "bold"), bg="#4c4c4c", fg="white").place(x=150 + i*200, y=60)

        self.goal_labels = []
        self.percent_labels = []

        for i, dist in enumerate(self.distance_categories):
            y = 90 + i * 20
            tk.Label(self.stats_window, text=dist, font=("Arial", 10), bg="#4c4c4c", fg="white").place(x=150, y=y)

            goal_lbl = tk.Label(self.stats_window, text=str(self.goal_counts[i]), font=("Arial", 10), bg="#4c4c4c", fg="white")
            goal_lbl.place(x=350, y=y)
            self.goal_labels.append(goal_lbl)

            percent_lbl = tk.Label(self.stats_window, text=f"{self.percentages[i]}%", font=("Arial", 10), bg="#4c4c4c", fg="white")
            percent_lbl.place(x=550, y=y)
            self.percent_labels.append(percent_lbl)

        self.total_goals_label = tk.Label(self.stats_window, text="Total Goals Made: 0", font=("Arial", 10),
                                          bg="#4c4c4c", fg="white")
        self.total_goals_label.place(x=150, y=430)

        self.insight_label = tk.Label(self.stats_window, text="You have made most goals in N/A", font=("Arial", 10),
                                      bg="#4c4c4c", fg="white")
        self.insight_label.place(x=150, y=450)

        reset_button = tk.Button(self.stats_window, text="RESET", command=self.reset_statistics,
                                 bg="#444444", fg="white", activebackground="#555555", activeforeground="white", relief="flat", font=("Arial", 10, "bold"))
        reset_button.place(x=20, y=90)

        back_button = tk.Button(self.stats_window, text="BACK", command=self.stats_window.destroy,
                                bg="#444444", fg="white", activebackground="#555555", activeforeground="white", relief="flat", font=("Arial", 10, "bold"))
        back_button.place(x=20, y=50)

        self.update_statistics_display()

    def update_statistics_display(self):
        """Refresh statistics on the UI."""
        total_goals = sum(self.goal_counts)

        # Ensure statistics window exists before updating UI elements
        if hasattr(self, "stats_window") and self.stats_window.winfo_exists():
            self.total_goals_label.config(text=f"Total Goals Made: {total_goals}")

            if total_goals > 0:
                self.percentages = [round((g / total_goals) * 100) for g in self.goal_counts]
            else:
                self.percentages = [0] * len(self.goal_counts)

            for i in range(len(self.goal_counts)):
                self.goal_labels[i].config(text=str(self.goal_counts[i]))
                self.percent_labels[i].config(text=f"{self.percentages[i]}%")

            if total_goals > 0:
                max_goals = max(self.goal_counts)
                max_index = self.goal_counts.index(max_goals)
                best_distance = self.distance_categories[max_index]
                rate = self.percentages[max_index]
                insight = f"You have made most goals in {best_distance}, which has a rate of {rate}% and a total of {max_goals} goals."
            else:
                insight = "You have made most goals in N/A"

            self.insight_label.config(text=insight)

            self.save_statistics_to_file()

    def save_statistics_to_file(self):
        """Save statistics to a text file in the same directory."""
        file_path = "data/statistics.txt"  # File will be saved in the same directory

        with open(file_path, "w") as file:
            file.write("Soccer Training Statistics\n")
            file.write("=========================\n\n")
            file.write(f"Total Goals Made: {sum(self.goal_counts)}\n\n")

            for i in range(len(self.distance_categories)):
                file.write(f"{self.distance_categories[i]}: {self.goal_counts[i]} goals ({self.percentages[i]:.2f}%)\n")

            if sum(self.goal_counts) > 0:
                max_goals = max(self.goal_counts)
                max_index = self.goal_counts.index(max_goals)
                best_distance = self.distance_categories[max_index]
                rate = self.percentages[max_index]
                file.write(f"\nMost goals scored in: {best_distance} ({rate:.2f}% success rate)\n")
            else:
                file.write("\nNo goals recorded yet.\n")

    def load_statistics_from_file(self):
        #Load saved statistics from a text file if it exists.
        file_path = "data/statistics.txt"

        try:
            with open(file_path, "r") as file:
                lines = file.readlines()
                for line in lines:
                    if "-" in line and "goals" in line:
                        parts = line.split(":")
                        category = parts[0].strip()
                        goal_data = parts[1].split()[0].strip()  # Extract number before "goals"

                        if goal_data.isdigit():
                            goal_count = int(goal_data)

                            if category in self.distance_categories:
                                index = self.distance_categories.index(category)
                                self.goal_counts[index] = goal_count

            # Ensure percentages update correctly
            total_goals = sum(self.goal_counts)
            self.percentages = [(g / total_goals) * 100 if total_goals > 0 else 0 for g in self.goal_counts]

        except FileNotFoundError:
            print("No previous statistics file found. Starting fresh.")

    def reset_statistics(self):
        confirm_window = tk.Toplevel(self.stats_window)
        confirm_window.title("Confirm Reset")
        confirm_window.attributes('-topmost', True)
        confirm_window.config(cursor="none")
        confirm_window.overrideredirect(True)
        confirm_window.geometry("400x200")
        confirm_window.configure(bg="#404040")
        confirm_window.transient(self.stats_window)
        confirm_window.grab_set()
        self.stats_window.update_idletasks()
        # Get the main window's position and size
        stats_x = self.stats_window.winfo_x()
        stats_y = self.stats_window.winfo_y()
        stats_width = self.stats_window.winfo_width()
        stats_height = self.stats_window.winfo_height()

        # Calculate the center position
        popup_width = 400  # Same as confirm_window's width
        popup_height = 200  # Same as confirm_window's height

        center_x = stats_x + (stats_width // 2) - (popup_width // 2)
        center_y = stats_y + (stats_height // 2) - (popup_height // 2)

        # Set the popup window's position
        confirm_window.geometry(f"{popup_width}x{popup_height}+{center_x}+{center_y}")
        
        msg_label = tk.Label(confirm_window, text="Are you sure you want to reset your statistics?",
                             font=("Arial", 12), bg="#404040", fg="white", wraplength=350, justify="center")
        msg_label.pack(pady=30)

        button_frame = tk.Frame(confirm_window, bg="#404040")
        button_frame.pack(pady=10)

        reset_btn = tk.Button(button_frame, text="RESET", width=10, command=lambda: self.perform_reset(confirm_window),
                              bg="#8B0000", fg="white", activebackground="#AA0000", relief="flat", font=("Arial", 10, "bold"))
        reset_btn.pack(side="left", padx=10)

        cancel_btn = tk.Button(button_frame, text="Keep my statistics", width=18, command=confirm_window.destroy,
                               bg="#444444", fg="white", activebackground="#555555", relief="flat", font=("Arial", 10, "bold"))
        cancel_btn.pack(side="left", padx=10)

    def perform_reset(self, window):
        self.goal_counts = [0] * len(self.distance_categories)
        self.percentages = [0] * len(self.distance_categories)
        self.update_statistics_display()
        window.destroy()


    def exit_app(self):
            # Ensure a clean exit by stopping the detection thread and closing all windows.
            self.stop_event.set()  # Stop detection thread

            if hasattr(self, 'cap') and self.cap.isOpened():
                self.cap.release()  # Close video capture
            
            if hasattr(self, 'detect_window') and self.detect_window.winfo_exists():
                self.detect_window.destroy()

            if hasattr(self, 'stats_window') and self.stats_window.winfo_exists():
                self.stats_window.destroy()

            cv2.destroyAllWindows()  # Ensure all OpenCV windows are closed
            
            if hasattr(self, 'detection_thread') and self.detection_thread.is_alive():
                self.detection_thread.join()  # Stop lingering thread

            self.root.after(100, self.root.destroy)  # Graceful Tkinter shutdown\
            os.system("sudo reboot")
if __name__ == "__main__":
    root = tk.Tk()
    app = SoccerTrainingApp(root)
    root.mainloop()
