import tkinter as tk
from tkinter import Canvas, Checkbutton, BooleanVar, Button, Entry, Label, filedialog, font, StringVar, OptionMenu, Radiobutton
from PIL import Image, ImageTk
import copy
import pickle

# Example data
data = [
    {
        "image_path": "/home/borisef/projects/pytorch3D/a10.png",
        "keypoints": {"left_eye": (100, 150), "right_eye": (200, 150), "nose": (150, 200)}
    },
    {
        "image_path": "/home/borisef/projects/pytorch3D/a100.png",
        "keypoints": {"left_eye": (110, 160), "right_eye": (210, 160), "nose": (160, 210)}
    }
    # Add more dictionaries as needed
]

# Projection data
projectionData = {'fov_x': 0, 'fov_y': 0, 'yaw': 0, 'pitch': 0, 'roll': 0, 't_x': 0, 't_y': 0, 't_z': 0}

class ImageKeypointsViewer:
    def __init__(self, root, data, projectionData):
        self.root = root
        self.data = data
        self.projectionData = projectionData
        self.index = 0
        self.zoom_scale = 1.0
        self.show_labels = BooleanVar(value=True)
        self.selected_keypoint = None
        self.original_data = [copy.deepcopy(entry) for entry in data]  # Make a deep copy of original data

        # Canvas for displaying images
        self.canvas = Canvas(root, width=800, height=600)
        self.canvas.pack()

        # Control buttons
        self.prev_button = tk.Button(root, text="Previous", command=self.show_prev_image)
        self.prev_button.pack(side=tk.LEFT, padx=10, pady=10)

        self.next_button = tk.Button(root, text="Next", command=self.show_next_image)
        self.next_button.pack(side=tk.LEFT, padx=10, pady=10)

        # Frame for vertically stacked buttons and projection data
        self.control_frame = tk.Frame(root)
        self.control_frame.pack(side=tk.LEFT, padx=10, pady=10)

        self.control_frame1 = tk.Frame(root)
        self.control_frame1.pack(side=tk.LEFT, padx=10, pady=10)

        # Add radio buttons for 2D/3D mode
        self.mode_var = StringVar(value="2D")
        self.mode_frame = tk.Frame(self.control_frame)
        self.mode_frame.pack(pady=10)
        Label(self.mode_frame, text="Mode").pack(side=tk.TOP)
        self.mode_2d_radio = Radiobutton(self.mode_frame, text="2D mode", variable=self.mode_var, value="2D", command=self.update_mode)
        self.mode_2d_radio.pack(side=tk.LEFT, padx=5)
        self.mode_3d_radio = Radiobutton(self.mode_frame, text="3D mode", variable=self.mode_var, value="3D", command=self.update_mode)
        self.mode_3d_radio.pack(side=tk.LEFT, padx=5)

        self.save_button = Button(root, text="Save Changes", command=self.save_changes)
        self.save_button.pack(pady=5)
        self.save_button.place(relx=1.0, rely=1.0, x=-10, y=-60, anchor='se')  # Place button at bottom right

        self.revert_button = Button(root, text="Revert Changes", command=self.revert_changes)
        self.revert_button.pack(pady=5)
        self.revert_button.place(relx=1.0, rely=1.0, x=-10, y=-100, anchor='se')  # Place button at bottom right

        # Add "Save Results" button with bold font
        bold_font = font.Font(weight="bold")
        self.save_results_button = Button(root, text="Save Results", command=self.save_results, font=bold_font)
        self.save_results_button.place(relx=1.0, rely=1.0, x=-10, y=-10, anchor='se')  # Place button at bottom right



        # Add "Overlay 3D" button
        self.overlay_3d_button = Button(root, text="Overlay 3D", command=self.overlay_3d)
        self.overlay_3d_button.pack(pady=5)
        self.overlay_3d_button.place(relx=1.0, rely=1.0, x=-10, y=-140, anchor='se')  # Place button at bottom right

        # Add dropdown menu for Edit 3D transform
        self.transform_var = StringVar(value="None")
        Label(self.control_frame, text="Edit 3D Transform").pack(pady=5)
        self.transform_dropdown = OptionMenu(self.control_frame, self.transform_var, "None", "XYZ", "yaw", "pitch", "roll")
        self.transform_dropdown.pack()

        # Add projection data input fields
        self.projection_vars = {}
        Label(self.control_frame1, text="Projection Data").pack(pady=5)
        for key in self.projectionData:
            frame = tk.Frame(self.control_frame1)
            frame.pack(fill=tk.X, pady=2)
            label = Label(frame, text=key, width=10)
            label.pack(side=tk.LEFT)
            var = tk.StringVar()
            var.set(str(self.projectionData[key]))
            var.trace('w', lambda name, index, mode, key=key, var=var: self.update_projection_data(key, var))
            entry = Entry(frame, width=10, textvariable=var)
            entry.pack(side=tk.LEFT)
            self.projection_vars[key] = var

        self.label_checkbox = Checkbutton(root, text="Show Labels", variable=self.show_labels, command=self.show_image)
        self.label_checkbox.place(relx=1.0, x=-10, y=10, anchor='ne')  # Place checkbox at top right

        # Bind keys for zooming
        self.root.bind("<KeyPress-A>", self.zoom_in)  # Capital 'A'
        self.root.bind("<KeyPress-a>", self.zoom_out)  # Lowercase 'a'

        # Bind mouse events for selecting and dragging keypoints
        self.canvas.bind("<Button-1>", self.select_keypoint)  # Left mouse button click
        self.canvas.bind("<B1-Motion>", self.move_keypoint)   # Left mouse button drag

        self.show_image()

    def show_image(self):
        # Clear canvas
        self.canvas.delete("all")

        # Load the image
        image_info = self.data[self.index]
        image_path = image_info["image_path"]
        keypoints = image_info["keypoints"]

        image = Image.open(image_path)

        # Apply zoom
        width, height = image.size
        new_size = (int(width * self.zoom_scale), int(height * self.zoom_scale))
        image = image.resize(new_size, Image.LANCZOS)

        self.photo = ImageTk.PhotoImage(image)

        # Display the image
        self.canvas.create_image(0, 20, anchor=tk.NW, image=self.photo)

        # Display the image path
        self.canvas.create_text(10, 10, anchor=tk.NW, text=image_path, fill="black")

        # Draw keypoints
        self.keypoint_items = {}
        for name, point in keypoints.items():
            x, y = point
            x = int(x * self.zoom_scale)
            y = int(y * self.zoom_scale) + 20  # Adjust for text offset
            keypoint_item = self.canvas.create_oval(x-5, y-5, x+5, y+5, fill="red", tags=name)
            if self.show_labels.get():
                self.canvas.create_text(x, y-10, text=name, fill="black")
            self.keypoint_items[name] = keypoint_item

    def show_prev_image(self):
        self.index = (self.index - 1) % len(self.data)
        self.show_image()

    def show_next_image(self):
        self.index = (self.index + 1) % len(self.data)
        self.show_image()

    def zoom_in(self, event=None):
        self.zoom_scale *= 1.1
        self.show_image()

    def zoom_out(self, event=None):
        self.zoom_scale /= 1.1
        self.show_image()

    def select_keypoint(self, event):
        x_click = event.x / self.zoom_scale
        y_click = (event.y - 20) / self.zoom_scale

        tolerance = 20

        for name, keypoint_item in self.keypoint_items.items():
            bbox = self.canvas.bbox(keypoint_item)
            if bbox:
                x0, y0, x1, y1 = bbox
                x0 /= self.zoom_scale
                y0 = (y0 - 20) / self.zoom_scale
                x1 /= self.zoom_scale
                y1 = (y1 - 20) / self.zoom_scale

                if (x0 - tolerance <= x_click <= x1 + tolerance and
                    y0 - tolerance <= y_click <= y1 + tolerance):
                    self.selected_keypoint = name
                    break
        else:
            self.selected_keypoint = None

        self.show_image()  # Update display to show selected keypoints

    def move_keypoint(self, event):
        if self.selected_keypoint:
            x_new = event.x
            y_new = event.y - 20  # Adjust for text offset
            self.canvas.coords(self.keypoint_items[self.selected_keypoint],
                               x_new-5, y_new-5, x_new+5, y_new+5)

            # Update keypoint data
            x_data = (x_new / self.zoom_scale)
            y_data = (y_new - 20) / self.zoom_scale
            self.data[self.index]["keypoints"][self.selected_keypoint] = (x_data, y_data)

            self.show_image()  # Update display

    def save_changes(self):
        self.original_data[self.index] = copy.deepcopy(self.data[self.index])

    def revert_changes(self):
        self.data[self.index] = copy.deepcopy(self.original_data[self.index])
        self.show_image()

    def save_results(self):
        save_path = filedialog.asksaveasfilename(defaultextension=".pkl", filetypes=[("Pickle files", "*.pkl")])
        if save_path:
            with open(save_path, 'wb') as f:
                pickle.dump(self.data, f)

    def update_projection_data(self, key, var):
        try:
            self.projectionData[key] = float(var.get())
            print(f"Updated {key}: {self.projectionData[key]}")
        except ValueError:
            print(f"Invalid input for {key}: {var.get()}")

    def overlay_3d(self):
        print("Overlay 3D button pressed")
        # Placeholder for actual 3D overlay functionality

    def update_mode(self):
        print(f"Mode changed to: {self.mode_var.get()}")
        # Placeholder for actual mode switch functionality

if __name__ == "__main__":
    root = tk.Tk()
    viewer = ImageKeypointsViewer(root, data, projectionData)
    root.mainloop()
