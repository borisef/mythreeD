import tkinter as tk
from tkinter import filedialog, messagebox
import numpy as np
import cv2
import trimesh
from PIL import Image, ImageTk


class MeshProjectionGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Mesh Projection GUI")

        # Default values
        self.default_image_path = "/home/borisef/projects/pytorch3D/data/a10.png"
        self.default_mesh_path = "/home/borisef/projects/pytorch3D/data/cow_mesh/cow.obj"
        self.default_camera_matrix = [[1000, 0, 400], [0, 1000, 300], [0, 0, 1]]
        self.default_tvec = [0, 0, 10]
        self.default_rvec = [0, 0, 0]

        # Default increments
        self.default_tvec_increment = 1
        self.default_rvec_increment = 0.05

        # Variables
        self.image = None
        self.mesh = None
        self.camera_matrix = np.array(self.default_camera_matrix, dtype=np.float32)
        self.tvec = np.array(self.default_tvec, dtype=np.float32).reshape(3, 1)
        self.rvec = np.array(self.default_rvec, dtype=np.float32).reshape(3, 1)
        self.interact_mode = tk.StringVar(value="tvec")  # Initialize interact_mode here
        self.tvec_increment = tk.DoubleVar(value=self.default_tvec_increment)
        self.rvec_increment = tk.DoubleVar(value=self.default_rvec_increment)

        # Variables for mouse dragging
        self.mouse_dragging = False
        self.last_mouse_x = None
        self.last_mouse_y = None

        # GUI components
        self.create_widgets()

        # Load defaults
        self.load_default_inputs()

        # Bind keys and mouse events for interactivity
        self.bind_keys_and_mouse()

    def create_widgets(self):
        # Image input
        tk.Label(self.root, text="Image (PNG)").grid(row=0, column=0, sticky="e")
        self.image_entry = tk.Entry(self.root, width=50)
        self.image_entry.grid(row=0, column=1)
        tk.Button(self.root, text="Browse", command=self.load_image).grid(row=0, column=2)

        # Mesh input
        tk.Label(self.root, text="Mesh (OBJ)").grid(row=1, column=0, sticky="e")
        self.mesh_entry = tk.Entry(self.root, width=50)
        self.mesh_entry.grid(row=1, column=1)
        tk.Button(self.root, text="Browse", command=self.load_mesh).grid(row=1, column=2)

        # Camera matrix input
        tk.Label(self.root, text="Camera Matrix (3x3)").grid(row=2, column=0, sticky="e")
        self.camera_matrix_entry = tk.Entry(self.root, width=50)
        self.camera_matrix_entry.grid(row=2, column=1)
        self.camera_matrix_entry.bind("<KeyRelease>", lambda event: self.update_camera_matrix())

        # Translation vector (tvec)
        tk.Label(self.root, text="tvec x").grid(row=3, column=0, sticky="e")
        self.tvec_x_entry = tk.Entry(self.root, width=10)
        self.tvec_x_entry.grid(row=3, column=1, sticky="w")
        self.tvec_x_entry.bind("<KeyRelease>", lambda event: self.update_tvec())

        tk.Label(self.root, text="tvec y").grid(row=3, column=1, sticky="e")
        self.tvec_y_entry = tk.Entry(self.root, width=10)
        self.tvec_y_entry.grid(row=3, column=2, sticky="w")
        self.tvec_y_entry.bind("<KeyRelease>", lambda event: self.update_tvec())

        tk.Label(self.root, text="tvec z").grid(row=3, column=2, sticky="e")
        self.tvec_z_entry = tk.Entry(self.root, width=10)
        self.tvec_z_entry.grid(row=3, column=3, sticky="w")
        self.tvec_z_entry.bind("<KeyRelease>", lambda event: self.update_tvec())

        # Rotation vector (rvec)
        tk.Label(self.root, text="rvec x").grid(row=4, column=0, sticky="e")
        self.rvec_x_entry = tk.Entry(self.root, width=10)
        self.rvec_x_entry.grid(row=4, column=1, sticky="w")
        self.rvec_x_entry.bind("<KeyRelease>", lambda event: self.update_rvec())

        tk.Label(self.root, text="rvec y").grid(row=4, column=1, sticky="e")
        self.rvec_y_entry = tk.Entry(self.root, width=10)
        self.rvec_y_entry.grid(row=4, column=2, sticky="w")
        self.rvec_y_entry.bind("<KeyRelease>", lambda event: self.update_rvec())

        tk.Label(self.root, text="rvec z").grid(row=4, column=2, sticky="e")
        self.rvec_z_entry = tk.Entry(self.root, width=10)
        self.rvec_z_entry.grid(row=4, column=3, sticky="w")
        self.rvec_z_entry.bind("<KeyRelease>", lambda event: self.update_rvec())

        # Interaction mode radio buttons
        tk.Radiobutton(self.root, text="Interact TVEC", variable=self.interact_mode, value="tvec",
                       command=self.bind_keys_and_mouse).grid(row=5, column=0, sticky="w")
        tk.Radiobutton(self.root, text="Interact RVEC", variable=self.interact_mode, value="rvec",
                       command=self.bind_keys_and_mouse).grid(row=6, column=0, sticky="w")

        # Combo boxes for increments
        tk.Label(self.root, text="TVEC Increment").grid(row=5, column=1, sticky="e")
        tvec_increment_choices = [0.05, 0.1, 0.5, 1, 5, 10]
        self.tvec_increment_combobox = tk.OptionMenu(self.root, self.tvec_increment, *tvec_increment_choices)
        self.tvec_increment_combobox.grid(row=5, column=2, sticky="w")
        self.tvec_increment_combobox.bind("<Configure>", self.update_key_bindings)

        tk.Label(self.root, text="RVEC Increment").grid(row=6, column=1, sticky="e")
        rvec_increment_choices = [0.005, 0.01, 0.05, 0.1]
        self.rvec_increment_combobox = tk.OptionMenu(self.root, self.rvec_increment, *rvec_increment_choices)
        self.rvec_increment_combobox.grid(row=6, column=2, sticky="w")
        self.rvec_increment_combobox.bind("<Configure>", self.update_key_bindings)

        # Canvas for displaying image and mesh
        self.canvas = tk.Canvas(self.root, width=800, height=600, bg="white")
        self.canvas.grid(row=8, column=0, columnspan=3, pady=10)

    def bind_keys_and_mouse(self):
        self.root.unbind("<Left>")
        self.root.unbind("<Right>")
        self.root.unbind("<Up>")
        self.root.unbind("<Down>")
        self.root.unbind("<minus>")
        self.root.unbind("<plus>")
        self.root.unbind("<KP_Subtract>")
        self.root.unbind("<KP_Add>")

        self.canvas.bind("<ButtonPress-1>", self.on_mouse_press)
        self.canvas.bind("<B1-Motion>", self.on_mouse_drag)
        self.canvas.bind("<MouseWheel>", self.on_mouse_wheel)

        tvec_increment_value = self.tvec_increment.get()
        rvec_increment_value = self.rvec_increment.get()

        if self.interact_mode.get() == "tvec":
            self.root.bind("<Left>", lambda event: self.update_vector(self.tvec, 0, -tvec_increment_value))
            self.root.bind("<Right>", lambda event: self.update_vector(self.tvec, 0, tvec_increment_value))
            self.root.bind("<Up>", lambda event: self.update_vector(self.tvec, 1, -tvec_increment_value))
            self.root.bind("<Down>", lambda event: self.update_vector(self.tvec, 1, tvec_increment_value))
            self.root.bind("<minus>", lambda event: self.update_vector(self.tvec, 2, -tvec_increment_value))
            self.root.bind("<plus>", lambda event: self.update_vector(self.tvec, 2, tvec_increment_value))
            self.root.bind("<KP_Subtract>", lambda event: self.update_vector(self.tvec, 2, -tvec_increment_value))
            self.root.bind("<KP_Add>", lambda event: self.update_vector(self.tvec, 2, tvec_increment_value))
        else:
            self.root.bind("<Left>", lambda event: self.update_vector(self.rvec, 0, -rvec_increment_value))
            self.root.bind("<Right>", lambda event: self.update_vector(self.rvec, 0, rvec_increment_value))
            self.root.bind("<Up>", lambda event: self.update_vector(self.rvec, 1, -rvec_increment_value))
            self.root.bind("<Down>", lambda event: self.update_vector(self.rvec, 1, rvec_increment_value))
            self.root.bind("<minus>", lambda event: self.update_vector(self.rvec, 2, -rvec_increment_value))
            self.root.bind("<plus>", lambda event: self.update_vector(self.rvec, 2, rvec_increment_value))
            self.root.bind("<KP_Subtract>", lambda event: self.update_vector(self.rvec, 2, -rvec_increment_value))
            self.root.bind("<KP_Add>", lambda event: self.update_vector(self.rvec, 2, rvec_increment_value))

    def on_mouse_press(self, event):
        self.mouse_dragging = True
        self.last_mouse_x = event.x
        self.last_mouse_y = event.y

    def on_mouse_drag(self, event):
        if self.mouse_dragging:
            dx = event.x - self.last_mouse_x
            dy = event.y - self.last_mouse_y
            self.last_mouse_x = event.x
            self.last_mouse_y = event.y

            increment_value = self.tvec_increment.get() if self.interact_mode.get() == "tvec" else self.rvec_increment.get()

            if self.interact_mode.get() == "tvec":
                self.update_vector(self.tvec, 0, dx * increment_value)
                self.update_vector(self.tvec, 1, dy * increment_value)
            else:
                self.update_vector(self.rvec, 0, dx * increment_value)
                self.update_vector(self.rvec, 1, dy * increment_value)

    def on_mouse_wheel(self, event):
        increment_value = self.tvec_increment.get() if self.interact_mode.get() == "tvec" else self.rvec_increment.get()

        if event.delta > 0:
            self.update_vector(self.tvec if self.interact_mode.get() == "tvec" else self.rvec, 2, increment_value)
        else:
            self.update_vector(self.tvec if self.interact_mode.get() == "tvec" else self.rvec, 2, -increment_value)

    def update_vector(self, vector, index, delta):
        vector[index] += delta
        self.update_entries()
        self.render_projection()

    def update_tvec(self):
        try:
            self.tvec[0, 0] = float(self.tvec_x_entry.get())
            self.tvec[1, 0] = float(self.tvec_y_entry.get())
            self.tvec[2, 0] = float(self.tvec_z_entry.get())
            self.render_projection()
        except ValueError:
            pass

    def update_rvec(self):
        try:
            self.rvec[0, 0] = float(self.rvec_x_entry.get())
            self.rvec[1, 0] = float(self.rvec_y_entry.get())
            self.rvec[2, 0] = float(self.rvec_z_entry.get())
            self.render_projection()
        except ValueError:
            pass

    def update_camera_matrix(self):
        try:
            matrix = eval(self.camera_matrix_entry.get())
            self.camera_matrix = np.array(matrix, dtype=np.float32)
            self.render_projection()
        except:
            pass

    def update_entries(self):
        self.tvec_x_entry.delete(0, tk.END)
        self.tvec_x_entry.insert(0, self.tvec[0, 0])
        self.tvec_y_entry.delete(0, tk.END)
        self.tvec_y_entry.insert(0, self.tvec[1, 0])
        self.tvec_z_entry.delete(0, tk.END)
        self.tvec_z_entry.insert(0, self.tvec[2, 0])

        self.rvec_x_entry.delete(0, tk.END)
        self.rvec_x_entry.insert(0, self.rvec[0, 0])
        self.rvec_y_entry.delete(0, tk.END)
        self.rvec_y_entry.insert(0, self.rvec[1, 0])
        self.rvec_z_entry.delete(0, tk.END)
        self.rvec_z_entry.insert(0, self.rvec[2, 0])

        self.render_projection()

    def render_projection(self):
        try:
            camera_matrix_str = self.camera_matrix_entry.get()
            self.camera_matrix = np.array(eval(camera_matrix_str), dtype=np.float32)
        except:
            self.camera_matrix = self.default_camera_matrix

        # Ensure projection matrix rendering updates are done based on tvec and rvec
        self.canvas.delete("all")  # Clear previous frame
        if self.image:
            self.canvas.create_image(0, 0, anchor="nw", image=self.image)

        if self.mesh:
            # Render the 3D mesh with the updated camera parameters
            self.render_mesh_projection()

    def render_mesh_projection(self):
        pass  # Add actual mesh rendering based on tvec and rvec transformation logic

    def load_default_inputs(self):
        # Load default values into the GUI entries
        self.image_entry.insert(0, self.default_image_path)
        self.mesh_entry.insert(0, self.default_mesh_path)
        self.camera_matrix_entry.insert(0, str(self.default_camera_matrix))
        self.tvec_x_entry.insert(0, self.tvec[0, 0])
        self.tvec_y_entry.insert(0, self.tvec[1, 0])
        self.tvec_z_entry.insert(0, self.tvec[2, 0])
        self.rvec_x_entry.insert(0, self.rvec[0, 0])
        self.rvec_y_entry.insert(0, self.rvec[1, 0])
        self.rvec_z_entry.insert(0, self.rvec[2, 0])

    def load_image(self):
        # Load an image from file dialog
        image_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.png *.jpg *.jpeg")])
        if image_path:
            self.image_entry.delete(0, tk.END)
            self.image_entry.insert(0, image_path)
            self.image = Image.open(image_path)
            self.image = self.image.resize((800, 600))  # Resize to fit canvas
            self.image_tk = ImageTk.PhotoImage(self.image)
            self.canvas.create_image(0, 0, anchor="nw", image=self.image_tk)
            self.render_projection()

    def load_mesh(self):
        # Load a mesh from file dialog
        mesh_path = filedialog.askopenfilename(filetypes=[("Mesh Files", "*.obj")])
        if mesh_path:
            self.mesh_entry.delete(0, tk.END)
            self.mesh_entry.insert(0, mesh_path)
            self.mesh = trimesh.load(mesh_path)
            self.render_projection()

    def render_projection(self):
        if self.image:
            # Clear the canvas and re-render the image
            self.canvas.delete("all")
            self.canvas.create_image(0, 0, anchor="nw", image=self.image_tk)

        if self.mesh:
            # Project mesh vertices to 2D using the camera parameters
            vertices = self.mesh.vertices
            projected_points, _ = cv2.projectPoints(vertices, self.rvec, self.tvec, self.camera_matrix, None)

            # Render projected points onto the canvas
            for point in projected_points:
                x, y = int(point[0][0]), int(point[0][1])
                self.canvas.create_oval(x - 2, y - 2, x + 2, y + 2, fill="red")

    def update_key_bindings(self, event=None):
        # Update the key bindings when the combo boxes for increments are changed
        tvec_increment_value = self.tvec_increment.get()
        rvec_increment_value = self.rvec_increment.get()

        if self.interact_mode.get() == "tvec":
            self.root.bind("<Left>", lambda event: self.update_vector(self.tvec, 0, -tvec_increment_value))
            self.root.bind("<Right>", lambda event: self.update_vector(self.tvec, 0, tvec_increment_value))
            self.root.bind("<Up>", lambda event: self.update_vector(self.tvec, 1, -tvec_increment_value))
            self.root.bind("<Down>", lambda event: self.update_vector(self.tvec, 1, tvec_increment_value))
            self.root.bind("<minus>", lambda event: self.update_vector(self.tvec, 2, -tvec_increment_value))
            self.root.bind("<plus>", lambda event: self.update_vector(self.tvec, 2, tvec_increment_value))
            self.root.bind("<KP_Subtract>", lambda event: self.update_vector(self.tvec, 2, -tvec_increment_value))
            self.root.bind("<KP_Add>", lambda event: self.update_vector(self.tvec, 2, tvec_increment_value))
        else:
            self.root.bind("<Left>", lambda event: self.update_vector(self.rvec, 0, -rvec_increment_value))
            self.root.bind("<Right>", lambda event: self.update_vector(self.rvec, 0, rvec_increment_value))
            self.root.bind("<Up>", lambda event: self.update_vector(self.rvec, 1, -rvec_increment_value))
            self.root.bind("<Down>", lambda event: self.update_vector(self.rvec, 1, rvec_increment_value))
            self.root.bind("<minus>", lambda event: self.update_vector(self.rvec, 2, -rvec_increment_value))
            self.root.bind("<plus>", lambda event: self.update_vector(self.rvec, 2, rvec_increment_value))
            self.root.bind("<KP_Subtract>", lambda event: self.update_vector(self.rvec, 2, -rvec_increment_value))
            self.root.bind("<KP_Add>", lambda event: self.update_vector(self.rvec, 2, rvec_increment_value))

    def update_vector(self, vector, index, delta):
        # Update the vector value at the specified index by delta
        vector[index] += delta
        self.update_entries()

    def update_entries(self):
        # Update the GUI entries to reflect the latest tvec and rvec values
        self.tvec_x_entry.delete(0, tk.END)
        self.tvec_x_entry.insert(0, self.tvec[0, 0])
        self.tvec_y_entry.delete(0, tk.END)
        self.tvec_y_entry.insert(0, self.tvec[1, 0])
        self.tvec_z_entry.delete(0, tk.END)
        self.tvec_z_entry.insert(0, self.tvec[2, 0])

        self.rvec_x_entry.delete(0, tk.END)
        self.rvec_x_entry.insert(0, self.rvec[0, 0])
        self.rvec_y_entry.delete(0, tk.END)
        self.rvec_y_entry.insert(0, self.rvec[1, 0])
        self.rvec_z_entry.delete(0, tk.END)
        self.rvec_z_entry.insert(0, self.rvec[2, 0])

        self.render_projection()


    def render_mesh_projection(self):
        # Add mesh projection logic based on tvec and rvec
        pass  # Replace this with actual projection handling


if __name__ == "__main__":
    root = tk.Tk()
    app = MeshProjectionGUI(root)
    root.mainloop()
