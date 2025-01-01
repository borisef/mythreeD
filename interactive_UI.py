import tkinter as tk
from tkinter import filedialog, messagebox
import numpy as np
import cv2
import trimesh
from PIL import Image, ImageTk

MAX_VERTICES = 5000


def combine_rot_vecs(rot_vec_old, rot_vec_new):
    #rotate first with rot_vec_old, after that with rot_vec_new
    #output:  final rot_vec
    # Convert rotation vectors to rotation matrices
    R_old, _ = cv2.Rodrigues(rot_vec_old)
    R_new, _ = cv2.Rodrigues(rot_vec_new)

    # Combine the rotations by multiplying the matrices
    R_combined = R_new @ R_old

    # Convert the resulting rotation matrix back to a rotation vector
    rot_vec_combined, _ = cv2.Rodrigues(R_combined)

    return rot_vec_combined

class MeshProjectionGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Mesh Projection GUI")

        # Default values
        self.default_image_path = "/home/borisef/projects/pytorch3D/data/a10.png"
        #self.default_mesh_path = "/home/borisef/projects/pytorch3D/data/cow_mesh/cow.obj"
        self.default_mesh_path = "/home/borisef/projects/pytorch3D/data/bixler/bixler.obj"
        self.default_camera_matrix = [[1000, 0, 400], [0, 1000, 300], [0, 0, 1]]
        self.default_tvec = [0, 0, 100]
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
        self.last_mouse_z = None
        self.last_mouse_yaw = None
        self.last_mouse_pitch = None
        self.last_mouse_roll = None

        # GUI components
        self.create_widgets()

        # Load defaults
        self.load_default_inputs()

        # Bind keys for interactivity
        self.bind_keys()

        # History for undo
        self.history = []

        self.render_projection()
        self.record_current_state()

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
                       command=self.bind_keys).grid(row=5, column=0, sticky="w")
        tk.Radiobutton(self.root, text="Interact RVEC", variable=self.interact_mode, value="rvec",
                       command=self.bind_keys).grid(row=6, column=0, sticky="w")

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

        # Record and Undo buttons
        tk.Button(self.root, text="Record", command=self.record_current_state).grid(row=7, column=0, pady=10)
        tk.Button(self.root, text="Undo", command=self.undo_last_change).grid(row=7, column=1, pady=10)

    def record_current_state(self):
        # Store the current values of tvec, rvec, and camera matrix
        state = {
            "camera_matrix": self.camera_matrix.copy(),
            "tvec": self.tvec.copy(),
            "rvec": self.rvec.copy()
        }
        self.history.append(state)
        self.default_tvec = self.tvec.copy()
        self.default_rvec = self.rvec.copy()
        self.default_camera_matrix = (self.camera_matrix.copy()).tolist()

        print("State recorded.")

    def undo_last_change(self):
        # if self.history:
        #     last_state = self.history.pop()
        #     self.camera_matrix = last_state["camera_matrix"]
        #     self.tvec = last_state["tvec"]
        #     self.rvec = last_state["rvec"]
        #     self.update_entries()
        #     self.render_projection()
        #     self.record_current_state()
        #     print("Reverted to last recorded state.")
        # else:
        #     print("No previous state to undo.")
        self.camera_matrix = np.array(self.default_camera_matrix, dtype=np.float32)

        self.tvec = self.default_tvec.copy()
        self.rvec = self.default_rvec.copy()
        self.update_entries()
        self.render_projection()


    def bind_keys(self):
        self.root.unbind("<Left>")
        self.root.unbind("<Right>")
        self.root.unbind("<Up>")
        self.root.unbind("<Down>")
        self.root.unbind("<minus>")
        self.root.unbind("<plus>")
        self.root.unbind("<KP_Subtract>")
        self.root.unbind("<KP_Add>")

        self.canvas.bind("<ButtonPress-1>", self.on_mouse_press1)
        self.canvas.bind("<ButtonPress-2>", self.on_mouse_press2)
        self.canvas.bind("<ButtonPress-3>", self.on_mouse_press3)
        self.canvas.bind("<B1-Motion>", self.on_mouse_drag1)
        self.canvas.bind("<B2-Motion>", self.on_mouse_drag2)
        self.canvas.bind("<B3-Motion>", self.on_mouse_drag3)

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


    def on_mouse_press1(self, event):
        self.mouse_dragging = True
        self.last_mouse_x = event.x
        self.last_mouse_y = event.y
        self.last_mouse_yaw = event.x

    def on_mouse_press2(self, event):
        self.mouse_dragging = True
        self.last_mouse_z = event.x
        self.last_mouse_y = event.y
        self.last_mouse_pitch = event.y

    def on_mouse_press3(self, event):
        self.mouse_dragging = True
        self.last_mouse_z = event.x
        self.last_mouse_roll = event.x

    def on_mouse_drag1(self, event):
        print("drag 1")
        if self.mouse_dragging:
            dx = event.x - self.last_mouse_x
            dy = event.y - self.last_mouse_y
            dyaw= event.x - self.last_mouse_yaw

            self.last_mouse_x = event.x
            self.last_mouse_y = event.y
            self.last_mouse_yaw = event.x

            increment_value = self.tvec_increment.get() if self.interact_mode.get() == "tvec" else self.rvec_increment.get()
            increment_value = increment_value * 0.1

            if self.interact_mode.get() == "tvec":
                self.update_vector(self.tvec, 0, dx * increment_value)
                self.update_vector(self.tvec, 1, dy * increment_value)
            else:
               # self.update_vector(self.rvec, 1, dyaw * increment_value)
                self.update_vector_rvec(self.rvec, 1, dyaw * increment_value)


    def on_mouse_drag2(self, event):
        print("drag 2")
        if self.mouse_dragging:
            dz = event.y - self.last_mouse_z
            dpitch = event.y - self.last_mouse_pitch
            self.last_mouse_z = event.y
            self.last_mouse_pitch = event.y


            increment_value = self.tvec_increment.get() if self.interact_mode.get() == "tvec" else self.rvec_increment.get()
            increment_value = increment_value * 0.1

            if self.interact_mode.get() == "tvec":
                self.update_vector(self.tvec, 2, dz * increment_value)

            else:

                #self.update_vector(self.rvec, 0, dpitch * increment_value)
                self.update_vector_rvec(self.rvec, 0, dpitch * increment_value)

    def on_mouse_drag3(self, event):
        print("drag 3")
        if self.mouse_dragging:
            dz = event.x - self.last_mouse_z
            droll = event.x - self.last_mouse_roll

            self.last_mouse_z = event.x
            self.last_mouse_roll = event.x


            increment_value = self.tvec_increment.get() if self.interact_mode.get() == "tvec" else self.rvec_increment.get()
            increment_value = increment_value * 0.1

            if self.interact_mode.get() == "tvec":
                self.update_vector(self.tvec, 2, dz * increment_value)

            else:
                #self.update_vector(self.rvec, 2, droll * increment_value)
                self.update_vector_rvec(self.rvec, 2, droll * increment_value)



    def on_mouse_wheel(self, event):
        increment_value = self.tvec_increment.get() if self.interact_mode.get() == "tvec" else self.rvec_increment.get()

        if event.delta > 0:
            self.update_vector(self.tvec if self.interact_mode.get() == "tvec" else self.rvec, 2, increment_value)
        else:
            self.update_vector(self.tvec if self.interact_mode.get() == "tvec" else self.rvec, 2, -increment_value)

    def update_key_bindings(self, event=None):
        """ Rebind keys after the increment value is changed in the combo boxes """
        self.bind_keys()

    def load_default_inputs(self):
        self.image_entry.insert(0, self.default_image_path)
        self.mesh_entry.insert(0, self.default_mesh_path)
        self.camera_matrix_entry.insert(0, str(self.default_camera_matrix))
        self.tvec_x_entry.insert(0, self.default_tvec[0])
        self.tvec_y_entry.insert(0, self.default_tvec[1])
        self.tvec_z_entry.insert(0, self.default_tvec[2])
        self.rvec_x_entry.insert(0, self.default_rvec[0])
        self.rvec_y_entry.insert(0, self.default_rvec[1])
        self.rvec_z_entry.insert(0, self.default_rvec[2])
        self.load_image()
        self.load_mesh()

    def load_image(self):
        file_path = self.image_entry.get()
        if file_path:
            self.image = cv2.imread(file_path)
            self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
            self.display_image()

    def load_mesh(self):
        file_path = self.mesh_entry.get()
        if file_path:
            self.mesh = trimesh.load(file_path)


        if(self.mesh.vertices.shape[0] > MAX_VERTICES):
            d = max(int(self.mesh.vertices.shape[0]/MAX_VERTICES),1)
            self.mesh.vertices = self.mesh.vertices[::d,:]

    def update_vector(self, vector, index, delta):
        vector[index] += delta
        self.update_entries()
        self.render_projection()

    def update_vector_rvec(self, rvec, index, delta,method = "target"):
        #"simple" - increment index
        #"camera" - rotate twice new(old)
        #"target" - rotate twice old(new)
        rvec_new = np.array([0, 0, 0], dtype=np.float32)
        rvec_new[index] = delta
        if(method == "simple"):
            rvec[index] += delta
        if(method =="camera"):
            self.rvec = combine_rot_vecs(self.rvec.copy(),rvec_new)
        if (method == "target"):
            self.rvec = combine_rot_vecs(rvec_new,self.rvec.copy())



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
            # matrix = eval(self.camera_matrix_entry.get())
            # self.camera_matrix = np.array(matrix, dtype=np.float32)
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

        #TODO: set camera
        self.camera_matrix_entry.delete(0, tk.END)
        stri = str(self.camera_matrix.copy().tolist())
        # # stri = s.replace("\n", "")
        self.camera_matrix_entry.insert(0, stri)

        self.render_projection()

    def render_projection(self):
        try:
            camera_matrix_str = self.camera_matrix_entry.get()
            self.camera_matrix = np.array(eval(camera_matrix_str), dtype=np.float32)

            self.tvec = np.array([
                float(self.tvec_x_entry.get()),
                float(self.tvec_y_entry.get()),
                float(self.tvec_z_entry.get())
            ], dtype=np.float32).reshape(3, 1)

            self.rvec = np.array([
                float(self.rvec_x_entry.get()),
                float(self.rvec_y_entry.get()),
                float(self.rvec_z_entry.get())
            ], dtype=np.float32).reshape(3, 1)
        except Exception as e:
            messagebox.showerror("Error", f"Invalid inputs: {e}")
            return

        if self.image is None or self.mesh is None or self.camera_matrix is None:
            messagebox.showerror("Error", "Please load image, mesh, and set camera parameters before rendering.")
            return

        # Project mesh vertices to image plane
        vertices = self.mesh.vertices
        projected_points, _ = cv2.projectPoints(vertices, self.rvec, self.tvec, self.camera_matrix, None)

        # Render
        overlay = self.image.copy()
        for point in projected_points:
            x, y = int(point[0][0]), int(point[0][1])
            cv2.circle(overlay, (x, y), 2, (255, 0, 0), -1)

        self.display_image(overlay)



    def display_image(self, image=None):
        if image is None:
            image = self.image

        image = Image.fromarray(image)
        image = image.resize((800, 600))
        photo = ImageTk.PhotoImage(image)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=photo)
        self.canvas.image = photo  # Keep a reference to avoid garbage collection


if __name__ == "__main__":
    root = tk.Tk()
    app = MeshProjectionGUI(root)
    root.mainloop()
