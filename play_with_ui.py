import tkinter as tk
from cgitb import enable
from tkinter import Canvas, Checkbutton, BooleanVar, Button, Entry, Label, filedialog, font, StringVar, OptionMenu, Radiobutton
from tkinter.tix import *

from PIL import Image, ImageTk
import copy
import pickle
import aux
from utils import *

import numpy as np

#global params
MAX_REPROJECTION_ERROR_3D_to_2D = 3
MIN_POINTS_3D_to_2D = 4
IFOVS_3D_to_2D = np.arange(start = 0.01, stop = 0.2,step = 0.005).tolist()
COLOR_PER_POINT = [(255,0,0),
                   (0,255,0),
                   (0,0,255),
                   (255,255,0),
                   (255,0,255),
                   (0,255,255),
                   (255,128,0),
                   (102, 255, 51)]*10

# Example data
data = [
    {
        "image_path": "/home/borisef/projects/pytorch3D/a10.png",
        "keypoints": {"nose_tip": (100, 150), "left_wing": (200, 150), "right_wing": (150, 200), 'cone_edge': (110,100),
                      "tip_of_vertical_stabilizer":(50,50), "left_tip_of_horizontal_stabilizer": (100,25),
                      "right_tip_of_horizontal_stabilizer": (100,250)}
    },
    {
        "image_path": "/home/borisef/projects/pytorch3D/a101.png",
        "keypoints": {"nose_tip": (120, 150), "left_wing": (210, 150), "right_wing": (160, 200), 'cone_edge': (110,100),
                      "tip_of_vertical_stabilizer":(50,60), "left_tip_of_horizontal_stabilizer": (110,25),
                      "right_tip_of_horizontal_stabilizer": (100,250)}
    }
    # Add more dictionaries as needed
]


data3D = [
    {
        "model_path": "/home/borisef/projects/pytorch3D/data/bixler/bixler.obj",
        "keypoints": {"nose_tip": (0, 16.25, 0.25), "left_wing": (-26, 1.22, 4.55), "right_wing": (26, 1.22, 4.55),
                      'cone_edge': (0,-15,0),
                      "tip_of_vertical_stabilizer": (0,-14.4,5.7),
                      "left_tip_of_horizontal_stabilizer": (-8.13,-15.21,0),
                      "right_tip_of_horizontal_stabilizer": (8.13,-15.21,0)},
        "projection": {"nose_tip": (100,100), "left_wing": (200,200), "right_wing": (300,300), 'cone_edge': (110,100),
                      "tip_of_vertical_stabilizer":(50,50), "left_tip_of_horizontal_stabilizer": (100,25),
                      "right_tip_of_horizontal_stabilizer": (100,250)},
        'valid_projection': False,
        'projection_params': None


    },
    {
        "model_path": "/home/borisef/projects/pytorch3D/data/bixler/bixler.obj",
        "keypoints": {"nose_tip": (0, 16.25, 0.25), "left_wing": (-26, 1.22, 4.55), "right_wing": (26, 1.22, 4.55),
                      'cone_edge': (0,-15,0),
                      "tip_of_vertical_stabilizer": (0,-14.4,5.7),
                      "left_tip_of_horizontal_stabilizer": (-8.13,-15.21,0),
                      "right_tip_of_horizontal_stabilizer": (8.13,-15.21,0)},
        "projection": {"nose_tip": (100,100), "left_wing": (200,200), "right_wing": (300,300), 'cone_edge': (110,100),
                      "tip_of_vertical_stabilizer":(50,50), "left_tip_of_horizontal_stabilizer": (100,25),
                      "right_tip_of_horizontal_stabilizer": (100,250)},
        'valid_projection': False,
        'projection_params': None
    },
    # Add more dictionaries as needed
]


# Projection data
projectionData = {'fov_xy': (-1,-1), 'ypr': (0,0,0), 't_xyz': (None,None,None), 'num_points': None, 'max_error': None }


class ImageKeypointsViewer:
    def __init__(self, root, data, data3D, projectionData):
        self.root = root
        self.data = data
        self.data3D = data3D
        self.projectionData = projectionData
        self.index = 0
        self.zoom_scale = 1.0
        self.translateX = 0
        self.translateY = 0

        self.show_labels = BooleanVar(value=True)
        self.use_all_points = BooleanVar(value=False)
        self.show_mesh = BooleanVar(value=False)
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
        #self.mode_var = StringVar(value="2D")
        #self.mode_frame = tk.Frame(self.control_frame)
        #self.mode_frame.pack(pady=10)
        #Label(self.mode_frame, text="Mode").pack(side=tk.TOP)
        #self.mode_2d_radio = Radiobutton(self.mode_frame, text="2D mode", variable=self.mode_var, value="2D", command=self.update_mode)
        #self.mode_2d_radio.pack(side=tk.LEFT, padx=5)
        #self.mode_3d_radio = Radiobutton(self.mode_frame, text="3D mode", variable=self.mode_var, value="3D", command=self.update_mode)
        #self.mode_3d_radio.pack(side=tk.LEFT, padx=5)

        self.save_button = Button(root, text="Save Changes", command=self.save_changes)
        self.save_button.pack(pady=5)
        self.save_button.place(relx=1.0, rely=1.0, x=-10, y=-60, anchor='se')  # Place button at bottom right
        CustomTooltip(self.save_button, "Save default for current image")


        self.revert_button = Button(root, text="Revert Changes", command=self.revert_changes)
        self.revert_button.pack(pady=5)
        self.revert_button.place(relx=1.0, rely=1.0, x=-10, y=-100, anchor='se')  # Place button at bottom right
        CustomTooltip(self.revert_button, "Back to last saved for current image")

        # Add "Save Results" button with bold font
        bold_font = font.Font(weight="bold")
        self.save_results_button = Button(root, text="Save Results", command=self.save_results, font=bold_font)
        self.save_results_button.place(relx=1.0, rely=1.0, x=-10, y=-10, anchor='se')  # Place button at bottom right
        CustomTooltip(self.save_results_button, "Save all results in file")



        # Add "Overlay 3D" button
        self.overlay_3d_button = Button(root, text="Overlay 3D", command=self.overlay_3d)
        self.overlay_3d_button.pack(pady=5)
        self.overlay_3d_button.place(relx=1.0, rely=1.0, x=-10, y=-140, anchor='se')  # Place button
        CustomTooltip(self.overlay_3d_button, "Compute 3D to 2D projection")

        # Add use all points for projection checkbox
        self.label_checkbox = Checkbutton(root, text="Use all points", variable=self.use_all_points, command=self.set_change_use_all_points)
        self.label_checkbox.place(relx=1.0, rely=1.0, x=-120, y=-140, anchor='se')  # Place checkbox at top right



        # Add projection data input fields
        self.projection_vars = {}
        self.projection_entries = {}
        Label(self.control_frame1, text="Projection Data").pack(pady=5)
        for key in self.projectionData:
            frame = tk.Frame(self.control_frame1)
            frame.pack(fill=tk.X, pady=2)
            label = Label(frame, text=key, width=10)
            label.pack(side=tk.LEFT)
            var = tk.StringVar()
            var.set(str(self.projectionData[key]))
            var.trace('w', lambda name, index, mode, key=key, var=var: self.update_projection_data(key, var))
            entry = Entry(frame, width=15, textvariable=var)
            entry.pack(side=tk.LEFT)
            self.projection_vars[key] = var
            self.projection_entries[key] = entry

        self.label_checkbox = Checkbutton(root, text="Show Labels", variable=self.show_labels, command=self.show_image)
        self.label_checkbox.place(relx=1.0, x=-10, y=10, anchor='ne')  # Place checkbox at top right

        # self.mesh_checkbox = Checkbutton(root, text="Show Mesh", variable=self.show_mesh, command=self.show_image)
        # self.mesh_checkbox.place(relx=1.0, x=-17, y=50, anchor='ne')  # Place checkbox at top right under previous

        self.adjust_2D_to_3D_button = tk.Button(root, text="Adjust 2D to 3D", command=self.set_adjust_2D_to_3D)
        self.adjust_2D_to_3D_button.place(relx=1.0, x=-17, y=150, anchor='ne')
        CustomTooltip(self.adjust_2D_to_3D_button, "Move all 2D markers on top of their 3D projections")

        self.mesh_button = tk.Button(root, text="Show Mesh", command=self.set_show_mesh, state = "disabled")
        self.mesh_button.place(relx=1.0, x=-17, y=50, anchor='ne')

        CustomTooltip(self.mesh_button, "Draw mesh on top of original image")

        # Bind keys for zooming
        self.root.bind("<KeyPress-A>", self.zoom_in)  # Capital 'A'
        self.root.bind("<KeyPress-a>", self.zoom_out)  # Lowercase 'a'

        # Bind keys for translating
        self.root.bind("<Left>", self.translate_left)
        self.root.bind("<Right>", self.translate_right)
        self.root.bind("<Up>", self.translate_up)
        self.root.bind("<Down>", self.translate_down)


        # Bind mouse events for selecting and dragging keypoints
        self.canvas.bind("<Button-1>", self.select_keypoint)  # Left mouse button click
        self.canvas.bind("<B1-Motion>", self.move_keypoint)   # Left mouse button drag

        self.show_image()

    def set_show_mesh(self):
        self.show_mesh.set(True)
        self.show_image()
        self.show_mesh.set(False)

    def set_adjust_2D_to_3D(self):
        self.Addjust2Dto3D()
        self.show_image()

    def set_change_use_all_points(self):
        self.data3D[self.index]["valid_projection"] = False
        self.mesh_button["state"] = "disabled"

    def show_image(self):
        # Clear canvas
        self.canvas.delete("all")

        # Load the image
        image_info = self.data[self.index]
        image_path = image_info["image_path"]
        orig_image_path = image_path
        keypoints = image_info["keypoints"]

        #Load 3D data
        theeD_info = self.data3D[self.index]
        keypoints3D = theeD_info["keypoints"]
        projection3D = theeD_info["projection"]

        #overlay the mesh above if needed
        if(self.data3D[self.index]["valid_projection"] and self.show_mesh.get()):
            projection_params = self.data3D[self.index]['projection_params']
            temp_img_with_mesh = "/home/borisef/projects/pytorch3D/data/output/temp.png"
            temp_vec = np.array([[0, 0, 0, 1]], dtype=np.float32)
            temp = np.concatenate([projection_params['rmat'], projection_params['tvec']], axis=1)
            Rt = np.concatenate([temp, temp_vec], axis=0)
            K = projection_params['camera_matrix']
            aux.render_3d_model_on_image(obj_path = self.data3D[self.index]['model_path'], K = K, Rt = Rt, image_size=None,
                                         output_path= temp_img_with_mesh, input_image=image_path,
                                         rvec = projection_params['rvec'],
                                         tvec = projection_params['tvec'])
            image_path = temp_img_with_mesh


        image = Image.open(image_path)
        image_path = orig_image_path
        # Apply zoom and translate
        width, height = image.size
        new_size = (int(width * self.zoom_scale), int(height * self.zoom_scale))

        image = pil_image_translate(image, self.translateX, self.translateY)# translate

        image = image.resize(new_size, Image.LANCZOS)

        self.photo = ImageTk.PhotoImage(image)

        # Display the image
        self.canvas.create_image(0, 20, anchor=tk.NW, image=self.photo)

        # Display the image path
        self.canvas.create_text(10, 10, anchor=tk.NW, text=image_path, fill="black")

        # Draw keypoints
        self.keypoint_items = {}
        ind = 0
        for name, point in keypoints.items():
            x, y = point
            x = x - self.translateX
            y = y - self.translateY
            x = int(x * self.zoom_scale)
            y = int(y * self.zoom_scale) + 20  # Adjust for text offset
            keypoint_item = self.canvas.create_oval(x-5, y-5, x+5, y+5, fill=rgb_to_hex(COLOR_PER_POINT[ind]), tags=name)
            if self.show_labels.get():
                self.canvas.create_text(x, y-10, text=name, fill="black")
            self.keypoint_items[name] = keypoint_item
            ind = ind+1

        if(self.data3D[self.index]["valid_projection"]):
            ind = 0
            for name, point in projection3D.items():
                x, y = point
                x = x - self.translateX
                y = y - self.translateY
                x = int(x * self.zoom_scale)
                y = int(y * self.zoom_scale) + 20  # Adjust for text offset
                #keypoint_3D_item = self.canvas.create_oval(x-5, y-5, x+5, y+5, fill="green", tags=name)
                self.canvas.create_rectangle((x-3,y-3,x+3,y+3), fill=rgb_to_hex(COLOR_PER_POINT[ind]),outline="black")
                # if self.show_labels.get():
                #     self.canvas.create_text(x, y-10, text=name, fill="black")
                #self.keypoint_items[name] = keypoint_item
                ind = ind + 1

    def update_proj_params(self, proj_params=None):
        entry= self.projection_entries['num_points']
        if(entry.get() is not None):
            entry.delete(0, tk.END)
        if(proj_params is not None):
            entry.insert(0,str(len(proj_params['selected_points'])))
        entry = self.projection_entries['max_error']
        if (entry.get() is not None):
            entry.delete(0, tk.END)
        if (proj_params is not None):
            entry.insert(0, str(proj_params['max_reprojection_error']))
        entry = self.projection_entries['t_xyz']
        if (entry.get() is not None):
            entry.delete(0, tk.END)
        if (proj_params is not None):
            entry.insert(0, str(proj_params['tvec'].flatten()))
        entry = self.projection_entries['fov_xy']
        if (entry.get() is not None):
            entry.delete(0, tk.END)
        if (proj_params is not None):
            entry.insert(0, str(proj_params['fov_xy_deg']))
        entry = self.projection_entries['ypr']
        if (entry.get() is not None):
            entry.delete(0, tk.END)
        if (proj_params is not None):
            entry.insert(0, str(proj_params['ypr_deg']))

    def Addjust2Dto3D(self):
        image_info = self.data[self.index]
        keypoints = image_info["keypoints"]

        theeD_info = self.data3D[self.index]
        projection3D = theeD_info["projection"]

        if (theeD_info['valid_projection']):
            for k in keypoints.keys():
                keypoints[k] = projection3D[k]

            for k in theeD_info['projection_params']:
                if("error" in k):
                    theeD_info['projection_params'][k] = 0
                if(k == 'selected_points'):
                    theeD_info['projection_params'][k] = list(range(len(keypoints)))
                if(k == "success"):
                    theeD_info['projection_params'][k] = True


            # self.data3D[self.index]['projection_params'] = out
            self.update_proj_params(theeD_info['projection_params'])





    def compute_3D_projection(self, min_points_2D_to_3D = MIN_POINTS_3D_to_2D):

        # Load the image
        image_info = self.data[self.index]
        image_path = image_info["image_path"]
        keypoints = image_info["keypoints"]



        image = Image.open(image_path)

        # Apply zoom
        W, H = image.size

        # Load 3D data
        theeD_info = self.data3D[self.index]
        keypoints3D = theeD_info["keypoints"]
        projection3D = theeD_info["projection"]



        # Intrinsic matrix K
        # K = np.array([
        #     [1000, 0, 512],
        #     [0, 1000, 384],
        #     [0, 0, 1]
        # ], dtype=np.float32) #TEMP

        #if no need to re-compute return
        if(theeD_info['valid_projection']):
            return

        #get 2D and 3D points
        NP = len(keypoints)
        object_points = np.zeros((NP,3), dtype=np.float32)
        image_points = np.zeros((NP,2), dtype=np.float32)
        for i,xy_xyz in enumerate(zip(keypoints.values(),keypoints3D.values())):
            object_points[i,:] = xy_xyz[1]
            image_points[i,:]=xy_xyz[0]

        #compute projection with PnP
        # camera_matrix = K
        # rmat, tvec, success, weighted_reprojection_error, avg_reprojection_error, max_reprojection_error, projected_points, rvec = \
        # aux.recover_camera_extrinsics_simple(object_points, image_points, camera_matrix)


        out = aux.run_multiple_recover_extrinsics(object_points,image_points,MAX_REPROJECTION_ERROR_3D_to_2D,
                                                  min_points_2D_to_3D, IFOVS_3D_to_2D,W,H)

        max_reprojection_error = out['max_reprojection_error']
        avg_reprojection_error = out['avg_reprojection_error']
        projected_points = out['all_projected_points'].copy()

        print('avg_reprojection_error' + str(avg_reprojection_error))
        print('max_reprojection_error' + str(max_reprojection_error))
        #TODO: compute error etc
        if(out['success']):
            for i, name in enumerate(projection3D.keys()):
                projection3D[name] = projected_points[i,:]

            theeD_info['projection_params'] = out
            #self.data3D[self.index]['projection_params'] = out
            self.update_proj_params(out)



    def show_prev_image(self):
        self.index = (self.index - 1) % len(self.data)
        self.show_image()

    def show_next_image(self):
        self.index = (self.index + 1) % len(self.data)
        self.show_image()

    def zoom_in(self, event=None):
        self.zoom_scale *= 1.1
        #print("self.zoom_scale =" + str(self.zoom_scale))
        self.show_image()

    def zoom_out(self, event=None):
        self.zoom_scale /= 1.1
        self.show_image()

    def translate_right(self, event=None):
        self.translateX -= 1
        #print("self.translateX =" + str(self.translateX ))
        self.show_image()

    def translate_left(self, event=None):
        self.translateX += 1
        self.show_image()
    def translate_up(self, event=None):
        self.translateY += 1
        self.show_image()

    def translate_down(self, event=None):
        self.translateY -= 1
        self.show_image()

    def select_keypoint(self, event):
        x_click = (event.x - self.translateX) / self.zoom_scale
        y_click = (event.y - self.translateY - 0*20) / self.zoom_scale

        tolerance = 5

        for name, keypoint_item in self.keypoint_items.items():
            bbox = self.canvas.bbox(keypoint_item)
            if bbox:
                x0, y0, x1, y1 = bbox
                x0 = x0 - self.translateX
                x1 = x1 - self.translateX
                y0 = y0 - self.translateY
                y1 = y1 - self.translateY

                x0 /= self.zoom_scale
                y0 = (y0 - 0*20) / self.zoom_scale
                x1 /= self.zoom_scale
                y1 = (y1 - 0*20) / self.zoom_scale

                if (x0 - tolerance <= x_click <= x1 + tolerance and
                    y0 - tolerance <= y_click <= y1 + tolerance):
                    self.selected_keypoint = name
                    break
        else:
            self.selected_keypoint = None

        self.show_image()  # Update display to show selected keypoints

    def move_keypoint(self, event):

       # print('Event:(' + str(int(event.x)) + "," + str(int(event.y)))


        if self.selected_keypoint:
            self.mesh_button["state"] = "disabled"
            x_new = event.x
            y_new = event.y - 20  # Adjust for text offset
            self.canvas.coords(self.keypoint_items[self.selected_keypoint],
                               x_new-5, y_new-5, x_new+5, y_new+5)

            # Update keypoint data
            x_data = ((x_new ) / self.zoom_scale) + self.translateX
            y_data = ((y_new ) - 0*20) / self.zoom_scale + self.translateY
            self.data[self.index]["keypoints"][self.selected_keypoint] = (x_data, y_data)

            self.data3D[self.index]["valid_projection"] = False
            self.update_proj_params()

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
                pickle.dump({'data': self.data, 'data3D': self.data3D}, f)

            #save to json
            save_path_json = save_path.replace(".pkl", ".json")
            save_data_2_json(self.data, self.data3D, save_path_json)

    def update_projection_data(self, key, var):
        try:
            self.projectionData[key] = float(var.get())
            print(f"Updated {key}: {self.projectionData[key]}")
        except ValueError:
            print(f"Invalid input for {key}: {var.get()}")

    def overlay_3d(self):
        print("Overlay 3D button pressed")
        # Placeholder for actual 3D overlay functionality
        if(self.use_all_points.get() == False):
            self.compute_3D_projection()
        else:
            print('Compute projection, use all points')
            nkp = len(self.data[self.index]["keypoints"])
            self.compute_3D_projection(min_points_2D_to_3D=nkp)
        self.data3D[self.index]["valid_projection"] = True
        self.mesh_button["state"] = "normal"
        print('Best projection computed')
        self.show_image()

    # def update_mode(self):
    #     print(f"Mode changed to: {self.mode_var.get()}")
    #     # Placeholder for actual mode switch functionality

if __name__ == "__main__":
    root = tk.Tk()
    viewer = ImageKeypointsViewer(root, data, data3D, projectionData)
    root.mainloop()
