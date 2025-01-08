import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk


class MainApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Main GUI")

        # Canvas to display the image
        self.canvas = tk.Canvas(self.root, width=500, height=500, bg="gray")
        self.canvas.pack(pady=10)
        self.canvas.bind("<Button-1>", self.get_main_canvas_coordinates)

        # Button to open an image
        self.open_btn = tk.Button(self.root, text="Open Image", command=self.open_image)
        self.open_btn.pack()

        # Text box to show coordinates
        self.coord_entry = tk.Entry(self.root)
        self.coord_entry.pack(pady=10)

        # Button to open secondary GUI
        self.secondary_btn = tk.Button(self.root, text="Open Secondary GUI", command=self.open_secondary_gui)
        self.secondary_btn.pack()

        self.image = None  # Store the loaded image

    def open_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.png;*.jpeg;*.bmp")])
        if file_path:
            # Load and display the image
            pil_image = Image.open(file_path)
            self.image = ImageTk.PhotoImage(pil_image.resize((500, 500)))
            self.canvas.create_image(0, 0, anchor=tk.NW, image=self.image)

    def open_secondary_gui(self):
        if self.image:
            secondary_app = SecondaryApp(self.root, self.image, self.update_coordinates)
            self.root.wait_window(secondary_app.master)  # Make main GUI wait until secondary is closed

    def update_coordinates(self, coords):
        self.coord_entry.delete(0, tk.END)
        self.coord_entry.insert(0, f"{coords[0]}, {coords[1]}")

    def get_main_canvas_coordinates(self, event):
        # Get the coordinates of the click on the main canvas
        coords = (event.x, event.y)
        self.update_coordinates(coords)


class SecondaryApp:
    def __init__(self, master, image, callback):
        self.master = tk.Toplevel(master)
        self.master.title("Secondary GUI")
        self.master.grab_set()  # Make the secondary GUI modal

        self.image = image
        self.callback = callback
        self.coords = (0, 0)

        # Canvas to display the image
        self.canvas = tk.Canvas(self.master, width=500, height=500, bg="gray")
        self.canvas.pack(pady=10)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.image)

        # Bind mouse click event to get coordinates
        self.canvas.bind("<Button-1>", self.get_coordinates)

        # Text box to display the coordinates
        self.coord_entry = tk.Entry(self.master)
        self.coord_entry.pack(pady=10)

        # Done button to return coordinates to main GUI
        self.done_btn = tk.Button(self.master, text="Done", command=self.done)
        self.done_btn.pack()

    def get_coordinates(self, event):
        # Get the coordinates of the click
        self.coords = (event.x, event.y)
        self.coord_entry.delete(0, tk.END)
        self.coord_entry.insert(0, f"{self.coords[0]}, {self.coords[1]}")

    def done(self):
        # Pass the coordinates back to the main GUI and close
        self.callback(self.coords)
        self.master.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = MainApp(root)
    root.mainloop()
