import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import webbrowser

class RobotLampUI:
    def __init__(self, root, on_start, on_stop, on_close, on_rest):
        """
        :param root: The Tk root window.
        :param on_start: Callback for when the Start button is clicked.
        :param on_stop: Callback for when the Stop button is clicked.
        :param on_close: Callback for when the window is closed.
        :param logo_path: Path to the logo image file.
        :param hyperlink: URL for the footer hyperlink.
        :param hyperlink_text: Text to display for the hyperlink.
        """
        self.root = root
        self.on_start = on_start
        self.on_stop = on_stop
        self.on_close = on_close
        self.on_rest = on_rest
        self.hyperlink = 'MeVirtuoso.com'
        self.hyperlink_text = 'MeVirtuoso.com'

        self._setup_ui()

    def _setup_ui(self):
        # Set up the main window
        self.root.title("Robot Lamp")
        self.root.geometry("800x650")
        self.root.configure(bg="#2b2b2b")

        # Set up ttk styling for a modern look
        style = ttk.Style(self.root)
        style.theme_use("clam")
        style.configure("Dark.TFrame", background="#2b2b2b")
        style.configure("Dark.TLabel", background="#2b2b2b", foreground="white")
        style.configure("Dark.TButton", font=("Helvetica", 11, "bold"), padding=10)
        style.configure("Start.TButton", background="#4CAF50", foreground="white")
        style.map("Start.TButton", background=[("active", "#45a049")])
        style.configure("Stop.TButton", background="#f44336", foreground="white")
        style.map("Stop.TButton", background=[("active", "#e53935")])
        style.configure("Rest.TButton", background="grey", foreground="white")
        style.map("Rest.TButton", background=[("active", "#333333")])


        # Configure grid rows and columns:
        # Row 0: header (logo + title)
        # Row 1: video feed area
        # Row 2: buttons
        # Row 3: buttons
        # Row 4: footer (hyperlink)
        self.root.grid_rowconfigure(1, minsize=500)  # video feed area
        self.root.grid_rowconfigure(2, minsize=50)   # buttons area
        self.root.grid_rowconfigure(3, minsize=50)   # buttons area
        self.root.grid_columnconfigure(0, weight=1) 
        self.root.grid_columnconfigure(1, weight=1)

        # 2) VIDEO FEED
        self.video_frame = ttk.Frame(self.root, style="Dark.TFrame", width=780, height=500)
        self.video_frame.grid(row=1, column=0, columnspan=2, padx=10, pady=10)
        self.video_frame.grid_propagate(False)

        self.video_label = ttk.Label(self.video_frame, style="Dark.TLabel")
        self.video_label.pack(fill="both", expand=True)

        # 3) BUTTONS
        self.start_button = ttk.Button(self.root, text="Start Camera", style="Start.TButton", command=self.on_start)
        self.start_button.grid(row=2, column=0, padx=10, pady=10, sticky="nsew")

        self.stop_button = ttk.Button(self.root, text="Stop Camera", style="Stop.TButton", command=self.on_stop)
        self.stop_button.grid(row=2, column=1, padx=10, pady=10, sticky="nsew")

        self.rest_button = ttk.Button(self.root, text="Rest", style="Rest.TButton", command=self.on_rest)
        self.rest_button.grid(row=3, column=0, columnspan=2 ,padx=80, pady=10, sticky="nsew")

        # 4) FOOTER (Hyperlink)
        self.footer_label = ttk.Label(self.root, text=self.hyperlink_text, style="Dark.TLabel", cursor="hand2")
        self.footer_label.grid(row=4, column=0, columnspan=2, pady=10)
        self.footer_label.bind("<Button-1>", self._open_link)

        # Bind the window close event
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def update_video_image(self, image):
        """Update the video label with a new image (ImageTk.PhotoImage)."""
        self.video_label.config(image=image)
        self.video_label.imgtk = image  # Prevent garbage collection

    def _open_link(self, event=None):
        """Open the hyperlink in the default web browser."""
        import webbrowser
        webbrowser.open_new_tab(self.hyperlink)


if __name__ == "__main__":
    root = tk.Tk()

     # Instantiate the UI, passing in the callback functions
    ui = RobotLampUI(
        root, None, None, None, None
    )
    root.mainloop()