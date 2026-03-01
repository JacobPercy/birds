import tkinter as tk
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageDraw, ImageOps, ImageTk


MODEL_SIZE = 256
CANVAS_SIZE = 512

DILATION_KERNEL_SIZE = 3
DILATION_ITERATIONS = 1
MODEL_LINE_WIDTH = DILATION_KERNEL_SIZE
DISPLAY_LINE_WIDTH = max(1, round(MODEL_LINE_WIDTH * CANVAS_SIZE / MODEL_SIZE))

ROOT = Path(__file__).resolve().parent
CHECKPOINT_PATH = ROOT / "models" / "pix2pix_birds2_best.pt"

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


class UNetDown(nn.Module):
    def __init__(self, in_c, out_c, normalize=True):
        super().__init__()
        layers = [nn.Conv2d(in_c, out_c, 4, 2, 1, bias=False)]
        if normalize:
            layers.append(nn.BatchNorm2d(out_c))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class UNetUp(nn.Module):
    def __init__(self, in_c, out_c, dropout=0.0):
        super().__init__()
        layers = [
            nn.ConvTranspose2d(in_c, out_c, 4, 2, 1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        ]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x, skip):
        x = self.model(x)
        return torch.cat((x, skip), 1)


class GeneratorUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.d1 = UNetDown(3, 64, normalize=False)
        self.d2 = UNetDown(64, 128)
        self.d3 = UNetDown(128, 256)
        self.d4 = UNetDown(256, 512)
        self.d5 = UNetDown(512, 512)
        self.d6 = UNetDown(512, 512)
        self.d7 = UNetDown(512, 512)
        self.d8 = UNetDown(512, 512, normalize=False)

        self.u1 = UNetUp(512, 512, 0.5)
        self.u2 = UNetUp(1024, 512, 0.5)
        self.u3 = UNetUp(1024, 512, 0.5)
        self.u4 = UNetUp(1024, 512)
        self.u5 = UNetUp(1024, 256)
        self.u6 = UNetUp(512, 128)
        self.u7 = UNetUp(256, 64)

        self.final = nn.Sequential(
            nn.ConvTranspose2d(128, 3, 4, 2, 1),
            nn.Tanh()
        )

    def forward(self, x):
        d1 = self.d1(x)
        d2 = self.d2(d1)
        d3 = self.d3(d2)
        d4 = self.d4(d3)
        d5 = self.d5(d4)
        d6 = self.d6(d5)
        d7 = self.d7(d6)
        d8 = self.d8(d7)

        u1 = self.u1(d8, d7)
        u2 = self.u2(u1, d6)
        u3 = self.u3(u2, d5)
        u4 = self.u4(u3, d4)
        u5 = self.u5(u4, d3)
        u6 = self.u6(u5, d2)
        u7 = self.u7(u6, d1)

        return self.final(u7)


def load_generator(checkpoint_path: Path):
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}")

    model = GeneratorUNet().to(DEVICE)
    ckpt = torch.load(checkpoint_path, map_location=DEVICE)

    if isinstance(ckpt, dict) and "G" in ckpt:
        state = ckpt["G"]
    else:
        state = ckpt

    if isinstance(state, dict):
        fixed = {}
        for k, v in state.items():
            if k.startswith("module."):
                fixed[k[len("module."):]] = v
            else:
                fixed[k] = v
        state = fixed

    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def preprocess_canvas_for_model(canvas_img_l: Image.Image):
    x = canvas_img_l.resize((MODEL_SIZE, MODEL_SIZE), Image.Resampling.BICUBIC)
    x = ImageOps.invert(x).convert("RGB")
    arr = np.asarray(x, dtype=np.float32) / 255.0
    t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    t = t * 2.0 - 1.0
    return t.to(DEVICE)


def postprocess_model_output(t: torch.Tensor):
    t = t.detach().cpu()
    if t.dim() == 4:
        t = t[0]
    t = (t * 0.5 + 0.5).clamp(0, 1)
    arr = (t.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    return Image.fromarray(arr)


class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Bird Sketch to Image")

        self.model = load_generator(CHECKPOINT_PATH)

        self.pending_generate = None
        self.prev_xy = None
        self.stroke_had_motion = False
        self.busy = False

        self.left_img = Image.new("L", (CANVAS_SIZE, CANVAS_SIZE), 255)
        self.left_draw = ImageDraw.Draw(self.left_img)

        self.right_img = Image.new("RGB", (CANVAS_SIZE, CANVAS_SIZE), (245, 245, 245))
        self.right_photo = None

        outer = tk.Frame(root)
        outer.pack(padx=12, pady=12)

        title = tk.Label(outer, text="Bird Sketch to Image", font=("Helvetica", 16, "bold"))
        title.grid(row=0, column=0, columnspan=2, pady=(0, 10))

        panels = tk.Frame(outer)
        panels.grid(row=1, column=0, columnspan=2)

        left_frame = tk.Frame(panels)
        left_frame.grid(row=0, column=0, padx=(0, 12))

        right_frame = tk.Frame(panels)
        right_frame.grid(row=0, column=1)

        tk.Label(left_frame, text="Draw (white bg, black lines)").pack(pady=(0, 6))
        self.canvas = tk.Canvas(
            left_frame,
            width=CANVAS_SIZE,
            height=CANVAS_SIZE,
            bg="white",
            highlightthickness=1,
            highlightbackground="#888"
        )
        self.canvas.pack()

        tk.Label(right_frame, text="Generated").pack(pady=(0, 6))
        self.out_label = tk.Label(
            right_frame,
            width=CANVAS_SIZE,
            height=CANVAS_SIZE,
            bd=1,
            relief="solid",
            bg="white"
        )
        self.out_label.pack()
        self.update_output_panel(self.right_img)

        controls = tk.Frame(outer)
        controls.grid(row=2, column=0, columnspan=2, pady=(10, 0))

        tk.Button(controls, text="Generate", width=12, command=self.generate_once).pack(side=tk.LEFT, padx=4)
        tk.Button(controls, text="Clear", width=12, command=self.clear).pack(side=tk.LEFT, padx=4)

        self.status = tk.Label(outer, text=f"device: {DEVICE.type} | line width: {DISPLAY_LINE_WIDTH}px")
        self.status.grid(row=3, column=0, columnspan=2, pady=(8, 0))

        self.canvas.bind("<Button-1>", self.on_press)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)

    def update_output_panel(self, pil_img_rgb: Image.Image):
        shown = pil_img_rgb.resize((CANVAS_SIZE, CANVAS_SIZE), Image.Resampling.NEAREST)
        self.right_photo = ImageTk.PhotoImage(shown)
        self.out_label.configure(image=self.right_photo)

    def draw_dot(self, x, y):
        r = DISPLAY_LINE_WIDTH // 2
        self.canvas.create_oval(x - r, y - r, x + r, y + r, fill="black", outline="black")
        self.left_draw.ellipse((x - r, y - r, x + r, y + r), fill=0, outline=0)

    def draw_segment(self, x0, y0, x1, y1):
        self.canvas.create_line(
            x0, y0, x1, y1,
            fill="black",
            width=DISPLAY_LINE_WIDTH,
            capstyle=tk.ROUND,
            joinstyle=tk.ROUND,
            smooth=True
        )
        self.left_draw.line((x0, y0, x1, y1), fill=0, width=DISPLAY_LINE_WIDTH)

    def on_press(self, event):
        self.prev_xy = (event.x, event.y)
        self.stroke_had_motion = False

    def on_drag(self, event):
        if self.prev_xy is None:
            self.prev_xy = (event.x, event.y)
            return
        x0, y0 = self.prev_xy
        x1, y1 = event.x, event.y
        self.draw_segment(x0, y0, x1, y1)
        self.prev_xy = (x1, y1)
        self.stroke_had_motion = True

    def on_release(self, event):
        if self.prev_xy is not None and not self.stroke_had_motion:
            self.draw_dot(event.x, event.y)
        self.prev_xy = None
        self.schedule_generate()

    def schedule_generate(self, delay_ms=60):
        if self.pending_generate is not None:
            self.root.after_cancel(self.pending_generate)
        self.pending_generate = self.root.after(delay_ms, self.generate_once)

    def clear(self):
        self.canvas.delete("all")
        self.left_img = Image.new("L", (CANVAS_SIZE, CANVAS_SIZE), 255)
        self.left_draw = ImageDraw.Draw(self.left_img)
        self.right_img = Image.new("RGB", (CANVAS_SIZE, CANVAS_SIZE), (245, 245, 245))
        self.update_output_panel(self.right_img)
        self.status.configure(text=f"device: {DEVICE.type} | line width: {DISPLAY_LINE_WIDTH}px")

    def generate_once(self):
        if self.busy:
            return
        self.pending_generate = None
        self.busy = True
        self.status.configure(text="generating...")
        self.root.update_idletasks()

        try:
            x = preprocess_canvas_for_model(self.left_img)
            with torch.inference_mode():
                y = self.model(x)
            out = postprocess_model_output(y)
            self.right_img = out
            self.update_output_panel(out)
            self.status.configure(text=f"device: {DEVICE.type} | generated")
        except Exception as e:
            self.status.configure(text=f"error: {e}")
        finally:
            self.busy = False


if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()