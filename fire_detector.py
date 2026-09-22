import os
import random
import threading
import urllib.request
import zipfile
import shutil
import base64
from collections import deque

import numpy as np
import cv2
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk

import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras import layers, Sequential
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
DATASET_ROOT = os.path.join(BASE_DIR, "fire_classification_dataset")
LAST_FOLDER_FILE = os.path.join(BASE_DIR, "last_tag_folder.txt")

TRAIN_FIRE_DIR = os.path.join(DATASET_ROOT, "train", "fire")
TRAIN_NOFIRE_DIR = os.path.join(DATASET_ROOT, "train", "no_fire")
TEST_FIRE_DIR = os.path.join(DATASET_ROOT, "test", "fire")
TEST_NOFIRE_DIR = os.path.join(DATASET_ROOT, "test", "no_fire")

for d in [OUTPUT_DIR, TRAIN_FIRE_DIR, TRAIN_NOFIRE_DIR, TEST_FIRE_DIR, TEST_NOFIRE_DIR]:
    os.makedirs(d, exist_ok=True)

IMG_SIZE = (96, 96)
MODEL_PATH = os.path.join(OUTPUT_DIR, "fire_classifier.keras")
PUBLIC_DATASET_URL = "https://github.com/DeepQuestAI/Fire-Smoke-Dataset/releases/download/v1/FIRE-SMOKE-DATASET.zip"

fire_model = None
app_window = None

#  پالت رنگی و استایل ظاهری برنامه
COLOR_BG = "#1b2430"        # سرمه‌ای تیره (نوار بالا/ناوبری)
COLOR_BG_LIGHT = "#f4f5f7"  # خاکستری روشن (پس‌زمینه‌ی محتوا)
COLOR_ACCENT = "#e64a19"    # نارنجی‌آتشی (رنگ اصلی برند)
COLOR_GREEN = "#2e7d32"
COLOR_TEXT = "#212121"
COLOR_MUTED = "#6b7280"
FONT_H2 = ("Segoe UI", 11, "bold")
FONT_BODY = ("Segoe UI", 10)
FONT_BTN = ("Segoe UI", 11, "bold")

# ---------------------------------------------------------------------
#  تصویر پس‌زمینه‌ی آتشین برای نوار بالای برنامه. این عکس مستقیماً
#  به‌صورت base64 داخل خود کد جاسازی شده (نه فایل جدا)، تا برنامه
#  کاملاً آفلاین و خودکفا بمونه و نیازی به فایل کنار برنامه نداشته
#  باشه. اگه به هر دلیلی (فایل خراب، حافظه کم) لود نشه، برنامه بدون
#  مشکل با پس‌زمینه‌ی رنگ ساده‌ی COLOR_BG ادامه می‌ده.
# ---------------------------------------------------------------------
_HEADER_BG_B64 = ""  # placeholder -- filled below via file read at build time
_HEADER_BG_PATH = os.path.join(OUTPUT_DIR, "_header_fire.jpg")


def _ensure_header_background():
    """
    عکس آتشِ جاسازی‌شده رو یک‌بار روی دیسک (تو پوشه‌ی outputs) می‌نویسه
    تا PIL/Tk بتونن بازش کنن. اگه قبلاً نوشته شده، دوباره نمی‌نویسه.
    """
    try:
        if os.path.exists(_HEADER_BG_PATH) and os.path.getsize(_HEADER_BG_PATH) > 0:
            return _HEADER_BG_PATH
        if not _HEADER_BG_B64:
            return None
        data = base64.b64decode(_HEADER_BG_B64)
        with open(_HEADER_BG_PATH, "wb") as f:
            f.write(data)
        return _HEADER_BG_PATH
    except Exception:
        return None


def imread_unicode(path):
    try:
        data = np.fromfile(path, dtype=np.uint8)
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    except Exception:
        return None


def imwrite_unicode(path, img):
    try:
        ext = os.path.splitext(path)[1] or ".jpg"
        ok, encoded = cv2.imencode(ext, img)
        if not ok:
            return False
        encoded.tofile(path)
        return True
    except Exception:
        return False


def list_images(folder):
    if not os.path.isdir(folder):
        return []
    exts = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
    return [os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(exts)]


def find_all_images_recursive(folder):
    exts = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
    paths = []
    for dirpath, _, filenames in os.walk(folder):
        for fn in filenames:
            if fn.lower().endswith(exts):
                paths.append(os.path.join(dirpath, fn))
    return sorted(paths)


def build_label_index(root_folder):
    index = {}
    for dirpath, _, filenames in os.walk(root_folder):
        for fn in filenames:
            if fn.lower().endswith('.txt'):
                stem = os.path.splitext(fn)[0].lower()
                index[stem] = os.path.join(dirpath, fn)
    return index

#تعمیم کلی میده مختصات عوض بشه کجاس
def parse_label_file(path, img_w, img_h):
    boxes = []
    if not path or not os.path.exists(path):
        return boxes
    try:
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = [ln.strip() for ln in f if ln.strip()]
    except Exception:
        return boxes

    for line in lines:
        parts = line.replace(',', ' ').split()
        nums = []
        for p in parts:
            try:
                nums.append(float(p))
            except ValueError:
                pass
        if len(nums) < 4:
            continue
        vals = nums[1:5] if len(nums) >= 5 else nums[:4]
        if all(0.0 <= v <= 1.0 for v in vals):
            xc, yc, w, h = vals
            x1 = (xc - w / 2) * img_w
            y1 = (yc - h / 2) * img_h
            x2 = (xc + w / 2) * img_w
            y2 = (yc + h / 2) * img_h
        else:
            x1, y1, x2, y2 = nums[-4:]
        x1, x2 = sorted([max(0.0, min(x1, img_w)), max(0.0, min(x2, img_w))])
        y1, y2 = sorted([max(0.0, min(y1, img_h)), max(0.0, min(y2, img_h))])
        if x2 - x1 < 2 or y2 - y1 < 2:
            continue
        boxes.append((int(x1), int(y1), int(x2), int(y2)))
    return boxes


def count_dataset():
    return {
        "train_fire": len(list_images(TRAIN_FIRE_DIR)),
        "train_no_fire": len(list_images(TRAIN_NOFIRE_DIR)),
        "test_fire": len(list_images(TEST_FIRE_DIR)),
        "test_no_fire": len(list_images(TEST_NOFIRE_DIR)),
    }


def refresh_dataset_count():
    c = count_dataset()
    dataset_count_label.config(
        text=(f"آموزش -> آتش: {c['train_fire']}   |   بدون آتش: {c['train_no_fire']}      "
              f"آزمون -> آتش: {c['test_fire']}   |   بدون آتش: {c['test_no_fire']}")
    )



#  ابزار برچسب‌گذاری دستی سریع


class QuickTagger:
    def __init__(self, parent):
        self.image_paths = []
        self.index = 0
        self.max_w, self.max_h = 640, 400
        self.tk_img = None

        top = tk.Frame(parent, bg=COLOR_BG_LIGHT)
        top.pack(fill="x", pady=8, padx=10)
        tk.Button(top, text="انتخاب پوشه‌ی تصاویر خام", font=FONT_BODY, bg=COLOR_ACCENT, fg="white",
                  activebackground="#c43e12", relief="flat", padx=10, pady=4,
                  command=self.choose_folder).pack(side="right", padx=5)
        self.info_label = tk.Label(top, text="پوشه‌ای انتخاب نشده", bg=COLOR_BG_LIGHT, fg=COLOR_MUTED,
                                    font=FONT_BODY)
        self.info_label.pack(side="right", padx=10)

        self.target_var = tk.StringVar(value="train")
        target_frame = tk.Frame(parent, bg=COLOR_BG_LIGHT)
        target_frame.pack(pady=4)
        tk.Label(target_frame, text="ذخیره در:", bg=COLOR_BG_LIGHT, font=FONT_BODY).pack(side="right")
        tk.Radiobutton(target_frame, text="آموزش", variable=self.target_var, value="train",
                        bg=COLOR_BG_LIGHT, font=FONT_BODY).pack(side="right")
        tk.Radiobutton(target_frame, text="آزمون", variable=self.target_var, value="test",
                        bg=COLOR_BG_LIGHT, font=FONT_BODY).pack(side="right")

        frame_border = tk.Frame(parent, bg=COLOR_BG, padx=3, pady=3)
        frame_border.pack(pady=10)
        self.image_label = tk.Label(frame_border, bg="#20242c")
        self.image_label.pack()

        self.status_label = tk.Label(parent, text="", bg=COLOR_BG_LIGHT, fg=COLOR_MUTED, font=FONT_BODY)
        self.status_label.pack(pady=3)

        btns = tk.Frame(parent, bg=COLOR_BG_LIGHT)
        btns.pack(pady=10)
        tk.Button(btns, text="این عکس آتش دارد", font=FONT_BTN, bg=COLOR_ACCENT, fg="white",
                  activebackground="#c43e12", relief="flat", padx=14, pady=8,
                  command=lambda: self.tag(True)).pack(side="right", padx=8)
        tk.Button(btns, text="این عکس آتش ندارد", font=FONT_BTN, bg=COLOR_GREEN, fg="white",
                  activebackground="#245c27", relief="flat", padx=14, pady=8,
                  command=lambda: self.tag(False)).pack(side="right", padx=8)

        nav = tk.Frame(parent, bg=COLOR_BG_LIGHT)
        nav.pack(pady=5)
        tk.Button(nav, text="< قبلی (بدون برچسب)", font=FONT_BODY, relief="flat", padx=8, pady=3,
                  command=self.prev_img).pack(side="right", padx=5)
        tk.Button(nav, text="بعدی (رد شود) >", font=FONT_BODY, relief="flat", padx=8, pady=3,
                  command=self.next_img).pack(side="right", padx=5)

        self._load_last_folder()

    def _load_last_folder(self):
        if os.path.exists(LAST_FOLDER_FILE):
            try:
                with open(LAST_FOLDER_FILE, "r", encoding="utf-8") as f:
                    folder = f.read().strip()
                if folder and os.path.isdir(folder):
                    self.image_paths = find_all_images_recursive(folder)
                    self.index = 0
                    self.info_label.config(text=f"{len(self.image_paths)} تصویر بارگذاری شد (پوشه‌ی قبلی)")
                    self.show_current()
            except Exception:
                pass

    def choose_folder(self):
        folder = filedialog.askdirectory(title="انتخاب پوشه‌ی تصاویر")
        if not folder:
            return
        self.image_paths = find_all_images_recursive(folder)
        self.index = 0
        self.info_label.config(text=f"{len(self.image_paths)} تصویر بارگذاری شد")
        try:
            with open(LAST_FOLDER_FILE, "w", encoding="utf-8") as f:
                f.write(folder)
        except Exception:
            pass
        self.show_current()

    def show_current(self):
        if not self.image_paths:
            return
        path = self.image_paths[self.index]
        img = imread_unicode(path)
        if img is None:
            self.next_img()
            return
        h, w = img.shape[:2]
        scale = min(self.max_w / w, self.max_h / h, 1.0) or 1.0
        disp = cv2.resize(img, (max(1, int(w * scale)), max(1, int(h * scale))))
        disp_rgb = cv2.cvtColor(disp, cv2.COLOR_BGR2RGB)
        self.tk_img = ImageTk.PhotoImage(Image.fromarray(disp_rgb))
        self.image_label.config(image=self.tk_img)
        self.status_label.config(text=f"{self.index + 1} از {len(self.image_paths)} -- {os.path.basename(path)}")

    def tag(self, is_fire):
        if not self.image_paths:
            return
        path = self.image_paths[self.index]
        img = imread_unicode(path)
        if img is not None:
            img_r = cv2.resize(img, IMG_SIZE)
            split = self.target_var.get()
            dest = (TRAIN_FIRE_DIR if is_fire else TRAIN_NOFIRE_DIR) if split == "train" else \
                   (TEST_FIRE_DIR if is_fire else TEST_NOFIRE_DIR)
            base = f"manual_{split}_{self.index}_{random.randint(0, 999999)}.jpg"
            imwrite_unicode(os.path.join(dest, base), img_r)
            refresh_dataset_count()
        self.next_img()

    def next_img(self):
        if not self.image_paths:
            return
        self.index = (self.index + 1) % len(self.image_paths)
        self.show_current()

    def prev_img(self):
        if not self.image_paths:
            return
        self.index = (self.index - 1) % len(self.image_paths)
        self.show_current()


#  درون‌ریزی دیتاست آماده (تصویر + txt)

def import_ready_classification(folder, dest_fire_dir, dest_nofire_dir, prefix, status_setter=None):
    images = find_all_images_recursive(folder)
    label_index = build_label_index(folder)
    total = len(images)
    n_fire, n_nofire, n_err = 0, 0, 0

    if status_setter:
        status_setter(f"{total} تصویر و {len(label_index)} فایل txt در «{folder}» پیدا شد...")

    for i, img_path in enumerate(images):
        img = imread_unicode(img_path)
        if img is None:
            n_err += 1
            continue
        h, w = img.shape[:2]
        stem = os.path.splitext(os.path.basename(img_path))[0].lower()
        label_path = label_index.get(stem)
        boxes = parse_label_file(label_path, w, h) if label_path else []
        is_fire = len(boxes) > 0

        img_r = cv2.resize(img, IMG_SIZE)
        dest_dir = dest_fire_dir if is_fire else dest_nofire_dir
        base = f"{prefix}_{i}.jpg"
        imwrite_unicode(os.path.join(dest_dir, base), img_r)
        if is_fire:
            n_fire += 1
        else:
            n_nofire += 1

        if status_setter and (i % 100 == 0 or i == total - 1):
            status_setter(f"در حال پردازش {prefix} ... {i + 1} از {total}")

    return {"total": total, "fire": n_fire, "no_fire": n_nofire, "errors": n_err, "labels_found": len(label_index)}


def choose_ready_train_folder():
    global selected_ready_train_folder
    folder = filedialog.askdirectory(title="انتخاب پوشه‌ی آموزش (تصویر + txt)")
    if folder:
        selected_ready_train_folder = folder
        ready_train_label.config(text=f"پوشه‌ی آموزش:\n{folder}")


def choose_ready_test_folder():
    global selected_ready_test_folder
    folder = filedialog.askdirectory(title="انتخاب پوشه‌ی آزمون (تصویر + txt)")
    if folder:
        selected_ready_test_folder = folder
        ready_test_label.config(text=f"پوشه‌ی آزمون:\n{folder}")


def _set_ready_status(text):
    app_window.after(0, lambda: ready_status_label.config(text=text))


def _run_ready_import():
    reports = []
    try:
        if selected_ready_train_folder:
            _set_ready_status("شروع درون‌ریزی پوشه‌ی آموزش...")
            r = import_ready_classification(selected_ready_train_folder, TRAIN_FIRE_DIR, TRAIN_NOFIRE_DIR,
                                             "ready_train", _set_ready_status)
            reports.append(("آموزش", r))
        if selected_ready_test_folder:
            _set_ready_status("شروع درون‌ریزی پوشه‌ی آزمون...")
            r = import_ready_classification(selected_ready_test_folder, TEST_FIRE_DIR, TEST_NOFIRE_DIR,
                                             "ready_test", _set_ready_status)
            reports.append(("آزمون", r))

        if not reports:
            app_window.after(0, lambda: messagebox.showwarning("خطا", "هیچ پوشه‌ای انتخاب نشده بود."))
            return

        lines = []
        for name, r in reports:
            lines.append(f"{name}: {r['total']} تصویر و {r['labels_found']} فایل txt پیدا شد | "
                          f"آتش: {r['fire']} | بدون آتش: {r['no_fire']} | خطا: {r['errors']}")
        _set_ready_status("درون‌ریزی کامل شد.")
        app_window.after(0, refresh_dataset_count)
        app_window.after(0, lambda: messagebox.showinfo("گزارش درون‌ریزی", "\n".join(lines)))
    except Exception as e:
        import traceback
        detail = traceback.format_exc()[-600:]
        app_window.after(0, lambda: messagebox.showerror("خطا در درون‌ریزی", f"{e}\n\n{detail}"))
    finally:
        app_window.after(0, lambda: ready_import_button.config(state="normal"))


def start_ready_import():
    if not selected_ready_train_folder and not selected_ready_test_folder:
        messagebox.showwarning("خطا", "حداقل یکی از پوشه‌های آموزش یا آزمون را انتخاب کنید.")
        return
    ready_import_button.config(state="disabled")
    threading.Thread(target=_run_ready_import, daemon=True).start()


def _set_public_status(text):
    app_window.after(0, lambda: public_status_label.config(text=text))


def _public_hook(block_num, block_size, total_size):
    if total_size > 0:
        pct = min(100.0, block_num * block_size * 100.0 / total_size)
        _set_public_status(f"در حال دانلود... {pct:.0f}٪")


def _run_public_download(limit_per_class):
    zip_path = os.path.join(DATASET_ROOT, "_public.zip")
    extract_dir = os.path.join(DATASET_ROOT, "_public_raw")
    try:
        if not os.path.exists(zip_path):
            _set_public_status("شروع دانلود دیتاست عمومی...")
            urllib.request.urlretrieve(PUBLIC_DATASET_URL, zip_path, reporthook=_public_hook)

        _set_public_status("در حال استخراج...")
        if os.path.isdir(extract_dir):
            shutil.rmtree(extract_dir)
        os.makedirs(extract_dir, exist_ok=True)
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(extract_dir)

        root_folder = os.path.join(extract_dir, "FIRE-SMOKE-DATASET")
        counts = {"train_fire": 0, "train_no_fire": 0, "test_fire": 0, "test_no_fire": 0}

        mapping = [("Train", "train"), ("Test", "test")]
        for src_split, dst_split in mapping:
            fire_src = os.path.join(root_folder, src_split, "Fire")
            dest_fire = TRAIN_FIRE_DIR if dst_split == "train" else TEST_FIRE_DIR
            dest_nofire = TRAIN_NOFIRE_DIR if dst_split == "train" else TEST_NOFIRE_DIR

            if os.path.isdir(fire_src):
                for i, fname in enumerate(sorted(os.listdir(fire_src))[:limit_per_class]):
                    img = imread_unicode(os.path.join(fire_src, fname))
                    if img is None:
                        continue
                    imwrite_unicode(os.path.join(dest_fire, f"public_{src_split}_{i}.jpg"),
                                     cv2.resize(img, IMG_SIZE))
                    counts[f"{dst_split}_fire"] += 1

            for neg_class in ["Neutral", "Smoke"]:
                neg_src = os.path.join(root_folder, src_split, neg_class)
                if os.path.isdir(neg_src):
                    n = max(1, limit_per_class // 2)
                    for i, fname in enumerate(sorted(os.listdir(neg_src))[:n]):
                        img = imread_unicode(os.path.join(neg_src, fname))
                        if img is None:
                            continue
                        imwrite_unicode(os.path.join(dest_nofire, f"public_{src_split}_{neg_class}_{i}.jpg"),
                                         cv2.resize(img, IMG_SIZE))
                        counts[f"{dst_split}_no_fire"] += 1

        _set_public_status("دانلود و آماده‌سازی کامل شد.")
        app_window.after(0, refresh_dataset_count)
        app_window.after(0, lambda: messagebox.showinfo(
            "دانلود کامل شد",
            f"آموزش -> آتش: {counts['train_fire']} | بدون آتش: {counts['train_no_fire']}\n"
            f"آزمون -> آتش: {counts['test_fire']} | بدون آتش: {counts['test_no_fire']}"
        ))
    except Exception as e:
        app_window.after(0, lambda: messagebox.showerror("خطا در دانلود", f"{e}"))
    finally:
        app_window.after(0, lambda: public_download_button.config(state="normal"))


def start_public_download():
    public_download_button.config(state="disabled")
    limit = public_limit_scale.get()
    threading.Thread(target=_run_public_download, args=(limit,), daemon=True).start()


def preview_dataset_samples(n=9):
    fire_files = list_images(TRAIN_FIRE_DIR)
    nofire_files = list_images(TRAIN_NOFIRE_DIR)
    if not fire_files and not nofire_files:
        messagebox.showwarning("خطا", "دیتاست خالی است.")
        return
    n_fire = min(n // 2, len(fire_files))
    n_nofire = min(n - n_fire, len(nofire_files))
    samples = [(p, "FIRE") for p in random.sample(fire_files, n_fire)] + \
              [(p, "NO FIRE") for p in random.sample(nofire_files, n_nofire)]
    random.shuffle(samples)

    cols = 3
    rows = max(1, (len(samples) + cols - 1) // cols)
    plt.figure(figsize=(12, 4 * rows))
    for i, (p, lab) in enumerate(samples):
        img = imread_unicode(p)
        if img is None:
            continue
        plt.subplot(rows, cols, i + 1)
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        # عمداً انگلیسی: عنوان‌های matplotlib فارسی رو درست رندر نمی‌کنن.
        plt.title(lab, fontsize=11, color=('red' if lab == "FIRE" else 'green'))
        plt.axis("off")
    plt.tight_layout()
    plt.show()



#  مدل CNN و آموزش

def build_fire_classifier(input_shape=(IMG_SIZE[1], IMG_SIZE[0], 3)):
    base = MobileNetV2(input_shape=input_shape, include_top=False, weights='imagenet')
    base.trainable = False

    model = Sequential([
        layers.Input(shape=input_shape),
        layers.Rescaling(2.0, offset=-1.0),
        base,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.3),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(1, activation='sigmoid'),
    ])
    model.compile(optimizer=Adam(1e-3), loss='binary_crossentropy', metrics=['accuracy'])
    return model, base  # base ro ham bar migardoonim ta too faze fine-tune behesh dastresi dashte bashim


def augment_image(img):
    if random.random() < 0.5:
        img = np.fliplr(img)
    if random.random() < 0.3:
        img = np.flipud(img)
    if random.random() < 0.5:
        h, w = img.shape[:2]
        angle = random.uniform(-20, 20)
        scale = random.uniform(0.9, 1.1)
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale)
        img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)
    if random.random() < 0.5:
        factor = random.uniform(0.7, 1.3)
        img = np.clip(img * factor, 0, 1)
    if random.random() < 0.3:
        noise = np.random.normal(0, 0.02, img.shape)
        img = np.clip(img + noise, 0, 1)
    if random.random() < 0.35:
        h, w = img.shape[:2]
        yy, xx = np.mgrid[0:h, 0:w]
        cy = h / 2 + random.uniform(-0.15, 0.15) * h
        cx = w / 2 + random.uniform(-0.15, 0.15) * w
        ry = h * random.uniform(0.35, 0.6)
        rx = w * random.uniform(0.35, 0.6)
        dist = ((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2
        vignette = np.clip(1.25 - dist, 0.0, 1.0).astype(np.float32)
        img = img * vignette[..., None]
    return np.ascontiguousarray(img, dtype=np.float32)

#خوندن و دسته دسته کردن عکسا با ضریب 
class ClassifierDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, fire_files, nofire_files, batch_size=32, augment=False, shuffle=True, repeat=1):
        items = [(p, 1.0) for p in fire_files] + [(p, 0.0) for p in nofire_files]
        self.items = items * max(1, repeat)
        self.batch_size = batch_size
        self.augment = augment
        self.shuffle = shuffle
        self.indices = np.arange(len(self.items))
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __len__(self):
        return max(1, len(self.items) // self.batch_size)

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __getitem__(self, idx):
        batch_idx = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        X, Y = [], []
        for i in batch_idx:
            path, label = self.items[i]
            img = imread_unicode(path)
            if img is None:
                continue
            img = cv2.resize(img, IMG_SIZE)
            img_f = (img[:, :, ::-1] / 255.0).astype(np.float32)
            if self.augment:
                img_f = augment_image(img_f)
            X.append(img_f)
            Y.append(label)
        if not X:
            return (np.zeros((1, IMG_SIZE[1], IMG_SIZE[0], 3), dtype=np.float32),
                    np.zeros((1,), dtype=np.float32))
        return np.array(X, dtype=np.float32), np.array(Y, dtype=np.float32)

#عکسا دستی هارو با پیشوند جدا میده ضریب بگیرن
def _oversample_manual_labels(file_list, factor=5):
    """
    عکس‌هایی که با ابزار برچسب‌گذاری دستی (QuickTagger) اضافه شدن، با
    پیشوند «manual_» ذخیره می‌شن. این تابع این فایل‌ها رو تو فهرست
    آموزش چندین برابر (پیش‌فرض ۵ برابر) تکرار می‌کنه تا سهمشون تو
    گرادیان آموزش، به‌جای گم‌شدن تو حجم دیتاست عمومی، واقعاً محسوس
    باشه.
    """
    boosted = []
    for f in file_list:
        boosted.append(f)
        if os.path.basename(f).startswith("manual_"):
            boosted.extend([f] * (factor - 1))
    return boosted

#قبل شروع اموزش شرایط و خوندن رو چک میکنه و تست و ارزیابی داره 15 درصد ارزیابی و بقیه اموزش 30تاکمتر نباشه هر کلاس ولی بالا دسته های 32 32 رو میخونه فرق این دو عدد ممکنه گیج بکنه مارو فرق دارن
def train_classifier():
    global fire_model
    fire_files = list_images(TRAIN_FIRE_DIR)
    nofire_files = list_images(TRAIN_NOFIRE_DIR)

    if len(fire_files) < 30 or len(nofire_files) < 30:
        messagebox.showerror(
            "داده کافی نیست",
            f"در حال حاضر دیتاست آموزش خیلی کوچک است (آتش: {len(fire_files)} / "
            f"بدون آتش: {len(nofire_files)}).\n\n"
            "با داده‌ی کم، مدل معمولاً فرومی‌پاشد و همه‌چیز را با اطمینان بالا «آتش» اعلام می‌کند.\n\n"
            "از تب «۲- دیتاست» دوباره دیتاست آماده را درون‌ریزی کنید یا دیتاست عمومی را دانلود کنید."
        )
        return

    random.shuffle(fire_files)
    random.shuffle(nofire_files)
    n_val_fire = max(1, int(len(fire_files) * 0.15))
    n_val_nofire = max(1, int(len(nofire_files) * 0.15))
    val_fire, train_fire = fire_files[:n_val_fire], fire_files[n_val_fire:]
    val_nofire, train_nofire = nofire_files[:n_val_nofire], nofire_files[n_val_nofire:]

    # oversampling نمونه‌های دستی فقط روی داده‌ی آموزش (نه validation)
    manual_factor = manual_weight_scale.get()
    n_manual_fire = sum(1 for f in train_fire if os.path.basename(f).startswith("manual_"))
    n_manual_nofire = sum(1 for f in train_nofire if os.path.basename(f).startswith("manual_"))
    train_fire = _oversample_manual_labels(train_fire, factor=manual_factor)
    train_nofire = _oversample_manual_labels(train_nofire, factor=manual_factor)

    multiplier = aug_scale.get()
    batch_size = 32
    train_gen = ClassifierDataGenerator(train_fire, train_nofire, batch_size=batch_size,
                                         augment=True, shuffle=True, repeat=multiplier)
    val_gen = ClassifierDataGenerator(val_fire, val_nofire, batch_size=batch_size,
                                       augment=False, shuffle=False, repeat=1)

    model, base = build_fire_classifier()
    checkpoint_cb = ModelCheckpoint(MODEL_PATH, monitor='val_loss', save_best_only=True, verbose=0)
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, min_lr=1e-6),
        checkpoint_cb,
    ]

    n_fire, n_nofire = len(train_fire), len(train_nofire)
    total = n_fire + n_nofire
    w_nofire = np.clip(total / (2.0 * n_nofire), 0.5, 3.0)
    w_fire = np.clip(total / (2.0 * n_fire), 0.5, 3.0)
    class_weight = {0: float(w_nofire), 1: float(w_fire)}

    try:
        history = model.fit(train_gen, validation_data=val_gen, epochs=epoch_scale.get(),
                             callbacks=callbacks, class_weight=class_weight, verbose=1)
    except Exception as e:
        import traceback
        detail = traceback.format_exc()[-800:]
        if os.path.exists(MODEL_PATH):
            try:
                fire_model = load_model(MODEL_PATH)
                messagebox.showwarning(
                    "آموزش ناتمام ماند",
                    f"آموزش وسط راه قطع شد:\n{e}\n\n"
                    "ولی خوشبختانه آخرین چک‌پوینتِ خوب (بهترین دوره تا همین لحظه) روی دیسک "
                    "ذخیره شده بود و همین الان لود شد. پس نباید همه‌چیز از دست رفته باشد -- "
                    "فقط دوره‌های بعد از آن انجام نشده‌اند.\n\n"
                    f"{detail}"
                )
            except Exception:
                messagebox.showerror("خطا در آموزش", f"{e}\n\n{detail}")
        else:
            messagebox.showerror("خطا در آموزش", f"{e}\n\n{detail}")
        return
#فاز اول تاریخچه رو نگه میداره رسم نمودار بزنه loss و ac
    all_loss = list(history.history['loss'])
    all_val_loss = list(history.history['val_loss'])
    all_acc = list(history.history['accuracy'])
    all_val_acc = list(history.history['val_accuracy'])
#فاز دوم فاین تیون
    fine_tune_epochs = fine_tune_scale.get()
    if fine_tune_epochs > 0:
        base.trainable = True
        fine_tune_at = max(0, len(base.layers) - 30)
        for layer in base.layers[:fine_tune_at]:
            layer.trainable = False

        model.compile(optimizer=Adam(1e-5), loss='binary_crossentropy', metrics=['accuracy'])

        try:
            history2 = model.fit(train_gen, validation_data=val_gen, epochs=fine_tune_epochs,
                                  callbacks=callbacks, class_weight=class_weight, verbose=1)
            all_loss += list(history2.history['loss'])
            all_val_loss += list(history2.history['val_loss'])
            all_acc += list(history2.history['accuracy'])
            all_val_acc += list(history2.history['val_accuracy'])
        except Exception as e:
            import traceback
            messagebox.showwarning(
                "هشدار فاز فاین-تیون",
                f"فاز فاین-تیون با خطا مواجه شد و رد شد (مدل فاز ۱ حفظ می‌شود):\n{e}\n\n"
                f"{traceback.format_exc()[-500:]}"
            )

    try:
        model.save(MODEL_PATH)
        fire_model = model
    except Exception as e:
        messagebox.showerror("خطا در ذخیره‌سازی", f"{e}")
        return

    plt.figure(figsize=(11, 4))
    plt.subplot(1, 2, 1)
    # برچسب‌های نمودار عمداً انگلیسی هستن چون ترکیب فارسی مشکل میخوره).
    plt.plot(all_loss, label='Train')
    plt.plot(all_val_loss, label='Validation')
    if fine_tune_epochs > 0:
        plt.axvline(x=len(history.history['loss']) - 0.5, color='gray', linestyle='--', label='Fine-tune start')
    plt.title("Loss")
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(all_acc, label='Train')
    plt.plot(all_val_acc, label='Validation')
    if fine_tune_epochs > 0:
        plt.axvline(x=len(history.history['accuracy']) - 0.5, color='gray', linestyle='--',
                    label='Fine-tune start')
    plt.title("Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.show()

    messagebox.showinfo(
        "آموزش کامل شد",
        f"مدل ذخیره شد در:\n{MODEL_PATH}\n\n"
        f"آموزش: آتش {len(train_fire)} / بدون آتش {len(train_nofire)} (ضریب افزایش داده: {multiplier})\n"
        f"نمونه‌های دستیِ oversample‌شده ({manual_factor} برابر): آتش {n_manual_fire} | "
        f"بدون آتش {n_manual_nofire}\n"
        f"ارزیابی: آتش {len(val_fire)} / بدون آتش {len(val_nofire)}\n"
        f"وزن کلاس (جبران عدم تعادل): بدون‌آتش={class_weight[0]:.2f} / آتش={class_weight[1]:.2f}\n"
        f"فاز فاین-تیون: {fine_tune_epochs} دوره اضافه"
    )

#اجرای بار اول لود کردن دیتاست

def try_load_model_on_startup():
    global fire_model
    if os.path.exists(MODEL_PATH):
        try:
            fire_model = load_model(MODEL_PATH)
        except Exception:
            fire_model = None
#سربرگ 3 لود شدن یا نشدن اون هست نشون دادن پیام به کاربر
def manual_load_model():
    global fire_model
    if not os.path.exists(MODEL_PATH):
        messagebox.showwarning("خطا", "هنوز مدلی آموزش داده نشده است.")
        return
    try:
        fire_model = load_model(MODEL_PATH)
        messagebox.showinfo("موفق", "مدل با موفقیت بارگذاری شد.")
    except Exception as e:
        messagebox.showerror("خطا", f"{e}")



#  استنباط: تک‌مقیاسی (TTA) + چندمقیاسی (Multiscale)
 
#فیلترا
def skin_fraction(img_bgr):
    """
    نسبت پیکسل‌های رنگ پوست انسان (فضای رنگی YCrCb) توی کل فریم یا یک
    crop مشخص. پوست دست/صورت زیر نور گرم می‌تونه شبیه نارنجی/قرمز
    شعله به‌نظر برسه.
    """
    ycrcb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycrcb)
    mask = (cr >= 133) & (cr <= 180) & (cb >= 77) & (cb <= 135) & (y > 60)
    return float(np.mean(mask))

#میزان بافته خاکستری میشه و بعدش بر اساس اون مثلا بافت شعله لبه عدد بالا= بافت زیاد داره ، پوست کمتر
def texture_score(crop_bgr):
    """
    واریانس لاپلاسین (میزان بافت/جزئیات). شعله‌ی واقعی و نزدیک، لبه‌های
    نامنظم و بافت زیادی داره، ولی سطوح کاملاً صاف و یکدست (مثل پوست
    ساده یا دیوار) بافت کمی دارن.
    """
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())

#یکنواختی رنگ گرم براساس تبدیل اچ اس وی و درصد ناحیه گرم + نوسان روشنایی همون وی   
# h=رنگم /s=خلوصخ/v=روشنایی
def warm_region_brightness_std(crop_bgr, min_pixels=30):
    """
    تشخیص «جسم جامد رنگ‌گرم» (مثل فندک خاموش نارنجی/قرمز) در برابر
    «شعله‌ی واقعی». شعله‌ی واقعی همیشه تلاطم/نوسان روشنایی داره، اما
    یک جسم جامد و یکدست، روشنایی نسبتاً یکنواخت‌تری داره.
    خروجی: (std, coverage)
    """
    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    mask = ((h <= 25) | (h >= 170)) & (s > 80) & (v > 100)
    coverage = float(np.mean(mask))
    if int(mask.sum()) < min_pixels:
        return None, coverage
    return float(np.std(v[mask])), coverage

#پیدا کردن اتش تو گوشه های مختلف-یک کروپ کلی-یک وسط زوم- اطراف هم پوشانی تا مرکز زوم بتونیم ببینیم برای تشخیص بهتر
def _make_grid_crops(img_bgr, crop_ratio=0.42, tight_ratio=0.28, tight_ratio2=0.16):
    """
    چند crop از فریم می‌سازه: کل فریم + یک شبکه‌ی هم‌پوشانی‌دار از
    crop‌های کوچک‌تر (برای گوشه‌های کادر) + دو crop زوم‌شده‌ی مرکزی با
    دو درجه‌ی زوم مختلف (برای شعله‌های کوچک/دور که وقتی کل فریم
    resize بشه، تقریباً گم می‌شن).
    """
    h, w = img_bgr.shape[:2]
    crops = [img_bgr]
    ch, cw = int(h * crop_ratio), int(w * crop_ratio)
    if ch >= 16 and cw >= 16:
        ys = sorted(set([0, max(0, (h - ch) // 2), max(0, h - ch)]))
        xs = sorted(set([0, max(0, (w - cw) // 2), max(0, w - cw)]))
        for y0 in ys:
            for x0 in xs:
                crop = img_bgr[y0:y0 + ch, x0:x0 + cw]
                if crop.size > 0:
                    crops.append(crop)

    for tr in (tight_ratio, tight_ratio2):
        tch, tcw = int(h * tr), int(w * tr)
        if tch >= 16 and tcw >= 16:
            ty0, tx0 = (h - tch) // 2, (w - tcw) // 2
            crop = img_bgr[ty0:ty0 + tch, tx0:tx0 + tcw]
            if crop.size > 0:
                crops.append(crop)

    return crops

#چندمقیاسی- با تابع بالا چند تا تکه عکس میگیره-هر تکیه جدا میده تا ببینه چقدر احتمال اتش-روی هر تکه سه تا فیلتر بالا بافت و پوست و... اعمال میکنه تا درصد بیاد پایین
def predict_prob_multiscale(img_bgr, top_k=3):
    """
    تشخیص چندمقیاسی: فریم رو به چند crop هم‌پوشانی‌دار تقسیم می‌کنه، هر
    crop رو جدا predict می‌کنه، و روی هر crop سه فیلتر اعمال می‌کنه:
      ۱) کاهش‌دهنده‌ی پوست (رنگ + بافت)
      ۲) کاهش‌دهنده‌ی بافت مستقل
      ۳) کاهش‌دهنده‌ی یکنواختی رنگ‌گرم (برای جسم جامد رنگ‌گرم، نه شعله)
    در آخر، به‌جای بیشینه (که خیلی حساس به خطای یک crop است)، میانگین
    ۳ تا crop برتر گرفته می‌شه -- یه رأی‌گیری نرم‌تر.
    """
    crops = _make_grid_crops(img_bgr)
    batch = []
    for c in crops:
        r = cv2.resize(c, IMG_SIZE)
        rgb = (r[:, :, ::-1] / 255.0).astype(np.float32)
        batch.append(rgb)
    batch = np.array(batch, dtype=np.float32)
    raw_preds = fire_model.predict(batch, verbose=0)[:, 0]

    TEXTURE_REF = 18.0
    SKIN_TEXTURE_SMOOTH_MAX = 20.0
    WARM_STD_FLAT_MAX = 17.0
    WARM_COVERAGE_MIN = 0.25
    per_crop_probs = []
    for ci, c in enumerate(crops):
        p = float(raw_preds[ci])
        tex = texture_score(c)

        skin_frac = skin_fraction(c)
        if skin_frac > 0.45:
            if tex <= SKIN_TEXTURE_SMOOTH_MAX:
                texture_gate = 1.0
            else:
                texture_gate = float(np.clip(
                    1.0 - (tex - SKIN_TEXTURE_SMOOTH_MAX) / 40.0, 0.15, 1.0))
            p *= max(0.0, 1.0 - 0.35 * skin_frac * texture_gate)

        texture_damp = float(np.clip(tex / TEXTURE_REF, 0.75, 1.0))
        p *= texture_damp

        warm_std, warm_coverage = warm_region_brightness_std(c)
        if warm_std is not None and warm_coverage > WARM_COVERAGE_MIN and warm_std < WARM_STD_FLAT_MAX:
            p *= 0.62

        per_crop_probs.append(p)

    per_crop_probs.sort(reverse=True)
    k = min(top_k, len(per_crop_probs))
    return float(np.mean(per_crop_probs[:k]))

#سریع مثل ویدیو
def predict_prob(img_bgr):
    """
    پیش‌بینی سریعِ تک‌مقیاسی روی کل تصویر (بدون crop) -- برای حالت
    «سریع».
    """
    img = cv2.resize(img_bgr, IMG_SIZE)
    rgb = (img[:, :, ::-1] / 255.0).astype(np.float32)
    return float(fire_model.predict(rgb[np.newaxis, ...], verbose=0)[0, 0])

#پایدار تر میانگین دو احتمال
def predict_prob_tta(img_bgr):
    """
    پیش‌بینی تک‌مقیاسی + TTA (میانگین با فلیپ افقی) روی کل تصویر.
    """
    img = cv2.resize(img_bgr, IMG_SIZE)
    rgb = (img[:, :, ::-1] / 255.0).astype(np.float32)
    rgb_flip = np.ascontiguousarray(np.fliplr(rgb))
    batch = np.stack([rgb, rgb_flip], axis=0)
    preds = fire_model.predict(batch, verbose=0)[:, 0]
    return float(np.mean(preds))

#ترکیبی عکس تکی
def predict_prob_combined(img_bgr):
    """
    فرمول واحد برای عکس تک: ۳۰٪ multiscale + ۷۰٪ کل‌عکس (TTA). ویدیو/
    وبکم جداگانه از predict_prob_multiscale خالص استفاده می‌کنن (بخش
    detect_fire_video/webcam رو ببین).
    """
    prob_multiscale = predict_prob_multiscale(img_bgr)
    prob_wholeimg = predict_prob_tta(img_bgr)
    return 0.30 * prob_multiscale + 0.70 * prob_wholeimg

#دسته ای
def _predict_batch(paths, batch_size=32, progress_cb=None):
    """
    به‌جای صدازدن model.predict برای هر عکس تک‌تک (کند)، عکس‌ها را در
    دسته‌های ۳۲تایی جمع می‌کند و یکجا predict می‌کند.
    """
    probs = []
    batch_imgs = []
    done = 0
    total = len(paths)
    for p in paths:
        img = imread_unicode(p)
        if img is None:
            done += 1
            continue
        img_r = cv2.resize(img, IMG_SIZE)
        batch_imgs.append((img_r[:, :, ::-1] / 255.0).astype(np.float32))
        if len(batch_imgs) >= batch_size:
            preds = fire_model.predict(np.array(batch_imgs), verbose=0)[:, 0]
            probs.extend(preds.tolist())
            done += len(batch_imgs)
            batch_imgs = []
            if progress_cb:
                progress_cb(done, total)
    if batch_imgs:
        preds = fire_model.predict(np.array(batch_imgs), verbose=0)[:, 0]
        probs.extend(preds.tolist())
        done += len(batch_imgs)
        if progress_cb:
            progress_cb(done, total)
    return probs

#ارزیابی روی دیتاست ازمون واقعی
def evaluate_on_test_set():
    if fire_model is None:
        messagebox.showwarning("خطا", "ابتدا مدل را آموزش دهید یا بارگذاری کنید.")
        return
    fire_files = list_images(TEST_FIRE_DIR)
    nofire_files = list_images(TEST_NOFIRE_DIR)
    if not fire_files and not nofire_files:
        messagebox.showwarning("خطا", "پوشه‌ی آزمون خالی است.")
        return

    def _run():
        total_n = len(fire_files) + len(nofire_files)
        app_window.after(0, lambda: eval_status_label.config(
            text=f"در حال ارزیابی روی {total_n} تصویر... ۰/{total_n}"))
        try:
            def _progress(done, sub_total, offset=0):
                app_window.after(0, lambda: eval_status_label.config(
                    text=f"در حال ارزیابی... {offset + done}/{total_n}"))

            fire_probs = _predict_batch(
                fire_files, progress_cb=lambda d, t: _progress(d, t, offset=0))
            nofire_probs = _predict_batch(
                nofire_files, progress_cb=lambda d, t: _progress(d, t, offset=len(fire_files)))

            tp = sum(1 for p in fire_probs if p >= 0.5)
            fn = sum(1 for p in fire_probs if p < 0.5)
            fp = sum(1 for p in nofire_probs if p >= 0.5)
            tn = sum(1 for p in nofire_probs if p < 0.5)

            total = tp + fp + tn + fn
            accuracy = (tp + tn) / total if total else 0
            precision = tp / (tp + fp) if (tp + fp) else 0
            recall = tp / (tp + fn) if (tp + fn) else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0

            app_window.after(0, lambda: eval_status_label.config(text="ارزیابی کامل شد."))
            app_window.after(0, lambda: messagebox.showinfo(
                "نتیجه‌ی ارزیابی روی دیتاست آزمون واقعی",
                f"تعداد نمونه: {total}\n"
                f"Accuracy: {accuracy:.3f}\nPrecision: {precision:.3f}\nRecall: {recall:.3f}\nF1: {f1:.3f}\n\n"
                f"TP={tp}  FP={fp}  TN={tn}  FN={fn}"
            ))
        except Exception as e:
            import traceback
            detail = traceback.format_exc()[-600:]
            app_window.after(0, lambda: messagebox.showerror("خطا در ارزیابی", f"{e}\n\n{detail}"))
        finally:
            app_window.after(0, lambda: eval_button.config(state="normal"))

    eval_button.config(state="disabled")
    threading.Thread(target=_run, daemon=True).start()

#ارزیابی روی اموزش
def self_test_on_training_samples(n=10):
    """
    خودآزمایی روی چند نمونه‌ی تصادفی از Train/fire و Train/no_fire
    (با هم).
    """
    if fire_model is None:
        messagebox.showwarning("خطا", "ابتدا مدل را آموزش دهید یا بارگذاری کنید.")
        return
    fire_files = list_images(TRAIN_FIRE_DIR)
    nofire_files = list_images(TRAIN_NOFIRE_DIR)
    if not fire_files and not nofire_files:
        messagebox.showwarning("خطا", "دیتاست آموزش خالی است.")
        return
    fire_sample = random.sample(fire_files, min(n, len(fire_files))) if fire_files else []
    nofire_sample = random.sample(nofire_files, min(n, len(nofire_files))) if nofire_files else []

    def _score(paths):
        raw_probs, ms_probs = [], []
        for p in paths:
            img = imread_unicode(p)
            if img is None:
                continue
            img_r = cv2.resize(img, IMG_SIZE)
            rgb = (img_r[:, :, ::-1] / 255.0).astype(np.float32)
            raw_probs.append(float(fire_model.predict(rgb[np.newaxis, ...], verbose=0)[0, 0]))
            ms_probs.append(predict_prob_multiscale(img))
        return raw_probs, ms_probs

    def _run():
        app_window.after(0, lambda: selftest_status_label.config(text="در حال خودآزمایی..."))
        try:
            fire_raw, fire_ms = _score(fire_sample)
            nofire_raw, nofire_ms = _score(nofire_sample)

            avg_fire_raw = float(np.mean(fire_raw)) if fire_raw else float('nan')
            avg_fire_ms = float(np.mean(fire_ms)) if fire_ms else float('nan')
            avg_nofire_raw = float(np.mean(nofire_raw)) if nofire_raw else float('nan')
            avg_nofire_ms = float(np.mean(nofire_ms)) if nofire_ms else float('nan')
            n_nofire_fp_raw = sum(1 for p in nofire_raw if p >= 0.5)
            n_nofire_fp_ms = sum(1 for p in nofire_ms if p >= 0.5)

            lines = []
            if fire_raw:
                lines.append(f"Train/fire ({len(fire_raw)} نمونه) -- انتظار: بالا باشد\n"
                              f"  خام: {avg_fire_raw:.3f}   |   نهایی (multiscale): {avg_fire_ms:.3f}")
            if nofire_raw:
                lines.append(f"\nTrain/no_fire ({len(nofire_raw)} نمونه) -- انتظار: پایین باشد\n"
                              f"  خام: {avg_nofire_raw:.3f} ({n_nofire_fp_raw}/{len(nofire_raw)} بالای 0.5)   |   "
                              f"نهایی (multiscale): {avg_nofire_ms:.3f} ({n_nofire_fp_ms}/{len(nofire_ms)} بالای 0.5)")

            if nofire_raw and avg_nofire_raw >= 0.5:
                diag = ("تشخیص: خود مدل خام هم روی نمونه‌های no_fire امتیاز بالا می‌دهد. این "
                        "مشکل فیلتر نیست -- مشکل آموزش است. دوباره آموزش دهید، و ضریب «اهمیت "
                        "عکس‌های دستی» را بالاتر ببرید.")
            elif nofire_raw and avg_nofire_ms >= 0.5 > avg_nofire_raw:
                diag = ("تشخیص: مدل خام خوب است (پایین)، ولی نتیجه‌ی نهایی (multiscale) بالاست. "
                        "مشکل از crop‌های زوم‌شده‌ی predict_prob_multiscale است -- اینجا باید "
                        "روی فیلترها کار کرد، نه آموزش دوباره.")
            else:
                diag = "نتیجه‌ها در بازه‌ی مطلوب هستند (آتش بالا، بدون‌آتش پایین)."

            app_window.after(0, lambda: selftest_status_label.config(text=""))
            app_window.after(0, lambda: messagebox.showinfo(
                "خودآزمایی روی نمونه‌های آموزش (آتش + بدون‌آتش)",
                "\n".join(lines) + f"\n\n{diag}"
            ))
        except Exception as e:
            import traceback
            detail = traceback.format_exc()[-600:]
            app_window.after(0, lambda: messagebox.showerror("خطا در خودآزمایی", f"{e}\n\n{detail}"))
            app_window.after(0, lambda: selftest_status_label.config(text=""))
        finally:
            app_window.after(0, lambda: selftest_button.config(state="normal"))

    selftest_button.config(state="disabled")
    threading.Thread(target=_run, daemon=True).start()


# =====================================================================
#  تشخیص روی عکس / ویدیو / وبکم
# =====================================================================
#تثبیت کننده از کلاس استا داره که تصمیم میگیره استفاده بشه بر اساس سه روش برای پایداری فریم ها لرزش ها(وزن دهی بیشتر به داده جدید مثلا سی درصد به راو و 70 تا با اسا اسفاده از داده قبلیو تعداد و روش سریع)
def overlay_result(frame, prob, threshold, detected_override=None):
    detected = detected_override if detected_override is not None else (prob >= threshold)
    out = frame.copy()
    color = (0, 0, 255) if detected else (0, 200, 0)
    cv2.rectangle(out, (0, 0), (out.shape[1] - 1, out.shape[0] - 1), color, 8)
    # عمداً انگلیسی: OpenCV فقط فونت ASCII (Hershey) پشتیبانی می‌کند و
    # حروف فارسی را اصلاً نشان نمی‌دهد.
    label = f"FIRE {prob * 100:.0f}%" if detected else f"no fire ({(1 - prob) * 100:.0f}%)"
    cv2.putText(out, label, (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    return out, detected


class FireStabilizer:
    """
    خروجی نویزی و لحظه‌ای مدل را به یک تصمیم پایدار «آتش هست/نیست»
    تبدیل می‌کند، با سه مکانیزم مکمل:
      ۱) EMA (میانگین متحرک نمایی)
      ۲) هیسترزیس (آستانه‌ی نامتقارن برای روشن/خاموش شدن)
      ۳) مسیر سریع برای شعله‌ی واضح (نسبت به آستانه‌ی خود کاربر تنظیم
         می‌شود، نه یک عدد ثابت)
    """

    def __init__(self, ema_alpha=0.3, window=8, on_count=4, off_count=2,
                 fast_path_prob=0.92, fast_path_min_consecutive=3):
        self.ema_alpha = ema_alpha
        self.smoothed = 0.0
        self.history = deque(maxlen=window)
        self.on_count = on_count
        self.off_count = off_count
        self.fast_path_prob = fast_path_prob
        self.fast_path_min_consecutive = fast_path_min_consecutive
        self._fast_streak = 0
        self.confirmed = False
#قلب استا تو ویدیو وبکم اول میانگین وزنی بعد اضافه لیست میکنه تعداد بالا زد اوکی میده یا بجز این اگه سه تا بالای 92 بود فریم پشت هم بگو اتیش در غیر این صورت همون دو تا اول تصمیم
    def update(self, raw_prob, threshold):
        self.smoothed = (self.ema_alpha * raw_prob) + (1 - self.ema_alpha) * self.smoothed
        self.history.append(self.smoothed)
        above = sum(1 for p in self.history if p >= threshold)

        if raw_prob >= self.fast_path_prob:
            self._fast_streak += 1
        else:
            self._fast_streak = 0

        if self._fast_streak >= self.fast_path_min_consecutive:
            self.confirmed = True
        elif not self.confirmed and above >= min(self.on_count, len(self.history)):
            self.confirmed = True
        elif self.confirmed and above <= self.off_count and self._fast_streak == 0:
            self.confirmed = False

        return self.smoothed, self.confirmed

#تشخیص عکس
def detect_fire_image():
    if fire_model is None:
        messagebox.showwarning("خطا", "ابتدا مدل را آموزش دهید یا بارگذاری کنید.")
        return
    path = filedialog.askopenfilename(title="انتخاب عکس",
                                       filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")])
    if not path:
        return
    img = imread_unicode(path)
    if img is None:
        messagebox.showerror("خطا", "تصویر خوانده نشد.")
        return

    threshold = thresh_scale.get() / 100.0 #استانه رو میخونه
    prob = predict_prob_combined(img)#روش نهایی
    prob_multiscale = predict_prob_multiscale(img)#چند مقیاسی
    prob_wholeimg = predict_prob_tta(img)#تی تی ای
    output, detected = overlay_result(img, prob, threshold)#رسم روی شکل

    out_path = os.path.join(OUTPUT_DIR, "fire_" + os.path.basename(path))
    imwrite_unicode(out_path, output)

    msg = "آتش تشخیص داده شد!" if detected else "آتشی تشخیص داده نشد."
    messagebox.showinfo("نتیجه",
                         f"{msg}\nاحتمال آتش: {prob:.3f} (آستانه: {threshold:.2f})\n"
                         f"  -- چندمقیاسی (crop به crop): {prob_multiscale:.3f}\n"
                         f"  -- کل عکس (TTA): {prob_wholeimg:.3f}\n"
                         f"ذخیره شد در:\n{out_path}")

    # عمداً انگلیسی: عنوان‌های matplotlib فارسی درست رندر نمی‌شن.
    plot_msg = "Fire Detected!" if detected else "No Fire Detected."
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title("Input")
    plt.axis("off")
    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
    plt.title(f"{plot_msg} (p={prob:.2f})")
    plt.axis("off")
    plt.tight_layout()
    plt.show()

#ویدیو
def detect_fire_video():
    if fire_model is None:
        messagebox.showwarning("خطا", "ابتدا مدل را آموزش دهید یا بارگذاری کنید.")
        return
    path = filedialog.askopenfilename(title="انتخاب ویدیو",
                                       filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv")])
    if not path:
        return
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        messagebox.showerror("خطا", "ویدیو باز نشد.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 25#نرخ فریم ها 
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_path = os.path.join(OUTPUT_DIR, "fire_" + os.path.splitext(os.path.basename(path))[0] + ".mp4")
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

    # عمداً انگلیسی: عنوان پنجره‌ی OpenCV هم محدودیت فونت داره.
    win_name = "Fire Detection - Video (ESC to exit)"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_name, max(w, 960), max(h, 640))
#تنظیمات دستی رو میخونه
    threshold = thresh_scale.get() / 100.0
    use_multiscale = highacc_var.get()#دقت بالا
    SKIP = 2 if use_multiscale else 3#سه فریم یا دوتا؟
    fast_path_prob = float(np.clip(threshold + 0.03, 0.90, 0.99))
    stabilizer = FireStabilizer(ema_alpha=0.30, window=8, on_count=4, off_count=2,
                                 fast_path_prob=fast_path_prob, fast_path_min_consecutive=3)
    last_prob = 0.0
    frame_i = 0
    fire_frames = 0
    total_frames = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        total_frames += 1#برای اینکه 2 تا رد کنه یا سه تا ایندکس هاشونو بدست میاره 
        if frame_i % SKIP == 0:
            last_prob = predict_prob_multiscale(frame) if use_multiscale else predict_prob(frame)
        frame_i += 1

        smoothed_prob, confirmed = stabilizer.update(last_prob, threshold)#تثبیت

        output, _ = overlay_result(frame, smoothed_prob, threshold, detected_override=confirmed)#اطلاعات اضاف
        cv2.putText(output, f"smoothed={smoothed_prob:.2f} raw={last_prob:.2f}",
                    (15, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 0), 2)
        if confirmed:
            fire_frames += 1
            cv2.putText(output, "FIRE CONFIRMED", (15, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
#ذخبره
        writer.write(output)
        cv2.imshow(win_name, output)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    writer.release()
    cv2.destroyAllWindows()

    messagebox.showinfo(
        "نتیجه‌ی تشخیص در ویدیو",
        f"فریم‌های با آتش تأییدشده: {fire_frames} از {total_frames}\nخروجی ذخیره شد در:\n{out_path}"
    )

#وبکم
def detect_fire_webcam():
    if fire_model is None:
        messagebox.showwarning("خطا", "ابتدا مدل را آموزش دهید یا بارگذاری کنید.")
        return
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        messagebox.showerror("خطا", "دوربین در دسترس نیست.")
        return

    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # عمداً انگلیسی: عنوان پنجره‌ی OpenCV هم محدودیت فونت داره.
    win_name = "Live Fire Detection - Webcam (ESC to exit)"
    cw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
    ch = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_name, cw, ch)

    threshold = thresh_scale.get() / 100.0
    use_multiscale = highacc_var.get()
    SKIP = 2 if use_multiscale else 3
    frame_i = 0
    last_prob = 0.0
    fast_path_prob = float(np.clip(threshold + 0.03, 0.90, 0.99))
    stabilizer = FireStabilizer(ema_alpha=0.30, window=8, on_count=4, off_count=2,
                                 fast_path_prob=fast_path_prob, fast_path_min_consecutive=3)
#خوندن فریم3یا2
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_i % SKIP == 0:
            last_prob = predict_prob_multiscale(frame) if use_multiscale else predict_prob(frame)
        frame_i += 1

        smoothed_prob, confirmed = stabilizer.update(last_prob, threshold)

        output, _ = overlay_result(frame, smoothed_prob, threshold, detected_override=confirmed)
        cv2.putText(output, f"smoothed={smoothed_prob:.2f} raw={last_prob:.2f}",
                    (15, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 0), 2)
        cv2.imshow(win_name, output)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


# =====================================================================
#  رابط گرافیکی (واکنش‌گرا)
# =====================================================================

def make_scrollable(parent):
    container = tk.Frame(parent, bg=COLOR_BG_LIGHT)#جعبه‌ی اصلی که همه چیز را نگه می‌دارد
    container.pack(fill="both", expand=True)#صفحه‌ای که محتوا را در خودش جای می‌دهد
    canvas = tk.Canvas(container, borderwidth=0, highlightthickness=0, bg=COLOR_BG_LIGHT)
    scrollbar = tk.Scrollbar(container, orient="vertical", command=canvas.yview)#نوار اسکرول در سمت راست
    scrollable_frame = tk.Frame(canvas, bg=COLOR_BG_LIGHT)#جایی که دکمه‌ها، اسلایدرها و متن‌ها قرار می‌گیرند

    inner_window = canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
    scrollable_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))

    def _on_canvas_resize(event):#بزرگ کوچیک شد اوکی باشه
        canvas.itemconfig(inner_window, width=event.width)

    canvas.bind("<Configure>", _on_canvas_resize)
    canvas.configure(yscrollcommand=scrollbar.set)#اتصال اسکرول سمت راست
    canvas.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side="right", fill="y")

    def _wheel(event):#چرخش موس 
        canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    canvas.bind("<Enter>", lambda _: canvas.bind_all("<MouseWheel>", _wheel))
    canvas.bind("<Leave>", lambda _: canvas.unbind_all("<MouseWheel>"))
    return scrollable_frame


def section_card(parent, title, icon=""):#مثلا تب دوم دیتاست و شکل کنارش و خط زیر
    """یک کارت بخش با عنوان، برای یکدست‌کردن ظاهر تب‌ها."""
    outer = tk.Frame(parent, bg=COLOR_BG_LIGHT)
    outer.pack(fill="x", padx=18, pady=(14, 4))#عرض قاب و محل
    tk.Label(outer, text=f"{icon}  {title}", font=FONT_H2, bg=COLOR_BG_LIGHT, fg=COLOR_ACCENT,
              anchor="w", justify="left").pack(fill="x")#فونت و رنگ اینا 
    tk.Frame(outer, bg=COLOR_ACCENT, height=2).pack(fill="x", pady=(2, 0))#خط زیرش
    return outer

#دکمه وکاراش و رنگش
def styled_button(parent, text, command, bg=COLOR_ACCENT, fg="white", big=False):
    return tk.Button(
        parent, text=text, command=command, bg=bg, fg=fg,
        activebackground=bg, activeforeground=fg,
        font=FONT_BTN if big else FONT_BODY, relief="flat",
        padx=16, pady=10 if big else 6, cursor="hand2",
        borderwidth=0, highlightthickness=0,
    )


def launch_gui():
    global app_window
    global selected_ready_train_folder, selected_ready_test_folder
    global ready_train_label, ready_test_label, ready_status_label, ready_import_button
    global public_status_label, public_download_button, public_limit_scale
    global dataset_count_label, epoch_scale, aug_scale, thresh_scale, fine_tune_scale
    global manual_weight_scale, highacc_var
    global eval_button, eval_status_label, selftest_button, selftest_status_label
    global _header_bg_photo

    selected_ready_train_folder = None
    selected_ready_test_folder = None
    _header_bg_photo = None

    window = tk.Tk()
    app_window = window
    window.title("تشخیص آتش با شبکه‌ی کانولوشنی (طبقه‌بندی)")
    window.geometry("960x800")
    window.minsize(720, 520)
    window.configure(bg=COLOR_BG_LIGHT)
    window.resizable(True, True)

    style = ttk.Style()
    try:
        style.theme_use("clam")
    except Exception:
        pass
    style.configure("TNotebook", background=COLOR_BG_LIGHT, borderwidth=0)
    style.configure("TNotebook.Tab", font=("Segoe UI", 10, "bold"), padding=[16, 10],
                     background="#dfe3e8", foreground=COLOR_TEXT)
    #هنگام انتخاب تب نارنجی شه متن سفید
    style.map("TNotebook.Tab",
              background=[("selected", COLOR_ACCENT)],
              foreground=[("selected", "white")])
    style.configure("TFrame", background=COLOR_BG_LIGHT)#ظاهر فریم ها 

    # ---------- نوار بالای برنامه (با پس‌زمینه‌ی آتشین) ----------
    HEADER_H = 110
    header = tk.Canvas(window, height=HEADER_H, highlightthickness=0, bd=0, bg=COLOR_BG)
    header.pack(fill="x", side="top")
    header.pack_propagate(False)

    bg_path = _ensure_header_background()
    if bg_path:
        try:
            src = Image.open(bg_path).convert("RGB")
            sw, sh = src.size
            scale = HEADER_H / sh
            new_w = max(1, int(sw * scale))
            src = src.resize((new_w, HEADER_H))
            # تیره‌کردن ملایم عکس تا متن سفید روش خوانا بمونه
            from PIL import ImageEnhance
            src = ImageEnhance.Brightness(src).enhance(0.55)
            _header_bg_photo = ImageTk.PhotoImage(src)

            def _draw_header_bg(event=None):
                header.delete("bgimg")
                win_w = header.winfo_width() or 960
                x = 0
                while x < win_w:
                    header.create_image(x, 0, image=_header_bg_photo, anchor="nw", tags="bgimg")
                    x += new_w
                header.tag_lower("bgimg")

            header.bind("<Configure>", _draw_header_bg)
            header.after(50, _draw_header_bg)
        except Exception:
            pass
#تایتلش
    header.create_text(24, HEADER_H // 2 - 16, anchor="w", text="تشخیص آتش با شبکه‌ی کانولوشنی",
                        font=("Segoe UI", 18, "bold"), fill="white")
    header.create_text(24, HEADER_H // 2 + 14, anchor="w", text="یادگیری انتقالی با MobileNetV2",
                        font=("Segoe UI", 11), fill="#ffe3cf")
#مختصات تگ ها 
    notebook = ttk.Notebook(window)
    notebook.pack(fill="both", expand=True, padx=14, pady=14)

    # ---------------- تب ۱: برچسب‌گذاری ----------------
    tab_tag = tk.Frame(notebook, bg=COLOR_BG_LIGHT)
    notebook.add(tab_tag, text="۱- برچسب‌گذاری دستی")
    QuickTagger(tab_tag)

    # ---------------- تب ۲: دیتاست ----------------
    tab_dataset = tk.Frame(notebook, bg=COLOR_BG_LIGHT)
    notebook.add(tab_dataset, text="۲- دیتاست")
    ds_scroll = make_scrollable(tab_dataset)

    section_card(ds_scroll, "وضعیت فعلی دیتاست", "📊")
    dataset_count_label = tk.Label(ds_scroll, text="", font=FONT_BODY, bg=COLOR_BG_LIGHT,
                                    fg=COLOR_TEXT, wraplength=820, justify="center")
    dataset_count_label.pack(pady=8)
    row1 = tk.Frame(ds_scroll, bg=COLOR_BG_LIGHT)
    row1.pack(pady=4)
    styled_button(row1, "به‌روزرسانی شمارش", refresh_dataset_count).pack(side="right", padx=6)
    styled_button(row1, "پیش‌نمایش نمونه‌ها", lambda: preview_dataset_samples(9),
                  bg=COLOR_GREEN).pack(side="right", padx=6)

    section_card(ds_scroll, "درون‌ریزی دیتاست آماده (تصویر + برچسب txt)", "📥")
    ready_frame = tk.Frame(ds_scroll, bg="white", padx=14, pady=14)
    ready_frame.pack(fill="x", padx=18, pady=6)
    styled_button(ready_frame, "انتخاب پوشه‌ی آموزش", choose_ready_train_folder).pack(pady=4)
    ready_train_label = tk.Label(ready_frame, text="پوشه‌ی آموزش انتخاب نشده", wraplength=760,
                                  fg=COLOR_MUTED, bg="white", font=FONT_BODY)
    ready_train_label.pack(pady=2)
    styled_button(ready_frame, "انتخاب پوشه‌ی آزمون", choose_ready_test_folder).pack(pady=4)
    ready_test_label = tk.Label(ready_frame, text="پوشه‌ی آزمون انتخاب نشده", wraplength=760,
                                 fg=COLOR_MUTED, bg="white", font=FONT_BODY)
    ready_test_label.pack(pady=2)
    ready_import_button = styled_button(ready_frame, "درون‌ریزی و ساخت دیتاست از txt‌ها",
                                         start_ready_import, big=True)
    ready_import_button.pack(pady=10, fill="x")
    ready_status_label = tk.Label(ready_frame, text="", fg=COLOR_ACCENT, bg="white",
                                   wraplength=760, font=FONT_BODY)
    ready_status_label.pack(pady=3)

    section_card(ds_scroll, "دانلود خودکار دیتاست عمومی (اختیاری)", "🌐")
    public_frame = tk.Frame(ds_scroll, bg="white", padx=14, pady=14)
    public_frame.pack(fill="x", padx=18, pady=6)
    tk.Label(public_frame, text="حداقل تعداد نمونه از هر کلاس:", bg="white", font=FONT_BODY).pack()
    public_limit_scale = tk.Scale(public_frame, from_=50, to=900, orient=tk.HORIZONTAL, length=420,
                                   bg="white", highlightthickness=0, troughcolor="#e0e0e0",
                                   activebackground=COLOR_ACCENT)
    public_limit_scale.set(400)
    public_limit_scale.pack()
    public_download_button = styled_button(public_frame, "دانلود و آماده‌سازی خودکار",
                                            start_public_download, big=True)
    public_download_button.pack(pady=10, fill="x")
    public_status_label = tk.Label(public_frame, text="", fg=COLOR_ACCENT, bg="white",
                                    wraplength=760, font=FONT_BODY)
    public_status_label.pack(pady=3)

    # ---------------- تب ۳: آموزش و ارزیابی ----------------
    tab_train = tk.Frame(notebook, bg=COLOR_BG_LIGHT)
    notebook.add(tab_train, text="۳- آموزش و ارزیابی")
    tr_scroll = make_scrollable(tab_train)

    section_card(tr_scroll, "پارامترهای آموزش", "⚙️")
    params_frame = tk.Frame(tr_scroll, bg="white", padx=14, pady=14)
    params_frame.pack(fill="x", padx=18, pady=6)

    tk.Label(params_frame, text="تعداد دوره‌های آموزش (epochs):", bg="white", font=FONT_BODY).pack(pady=(6, 0))
    epoch_scale = tk.Scale(params_frame, from_=5, to=100, orient=tk.HORIZONTAL, length=420,
                            bg="white", highlightthickness=0, troughcolor="#e0e0e0",
                            activebackground=COLOR_ACCENT)
    epoch_scale.set(20)
    epoch_scale.pack()

    tk.Label(params_frame, text="ضریب افزایش داده (Augmentation):", bg="white", font=FONT_BODY).pack(pady=(14, 0))
    aug_scale = tk.Scale(params_frame, from_=1, to=10, orient=tk.HORIZONTAL, length=420,
                          bg="white", highlightthickness=0, troughcolor="#e0e0e0",
                          activebackground=COLOR_ACCENT)
    aug_scale.set(2)
    aug_scale.pack()

    tk.Label(params_frame, text="ضریب اهمیت عکس‌های دستی (Manual Oversample):",
             bg="white", font=FONT_BODY).pack(pady=(14, 0))
    manual_weight_scale = tk.Scale(params_frame, from_=1, to=15, orient=tk.HORIZONTAL, length=420,
                                    bg="white", highlightthickness=0, troughcolor="#e0e0e0",
                                    activebackground=COLOR_ACCENT)
    manual_weight_scale.set(5)
    manual_weight_scale.pack()
    tk.Label(params_frame,
             text="هر عکسی که توی تب ۱ دستی برچسب زدی، به این تعداد توی\n"
                  "آموزش تکرار می‌شود -- عدد بیشتر یعنی مدل کمتر از\n"
                  "رویش می‌گذرد و بهتر یادش می‌گیرد.",
             fg=COLOR_MUTED, bg="white", wraplength=700, justify="center", font=FONT_BODY).pack(pady=(4, 4))

    tk.Label(params_frame, text="دوره‌های فاین-تیون لایه‌های آخر MobileNetV2 (۰ = خاموش):",
             bg="white", font=FONT_BODY).pack(pady=(14, 0))
    fine_tune_scale = tk.Scale(params_frame, from_=0, to=40, orient=tk.HORIZONTAL, length=420,
                                bg="white", highlightthickness=0, troughcolor="#e0e0e0",
                                activebackground=COLOR_ACCENT)
    fine_tune_scale.set(10)
    fine_tune_scale.pack()
    tk.Label(params_frame,
             text="مدل بعد از آموزشِ سرِ طبقه‌بندی، چند لایه‌ی آخر backbone را\n"
                  "هم با نرخ یادگیری خیلی کم تنظیم می‌کند تا با ظاهر دقیق آتش/دودِ\n"
                  "ویدیوی خودت سازگار شود.",
             fg=COLOR_MUTED, bg="white", wraplength=700, justify="center", font=FONT_BODY).pack(pady=(6, 10))

    section_card(tr_scroll, "آموزش و ارزیابی", "🎓")
    action_frame = tk.Frame(tr_scroll, bg=COLOR_BG_LIGHT)
    action_frame.pack(fill="x", padx=18, pady=6)
    styled_button(action_frame, "آموزش مدل طبقه‌بندی آتش", train_classifier,
                  big=True).pack(pady=8, fill="x")
    styled_button(action_frame, "بارگذاری مدل ذخیره‌شده", manual_load_model,
                  bg="#607d8b").pack(pady=6, fill="x")

    eval_button = styled_button(action_frame, "ارزیابی روی دیتاست آزمون واقعی", evaluate_on_test_set,
                                 bg=COLOR_GREEN, big=True)
    eval_button.pack(pady=6, fill="x")
    eval_status_label = tk.Label(action_frame, text="", fg=COLOR_ACCENT, bg=COLOR_BG_LIGHT, font=FONT_BODY)
    eval_status_label.pack(pady=2)

    selftest_button = styled_button(action_frame, "خودآزمایی روی نمونه‌های آموزش",
                                     lambda: self_test_on_training_samples(10),
                                     bg="#fbc02d", fg="#212121")
    selftest_button.pack(pady=6, fill="x")
    selftest_status_label = tk.Label(action_frame, text="", fg=COLOR_ACCENT, bg=COLOR_BG_LIGHT, font=FONT_BODY)
    selftest_status_label.pack(pady=2)

    # ---------------- تب ۴: تشخیص ----------------
    tab_detect = tk.Frame(notebook, bg=COLOR_BG_LIGHT)
    notebook.add(tab_detect, text="۴- تشخیص آتش")
    dt_scroll = make_scrollable(tab_detect)

    section_card(dt_scroll, "تنظیمات تشخیص", "🎯")
    settings_frame = tk.Frame(dt_scroll, bg="white", padx=14, pady=14)
    settings_frame.pack(fill="x", padx=18, pady=6)

    tk.Label(settings_frame, text="آستانه‌ی تشخیص (٪):", bg="white", font=FONT_BODY).pack(pady=(6, 0))
    thresh_scale = tk.Scale(settings_frame, from_=10, to=95, orient=tk.HORIZONTAL, length=420,
                             bg="white", highlightthickness=0, troughcolor="#e0e0e0",
                             activebackground=COLOR_ACCENT)
    thresh_scale.set(50)
    thresh_scale.pack(pady=(0, 12))

    highacc_var = tk.BooleanVar(value=True)
    tk.Checkbutton(
        settings_frame, variable=highacc_var,
        text="حالت دقت بالا (چندمقیاسی) -- کندتر ولی در برابر انسان/حرکت محکم‌تر است",
        font=FONT_BODY, bg="white", activebackground="white"
    ).pack(pady=(4, 8))

    section_card(dt_scroll, "اجرای تشخیص", "🔥")
    detect_frame = tk.Frame(dt_scroll, bg=COLOR_BG_LIGHT)
    detect_frame.pack(fill="x", padx=18, pady=6)
    styled_button(detect_frame, "تشخیص آتش در عکس", detect_fire_image, big=True).pack(pady=8, fill="x")
    styled_button(detect_frame, "تشخیص آتش در ویدیو", detect_fire_video, big=True).pack(pady=8, fill="x")
    styled_button(detect_frame, "تشخیص آتش زنده (وبکم)", detect_fire_webcam,
                  bg="#ff7043", big=True).pack(pady=8, fill="x")

    tk.Label(dt_scroll, text="در حین پخش ویدیو/وبکم، کلید ESC برای خروج است.\n"
                             "پنجره‌ی ویدیو/وبکم را می‌شود با موس resize کرد، خودش scale می‌شود.",
             fg=COLOR_MUTED, bg=COLOR_BG_LIGHT, font=FONT_BODY, justify="center").pack(pady=(6, 16))

    try_load_model_on_startup()
    refresh_dataset_count()
    window.mainloop()


def main():
    launch_gui()


if __name__ == "__main__":
    main()