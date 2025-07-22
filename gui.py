# -*- coding: utf-8 -*-
import sys
import time
from PyQt5.QtGui import (
    QBrush,
    QPainter,
    QPen,
    QPixmap,
    QKeySequence,
    QPen,
    QBrush,
    QColor,
    QImage,
)
from PyQt5.QtWidgets import (
    QFileDialog,
    QApplication,
    QGraphicsEllipseItem,
    QGraphicsItem,
    QGraphicsRectItem,
    QGraphicsScene,
    QGraphicsView,
    QGraphicsPixmapItem,
    QHBoxLayout,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
    QShortcut,
)
""" 07-22-2025 update
 - updated path to model checkpoint"""

import numpy as np
from skimage import transform, io
import torch
import torch.nn as nn
from torch.nn import functional as F
from PIL import Image
from segment_anything import sam_model_registry

# freeze seeds
torch.manual_seed(2023)
torch.cuda.empty_cache()
torch.cuda.manual_seed(2023)
np.random.seed(2023)

SAM_MODEL_TYPE = "vit_b"
MedSAM_CKPT_PATH = "/home/medsam-vit-b/medsam_vit_b.pth"
MEDSAM_IMG_INPUT_SIZE = 1024

if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def medsam_inference(medsam_model, img_embed, box_1024, height, width):
    box_torch = torch.as_tensor(box_1024, dtype=torch.float, device=img_embed.device)
    if len(box_torch.shape) == 2:
        box_torch = box_torch[:, None, :]  # (B, 1, 4)

    sparse_embeddings, dense_embeddings = medsam_model.prompt_encoder(
        points=None,
        boxes=box_torch,
        masks=None,
    )
    low_res_logits, _ = medsam_model.mask_decoder(
        image_embeddings=img_embed,  # (B, 256, 64, 64)
        image_pe=medsam_model.prompt_encoder.get_dense_pe(),  # (1, 256, 64, 64)
        sparse_prompt_embeddings=sparse_embeddings,  # (B, 2, 256)
        dense_prompt_embeddings=dense_embeddings,  # (B, 256, 64, 64)
        multimask_output=False,
    )

    low_res_pred = torch.sigmoid(low_res_logits)  # (1, 1, 256, 256)

    low_res_pred = F.interpolate(
        low_res_pred,
        size=(height, width),
        mode="bilinear",
        align_corners=False,
    )  # (1, 1, gt.shape)
    low_res_pred = low_res_pred.squeeze().cpu().numpy()  # (256, 256)
    medsam_seg = (low_res_pred > 0.5).astype(np.uint8)
    return medsam_seg


print("Loading MedSAM model, a sec.")
tic = time.perf_counter()

# set up model
medsam_model = sam_model_registry["vit_b"](checkpoint=MedSAM_CKPT_PATH).to(device)
medsam_model.eval()

print(f"Done, took {time.perf_counter() - tic}")


def np2pixmap(np_img):
    height, width, channel = np_img.shape
    bytesPerLine = 3 * width
    qImg = QImage(np_img.data, width, height, bytesPerLine, QImage.Format_RGB888)
    return QPixmap.fromImage(qImg)


colors = [
    (255, 0, 0),
    (0, 255, 0),
    (0, 0, 255),
    (255, 255, 0),
    (255, 0, 255),
    (0, 255, 255),
    (128, 0, 0),
    (0, 128, 0),
    (0, 0, 128),
    (128, 128, 0),
    (128, 0, 128),
    (0, 128, 128),
    (255, 255, 255),
    (192, 192, 192),
    (64, 64, 64),
    (255, 0, 255),
    (0, 255, 255),
    (255, 255, 0),
    (0, 0, 127),
    (192, 0, 192),
]


class Window(QWidget):
    def __init__(self):
        super().__init__()

        # configs
        self.half_point_size = 5  # radius of bbox starting and ending points

        # app stats
        self.image_path = None
        self.color_idx = 0
        self.bg_img = None
        self.is_mouse_down = False
        self.rect = None
        self.point_size = self.half_point_size * 2
        self.start_point = None
        self.end_point = None
        self.start_pos = (None, None)
        self.embedding = None
        self.prev_mask = None

        self.view = QGraphicsView()
        self.view.setRenderHint(QPainter.Antialiasing)

        #pixmap = self.load_image() Don't autoload image anymore after edits

        vbox = QVBoxLayout(self)
        vbox.addWidget(self.view)

        load_button = QPushButton("Load Image")
        save_button = QPushButton("Save Mask")

        hbox = QHBoxLayout(self)
        hbox.addWidget(load_button)
        hbox.addWidget(save_button)

        vbox.addLayout(hbox)

        self.setLayout(vbox)

        # keyboard shortcuts
        self.quit_shortcut = QShortcut(QKeySequence("Ctrl+Q"), self)
        self.quit_shortcut.activated.connect(lambda: quit())

        self.undo_shortcut = QShortcut(QKeySequence("Ctrl+Z"), self)
        self.undo_shortcut.activated.connect(self.undo)

        load_button.clicked.connect(self.load_image)
        save_button.clicked.connect(self.save_mask)

    def undo(self):
        if self.prev_mask is None:
            print("No previous mask record")
            return

        self.color_idx -= 1

        bg = Image.fromarray(self.img_3c.astype("uint8"), "RGB")
        mask = Image.fromarray(self.prev_mask.astype("uint8"), "RGB")
        img = Image.blend(bg, mask, 0.2)

        self.scene.removeItem(self.bg_img)
        self.bg_img = self.scene.addPixmap(np2pixmap(np.array(img)))

        self.mask_c = self.prev_mask
        self.prev_mask = None
#------------Updated image loading method 07-22-2025----------------

    def add_navigation_buttons(self):
        """Add navigation buttons for dataset browsing"""
        from PyQt5.QtWidgets import QInputDialog
        
        # Create navigation layout if it doesn't exist
        if not hasattr(self, 'nav_layout'):
            nav_layout = QHBoxLayout()
            
            prev_btn = QPushButton("← Previous")
            next_btn = QPushButton("Next →")
            goto_btn = QPushButton("Go to...")
            info_btn = QPushButton("Info")
            
            prev_btn.clicked.connect(self.prev_dataset_image)
            next_btn.clicked.connect(self.next_dataset_image)
            goto_btn.clicked.connect(self.goto_dataset_image)
            info_btn.clicked.connect(self.show_dataset_info)
            
            nav_layout.addWidget(prev_btn)
            nav_layout.addWidget(next_btn)
            nav_layout.addWidget(goto_btn)
            nav_layout.addWidget(info_btn)
            
            # Insert navigation at the top
            self.layout().insertLayout(0, nav_layout)
            self.nav_layout = nav_layout

    def prev_dataset_image(self):
        """Load previous image in dataset"""
        if hasattr(self, 'dataset') and self.current_idx > 0:
            self.current_idx -= 1
            self.load_dataset_image_by_index(self.current_idx)
        else:
            print("Already at first image or no dataset loaded")

    def next_dataset_image(self):
        """Load next image in dataset"""
        if hasattr(self, 'dataset') and self.current_idx < len(self.dataset[self.current_split]) - 1:
            self.current_idx += 1
            self.load_dataset_image_by_index(self.current_idx)
        else:
            print("Already at last image or no dataset loaded")

    def goto_dataset_image(self):
        """Go to specific image in dataset"""
        if not hasattr(self, 'dataset'):
            print("No dataset loaded")
            return
            
        from PyQt5.QtWidgets import QInputDialog
        max_idx = len(self.dataset[self.current_split]) - 1
        idx, ok = QInputDialog.getInt(self, 'Go to Image', 
                                    f'Enter index (0-{max_idx}):', 
                                    self.current_idx, 0, max_idx)
        if ok:
            self.current_idx = idx
            self.load_dataset_image_by_index(idx)

    def show_dataset_info(self):
        """Show current dataset info"""
        if hasattr(self, 'dataset'):
            total = len(self.dataset[self.current_split])
            print(f"Dataset Info: Image {self.current_idx + 1}/{total} in split '{self.current_split}'")
        else:
            print("No dataset loaded")

    def load_dataset_image_by_index(self, idx):
        """Load specific image from dataset"""
        sample = self.dataset[self.current_split][idx]
        pil_image = sample['image']
        img_3c = np.array(pil_image)
        
        if len(img_3c.shape) == 2:
            img_3c = np.repeat(img_3c[:, :, None], 3, axis=-1)
            
        virtual_path = f"dataset_{self.current_split}_{idx}.png"
        print(f"Loading image {idx + 1}/{len(self.dataset[self.current_split])}")
        self.setup_image(img_3c, virtual_path)

    def load_image(self):
        # Choice dialog: file browser or HuggingFace dataset
        from PyQt5.QtWidgets import QMessageBox
        
        reply = QMessageBox.question(self, 'Image Source', 
                                    'Load from:\n"File" - Browse local files\n"Dataset" - Use HuggingFace dataset',
                                    QMessageBox.No | QMessageBox.Yes, 
                                    QMessageBox.Yes)
        
        if reply == QMessageBox.Yes:  # Dataset
            self.load_from_dataset()
        else:  # File
            self.load_from_file()

    def load_from_file(self):
        """Original file loading method"""
        file_path, file_type = QFileDialog.getOpenFileName(
            self, "Choose Image to Segment", ".", "Image Files (*.png *.jpg *.bmp)"
        )

        if file_path is None or len(file_path) == 0:
            print("No image path specified, plz select an image")
            return

        img_np = io.imread(file_path)
        if len(img_np.shape) == 2:
            img_3c = np.repeat(img_np[:, :, None], 3, axis=-1)
        else:
            img_3c = img_np

        self.setup_image(img_3c, file_path)

    def load_from_dataset(self):
        """Load from HuggingFace dataset"""
        # Import here to avoid dependency issues if not using dataset
        try:
            from datasets import load_dataset
            import huggingface_hub
            from PyQt5.QtWidgets import QInputDialog
            
            # Get dataset info
            dataset_name, ok = QInputDialog.getText(self, 'Dataset Name', 'Enter HuggingFace dataset name:')
            if not ok or not dataset_name:
                return
                
            # Load dataset
            print(f"Loading dataset: {dataset_name}")
            dataset = load_dataset(dataset_name, token=True)
            
            # Get available splits
            splits = list(dataset.keys())
            split_name, ok = QInputDialog.getItem(self, 'Dataset Split', 'Choose split:', splits, 0, False)
            if not ok:
                return
                
            # Get image index
            max_idx = len(dataset[split_name]) - 1
            img_idx, ok = QInputDialog.getInt(self, 'Image Index', f'Enter image index (0-{max_idx}):', 0, 0, max_idx)
            if not ok:
                return
                
            # Load the image
            sample = dataset[split_name][img_idx]
            pil_image = sample['image']
            img_3c = np.array(pil_image)
            
            # Convert grayscale to RGB if needed
            if len(img_3c.shape) == 2:
                img_3c = np.repeat(img_3c[:, :, None], 3, axis=-1)
                
            # Create a virtual path for saving
            virtual_path = f"{dataset_name}_{split_name}_{img_idx}.png"
            
            print(f"Loaded image #{img_idx} from {dataset_name}/{split_name}")
            self.setup_image(img_3c, virtual_path)
            
            # Store dataset info for navigation
            self.dataset = dataset
            self.current_split = split_name
            self.current_idx = img_idx
            
        except ImportError:
            print("datasets library not available. Install with: pip install datasets")
        except Exception as e:
            print(f"Error loading dataset: {e}")

    def setup_image(self, img_3c, image_path):
        """Common image setup code"""
        self.img_3c = img_3c
        self.image_path = image_path
        self.get_embeddings()
        pixmap = np2pixmap(self.img_3c)

        H, W, _ = self.img_3c.shape

        self.scene = QGraphicsScene(0, 0, W, H)
        self.end_point = None
        self.rect = None
        self.bg_img = self.scene.addPixmap(pixmap)
        self.bg_img.setPos(0, 0)
        self.mask_c = np.zeros((*self.img_3c.shape[:2], 3), dtype="uint8")
        self.view.setScene(self.scene)

        # events
        self.scene.mousePressEvent = self.mouse_press
        self.scene.mouseMoveEvent = self.mouse_move
        self.scene.mouseReleaseEvent = self.mouse_release
        
        # Add navigation buttons if dataset is loaded
        if hasattr(self, 'dataset'):
            self.add_navigation_buttons()       
 #-------------------------------------------------------   
    # def load_image(self): *original script*
    #     file_path, file_type = QFileDialog.getOpenFileName(
    #         self, "Choose Image to Segment", ".", "Image Files (*.png *.jpg *.bmp)"
    #     )

    #     if file_path is None or len(file_path) == 0:
    #         print("No image path specified, plz select an image")
    #         exit()

    #     img_np = io.imread(file_path)
    #     if len(img_np.shape) == 2:
    #         img_3c = np.repeat(img_np[:, :, None], 3, axis=-1)
    #     else:
    #         img_3c = img_np

    #     self.img_3c = img_3c
    #     self.image_path = file_path
    #     self.get_embeddings()
    #     pixmap = np2pixmap(self.img_3c)

    #     H, W, _ = self.img_3c.shape

    #     self.scene = QGraphicsScene(0, 0, W, H)
    #     self.end_point = None
    #     self.rect = None
    #     self.bg_img = self.scene.addPixmap(pixmap)
    #     self.bg_img.setPos(0, 0)
    #     self.mask_c = np.zeros((*self.img_3c.shape[:2], 3), dtype="uint8")
    #     self.view.setScene(self.scene)

    #     # events
    #     self.scene.mousePressEvent = self.mouse_press
    #     self.scene.mouseMoveEvent = self.mouse_move
    #     self.scene.mouseReleaseEvent = self.mouse_release

    def mouse_press(self, ev):
        x, y = ev.scenePos().x(), ev.scenePos().y()
        self.is_mouse_down = True
        self.start_pos = ev.scenePos().x(), ev.scenePos().y()
        self.start_point = self.scene.addEllipse(
            x - self.half_point_size,
            y - self.half_point_size,
            self.point_size,
            self.point_size,
            pen=QPen(QColor("red")),
            brush=QBrush(QColor("red")),
        )

    def mouse_move(self, ev):
        if not self.is_mouse_down:
            return

        x, y = ev.scenePos().x(), ev.scenePos().y()

        if self.end_point is not None:
            self.scene.removeItem(self.end_point)
        self.end_point = self.scene.addEllipse(
            x - self.half_point_size,
            y - self.half_point_size,
            self.point_size,
            self.point_size,
            pen=QPen(QColor("red")),
            brush=QBrush(QColor("red")),
        )

        if self.rect is not None:
            self.scene.removeItem(self.rect)
        sx, sy = self.start_pos
        xmin = min(x, sx)
        xmax = max(x, sx)
        ymin = min(y, sy)
        ymax = max(y, sy)
        self.rect = self.scene.addRect(
            xmin, ymin, xmax - xmin, ymax - ymin, pen=QPen(QColor("red"))
        )

    def mouse_release(self, ev):
        x, y = ev.scenePos().x(), ev.scenePos().y()
        sx, sy = self.start_pos
        xmin = min(x, sx)
        xmax = max(x, sx)
        ymin = min(y, sy)
        ymax = max(y, sy)

        self.is_mouse_down = False

        H, W, _ = self.img_3c.shape
        box_np = np.array([[xmin, ymin, xmax, ymax]])
        # print("bounding box:", box_np)
        box_1024 = box_np / np.array([W, H, W, H]) * 1024

        sam_mask = medsam_inference(medsam_model, self.embedding, box_1024, H, W)

        self.prev_mask = self.mask_c.copy()
        self.mask_c[sam_mask != 0] = colors[self.color_idx % len(colors)]
        self.color_idx += 1

        bg = Image.fromarray(self.img_3c.astype("uint8"), "RGB")
        mask = Image.fromarray(self.mask_c.astype("uint8"), "RGB")
        img = Image.blend(bg, mask, 0.2)

        self.scene.removeItem(self.bg_img)
        self.bg_img = self.scene.addPixmap(np2pixmap(np.array(img)))

    def save_mask(self):
        out_path = f"{self.image_path.split('.')[0]}_mask.png"
        io.imsave(out_path, self.mask_c)

    @torch.no_grad()
    def get_embeddings(self):
        print("Calculating embedding, gui may be unresponsive.")
        img_1024 = transform.resize(
            self.img_3c, (1024, 1024), order=3, preserve_range=True, anti_aliasing=True
        ).astype(np.uint8)
        img_1024 = (img_1024 - img_1024.min()) / np.clip(
            img_1024.max() - img_1024.min(), a_min=1e-8, a_max=None
        )  # normalize to [0, 1], (H, W, 3)
        # convert the shape to (3, H, W)
        img_1024_tensor = (
            torch.tensor(img_1024).float().permute(2, 0, 1).unsqueeze(0).to(device)
        )

        # if self.embedding is None:
        with torch.no_grad():
            self.embedding = medsam_model.image_encoder(
                img_1024_tensor
            )  # (1, 256, 64, 64)
        print("Done.")


app = QApplication(sys.argv)

w = Window()
w.show()

app.exec()
