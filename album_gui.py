import sys
import os
from pathlib import Path
from typing import List, Dict, Optional

from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QListWidget, QListWidgetItem, 
                             QGraphicsView, QGraphicsScene, QGraphicsRectItem, 
                             QGraphicsPixmapItem, QFileDialog, QLabel, QProgressBar,
                             QSplitter, QMessageBox, QFrame)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QSize, QMimeData, QPointF
from PyQt6.QtGui import QPixmap, QDrag, QImage, QPainter, QColor, QPen, QIcon

import sa_advanced as sa
from PIL import Image, ImageOps

class OptimizationThread(QThread):
    progress = pyqtSignal(int, int)
    finished_optim = pyqtSignal(list)
    
    def __init__(self, root, page_W, page_H, images, prefs, locked_leaves):
        super().__init__()
        self.root = root
        self.page_W = page_W
        self.page_H = page_H
        self.images = images
        self.prefs = prefs
        self.locked_leaves = locked_leaves
        
    def run(self):
        # We don't need snapshots for the optimization run itself unless we want to display them
        # Let's just run it.
        perm, _ = sa.anneal_with_snapshots(
            root=self.root,
            page_W=self.page_W,
            page_H=self.page_H,
            images=self.images,
            prefs=self.prefs,
            page_margin_px=50, # Scaled down for GUI? Or full size? 
                          # Ideally we use full size logic but simple coordinates.
            gap_px=20,
            steps=5000, # Faster for interactive
            snapshots_count=0,
            seed=42, # Make random?
            desc="Optimizing",
            locked_leaves=self.locked_leaves,
            progress_callback=self.emit_progress
        )
        self.finished_optim.emit(perm)
        
    def emit_progress(self, step, total):
        self.progress.emit(step, total)

class LeafItem(QGraphicsRectItem):
    def __init__(self, x, y, w, h, leaf_id, parent_gui):
        super().__init__(x, y, w, h)
        self.leaf_id = leaf_id
        self.parent_gui = parent_gui
        self.setAcceptDrops(True)
        self.setPen(QPen(Qt.GlobalColor.black))
        self.setBrush(QColor(240, 240, 240))
        
        self.pixmap_item = QGraphicsPixmapItem(self)
        self.pixmap_item.setTransformationMode(Qt.TransformationMode.SmoothTransformation)
        
        # Lock icon (simple red border or overlay for now)
        self.is_locked = False
        
    def paint(self, painter, option, widget):
        super().paint(painter, option, widget)
        if self.is_locked:
            painter.setPen(QPen(Qt.GlobalColor.red, 4))
            painter.drawRect(self.rect())
            
    def dropEvent(self, event):
        if event.mimeData().hasFormat("application/x-image-idx"):
            data = event.mimeData().data("application/x-image-idx")
            img_idx = int(data.data().decode())
            self.parent_gui.handle_drop(self.leaf_id, img_idx)
            event.accept()
        else:
            event.ignore()

class AlbumWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Esthetic Album Designer")
        self.resize(1200, 800)
        
        # State
        self.image_paths: List[Path] = []
        self.root: Optional[sa.Node] = None
        self.perm: List[int] = [] # Maps leaf_id -> image_idx
        self.locked_leaves: Dict[int, int] = {} # leaf_id -> image_idx
        
        self.page_W = 1000  # Internal logic size
        self.page_H = 1414  # ~A4 Aspect
        
        self.init_ui()
        
    def init_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QHBoxLayout(main_widget)
        
        # Left: Images
        left_layout = QVBoxLayout()
        self.load_btn = QPushButton("Load Folder")
        self.load_btn.clicked.connect(self.load_images_dialog)
        left_layout.addWidget(self.load_btn)
        
        self.image_list = QListWidget()
        self.image_list.setIconSize(QSize(100, 100))
        self.image_list.setDragEnabled(True)
        self.image_list.setDragDropMode(QListWidget.DragDropMode.DragOnly)
        left_layout.addWidget(self.image_list)
        
        # Center: Canvas
        self.scene = QGraphicsScene()
        self.view = QGraphicsView(self.scene)
        self.view.setBackgroundBrush(QColor(50, 50, 50))
        
        # Right: Controls
        right_layout = QVBoxLayout()
        self.optimize_btn = QPushButton("✨ Optimize Layout")
        self.optimize_btn.clicked.connect(self.start_optimization)
        self.optimize_btn.setEnabled(False)
        self.reset_btn = QPushButton("Reset Layout")
        self.reset_btn.clicked.connect(self.reset_layout)
        
        self.progress_bar = QProgressBar()
        
        right_layout.addWidget(self.optimize_btn)
        right_layout.addWidget(self.reset_btn)
        right_layout.addStretch()
        right_layout.addWidget(self.progress_bar)
        
        splitter = QSplitter(Qt.Orientation.Horizontal)
        left_widget = QWidget()
        left_widget.setLayout(left_layout)
        
        right_widget = QWidget()
        right_widget.setLayout(right_layout)
        
        splitter.addWidget(left_widget)
        splitter.addWidget(self.view)
        splitter.addWidget(right_widget)
        splitter.setStretchFactor(1, 2)
        
        layout.addWidget(splitter)

    def load_images_dialog(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Image Folder")
        if folder:
            self.load_images(folder)

    def load_images(self, folder):
        self.image_paths = []
        self.image_list.clear()
        
        folder_path = Path(folder)
        exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
        files = sorted([p for p in folder_path.rglob("*") if p.suffix.lower() in exts])
        
        if not files:
            return

        # For demo, limit to e.g. 16 images for one page or handle paging?
        # Let's take first 16 or nearest power of 2 for simplicity
        # Code supports arbitrary power of 2.
        
        count = len(files)
        # Find largest power of 2 <= count for one page
        if count == 0: return
        # Limit to 32 max for this demo
        count = min(count, 32)
        k = sa.largest_power_of_two_leq(count)
        self.image_paths = files[:k]
        
        for idx, p in enumerate(self.image_paths):
            # Load thumbnail
            item = QListWidgetItem()
            # Efficient thumbnail loading?
            pix = QPixmap(str(p)).scaled(100, 100, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
            item.setIcon(QIcon(pix))
            item.setData(Qt.ItemDataRole.UserRole, idx) # Store index
            item.setText(p.name)
            self.image_list.addItem(item)
            
        self.init_tree()
        self.optimize_btn.setEnabled(True)

    def init_tree(self):
        L = len(self.image_paths)
        if L == 0: return
        
        try:
            self.root = sa.build_full_tree(L, seed=42)
        except AssertionError:
             # Retry with smaller power of 2
             pass
             
        # Initial perm: random
        import random
        self.perm = list(range(L))
        random.shuffle(self.perm)
        self.locked_leaves = {}
        
        self.draw_layout()

    def draw_layout(self):
        self.scene.clear()
        if not self.root: return
        
        margin = 20
        W, H = self.page_W, self.page_H
        in_W = W - 2*margin
        in_H = H - 2*margin
        
        boxes = sa.decode_region(self.root, margin, margin, in_W, in_H)
        
        gap = 10
        
        for leaf_id, (x, y, w, h) in boxes.items():
            # Apply gap
            rect_item = LeafItem(
                0, 0, w - gap, h - gap, 
                leaf_id, self
            )
            rect_item.setPos(x + gap/2, y + gap/2)
            
            img_idx = self.perm[leaf_id]
            
            # Load image to display in rect
            path = self.image_paths[img_idx]
            # We need to fit image
            pix = QPixmap(str(path))
            if not pix.isNull():
                 # Scale exactly to fit? Or AspectFit? 
                 # sa logic uses ImageOps.fit (crop to fill).
                 # Let's emulate crop to fill for preview.
                 scaled = pix.scaled(int(w-gap), int(h-gap), Qt.AspectRatioMode.KeepAspectRatioByExpanding, Qt.TransformationMode.SmoothTransformation)
                 # Crop center
                 copy = scaled.copy(
                     (scaled.width() - int(w-gap)) // 2,
                     (scaled.height() - int(h-gap)) // 2,
                     int(w-gap), int(h-gap)
                 )
                 rect_item.pixmap_item.setPixmap(copy)
            
            if leaf_id in self.locked_leaves:
                rect_item.is_locked = True
                
            self.scene.addItem(rect_item)
            
        self.view.fitInView(self.scene.itemsBoundingRect(), Qt.AspectRatioMode.KeepAspectRatio)

    def handle_drop(self, leaf_id, img_idx):
        # User dropped image `img_idx` onto `leaf_id`.
        # Constraint: Image `img_idx` MUST be at `leaf_id`.
        
        # 1. Update locked stats
        self.locked_leaves[leaf_id] = img_idx
        
        # 2. Update permutation to satisfy this lock immediately
        # Find where img_idx currently is
        current_leaf_of_img = -1
        for l_id, im_id in enumerate(self.perm):
            if im_id == img_idx:
                current_leaf_of_img = l_id
                break
        
        # If it's already there, great.
        if current_leaf_of_img != leaf_id:
            # Swap whatever is at leaf_id with img_idx
            # Wait, if `leaf_id` was already locked to something else? 
            # Overwrite lock.
            
            # The image currently at `leaf_id`
            old_img_at_target = self.perm[leaf_id]
            
            # Swap
            self.perm[leaf_id] = img_idx
            self.perm[current_leaf_of_img] = old_img_at_target
            
        # Re-draw
        self.draw_layout()

    def start_optimization(self):
        L = len(self.image_paths)
        if L == 0: return
        
        prefs = [sa.pref_aspect_for(p) for p in self.image_paths]
        
        self.worker = OptimizationThread(
            self.root, self.page_W, self.page_H,
            self.image_paths, prefs, self.locked_leaves
        )
        self.worker.progress.connect(self.progress_bar.setValue)
        self.worker.finished_optim.connect(self.on_optim_finished)
        
        self.optimize_btn.setEnabled(False)
        self.worker.start()

    def on_optim_finished(self, perm):
        self.perm = perm
        self.draw_layout()
        self.optimize_btn.setEnabled(True)
        QMessageBox.information(self, "Done", "Optimization Complete!")

    def reset_layout(self):
        self.locked_leaves = {}
        self.init_tree() # Re-randomize

# Custom Drag for ListWidget
# We need to subclass QListWidget to support custom mime data easily, 
# or just rely on internal model?
# Actually, standard QListWidget drag puts mime data.
# Often easier to override startDrag in QListWidget
# But let's try to patch QListWidget instance methods or subclass properly.

# Monkey-patching for simplicity in one file
def startDrag(self, actions):
    drag = QDrag(self)
    items = self.selectedItems()
    if not items: return
    item = items[0]
    idx = item.data(Qt.ItemDataRole.UserRole)
    
    mime = QMimeData()
    mime.setData("application/x-image-idx", str(idx).encode())
    drag.setMimeData(mime)
    
    drag.exec(Qt.DropAction.CopyAction)

QListWidget.startDrag = startDrag


def main():
    app = QApplication(sys.argv)
    window = AlbumWindow()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()