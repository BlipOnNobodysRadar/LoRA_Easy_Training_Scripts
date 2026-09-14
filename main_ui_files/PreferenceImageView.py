"""Image inspection controls; transformations affect only the display."""
from PySide6.QtCore import Qt, QPointF, QRectF, QSizeF, Signal
from PySide6.QtGui import QPainter, QPixmap
from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QPushButton, QSizePolicy, QVBoxLayout, QWidget


class PreferenceImageView(QFrame):
    doubleClicked = Signal()
    viewChanged = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._pixmap = None
        self._scale = 1.0
        self._offset = QPointF()
        self._fit_mode = True
        self._drag_position = None
        self._message = "No image"
        self.setMinimumSize(220, 220)
        self.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.setFrameShape(QFrame.StyledPanel)
        self.setToolTip("Mouse wheel: zoom at pointer. Drag: pan. Double-click: larger preview.")
        self.setAccessibleName("Image preview; mouse wheel to zoom, drag to pan")

    def set_image(self, path):
        pixmap = path if isinstance(path, QPixmap) else QPixmap(str(path)) if path else QPixmap()
        self._pixmap = None if pixmap.isNull() else pixmap
        self._message = "Image not found:\n%s" % (path or "(none)")
        self._drag_position = None
        self.fit_to_view()

    def _fit_scale(self):
        if self._pixmap is None:
            return 1.0
        rect = self.contentsRect()
        return min(max(1, rect.width()) / self._pixmap.width(),
                   max(1, rect.height()) / self._pixmap.height())

    def image_point(self, position):
        return (QPointF(position) - self._offset) / self._scale

    def fit_to_view(self):
        self._fit_mode = True
        self._scale = self._fit_scale()
        self._offset = QPointF(self.contentsRect().center())
        if self._pixmap is not None:
            self._offset -= QPointF(self._pixmap.width(), self._pixmap.height()) * (self._scale / 2)
        self._update_view()

    def actual_size(self):
        self._fit_mode = False
        self._set_scale(1.0, QPointF(self.contentsRect().center()))

    def _set_scale(self, scale, anchor):
        if self._pixmap is None:
            return
        point = self.image_point(anchor)
        self._scale = max(min(self._fit_scale(), 1.0), min(scale, max(self._fit_scale(), 8.0)))
        self._offset = anchor - point * self._scale
        self._update_view()

    def _update_view(self):
        if self._pixmap is not None:
            rect = self.contentsRect()
            for axis, extent, origin, size in (
                ("x", self._pixmap.width() * self._scale, rect.x(), rect.width()),
                ("y", self._pixmap.height() * self._scale, rect.y(), rect.height()),
            ):
                offset = getattr(self._offset, axis)()
                value = origin + (size - extent) / 2 if extent <= size else max(origin + size - extent, min(origin, offset))
                getattr(self._offset, "set" + axis.upper())(value)
        movable = self._pixmap is not None and self._scale > self._fit_scale() + 1e-6
        self.setCursor(Qt.ClosedHandCursor if self._drag_position is not None else
                       Qt.OpenHandCursor if movable else Qt.ArrowCursor)
        self.update()
        self.viewChanged.emit()

    def paintEvent(self, event):
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setClipRect(self.contentsRect())
        if self._pixmap is None:
            painter.setPen(self.palette().windowText().color())
            painter.drawText(self.contentsRect(), Qt.AlignCenter | Qt.TextWordWrap, self._message)
        else:
            painter.setRenderHint(QPainter.SmoothPixmapTransform)
            target = QRectF(self._offset, QSizeF(self._pixmap.size()) * self._scale)
            painter.drawPixmap(target, self._pixmap, QRectF(self._pixmap.rect()))

    def resizeEvent(self, event):
        if self._fit_mode:
            self.fit_to_view()
        else:
            center = self.image_point(QPointF(event.oldSize().width() / 2, event.oldSize().height() / 2))
            self._offset = QPointF(self.width() / 2, self.height() / 2) - center * self._scale
            self._update_view()
        super().resizeEvent(event)

    def wheelEvent(self, event):
        if self._pixmap is None:
            event.ignore()
            return
        delta = event.angleDelta().y() / 120 if event.angleDelta().y() else event.pixelDelta().y() / 100
        self._fit_mode = False
        self._set_scale(self._scale * 1.2 ** max(-20, min(20, delta)), event.position())
        if abs(self._scale - self._fit_scale()) < 1e-6:
            self._fit_mode = True
        event.accept()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton and self._pixmap is not None:
            self._drag_position = event.position()
            self._update_view()
            event.accept()
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._drag_position is not None:
            self._offset += event.position() - self._drag_position
            self._drag_position = event.position()
            self._update_view()
            event.accept()
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._drag_position = None
            self._update_view()
        super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event):
        if event.button() == Qt.LeftButton and self._pixmap is not None:
            self._drag_position = None
            self._update_view()
            self.doubleClicked.emit()
            event.accept()
        else:
            super().mouseDoubleClickEvent(event)


def image_panel(title, view):
    view.setAccessibleName(title + "; mouse wheel to zoom, drag to pan")
    panel = QWidget()
    box = QVBoxLayout(panel)
    header = QHBoxLayout()
    header.addWidget(QLabel(title), 1)
    percentage = QLabel()
    header.addWidget(percentage)
    fit = QPushButton("Fit")
    fit.setToolTip("Fit the whole image in this pane")
    fit.clicked.connect(view.fit_to_view)
    actual = QPushButton("100%")
    actual.setToolTip("Show one image pixel per display coordinate")
    actual.clicked.connect(view.actual_size)
    for button in (fit, actual):
        button.setAutoDefault(False)
        header.addWidget(button)
    view.viewChanged.connect(lambda: percentage.setText(f"{view._scale:.0%}" if view._pixmap else ""))
    box.addLayout(header)
    box.addWidget(view, 1)
    return panel
