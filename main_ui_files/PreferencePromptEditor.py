"""Per-prompt generation controls; no model or dataset side effects."""

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QFormLayout, QGroupBox, QHBoxLayout, QLabel, QPlainTextEdit,
    QPushButton, QSpinBox, QVBoxLayout, QWidget,
)


class PreferencePromptEditor(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.rows = []
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.row_layout = QVBoxLayout()
        layout.addLayout(self.row_layout)
        footer = QHBoxLayout()
        self.add_btn = QPushButton("Add prompt")
        self.add_btn.setAutoDefault(False)
        self.add_btn.clicked.connect(lambda: self.add_prompt())
        self.summary_label = QLabel()
        footer.addWidget(self.add_btn)
        footer.addWidget(self.summary_label, 1)
        layout.addLayout(footer)
        self._refresh()

    def set_entries(self, entries):
        for row in self.rows[:]:
            self._remove(row)
        for entry in entries:
            self.add_prompt(entry, focus=False)

    def add_prompt(self, entry=None, *, focus=True):
        entry = entry or {"prompt": "", "negative_prompt": "", "pairs": 1}
        row = QGroupBox()
        form = QFormLayout(row)
        row.prompt_edit = QPlainTextEdit(entry["prompt"])
        row.prompt_edit.setPlaceholderText("Describe this prompt; multiple lines stay together")
        row.prompt_edit.setFixedHeight(76)
        row.negative_edit = QPlainTextEdit(entry["negative_prompt"])
        row.negative_edit.setPlaceholderText("Optional negative prompt for these pairs")
        row.negative_edit.setFixedHeight(54)
        row.pairs_spin = QSpinBox()
        row.pairs_spin.setRange(1, max(1_000_000, entry["pairs"]))
        row.pairs_spin.setValue(entry["pairs"])
        # Scrolling the long form must not silently change pair counts.
        row.pairs_spin.setFocusPolicy(Qt.StrongFocus)
        row.pairs_spin.installEventFilter(self)
        row.remove_btn = QPushButton("Remove prompt")
        row.remove_btn.setAutoDefault(False)
        row.remove_btn.clicked.connect(lambda: self._remove(row))
        controls = QHBoxLayout()
        controls.addWidget(row.pairs_spin)
        controls.addStretch(1)
        controls.addWidget(row.remove_btn)
        form.addRow("Positive", row.prompt_edit)
        form.addRow("Negative", row.negative_edit)
        form.addRow("Pairs (2 images each)", controls)
        self.rows.append(row)
        self.row_layout.addWidget(row)
        row.pairs_spin.valueChanged.connect(self._refresh)
        self._refresh()
        if focus:
            # Let the enclosing scroll area lay out the row before focus scrolls.
            QTimer.singleShot(0, row.prompt_edit.setFocus)
        return row

    def eventFilter(self, watched, event):
        from PySide6.QtCore import QEvent
        if event.type() == QEvent.Wheel and not watched.hasFocus():
            event.ignore()
            return True
        return super().eventFilter(watched, event)

    def _remove(self, row):
        self.rows.remove(row)
        self.row_layout.removeWidget(row)
        row.hide()
        row.deleteLater()
        self._refresh()

    def _refresh(self):
        for number, row in enumerate(self.rows, 1):
            row.setTitle(f"Prompt {number}")
            for widget, field in ((row.prompt_edit, "positive"), (row.negative_edit, "negative"),
                                  (row.pairs_spin, "pairs"), (row.remove_btn, "remove")):
                widget.setAccessibleName(f"Prompt {number} {field}")
        pairs = sum(row.pairs_spin.value() for row in self.rows)
        self.summary_label.setText(f"Total: {pairs:,} pairs / {pairs * 2:,} images")

    def entries(self):
        return [{"prompt": row.prompt_edit.toPlainText(),
                 "negative_prompt": row.negative_edit.toPlainText(),
                 "pairs": row.pairs_spin.value()} for row in self.rows]
