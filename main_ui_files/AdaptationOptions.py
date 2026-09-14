"""Standalone PySide6 widget exposing adaptation-method training options."""

from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QSpinBox,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

DPO = "dpo"
ADDIFT = "addift"
LECO = "leco"
MODES = (DPO, ADDIFT, LECO)
MODE_LABELS = {
    DPO: "Diffusion-DPO",
    ADDIFT: "ADDifT (aligned image edits)",
    LECO: "LECO (text concept)",
}
ACTIONS = ("enhance", "erase")


def _as_dict(value):
    return value if isinstance(value, dict) else {}


def _number(value, default, caster):
    if value is None or isinstance(value, bool):
        return default
    try:
        return caster(value)
    except (TypeError, ValueError):
        return default


def _text(value, default=""):
    return default if value is None else str(value)


def _wrapped(text, parent=None):
    label = QLabel(text, parent)
    label.setWordWrap(True)
    return label


class AdaptationOptions(QWidget):
    """Method selector plus per-method settings pages (no file pickers)."""

    objective_changed = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.method_combo = QComboBox(self)
        for mode in MODES:
            self.method_combo.addItem(MODE_LABELS[mode], mode)

        self.pages = QStackedWidget(self)
        self.pages.addWidget(self._dpo_page())
        self.pages.addWidget(self._addift_page())
        self.pages.addWidget(self._leco_page())

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Adaptation objective", self))
        layout.addWidget(self.method_combo)
        layout.addWidget(self.pages)

        self.method_combo.currentIndexChanged.connect(self._on_method_changed)
        for button in self.findChildren(QPushButton):
            button.setAutoDefault(False)
            button.setDefault(False)
        self._on_method_changed(self.method_combo.currentIndex())

    # ---------------------------------------------------------------- pages
    def _dpo_page(self):
        page = QWidget(self)
        layout = QVBoxLayout(page)
        layout.addWidget(_wrapped(
            "Training uses the winner and preference strength. Quality labels "
            "and reasons are metadata only. Ties/skips are excluded.", page))
        layout.addStretch(1)
        return page

    def _addift_page(self):
        page = QWidget(self)
        layout = QVBoxLayout(page)
        layout.addWidget(_wrapped(
            "Uses only explicitly imported aligned before/after pairs; the "
            "winner defines the desired image and requires a human A/B rating.",
            page))
        form = QFormLayout()

        self.min_timestep = QSpinBox(page)
        self.min_timestep.setRange(0, 999)
        self.min_timestep.setValue(400)
        form.addRow("Minimum timestep", self.min_timestep)

        self.max_timestep = QSpinBox(page)
        self.max_timestep.setRange(1, 1000)
        self.max_timestep.setValue(900)
        # Exclusive upper bound: sampling stays strictly below this value.
        form.addRow("Maximum timestep (exclusive)", self.max_timestep)

        self.reverse = QCheckBox("Alternate inverse direction", page)
        self.reverse.setChecked(True)
        form.addRow(self.reverse)
        layout.addLayout(form)
        layout.addWidget(_wrapped(
            "Shared noise and a deterministic VAE mean are used. Automatic "
            "feature isolation is not provided.", page))
        layout.addStretch(1)
        return page

    def _leco_page(self):
        page = QWidget(self)
        layout = QVBoxLayout(page)
        layout.addWidget(_wrapped(
            "Text-only; no rated image pairs required. Enhance adds concept "
            "minus contrast to neutral; erase subtracts it. Target is the "
            "conditioning prompt learned by the adapter. Uses generation "
            "width and height.", page))
        form = QFormLayout()

        self.target_edit = self._prompt(page)
        form.addRow("Prompt to change", self.target_edit)
        self.positive_edit = self._prompt(page)
        form.addRow("Concept to enhance/erase", self.positive_edit)
        self.neutral_edit = self._prompt(page)
        form.addRow("Neutral prompt (empty is allowed)", self.neutral_edit)
        self.unconditional_edit = self._prompt(page)
        form.addRow("Contrast baseline (empty is allowed)", self.unconditional_edit)

        self.action_combo = QComboBox(page)
        self.action_combo.addItem("Enhance", "enhance")
        self.action_combo.addItem("Erase", "erase")
        form.addRow("Action", self.action_combo)

        self.guidance_scale = QDoubleSpinBox(page)
        self.guidance_scale.setRange(0.01, 20.0)
        self.guidance_scale.setDecimals(2)
        self.guidance_scale.setValue(1.0)
        form.addRow("Concept change strength", self.guidance_scale)

        self.denoising_steps = QSpinBox(page)
        self.denoising_steps.setRange(2, 100)
        self.denoising_steps.setValue(20)
        form.addRow("Partial sampling steps", self.denoising_steps)

        self.denoise_cfg = QDoubleSpinBox(page)
        self.denoise_cfg.setRange(1.0, 20.0)
        self.denoise_cfg.setDecimals(2)
        self.denoise_cfg.setValue(3.0)
        form.addRow("Partial sampling CFG", self.denoise_cfg)

        layout.addLayout(form)
        layout.addStretch(1)
        return page

    @staticmethod
    def _prompt(parent):
        editor = QPlainTextEdit(parent)
        editor.setFixedHeight(60)
        return editor

    # ------------------------------------------------------------- behaviour
    def _on_method_changed(self, index):
        mode = self.method_combo.itemData(index)
        if mode in MODES:
            self.pages.setCurrentIndex(MODES.index(mode))
            self.pages.setMaximumHeight(self.pages.currentWidget().sizeHint().height())
        self.objective_changed.emit(mode)

    def current_mode(self):
        return self.method_combo.currentData()

    def set_settings(self, training):
        data = _as_dict(training)
        raw = data.get("objective", data.get("mode"))
        mode = DPO if raw is None else raw
        if mode not in MODES:
            raise ValueError("unknown adaptation objective: %r" % (mode,))
        self.method_combo.setCurrentIndex(MODES.index(mode))

        add = _as_dict(data.get("addift"))
        self.min_timestep.setValue(_number(add.get("min_timestep"), 400, int))
        self.max_timestep.setValue(_number(add.get("max_timestep"), 900, int))
        inverse = add.get("alternate_inverse")
        self.reverse.setChecked(True if inverse is None else bool(inverse))

        leco = _as_dict(data.get("leco"))
        self.target_edit.setPlainText(_text(leco.get("target")))
        self.positive_edit.setPlainText(_text(leco.get("positive")))
        self.neutral_edit.setPlainText(_text(leco.get("neutral")))
        self.unconditional_edit.setPlainText(_text(leco.get("unconditional")))
        action = leco.get("action", "enhance")
        self.action_combo.setCurrentIndex(
            ACTIONS.index(action) if action in ACTIONS else ACTIONS.index("enhance"))
        self.guidance_scale.setValue(_number(leco.get("guidance_scale"), 1.0, float))
        self.denoising_steps.setValue(_number(leco.get("denoising_steps"), 20, int))
        self.denoise_cfg.setValue(_number(leco.get("denoise_cfg"), 3.0, float))

    def settings(self):
        return {
            "objective": self.current_mode(),
            "addift": {
                "min_timestep": self.min_timestep.value(),
                "max_timestep": self.max_timestep.value(),
                "alternate_inverse": bool(self.reverse.isChecked()),
            },
            "leco": {
                "target": self.target_edit.toPlainText(),
                "positive": self.positive_edit.toPlainText(),
                "neutral": self.neutral_edit.toPlainText(),
                "unconditional": self.unconditional_edit.toPlainText(),
                "action": self.action_combo.currentData(),
                "guidance_scale": self.guidance_scale.value(),
                "denoising_steps": self.denoising_steps.value(),
                "denoise_cfg": self.denoise_cfg.value(),
            },
        }