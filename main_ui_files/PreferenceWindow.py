"""Practical Qt preference-rating UI for the SDXL preference subsystem.

Standalone entry point::

    python -m main_ui_files.PreferenceWindow [--config PATH]

The module deliberately imports ``backend.preference`` at import time; those
imports resolve once the repository is integrated.  No GPU/model code is
imported here, and generation/training always run out-of-process via QProcess.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

from PySide6.QtCore import Qt, QProcess, Signal
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QButtonGroup,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QSpinBox,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from backend.preference.store import PreferenceStore
from backend.preference.config import atomic_json, load_config

SAMPLERS = ["euler", "ddim", "dpmpp_2m"]
QUALITY_CHOICES = [
    ("Unrated", None),
    ("Excellent", "excellent"),
    ("Good", "good"),
    ("Acceptable", "acceptable"),
    ("Bad", "bad"),
    ("Especially bad", "especially_bad"),
]
# label, preference, strength
PREF_BUTTONS = [
    ("A strong", "a", "strong"),
    ("A slight", "a", "slight"),
    ("Tie", "tie", None),
    ("B slight", "b", "slight"),
    ("B strong", "b", "strong"),
    ("No preference", "unrated", None),
]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _repo_root() -> Path:
    # Keep a short Windows junction path: resolving it can push venv imports
    # beyond MAX_PATH even when the executable itself still launches.
    return Path(__file__).absolute().parents[1]


def _backend_python() -> str:
    root = _repo_root()
    for candidate in (
        root / "backend" / "sd_scripts" / "venv" / "Scripts" / "python.exe",
        root / "backend" / "sd_scripts" / "venv" / "bin" / "python",
    ):
        if candidate.exists():
            return str(candidate)
    return sys.executable


class _ImageLabel(QLabel):
    """Aspect-ratio preserving image preview; double-click shows full size."""

    doubleClicked = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._pixmap = None
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(220, 220)
        self.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.setFrameShape(QFrame.StyledPanel)
        self.setWordWrap(True)
        self.setText("No image")

    def set_image(self, path):
        pixmap = QPixmap(str(path)) if path else QPixmap()
        self._pixmap = None if pixmap.isNull() else pixmap
        if self._pixmap is None:
            self.setPixmap(QPixmap())
            self.setText("Image not found:\n%s" % (path or "(none)"))
        self._rescale()

    def _rescale(self):
        if self._pixmap is None:
            return
        self.setPixmap(
            self._pixmap.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        )

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._rescale()

    def mouseDoubleClickEvent(self, event):
        if self._pixmap is not None:
            self.doubleClicked.emit()
        super().mouseDoubleClickEvent(event)


class PreferenceWindow(QDialog):
    """Non-modal rating + configuration dialog."""

    def __init__(self, parent=None, config_path=None):
        super().__init__(parent)
        self.setWindowTitle("Preference Rating")
        self.setModal(False)
        self.resize(1100, 760)

        self._config = {}
        local = _repo_root() / "preference.local.json"
        self._config_path = Path(config_path).resolve() if config_path else (local if local.exists() else None)
        self._store = None
        self._comparisons = []
        self._index = -1
        self._sel_pref = None
        self._sel_strength = None
        self._process = None
        self._stop_file = None
        self._close_when_done = False

        tabs = self.tabs = QTabWidget(self)
        tabs.addTab(self._build_rate_tab(), "Rate")
        tabs.addTab(self._build_config_tab(), "Configuration")
        tabs.currentChanged.connect(lambda _: self._reload_store() if tabs.currentIndex() == 0 else None)
        root = QVBoxLayout(self)
        root.addWidget(tabs)
        root.addWidget(self.status_label)

        self._load_config()
        self._reload_store()

    # ------------------------------------------------------------------ rate
    def _build_rate_tab(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)

        top = QHBoxLayout()
        top.addWidget(QLabel("Review:"))
        self.review_combo = QComboBox()
        self.review_combo.addItem("All comparisons", "all")
        self.review_combo.addItem("Unrated only", "unrated")
        self.review_combo.currentIndexChanged.connect(self._reload_store)
        top.addWidget(self.review_combo)
        self.counts_label = QLabel("No store loaded")
        top.addWidget(self.counts_label, 1)
        top.addWidget(QLabel("Position:"))
        self.position_label = QLabel("-")
        top.addWidget(self.position_label)
        layout.addLayout(top)

        self.prompt_label = QLabel("No comparison selected")
        self.prompt_label.setWordWrap(True)
        self.prompt_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        layout.addWidget(self.prompt_label)

        splitter = QSplitter(Qt.Horizontal)
        self.image_a = _ImageLabel()
        self.image_b = _ImageLabel()
        for label, text in ((self.image_a, "A"), (self.image_b, "B")):
            label.doubleClicked.connect(lambda lab=label: self._show_full_size(lab))
            wrapper = QWidget()
            box = QVBoxLayout(wrapper)
            box.addWidget(QLabel("Image %s" % text))
            box.addWidget(label, 1)
            splitter.addWidget(wrapper)
        layout.addWidget(splitter, 1)

        pref_row = QHBoxLayout()
        pref_row.addWidget(QLabel("Relative preference:"))
        self.pref_group = QButtonGroup(self)
        self.pref_group.setExclusive(True)
        self._pref_buttons = []
        for label, pref, strength in PREF_BUTTONS:
            btn = QPushButton(label)
            btn.setCheckable(True)
            btn.setToolTip(
                "Selects '%s' without saving. Use Save or Save & Next to commit."
                % label
            )
            self.pref_group.addButton(btn)
            self._pref_buttons.append((btn, pref, strength))
            pref_row.addWidget(btn)
        self.pref_group.buttonClicked.connect(self._on_pref_clicked)
        layout.addLayout(pref_row)

        qual_row = QHBoxLayout()
        qual_row.addWidget(QLabel("Quality A:"))
        self.quality_a = QComboBox()
        qual_row.addWidget(self.quality_a)
        qual_row.addWidget(QLabel("Quality B:"))
        self.quality_b = QComboBox()
        qual_row.addWidget(self.quality_b)
        for combo in (self.quality_a, self.quality_b):
            for label, value in QUALITY_CHOICES:
                combo.addItem(label, value)
        qual_row.addStretch(1)
        layout.addLayout(qual_row)

        reason_row = QHBoxLayout()
        reason_row.addWidget(QLabel("Reasons (comma separated):"))
        self.reasons_edit = QLineEdit()
        self.reasons_edit.setPlaceholderText("e.g. better composition, sharper detail")
        reason_row.addWidget(self.reasons_edit, 1)
        layout.addLayout(reason_row)

        actions = QHBoxLayout()
        self.save_btn = QPushButton("Save")
        self.save_next_btn = QPushButton("Save & Next")
        self.next_btn = QPushButton("Next")
        self.prev_btn = QPushButton("Previous")
        self.skip_btn = QPushButton("Skip (save skip & next)")
        self.undo_btn = QPushButton("Undo last feedback")
        self.save_btn.clicked.connect(lambda: self._save_current(advance=False))
        self.save_next_btn.clicked.connect(lambda: self._save_current(advance=True))
        self.next_btn.clicked.connect(lambda: self._navigate(1))
        self.prev_btn.clicked.connect(lambda: self._navigate(-1))
        self.skip_btn.clicked.connect(self._skip_current)
        self.undo_btn.clicked.connect(self._undo)
        for btn in (self.save_btn, self.save_next_btn, self.next_btn, self.prev_btn,
                    self.skip_btn, self.undo_btn):
            actions.addWidget(btn)
        actions.addStretch(1)
        layout.addLayout(actions)

        hint = QLabel(
            "Relative buttons only select a choice. Nothing is written until Save, "
            "Save & Next, or Skip. Navigation never stores a rating."
        )
        hint.setWordWrap(True)
        layout.addWidget(hint)

        self.status_label = QLabel("")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)
        return page

    # --------------------------------------------------------------- config
    def _build_config_tab(self) -> QWidget:
        page = QWidget()
        outer = QVBoxLayout(page)
        form_widget = QWidget()
        form = QFormLayout(form_widget)

        self.cfg_checkpoint = self._file_row(form, "Checkpoint (.safetensors)", "*.safetensors")
        self.cfg_lora_path = self._file_row(form, "Original LoRA path (first, if stacked)", "*.safetensors")
        self.cfg_lora_weight = self._dspin(form, "Original LoRA weight", -10.0, 10.0, 1.0)
        self.cfg_pref_lora = self._file_row(form, "Preference adapter (optional)", "*.safetensors")
        self.cfg_pref_weight = self._dspin(form, "Preference adapter weight (inference)", -10.0, 10.0, 1.0)
        self.cfg_dataset_dir = self._dir_row(form, "Dataset directory")
        self.cfg_output_dir = self._dir_row(form, "Output directory (new run parent)")
        self.cfg_resume_dir = self._dir_row(form, "Resume checkpoint directory")

        self.cfg_prompts = QPlainTextEdit()
        self.cfg_prompts.setPlaceholderText("One prompt per line")
        self.cfg_prompts.setFixedHeight(90)
        form.addRow("Prompts (one per line)", self.cfg_prompts)
        self.cfg_negative = QPlainTextEdit()
        self.cfg_negative.setFixedHeight(50)
        form.addRow("Negative prompt", self.cfg_negative)

        res_row = QWidget()
        res_layout = QHBoxLayout(res_row)
        res_layout.setContentsMargins(0, 0, 0, 0)
        self.cfg_width = QSpinBox(); self.cfg_width.setRange(256, 1536); self.cfg_width.setSingleStep(64)
        self.cfg_height = QSpinBox(); self.cfg_height.setRange(256, 1536); self.cfg_height.setSingleStep(64)
        res_layout.addWidget(self.cfg_width)
        res_layout.addWidget(QLabel("x"))
        res_layout.addWidget(self.cfg_height)
        res_layout.addStretch(1)
        form.addRow("Resolution (width x height)", res_row)

        self.cfg_steps = QSpinBox(); self.cfg_steps.setRange(1, 500)
        form.addRow("Sampling steps", self.cfg_steps)
        self.cfg_cfg = self._dspin(form, "CFG", 0.0, 50.0, 7.0)
        self.cfg_sampler = QComboBox(); self.cfg_sampler.addItems(SAMPLERS)
        form.addRow("Sampler", self.cfg_sampler)
        self.cfg_seed = QLineEdit()
        form.addRow("Seed", self.cfg_seed)
        self.cfg_pairs = QSpinBox(); self.cfg_pairs.setRange(1, 64)
        form.addRow("Pairs per prompt", self.cfg_pairs)
        self.cfg_split_seed = QLineEdit()
        form.addRow("Split seed", self.cfg_split_seed)

        self.cfg_rank = QSpinBox(); self.cfg_rank.setRange(1, 512)
        form.addRow("DPO rank", self.cfg_rank)
        self.cfg_alpha = QDoubleSpinBox(); self.cfg_alpha.setRange(0.001, 8192); self.cfg_alpha.setDecimals(3)
        form.addRow("DPO alpha", self.cfg_alpha)
        self.cfg_lr = QLineEdit()
        form.addRow("Learning rate", self.cfg_lr)
        self.cfg_beta = QLineEdit()
        form.addRow("DPO beta", self.cfg_beta)
        self.cfg_max_steps = QSpinBox(); self.cfg_max_steps.setRange(1, 1_000_000)
        form.addRow("Max steps", self.cfg_max_steps)
        self.cfg_grad_accum = QSpinBox(); self.cfg_grad_accum.setRange(1, 512)
        form.addRow("Gradient accumulation", self.cfg_grad_accum)
        self.cfg_ckpt_every = QSpinBox(); self.cfg_ckpt_every.setRange(1, 100000)
        form.addRow("Checkpoint every", self.cfg_ckpt_every)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(form_widget)
        outer.addWidget(scroll, 3)

        btns = QHBoxLayout()
        load_btn = QPushButton("Load config...")
        save_btn = QPushButton("Save config")
        gen_btn = QPushButton("Generate pairs")
        train_btn = QPushButton("Train")
        stop_btn = QPushButton("Request stop")
        load_btn.clicked.connect(self._choose_and_load_config)
        save_btn.clicked.connect(lambda: self._save_config())
        gen_btn.clicked.connect(self._launch_generate)
        train_btn.clicked.connect(self._launch_train)
        stop_btn.clicked.connect(self._request_stop)
        for btn in (load_btn, save_btn, gen_btn, train_btn, stop_btn):
            btns.addWidget(btn)
        btns.addStretch(1)
        outer.addLayout(btns)

        self.job_label = QLabel("No job running")
        self.job_label.setWordWrap(True)
        outer.addWidget(self.job_label)
        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMaximumBlockCount(5000)
        outer.addWidget(self.log_view, 1)
        return page

    def _file_row(self, form, label, filt):
        row = QWidget()
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        edit = QLineEdit()
        browse = QPushButton("Browse...")
        browse.clicked.connect(lambda: self._pick_file(edit, filt))
        layout.addWidget(edit, 1)
        layout.addWidget(browse)
        form.addRow(label, row)
        return edit

    def _dir_row(self, form, label):
        row = QWidget()
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        edit = QLineEdit()
        browse = QPushButton("Browse...")
        browse.clicked.connect(lambda: self._pick_dir(edit))
        layout.addWidget(edit, 1)
        layout.addWidget(browse)
        form.addRow(label, row)
        return edit

    def _dspin(self, form, label, lo, hi, value):
        spin = QDoubleSpinBox()
        spin.setRange(lo, hi)
        spin.setDecimals(6)
        spin.setValue(value)
        form.addRow(label, spin)
        return spin

    def _pick_file(self, edit, filt):
        path, _ = QFileDialog.getOpenFileName(self, "Select file", edit.text() or str(_repo_root()), filt)
        if path:
            edit.setText(path)

    def _pick_dir(self, edit):
        path = QFileDialog.getExistingDirectory(self, "Select directory", edit.text() or str(_repo_root()))
        if path:
            edit.setText(path)

    # ---------------------------------------------------------------- config
    def _load_config(self):
        if self._config_path and self._config_path.exists():
            try:
                self._config = json.loads(self._config_path.read_text(encoding="utf-8-sig"))
                # Resolve the same relative paths the backend uses.
                for obj, key in [(self._config, "dataset_dir"), (self._config.get("model", {}), "checkpoint"),
                                 (self._config.get("model", {}), "preference_lora"),
                                 (self._config.get("training", {}), "output_dir")]:
                    if obj.get(key) and not Path(obj[key]).is_absolute():
                        obj[key] = str((self._config_path.parent / obj[key]).resolve())
                for item in self._config.get("model", {}).get("base_loras", []):
                    if item.get("path") and not Path(item["path"]).is_absolute():
                        item["path"] = str((self._config_path.parent / item["path"]).resolve())
            except (OSError, ValueError) as exc:
                self._set_status("Could not load config: %s" % exc, error=True)
                self._config = {}
        else:
            self._config = {}
        self._apply_config_to_widgets()

    def _apply_config_to_widgets(self):
        cfg = self._config
        model = cfg.get("model", {})
        gen = cfg.get("generation", {})
        train = cfg.get("training", {})

        self.cfg_checkpoint.setText(str(model.get("checkpoint") or ""))
        loras = model.get("base_loras") or []
        first = loras[0] if loras else {}
        self.cfg_lora_path.setText(str(first.get("path") or ""))
        self.cfg_lora_weight.setValue(float(first.get("weight", 1.0)))
        self.cfg_pref_lora.setText(str(model.get("preference_lora") or ""))
        self.cfg_pref_weight.setValue(float(model.get("preference_weight", 1.0)))

        self.cfg_dataset_dir.setText(str(cfg.get("dataset_dir") or ""))
        self.cfg_output_dir.setText(str(train.get("output_dir") or ""))
        self.cfg_prompts.setPlainText("\n".join(gen.get("prompts") or []))
        self.cfg_negative.setPlainText(str(gen.get("negative_prompt") or ""))
        self.cfg_width.setValue(int(gen.get("width", 1024)))
        self.cfg_height.setValue(int(gen.get("height", 1024)))
        self.cfg_steps.setValue(int(gen.get("steps", 25)))
        self.cfg_cfg.setValue(float(gen.get("cfg", 7.0)))
        sampler = str(gen.get("sampler", "euler"))
        if sampler in SAMPLERS:
            self.cfg_sampler.setCurrentText(sampler)
        self.cfg_seed.setText(str(gen.get("seed", 9796)))
        self.cfg_pairs.setValue(int(gen.get("pairs_per_prompt", 1)))
        self.cfg_split_seed.setText(str(gen.get("split_seed", 9796)))

        self.cfg_rank.setValue(int(train.get("rank", 16)))
        self.cfg_alpha.setValue(float(train.get("alpha", 16)))
        self.cfg_lr.setText(str(train.get("learning_rate", 0.00001)))
        self.cfg_beta.setText(str(train.get("beta", 5000)))
        self.cfg_max_steps.setValue(int(train.get("max_steps", 50)))
        self.cfg_grad_accum.setValue(int(train.get("gradient_accumulation", 4)))
        self.cfg_ckpt_every.setValue(int(train.get("checkpoint_every", 25)))

    def _collect_config(self) -> dict:
        """Return the loaded JSON with only exposed fields updated."""
        cfg = copy.deepcopy(self._config) if self._config else {}
        cfg.setdefault("schema_version", 1)
        model = cfg.setdefault("model", {})
        gen = cfg.setdefault("generation", {})
        train = cfg.setdefault("training", {})

        model["family"] = model.get("family", "sdxl")
        model["checkpoint"] = self.cfg_checkpoint.text().strip()
        lora_path = self.cfg_lora_path.text().strip()
        additional = model.get("base_loras", [])[1:]
        if lora_path:
            model["base_loras"] = [{"path": lora_path, "weight": self.cfg_lora_weight.value()}] + additional
        else:
            model["base_loras"] = []
        model["preference_lora"] = self.cfg_pref_lora.text().strip() or None
        model["preference_weight"] = self.cfg_pref_weight.value()

        cfg["dataset_dir"] = self.cfg_dataset_dir.text().strip()
        gen["prompts"] = [p.strip() for p in self.cfg_prompts.toPlainText().splitlines() if p.strip()]
        gen["negative_prompt"] = self.cfg_negative.toPlainText().strip()
        gen["width"] = self.cfg_width.value()
        gen["height"] = self.cfg_height.value()
        gen["steps"] = self.cfg_steps.value()
        gen["cfg"] = self.cfg_cfg.value()
        gen["sampler"] = self.cfg_sampler.currentText()
        gen["seed"] = int(self.cfg_seed.text())
        gen["pairs_per_prompt"] = self.cfg_pairs.value()
        gen["split_seed"] = int(self.cfg_split_seed.text())

        train["output_dir"] = self.cfg_output_dir.text().strip()
        train["rank"] = self.cfg_rank.value()
        train["alpha"] = self.cfg_alpha.value()
        train["learning_rate"] = self._as_number(self.cfg_lr.text(), train.get("learning_rate", 0.00001))
        train["beta"] = self._as_number(self.cfg_beta.text(), train.get("beta", 5000))
        train["max_steps"] = self.cfg_max_steps.value()
        train["gradient_accumulation"] = self.cfg_grad_accum.value()
        train["checkpoint_every"] = self.cfg_ckpt_every.value()
        return cfg

    @staticmethod
    def _as_number(text, fallback):
        try:
            value = float(text)
            return int(value) if value.is_integer() and isinstance(fallback, int) else value
        except (TypeError, ValueError):
            raise ValueError("Learning rate and beta must be valid numbers")

    def _choose_and_load_config(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load config", str(_repo_root()), "JSON (*.json)")
        if path:
            self._config_path = Path(path)
            self._load_config()
            self._reload_store()
            self._set_status("Loaded config %s" % path)

    def _save_config(self, silent=False) -> bool:
        if self._config_path is None:
            path, _ = QFileDialog.getSaveFileName(self, "Save config", str(_repo_root()), "JSON (*.json)")
            if not path:
                return False
            self._config_path = Path(path)
        try:
            if self._config_path.suffix.lower() != ".json":
                raise ValueError("Configuration filename must end in .json")
            self._config = self._collect_config()
            atomic_json(self._config_path, self._config)
            self._load_config()
        except (OSError, ValueError) as exc:
            self._set_status("Could not save config: %s" % exc, error=True)
            return False
        if not silent:
            self._set_status("Saved config %s" % self._config_path)
        return True

    # ----------------------------------------------------------------- store
    def _reload_store(self):
        dataset_dir = self.cfg_dataset_dir.text().strip() if hasattr(self, "cfg_dataset_dir") else ""
        if not dataset_dir:
            self._store = None
            self._comparisons = []
            self._index = -1
            self.counts_label.setText("No dataset directory configured")
            self._show_current()
            return
        try:
            self._store = PreferenceStore(dataset_dir)
            status = self.review_combo.currentData() or "all"
            self._comparisons = self._store.list_comparisons(status=status)
        except Exception as exc:  # store contract raises ValueError/OSError
            self._store = None
            self._comparisons = []
            self._index = -1
            self._set_status("Store error: %s" % exc, error=True)
            self._show_current()
            return
        self._index = 0 if self._comparisons else -1
        self._refresh_counts()
        self._show_current()

    def _refresh_counts(self):
        if self._store is None:
            return
        try:
            counts = self._store.counts()
        except Exception as exc:
            self._set_status("Could not read counts: %s" % exc, error=True)
            return
        self.counts_label.setText(
            "total {total} | unrated {unrated} | rated {rated} | eligible {eligible}"
            " | ties {ties} | skipped {skipped}".format(**counts)
        )

    def _current(self):
        if 0 <= self._index < len(self._comparisons):
            return self._comparisons[self._index]
        return None

    def _show_current(self):
        record = self._current()
        self._clear_selection()
        if record is None:
            self.prompt_label.setText("No comparison to display")
            self.position_label.setText("-")
            self.image_a.set_image(None)
            self.image_b.set_image(None)
            return
        self.position_label.setText("%d / %d" % (self._index + 1, len(self._comparisons)))
        prompt = record.get("prompt") or "(no prompt)"
        negative = record.get("negative_prompt") or ""
        text = "Prompt: %s" % prompt
        if negative:
            text += "\nNegative: %s" % negative
        text += "\nSplit: %s | id: %s" % (record.get("split", "?"), record.get("id", "?"))
        self.prompt_label.setText(text)

        root = Path(self._store.root) if self._store else Path(".")
        by_id = {img["id"]: img for img in record.get("images", [])}
        images = [by_id[k] for k in ("a", "b") if k in by_id]
        labels = [self.image_a, self.image_b]
        for label, image in zip(labels, images[:2]):
            path = image.get("path")
            label.set_image(root / path if path else None)
        for label in labels[len(images[:2]):]:
            label.set_image(None)

        feedback = record.get("feedback")
        if feedback:
            self._apply_feedback_to_widgets(feedback)

    def _apply_feedback_to_widgets(self, feedback):
        pref = feedback.get("preference")
        strength = feedback.get("strength")
        self._sel_pref, self._sel_strength = pref, strength
        for btn, btn_pref, btn_strength in self._pref_buttons:
            if btn_pref == pref and btn_strength == strength:
                btn.setChecked(True)
                self._sel_pref, self._sel_strength = pref, strength
                break
        self._set_combo(self.quality_a, feedback.get("quality_a"))
        self._set_combo(self.quality_b, feedback.get("quality_b"))
        reasons = feedback.get("reasons") or []
        self.reasons_edit.setText(", ".join(str(r) for r in reasons))

    @staticmethod
    def _set_combo(combo, value):
        index = combo.findData(value)
        combo.setCurrentIndex(index if index >= 0 else 0)

    def _clear_selection(self):
        self.pref_group.setExclusive(False)
        for btn, _, _ in self._pref_buttons:
            btn.setChecked(False)
        self.pref_group.setExclusive(True)
        self._sel_pref = None
        self._sel_strength = None
        self.quality_a.setCurrentIndex(0)
        self.quality_b.setCurrentIndex(0)
        self.reasons_edit.clear()

    def _on_pref_clicked(self, button):
        for btn, pref, strength in self._pref_buttons:
            if btn is button:
                self._sel_pref, self._sel_strength = pref, strength
                self._set_status("Selected '%s' - press Save or Save & Next to store." % button.text())
                return

    def _build_feedback(self) -> dict:
        pref = self._sel_pref or "unrated"
        if pref in ("a", "b"):
            if self._sel_strength not in ("slight", "normal", "strong"):
                raise ValueError("Relative preference requires a strength")
            strength = self._sel_strength
        else:
            if pref not in ("tie", "skip", "unrated"):
                raise ValueError("No preference selected")
            strength = None
        reasons = [r.strip() for r in self.reasons_edit.text().split(",") if r.strip()]
        return {
            "preference": pref,
            "strength": strength,
            "quality_a": self.quality_a.currentData(),
            "quality_b": self.quality_b.currentData(),
            "reasons": reasons,
            "updated_at": _utc_now(),
        }

    def _save_current(self, advance=False) -> bool:
        record = self._current()
        if record is None or self._store is None:
            self._set_status("Nothing to save", error=True)
            return False
        try:
            feedback = self._build_feedback()
            self._store.put_feedback(record["id"], feedback)
        except (ValueError, KeyError) as exc:
            self._set_status("Cannot save: %s" % exc, error=True)
            return False
        except Exception as exc:
            self._set_status("Save failed: %s" % exc, error=True)
            return False
        record["feedback"] = feedback
        self._refresh_counts()
        self._set_status("Saved feedback for %s" % record["id"])
        if self.review_combo.currentData() == "unrated" and feedback["preference"] != "unrated":
            self._comparisons.pop(self._index)
            self._index = min(self._index, len(self._comparisons) - 1)
            self._show_current()
        elif advance:
            self._navigate(1)
        return True

    def _skip_current(self):
        record = self._current()
        if record is None or self._store is None:
            self._set_status("Nothing to skip", error=True)
            return
        feedback = {
            "preference": "skip",
            "strength": None,
            "quality_a": self.quality_a.currentData(),
            "quality_b": self.quality_b.currentData(),
            "reasons": [r.strip() for r in self.reasons_edit.text().split(",") if r.strip()],
            "updated_at": _utc_now(),
        }
        try:
            self._store.put_feedback(record["id"], feedback)
        except Exception as exc:
            self._set_status("Skip failed: %s" % exc, error=True)
            return
        record["feedback"] = feedback
        self._refresh_counts()
        self._set_status("Recorded skip for %s" % record["id"])
        if self.review_combo.currentData() == "unrated":
            self._comparisons.pop(self._index)
            self._index = min(self._index, len(self._comparisons) - 1)
            self._show_current()
        else:
            self._navigate(1)

    def _undo(self):
        if self._store is None:
            self._set_status("No store loaded", error=True)
            return
        try:
            comparison_id = self._store.undo_last_feedback()
        except Exception as exc:
            self._set_status("Undo failed: %s" % exc, error=True)
            return
        if comparison_id is None:
            self._set_status("Nothing to undo")
            return
        self._reload_store()
        for i, record in enumerate(self._comparisons):
            if record.get("id") == comparison_id:
                self._index = i
                self._show_current()
                break
        self._set_status("Undid feedback for %s" % comparison_id)

    def _navigate(self, delta):
        if not self._comparisons:
            self._set_status("No comparisons available")
            return
        self._index = (self._index + delta) % len(self._comparisons)
        self._show_current()

    def _show_full_size(self, label):
        if label._pixmap is None:
            return
        dialog = QDialog(self)
        dialog.setWindowTitle("Full size preview")
        layout = QVBoxLayout(dialog)
        view = QLabel()
        view.setPixmap(label._pixmap)
        scroll = QScrollArea()
        scroll.setWidget(view)
        layout.addWidget(scroll)
        dialog.resize(min(label._pixmap.width() + 40, 1400),
                      min(label._pixmap.height() + 40, 1000))
        dialog.show()

    # ------------------------------------------------------------------ jobs
    def _jobs_dir(self) -> Path:
        dataset_dir = self.cfg_dataset_dir.text().strip()
        if not dataset_dir:
            raise ValueError("Dataset directory is required before launching a job")
        jobs = Path(dataset_dir) / "jobs"
        jobs.mkdir(parents=True, exist_ok=True)
        return jobs

    def _job_running(self) -> bool:
        return self._process is not None and self._process.state() != QProcess.NotRunning

    def _launch_generate(self):
        self._launch(["generate"])

    def _launch_train(self):
        args = ["train"]
        resume = self.cfg_resume_dir.text().strip()
        if resume:
            args += ["--resume", resume]
        self._launch(args)

    def _launch(self, command):
        if self._job_running():
            self._set_status("A job is already running; wait for it to finish.", error=True)
            return
        if not self._save_config(silent=True):
            return
        try:
            load_config(self._config_path)
            jobs = self._jobs_dir()
        except (ValueError, OSError) as exc:
            self._set_status(str(exc), error=True)
            return
        self._stop_file = jobs / ("stop-%s.txt" % uuid.uuid4().hex)
        if self._stop_file.exists():
            self._stop_file.unlink()

        program = _backend_python()
        args = ["-X", "utf8", "-u", "-m", "backend.preference.cli"] + list(command) + [
            "--config", str(self._config_path),
            "--stop-file", str(self._stop_file),
        ]
        self.log_view.clear()
        self.log_view.appendPlainText("$ %s %s" % (program, " ".join(args)))
        process = QProcess(self)
        process.setProcessChannelMode(QProcess.MergedChannels)
        process.setWorkingDirectory(str(_repo_root()))
        process.readyReadStandardOutput.connect(self._read_process_output)
        process.finished.connect(self._on_process_finished)
        process.errorOccurred.connect(self._on_process_error)
        self._process = process
        self.job_label.setText("Running: %s" % " ".join(command))
        self._set_status("Started job (%s)." % " ".join(command))
        process.start(program, args)

    def _read_process_output(self):
        if self._process is None:
            return
        data = bytes(self._process.readAllStandardOutput()).decode("utf-8", "replace")
        if data:
            self.log_view.appendPlainText(data.rstrip("\n"))
            self.log_view.verticalScrollBar().setValue(
                self.log_view.verticalScrollBar().maximum()
            )

    def _on_process_error(self, error):
        self.log_view.appendPlainText("QProcess error: %s" % error)
        self._set_status("Process error: %s" % error, error=True)

    def _on_process_finished(self, exit_code, exit_status):
        self.log_view.appendPlainText("[process finished: exit code %s]" % exit_code)
        self.job_label.setText("No job running")
        self._set_status("Job finished with exit code %s (zero means completion or orderly stop)." % exit_code)
        self._process = None
        self._stop_file = None
        self._reload_store()
        if self._close_when_done:
            self._close_when_done = False
            self.close()

    def _request_stop(self):
        if not self._job_running():
            self._set_status("No job running to stop")
            return
        if self._stop_file is None:
            return
        try:
            self._stop_file.write_text("stop requested at %s\n" % _utc_now(), encoding="utf-8")
        except OSError as exc:
            self._set_status("Could not write stop file: %s" % exc, error=True)
            return
        self._set_status("Requested cooperative stop via %s" % self._stop_file)

    def closeEvent(self, event):
        if self._job_running():
            self._request_stop()
            self._close_when_done = True
            self._set_status(
                "Job still running; requested cooperative stop. "
                "Window closes when the process exits.", error=True)
            event.ignore()
            return
        super().closeEvent(event)

    # ----------------------------------------------------------------- misc
    def _set_status(self, message, error=False):
        self.status_label.setText(message)
        self.status_label.setStyleSheet("color: #b00020;" if error else "")


def main(argv=None):
    parser = argparse.ArgumentParser(description="SDXL preference rating UI")
    parser.add_argument("--config", help="Path to a preference configuration JSON file")
    args = parser.parse_args(argv)
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication(sys.argv[:1])
    window = PreferenceWindow(config_path=args.config)
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
