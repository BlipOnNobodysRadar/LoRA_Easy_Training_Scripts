"""Qt integration acceptance without loading the GPU or modifying real ratings."""
import copy
import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
from PySide6.QtGui import QImage, QColor
from PySide6.QtWidgets import QApplication
from main_ui_files.PreferenceWindow import PreferenceWindow
from backend.preference.config import utc_now
from backend.preference.store import PreferenceStore

app = QApplication.instance() or QApplication([])


class PreferenceUITests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.store = PreferenceStore(self.root / 'data')
        images = []
        for side, color in [('a', 'red'), ('b', 'blue')]:
            img = QImage(256, 256, QImage.Format_RGB32)
            img.fill(QColor(color))
            path = self.store.root / (side + '.png')
            img.save(str(path))
            images.append(dict(id=side, seed=0, path=path.name, sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        for i in range(2):
            self.store.add_comparison(dict(id=str(i), created_at=utc_now(), session_id='test',
                group_id=str(i), split='train', synthetic=True, prompt='fixture', negative_prompt='',
                images=list(reversed(images)), model={}, generation_settings={}))
        self.config = self.root / 'fixture.json'
        self.config.write_text(json.dumps(dict(dataset_dir='data', model=dict(base_loras=[
            dict(path='first.safetensors', weight=0), dict(path='second.safetensors', weight=.5)]),
            generation=dict(seed=2**40), training=dict(strength_weights=dict(slight=.3), alpha=12.5))))
        self.window = PreferenceWindow(config_path=self.config)

    def tearDown(self):
        self.window.close()
        self.window.deleteLater()
        app.processEvents()
        self.temp.cleanup()

    def test_image_identity_and_quality_without_winner(self):
        w = self.window
        self.assertEqual(w.image_a._pixmap.toImage().pixelColor(0, 0).name(), '#ff0000')
        w.quality_a.setCurrentIndex(w.quality_a.findData('bad'))
        w.quality_b.setCurrentIndex(w.quality_b.findData('especially_bad'))
        self.assertTrue(w._save_current())
        feedback = self.store.get('0')['feedback']
        self.assertEqual(feedback['preference'], 'unrated')
        self.assertEqual(feedback['quality_b'], 'especially_bad')
        self.assertEqual(self.store.counts()['eligible'], 0)

    def test_selection_navigation_never_saves_and_undo_works(self):
        w = self.window
        w._pref_buttons[0][0].click()
        w._navigate(1)
        self.assertEqual(self.store.counts()['rated'], 0)
        w._pref_buttons[2][0].click()
        w._save_current()
        self.assertEqual(self.store.counts()['ties'], 1)
        w._undo()
        self.assertEqual(self.store.counts()['ties'], 0)

    def test_unrated_filter_removes_saved_choice(self):
        w = self.window
        w.review_combo.setCurrentIndex(1)
        w._pref_buttons[0][0].click()
        w._save_current(True)
        self.assertEqual(len(w._comparisons), 1)
        self.assertEqual(w._current()['id'], '1')

    def test_configuration_preserves_precision_and_advanced_fields(self):
        cfg = self.window._collect_config()
        self.assertEqual(cfg['model']['base_loras'][0]['weight'], 0)
        self.assertEqual(len(cfg['model']['base_loras']), 2)
        self.assertEqual(cfg['generation']['seed'], 2**40)
        self.assertEqual(cfg['training']['alpha'], 12.5)
        self.assertEqual(cfg['training']['strength_weights'], {'slight':.3})


if __name__ == '__main__':
    unittest.main()
