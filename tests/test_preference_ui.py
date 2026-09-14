"""Qt integration acceptance without loading the GPU or modifying real ratings."""
import copy
import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
from PySide6.QtCore import Qt, QPoint, QPointF
from PySide6.QtGui import QImage, QColor, QWheelEvent
from PySide6.QtTest import QTest
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

    def test_wheel_zooms_at_pointer_without_changing_other_image_or_rating(self):
        w = self.window
        w.show()
        app.processEvents()
        a, b = w.image_a, w.image_b
        original = a._pixmap.toImage()
        b_scale = b._scale
        anchor = QPointF(a.width() / 2, a.height() / 2) + QPointF(0, 20)
        before = a.image_point(anchor)
        event = QWheelEvent(anchor, QPointF(a.mapToGlobal(anchor.toPoint())), QPoint(),
                            QPoint(0, 120), Qt.NoButton, Qt.NoModifier, Qt.ScrollUpdate, False)
        app.sendEvent(a, event)
        self.assertGreater(a._scale, a._fit_scale())
        self.assertLess((a.image_point(anchor) - before).manhattanLength(), 1e-6)
        self.assertEqual(b._scale, b_scale)
        self.assertEqual(a._pixmap.toImage(), original)
        self.assertEqual(self.store.counts()['rated'], 0)

    def test_drag_pan_actual_size_and_fit_reset(self):
        w = self.window
        w.show()
        app.processEvents()
        a = w.image_a
        center = a.contentsRect().center()
        event = QWheelEvent(QPointF(center), QPointF(a.mapToGlobal(center)), QPoint(),
                            QPoint(0, 600), Qt.NoButton, Qt.NoModifier, Qt.ScrollUpdate, False)
        app.sendEvent(a, event)
        before = a.image_point(center)
        QTest.mousePress(a, Qt.LeftButton, pos=center)
        QTest.mouseMove(a, center + QPoint(15, 15))
        QTest.mouseRelease(a, Qt.LeftButton, pos=center + QPoint(15, 15))
        self.assertGreater((a.image_point(center) - before).manhattanLength(), 1)
        a.actual_size()
        self.assertEqual(a._scale, 1)
        w._navigate(1)
        self.assertTrue(a._fit_mode)
        self.assertAlmostEqual(a._scale, a._fit_scale())

    def test_fullscreen_escape_restores_maximized_window_and_unsaved_rating(self):
        w = self.window
        self.assertTrue(w.windowFlags() & Qt.WindowMaximizeButtonHint)
        w.showMaximized()
        app.processEvents()
        w._pref_buttons[1][0].click()
        w.fullscreen_btn.click()
        app.processEvents()
        self.assertTrue(w.isFullScreen())
        QTest.keyClick(w, Qt.Key_Escape)
        app.processEvents()
        self.assertFalse(w.isFullScreen())
        self.assertTrue(w.isMaximized())
        self.assertEqual((w._sel_pref, w._sel_strength), ('a', 'slight'))
        QTest.keyClick(w, Qt.Key_Escape)
        self.assertTrue(w.isVisible())
        self.assertEqual(self.store.counts()['rated'], 0)


if __name__ == '__main__':
    unittest.main()
