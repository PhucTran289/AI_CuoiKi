

from __future__ import annotations
import sys, os, csv, io, math
from datetime import datetime
from typing import List, Optional, Tuple

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QStackedWidget,
    QVBoxLayout, QHBoxLayout, QGridLayout, QFormLayout,
    QLabel, QPushButton, QTableWidget, QTableWidgetItem,
    QComboBox, QSpinBox, QDoubleSpinBox, QHeaderView,
    QMessageBox, QFileDialog, QDialog, QDialogButtonBox,
    QFrame, QSizePolicy, QProgressBar, QScrollArea,
    QCheckBox, QToolButton, QTextEdit, QGroupBox,
)
from PyQt6.QtCore  import Qt, QThread, pyqtSignal, QPropertyAnimation, QEasingCurve, QSize
from PyQt6.QtGui   import QColor, QFont, QPainter, QPen, QBrush, QLinearGradient, QPixmap, QIcon, QPalette

try:
    import matplotlib
    matplotlib.use("QtAgg")
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.figure import Figure
    import matplotlib.pyplot as plt
    MPL = True
except ImportError:
    MPL = False

try:
    from core   import create_problem, generate_random_problem, KnapsackProblem, KnapsackResult, evaluate, is_feasible
    from greedy import greedy_ratio, greedy_best_first_search
    from bat    import bat_algorithm
    BACKEND_OK = True;  BACKEND_ERR = ""
except ImportError as e:
    BACKEND_OK = False; BACKEND_ERR = str(e)


# ══════════════════════════════════════════════════════════════════
#  PALETTE & CONSTANTS
# ══════════════════════════════════════════════════════════════════
C = {
    "bg":        "#0d1117",
    "bg2":       "#0f1923",
    "surface":   "#111d2c",
    "surface2":  "#162032",
    "border":    "#1e3448",
    "border2":   "#243d54",
    "teal":      "#00c896",
    "teal_dim":  "#009e77",
    "teal_glow": "#00c89630",
    "blue":      "#3d8ef0",
    "blue_dim":  "#1e5fb8",
    "purple":    "#9b6dff",
    "purple_dim":"#6b4dcc",
    "text":      "#e2e8f0",
    "text2":     "#94a3b8",
    "text3":     "#475569",
    "warn":      "#f59e0b",
    "danger":    "#ef4444",
    "success":   "#10b981",
    "card":      "#0f1c2a",
    "card2":     "#13233a",
    "highlight": "#1a3a5c",
}

ALGO_COLORS = {
    "Greedy (Ratio)":              C["blue"],
    "Greedy Best-First Search":    C["teal"],
    "Bat Algorithm":               C["purple"],
}
ALGO_ICONS = {
    "Greedy (Ratio)":           "🏆",
    "Greedy Best-First Search": "⭐",
    "Bat Algorithm":            "🦇",
}


# ══════════════════════════════════════════════════════════════════
#  GLOBAL STYLESHEET
# ══════════════════════════════════════════════════════════════════
STYLE = f"""
* {{ box-sizing: border-box; }}

QMainWindow, QDialog {{
    background: {C['bg']};
}}
QWidget {{
    background: transparent;
    color: {C['text']};
    font-family: 'Segoe UI', 'SF Pro Display', sans-serif;
    font-size: 13px;
}}

/* ── Scroll ─────────────────────────── */
QScrollArea {{ border: none; background: transparent; }}
QScrollBar:vertical {{
    background: {C['surface']}; width: 6px; border-radius: 3px;
}}
QScrollBar::handle:vertical {{
    background: {C['border2']}; border-radius: 3px; min-height: 30px;
}}
QScrollBar::handle:vertical:hover {{ background: {C['teal_dim']}; }}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}

/* ── Tables ─────────────────────────── */
QTableWidget {{
    background: {C['surface']}; color: {C['text']};
    border: 1px solid {C['border']}; border-radius: 8px;
    gridline-color: {C['border']};
    selection-background-color: {C['highlight']};
    alternate-background-color: {C['surface2']};
}}
QTableWidget::item {{ padding: 6px 12px; }}
QTableWidget::item:selected {{ color: {C['text']}; background: {C['highlight']}; }}
QHeaderView::section {{
    background: {C['surface2']}; color: {C['text2']};
    border: none; border-bottom: 1px solid {C['border']};
    padding: 8px 12px; font-weight: 600; font-size: 12px;
}}
QHeaderView {{ background: {C['surface2']}; }}

/* ── Inputs ─────────────────────────── */
QSpinBox, QDoubleSpinBox, QComboBox {{
    background: {C['surface']}; color: {C['text']};
    border: 1px solid {C['border2']}; border-radius: 6px;
    padding: 6px 10px; font-size: 13px;
}}
QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {{
    border-color: {C['teal']};
}}
QSpinBox::up-button, QSpinBox::down-button,
QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {{
    background: {C['surface2']}; border: none; width: 18px; border-radius: 3px;
}}
QComboBox::drop-down {{ border: none; width: 24px; }}
QComboBox QAbstractItemView {{
    background: {C['surface']}; border: 1px solid {C['border2']};
    selection-background-color: {C['highlight']}; color: {C['text']};
}}

/* ── Labels ─────────────────────────── */
QLabel {{ background: transparent; }}

/* ── Tooltip ─────────────────────────── */
QToolTip {{
    background: {C['surface2']}; color: {C['text']};
    border: 1px solid {C['border2']}; border-radius: 4px; padding: 4px 8px;
}}

/* ── CheckBox ─────────────────────────── */
QCheckBox {{ spacing: 6px; color: {C['text2']}; font-size: 12px; }}
QCheckBox::indicator {{
    width: 16px; height: 16px; border-radius: 4px;
    border: 1px solid {C['border2']}; background: {C['surface']};
}}
QCheckBox::indicator:checked {{
    background: {C['teal']}; border-color: {C['teal']};
}}

/* ── Progress ─────────────────────────── */
QProgressBar {{
    border: 1px solid {C['border']}; border-radius: 4px;
    background: {C['surface']}; height: 6px; text-align: center;
}}
QProgressBar::chunk {{ background: {C['teal']}; border-radius: 4px; }}

/* ── Message Box ─────────────────────── */
QMessageBox {{ background: {C['surface']}; }}
QMessageBox QPushButton {{
    background: {C['surface2']}; color: {C['text']};
    border: 1px solid {C['border2']}; border-radius: 6px;
    padding: 6px 16px; min-width: 80px;
}}
QMessageBox QPushButton:hover {{ border-color: {C['teal']}; color: {C['teal']}; }}
"""


# ══════════════════════════════════════════════════════════════════
#  REUSABLE STYLED WIDGETS
# ══════════════════════════════════════════════════════════════════

def _lbl(text="", size=13, color=None, bold=False, wrap=False) -> QLabel:
    l = QLabel(text)
    l.setWordWrap(wrap)
    style = f"font-size:{size}px; color:{color or C['text']};"
    if bold: style += "font-weight:600;"
    l.setStyleSheet(style)
    return l

def _sep(vertical=False) -> QFrame:
    f = QFrame()
    f.setFrameShape(QFrame.Shape.VLine if vertical else QFrame.Shape.HLine)
    f.setStyleSheet(f"color:{C['border']}; background:{C['border']};")
    f.setFixedWidth(1) if vertical else f.setFixedHeight(1)
    return f

class GlowButton(QPushButton):
    """Button với hiệu ứng glow khi hover."""
    def __init__(self, text, color=None, icon_txt="", parent=None):
        super().__init__(text, parent)
        self._color = color or C["teal"]
        self._icon  = icon_txt
        self._base_style()

    def _base_style(self):
        c = self._color
        # lighten for hover
        self.setStyleSheet(f"""
            QPushButton {{
                background: transparent;
                color: {c};
                border: 1px solid {c};
                border-radius: 8px;
                padding: 8px 20px;
                font-size: 13px;
                font-weight: 600;
            }}
            QPushButton:hover {{
                background: {c}22;
                border-color: {c};
            }}
            QPushButton:pressed {{
                background: {c}44;
            }}
            QPushButton:disabled {{
                color: {C['text3']};
                border-color: {C['border']};
            }}
        """)

class SolidButton(QPushButton):
    """Button nền đặc."""
    def __init__(self, text, color=None, text_color="#0d1117", parent=None):
        super().__init__(text, parent)
        c  = color or C["teal"]
        tc = text_color
        self.setStyleSheet(f"""
            QPushButton {{
                background: qlineargradient(x1:0,y1:0,x2:1,y2:0,
                    stop:0 {c}, stop:1 {c}cc);
                color: {tc};
                border: none;
                border-radius: 8px;
                padding: 10px 28px;
                font-size: 14px;
                font-weight: 700;
            }}
            QPushButton:hover {{
                background: qlineargradient(x1:0,y1:0,x2:1,y2:0,
                    stop:0 {c}dd, stop:1 {c}aa);
            }}
            QPushButton:pressed {{ background: {c}99; }}
            QPushButton:disabled {{
                background: {C['surface2']}; color: {C['text3']};
            }}
        """)

class CardFrame(QFrame):
    """Card container với border + background."""
    def __init__(self, parent=None, accent=None, glow=False):
        super().__init__(parent)
        border = accent or C["border"]
        bg     = C["card"]
        extra  = f"box-shadow: 0 0 20px {accent}33;" if glow and accent else ""
        self.setStyleSheet(f"""
            QFrame {{
                background: {bg};
                border: 1px solid {border};
                border-radius: 12px;
                {extra}
            }}
        """)

class StatCard(QFrame):
    """Mini stat card: label trên, value to dưới."""
    def __init__(self, label, value="—", color=None, parent=None):
        super().__init__(parent)
        self.setStyleSheet(f"""
            QFrame {{
                background: {C['surface']};
                border: 1px solid {C['border']};
                border-radius: 8px;
            }}
        """)
        v = QVBoxLayout(self)
        v.setContentsMargins(12, 10, 12, 10)
        v.setSpacing(2)
        self._lbl_title = _lbl(label, 11, C["text2"])
        self._lbl_value = _lbl(value, 18, color or C["teal"], bold=True)
        v.addWidget(self._lbl_title)
        v.addWidget(self._lbl_value)

    def set_value(self, text, color=None):
        self._lbl_value.setText(text)
        if color:
            self._lbl_value.setStyleSheet(
                f"font-size:18px; font-weight:600; color:{color};")


# ══════════════════════════════════════════════════════════════════
#  STEP INDICATOR (top progress bar)
# ══════════════════════════════════════════════════════════════════
class StepIndicator(QWidget):
    STEPS = ["IMPORT DATA", "RESULT", "ANALYSIS"]

    def __init__(self, parent=None):
        super().__init__(parent)
        self._current = 0
        self.setFixedHeight(64)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self._circles: List[QLabel] = []
        self._labels:  List[QLabel] = []
        self._lines:   List[QFrame] = []

        for i, name in enumerate(self.STEPS):
            # Circle
            circle = QLabel(str(i + 1))
            circle.setAlignment(Qt.AlignmentFlag.AlignCenter)
            circle.setFixedSize(36, 36)
            self._circles.append(circle)

            col = QVBoxLayout()
            col.setSpacing(4)
            col.setAlignment(Qt.AlignmentFlag.AlignCenter)

            row_c = QHBoxLayout()
            row_c.addStretch()
            row_c.addWidget(circle)
            row_c.addStretch()
            col.addLayout(row_c)

            lbl = _lbl(name, 10, C["text3"], bold=True)
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self._labels.append(lbl)
            col.addWidget(lbl)

            w = QWidget(); w.setLayout(col)
            layout.addWidget(w)

            if i < len(self.STEPS) - 1:
                line = QFrame()
                line.setFrameShape(QFrame.Shape.HLine)
                line.setFixedHeight(2)
                line.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
                self._lines.append(line)
                layout.addWidget(line)

        self.set_step(0)

    def set_step(self, step: int):
        self._current = step
        for i, (circle, lbl) in enumerate(zip(self._circles, self._labels)):
            if i < step:  # done
                circle.setText("✓")
                circle.setStyleSheet(f"""
                    QLabel {{ background:{C['teal']}; color:#0d1117;
                    border-radius:18px; font-size:14px; font-weight:700; }}
                """)
                lbl.setStyleSheet(f"font-size:10px; color:{C['teal']}; font-weight:700;")
            elif i == step:  # active
                circle.setText(str(i + 1))
                circle.setStyleSheet(f"""
                    QLabel {{ background:{C['teal']}; color:#0d1117;
                    border-radius:18px; font-size:14px; font-weight:700; }}
                """)
                lbl.setStyleSheet(f"font-size:10px; color:{C['teal']}; font-weight:700;")
            else:  # future
                circle.setText(str(i + 1))
                circle.setStyleSheet(f"""
                    QLabel {{ background:{C['surface']}; color:{C['text3']};
                    border:2px solid {C['border2']}; border-radius:18px;
                    font-size:14px; font-weight:700; }}
                """)
                lbl.setStyleSheet(f"font-size:10px; color:{C['text3']}; font-weight:600;")
        for i, line in enumerate(self._lines):
            done = (i < step)
            line.setStyleSheet(
                f"background:{C['teal']}; border:none;" if done
                else f"background:{C['border']}; border:none;"
            )


# ══════════════════════════════════════════════════════════════════
#  CSV PARSER
# ══════════════════════════════════════════════════════════════════
class CsvParseResult:
    def __init__(self):
        self.capacity: Optional[float] = None
        self.names:    List[str]   = []
        self.weights:  List[float] = []
        self.values:   List[float] = []
        self.warnings: List[str]   = []
        self.error:    str         = ""

def parse_knapsack_csv(filepath: str) -> CsvParseResult:
    r = CsvParseResult()
    try:
        with open(filepath, newline="", encoding="utf-8-sig") as f:
            raw = f.read()
    except Exception as e:
        r.error = f"Không đọc được file: {e}"; return r

    sample = raw[:2000]
    delim  = "," if sample.count(",") >= sample.count(";") else ";"
    reader = csv.reader(io.StringIO(raw), delimiter=delim)
    rows   = [row for row in reader if any(c.strip() for c in row)]
    if not rows:
        r.error = "File CSV rỗng."; return r

    cursor = 0
    first  = [c.strip() for c in rows[0]]
    if len(first) >= 2 and first[0].lower() in ("capacity","cap","w_cap"):
        try: r.capacity = float(first[1]); cursor = 1
        except ValueError: pass
    elif len(first) == 1:
        try: r.capacity = float(first[0]); cursor = 1
        except ValueError: pass

    if cursor < len(rows):
        cand = [c.strip().lower() for c in rows[cursor]]
        if any(h in cand for h in ("name","weight","value","w","v","tên","item")):
            cursor += 1

    skipped = 0
    for i, row in enumerate(rows[cursor:]):
        row = [c.strip() for c in row if c.strip()]
        if not row: continue
        try:
            if len(row) >= 3:
                name, w, v = row[0], float(row[1]), float(row[2])
            elif len(row) == 2:
                name, w, v = f"Item_{len(r.names):03d}", float(row[0]), float(row[1])
            else:
                skipped += 1; continue
            if w <= 0: r.warnings.append(f"Dòng {cursor+i+1}: weight≤0, bỏ qua."); skipped+=1; continue
            if v <  0: r.warnings.append(f"Dòng {cursor+i+1}: value<0, bỏ qua.");  skipped+=1; continue
            r.names.append(name); r.weights.append(w); r.values.append(v)
        except ValueError:
            skipped += 1
            r.warnings.append(f"Dòng {cursor+i+1}: không parse được, bỏ qua.")
    if skipped:
        r.warnings.append(f"Tổng cộng bỏ qua {skipped} dòng.")
    if not r.names:
        r.error = "Không có dòng dữ liệu hợp lệ nào."
    return r


# ══════════════════════════════════════════════════════════════════
#  CAPACITY DIALOG
# ══════════════════════════════════════════════════════════════════
class CapacityDialog(QDialog):
    def __init__(self, total_w: float, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Nhập Capacity")
        self.setMinimumWidth(360)
        self.setStyleSheet(f"background:{C['surface']}; color:{C['text']};")
        v = QVBoxLayout(self)
        v.setSpacing(12); v.setContentsMargins(20,20,20,20)
        v.addWidget(_lbl(
            f"File CSV không có dòng <b>capacity</b>.<br>"
            f"Tổng trọng lượng = <b>{total_w:.2f}</b>.<br>Nhập sức chứa:",
            13, wrap=True))
        self.spin = QDoubleSpinBox()
        self.spin.setRange(0.01, 1e9); self.spin.setDecimals(2)
        self.spin.setValue(round(total_w * 0.5, 2))
        v.addWidget(self.spin)
        v.addWidget(_lbl("Gợi ý: 50% tổng trọng lượng", 11, C["text2"]))
        btns = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok |
                                QDialogButtonBox.StandardButton.Cancel)
        btns.button(QDialogButtonBox.StandardButton.Ok).setStyleSheet(
            f"background:{C['teal']}; color:#0d1117; border:none; border-radius:6px; padding:6px 16px; font-weight:700;")
        btns.accepted.connect(self.accept); btns.rejected.connect(self.reject)
        v.addWidget(btns)
    def value(self): return self.spin.value()


# ══════════════════════════════════════════════════════════════════
#  SOLVER WORKER THREAD
# ══════════════════════════════════════════════════════════════════
class SolverWorker(QThread):
    finished = pyqtSignal(list)
    error    = pyqtSignal(str)

    def __init__(self, problem, algorithms: List[str], bat_params: dict):
        super().__init__()
        self.problem    = problem
        self.algorithms = algorithms
        self.bat_params = bat_params

    def run(self):
        try:
            results = []
            for algo in self.algorithms:
                if   algo == "greedy_ratio": r = greedy_ratio(self.problem)
                elif algo == "gbfs":         r = greedy_best_first_search(self.problem)
                elif algo == "bat":          r = bat_algorithm(self.problem, **self.bat_params)
                else: continue
                results.append(r)
            self.finished.emit(results)
        except Exception as e:
            self.error.emit(str(e))


# ══════════════════════════════════════════════════════════════════
#  MATPLOTLIB CHART HELPERS
# ══════════════════════════════════════════════════════════════════
def _make_figure(ncols=1, nrows=1, figsize=(5,3)):
    fig = Figure(figsize=figsize, facecolor=C["card"])
    fig.subplots_adjust(left=0.12, right=0.97, top=0.88, bottom=0.18)
    return fig

def _style_ax(ax, title="", xlabel="", ylabel=""):
    ax.set_facecolor(C["surface"])
    ax.set_title(title, color=C["text"], fontsize=11, pad=8, fontweight="bold")
    ax.set_xlabel(xlabel, color=C["text2"], fontsize=9)
    ax.set_ylabel(ylabel, color=C["text2"], fontsize=9)
    ax.tick_params(colors=C["text2"], labelsize=8)
    for sp in ax.spines.values(): sp.set_edgecolor(C["border"])
    ax.tick_params(axis="x", colors=C["text2"])
    ax.tick_params(axis="y", colors=C["text2"])


class BarChart(QWidget):
    def __init__(self, title="", xlabel="", ylabel="", figsize=(4,2.8), parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self); layout.setContentsMargins(0,0,0,0)
        if not MPL:
            layout.addWidget(_lbl("matplotlib not installed", 12, C["text3"]))
            self.canvas = None; return
        self.fig = _make_figure(figsize=figsize)
        self.ax  = self.fig.add_subplot(111)
        _style_ax(self.ax, title, xlabel, ylabel)
        self.canvas = FigureCanvas(self.fig)
        layout.addWidget(self.canvas)
        self._title = title; self._xlabel = xlabel; self._ylabel = ylabel

    def update(self, names, values, colors):
        if not self.canvas: return
        self.ax.clear(); _style_ax(self.ax, self._title, self._xlabel, self._ylabel)
        bars = self.ax.bar(names, values, color=colors, width=0.55, zorder=3)
        for bar, v in zip(bars, values):
            self.ax.text(bar.get_x()+bar.get_width()/2,
                         bar.get_height() + max(values)*0.015,
                         f"{v:.4f}" if v < 10000 else f"{v:.0f}",
                         ha="center", va="bottom", color=C["text"], fontsize=9)
        self.ax.set_axisbelow(True)
        self.ax.yaxis.grid(True, color=C["border"], linestyle="--", alpha=0.5)
        self.canvas.draw()


class HBarChart(QWidget):
    """Horizontal bar — thời gian."""
    def __init__(self, figsize=(4,2.8), parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self); layout.setContentsMargins(0,0,0,0)
        if not MPL:
            layout.addWidget(_lbl("matplotlib not installed", 12, C["text3"])); self.canvas = None; return
        self.fig = _make_figure(figsize=figsize)
        self.ax  = self.fig.add_subplot(111)
        self.canvas = FigureCanvas(self.fig)
        layout.addWidget(self.canvas)

    def update(self, names, times_ms, colors):
        if not self.canvas: return
        self.ax.clear()
        _style_ax(self.ax, "SO SÁNH THỜI GIAN THỰC THI (ms)",
                  xlabel="Thời gian (ms)", ylabel="")
        bars = self.ax.barh(names, times_ms, color=colors, height=0.5, zorder=3)
        for bar, t in zip(bars, times_ms):
            self.ax.text(bar.get_width() + max(times_ms)*0.01,
                         bar.get_y() + bar.get_height()/2,
                         f"{t:.3f} ms", va="center", color=C["text"], fontsize=9)
        self.ax.set_axisbelow(True)
        self.ax.xaxis.grid(True, color=C["border"], linestyle="--", alpha=0.5)
        note = _lbl("Thời gian càng nhỏ càng tốt", 9, C["text3"])
        self.canvas.draw()


class LineChart(QWidget):
    """Convergence line chart."""
    def __init__(self, figsize=(4,2.8), parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self); layout.setContentsMargins(0,0,0,0)
        if not MPL:
            layout.addWidget(_lbl("matplotlib not installed", 12, C["text3"])); self.canvas = None; return
        self.fig = _make_figure(figsize=figsize)
        self.ax  = self.fig.add_subplot(111)
        self.canvas = FigureCanvas(self.fig)
        layout.addWidget(self.canvas)

    def update(self, results):
        if not self.canvas: return
        self.ax.clear()
        _style_ax(self.ax, "BIỂU ĐỒ HỘI TỤ – BAT ALGORITHM",
                  xlabel="Vòng lặp", ylabel="Best Value")
        has_data = False
        for r in results:
            if hasattr(r, "extra") and "convergence" in (r.extra or {}):
                conv = r.extra["convergence"]
                self.ax.plot(conv, color=C["purple"], linewidth=1.8, label="Best Value")
                has_data = True
        if not has_data:
            self.ax.text(0.5, 0.5, "Không có dữ liệu hội tụ\n(chưa chạy Bat Algorithm)",
                         ha="center", va="center", color=C["text2"], fontsize=11,
                         transform=self.ax.transAxes)
        else:
            self.ax.legend(facecolor=C["surface"], labelcolor=C["text2"], fontsize=9,
                           framealpha=0.8, edgecolor=C["border"])
        self.ax.set_axisbelow(True)
        self.ax.yaxis.grid(True, color=C["border"], linestyle="--", alpha=0.5)
        self.canvas.draw()


# ══════════════════════════════════════════════════════════════════
#  STEP 1: IMPORT DATA PAGE
# ══════════════════════════════════════════════════════════════════
class ImportPage(QWidget):
    run_requested = pyqtSignal()  # emitted when user clicks "Chạy thuật toán"

    def __init__(self, parent=None):
        super().__init__(parent)
        self._last_dir = os.path.expanduser("~")
        self._build()

    def _build(self):
        root = QHBoxLayout(self)
        root.setContentsMargins(24, 16, 24, 16)
        root.setSpacing(20)

        # ── LEFT: Input Data ──────────────────────────────────
        left = QVBoxLayout(); left.setSpacing(12)

        # Header
        hdr = QHBoxLayout()
        icon_lbl = _lbl("📋", 16); icon_lbl.setFixedWidth(24)
        hdr.addWidget(icon_lbl)
        hdr.addWidget(_lbl("INPUT DATA", 13, C["text"], bold=True))
        hdr.addStretch()
        left.addLayout(hdr)

        # Capacity row
        cap_card = CardFrame()
        cap_lay  = QHBoxLayout(cap_card)
        cap_lay.setContentsMargins(14,10,14,10)
        cap_lay.addWidget(_lbl("Sức chứa (capacity):", 12, C["text2"]))
        self.spin_capacity = QDoubleSpinBox()
        self.spin_capacity.setRange(0.01, 1e9)
        self.spin_capacity.setValue(15.0)
        self.spin_capacity.setDecimals(2)
        self.spin_capacity.setSingleStep(1.0)
        self.spin_capacity.setFixedWidth(120)
        cap_lay.addWidget(self.spin_capacity)
        cap_lay.addStretch()

        btn_random = GlowButton("🎲  Random Data", C["teal"])
        btn_import = GlowButton("📂  Import CSV",  C["blue"])
        btn_clear  = GlowButton("✖  Xóa dữ liệu", C["danger"])
        btn_random.setFixedHeight(34)
        btn_import.setFixedHeight(34)
        btn_clear.setFixedHeight(34)
        btn_random.clicked.connect(self._generate_random)
        btn_import.clicked.connect(self._import_csv)
        btn_clear.clicked.connect(self._clear_table)
        cap_lay.addWidget(btn_random)
        cap_lay.addWidget(btn_import)
        cap_lay.addWidget(btn_clear)
        left.addWidget(cap_card)

        # Source label
        self.lbl_source = _lbl("📄  Nguồn: nhập tay / random", 11, C["text3"])
        left.addWidget(self.lbl_source)

        # Table
        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["#", "Tên", "Trọng lượng", ])
        self.table.setHorizontalHeaderLabels(["#", "Tên vật phẩm", "Trọng lượng", ])
        self.table.setColumnCount(3)
        self.table.setHorizontalHeaderLabels(["Tên", "Trọng lượng", "Giá trị"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.setAlternatingRowColors(True)
        self.table.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        left.addWidget(self.table)

        # Row buttons
        btn_row = QHBoxLayout(); btn_row.setSpacing(8)
        btn_add = GlowButton("+ Thêm hàng", C["teal"]); btn_add.setFixedHeight(32)
        btn_del = GlowButton("🗑  Xóa hàng", C["text3"]); btn_del.setFixedHeight(32)
        btn_add.clicked.connect(self._add_row)
        btn_del.clicked.connect(self._delete_row)
        btn_row.addWidget(btn_add); btn_row.addWidget(btn_del); btn_row.addStretch()
        left.addLayout(btn_row)

        # Random config
        rand_row = QHBoxLayout(); rand_row.setSpacing(8)
        rand_row.addWidget(_lbl("Số vật ngẫu nhiên:", 11, C["text2"]))
        self.spin_rand_n = QSpinBox(); self.spin_rand_n.setRange(2,2000); self.spin_rand_n.setValue(10)
        self.spin_rand_n.setFixedWidth(70)
        rand_row.addWidget(self.spin_rand_n)
        rand_row.addSpacing(8)
        rand_row.addWidget(_lbl("Seed:", 11, C["text2"]))
        self.spin_seed = QSpinBox(); self.spin_seed.setRange(-1,99999); self.spin_seed.setValue(42)
        self.spin_seed.setSpecialValueText("Random"); self.spin_seed.setFixedWidth(80)
        rand_row.addWidget(self.spin_seed)
        rand_row.addStretch()
        left.addLayout(rand_row)

        # Hint
        hint = _lbl("💡  Bạn có thể nhập dữ liệu thủ công hoặc tạo ngẫu nhiên để bắt đầu.", 11, C["text3"], wrap=True)
        left.addWidget(hint)

        # ── RIGHT: Settings ──────────────────────────────────
        right = QVBoxLayout(); right.setSpacing(12)
        right.addWidget(_lbl("⚙  SETTING – BAT ALGORITHM", 13, C["text"], bold=True))

        bat_card = CardFrame()
        bat_layout = QFormLayout(bat_card)
        bat_layout.setContentsMargins(16,14,16,14)
        bat_layout.setSpacing(10)
        bat_layout.setLabelAlignment(Qt.AlignmentFlag.AlignLeft)

        def dsp(lo,hi,val,step,dec):
            s = QDoubleSpinBox(); s.setRange(lo,hi); s.setValue(val)
            s.setDecimals(dec); s.setSingleStep(step); return s
        def isp(lo,hi,val):
            s = QSpinBox(); s.setRange(lo,hi); s.setValue(val); return s

        self.bat_n_bats   = isp(5,1000,30)
        self.bat_max_iter = isp(50,1000,500)  
        self.bat_alpha    = dsp(0.01,1.0,0.90,0.05,2)
        self.bat_gamma    = dsp(0.01,5.0,0.90,0.10,2)
        self.bat_mut_rate = dsp(0.0,0.05,0.020,0.005,3)
        self.bat_f_min    = dsp(0.0,10.0,0.0,0.5,1)
        self.bat_f_max    = dsp(0.1,10.0,2.0,0.5,1)

        fields = [
            ("Số dơi (n_bats):",     self.bat_n_bats,   "Số lượng dơi trong quần thể"),
            ("Vòng lặp (max_iter):", self.bat_max_iter,  "Số vòng lặp tối đa"),
            ("Alpha (loudness):",    self.bat_alpha,     "Độ âm thanh (0 – 1)"),
            ("Gamma (pulse):",       self.bat_gamma,     "Tần số xung (0 – 1)"),
            ("Đột biến (≤ 5%):",     self.bat_mut_rate,  "Tỉ lệ đột biến (0 – 0.05)"),
            ("f_min:",               self.bat_f_min,     "Tần số tối thiểu"),
            ("f_max:",               self.bat_f_max,     "Tần số tối đa"),
        ]
        for label, widget, tip in fields:
            lbl = _lbl(label, 12, C["text2"])
            lbl.setToolTip(tip)
            widget.setToolTip(tip)
            bat_layout.addRow(lbl, widget)

        right.addWidget(bat_card)

        # Mô tả tham số
        desc_card = CardFrame()
        desc_lay  = QVBoxLayout(desc_card); desc_lay.setContentsMargins(14,12,14,12)
        desc_lay.addWidget(_lbl("ℹ  Mô tả tham số", 12, C["teal"], bold=True))
        desc_lay.addWidget(_lbl(
            "Các tham số ảnh hưởng đến khả năng khám phá và khai thác của thuật toán.\n"
            "Bạn có thể giữ mặc định hoặc điều chỉnh để tìm kết quả tốt hơn.",
            11, C["text2"], wrap=True))
        right.addWidget(desc_card)
        right.addStretch()

        # ── Thuật toán chọn ──────────────────────────────────
        algo_card = CardFrame()
        algo_lay  = QHBoxLayout(algo_card); algo_lay.setContentsMargins(14,10,14,10)
        algo_lay.addWidget(_lbl("Chọn thuật toán:", 12, C["text2"]))
        self.combo_algo = QComboBox()
        self.combo_algo.addItems([
            "Greedy (Ratio)",
            "Greedy Best-First Search (GBFS)",
            "Bat Algorithm",
            "Tất cả (so sánh)",
        ])
        self.combo_algo.setFixedWidth(240)
        algo_lay.addWidget(self.combo_algo)
        algo_lay.addStretch()
        right.insertWidget(1, algo_card)

        # ── RUN BUTTON ───────────────────────────────────────
        self.btn_run = SolidButton("▶   CHẠY THUẬT TOÁN\nBắt đầu tìm kiếm giải pháp tối ưu",
                                   C["teal"], "#0d1117")
        self.btn_run.setMinimumHeight(56)
        self.btn_run.clicked.connect(self.run_requested.emit)

        self.progress = QProgressBar(); self.progress.setRange(0,0)
        self.progress.setVisible(False); self.progress.setFixedHeight(4)

        # ── Assemble ─────────────────────────────────────────
        left_w = QWidget(); left_w.setLayout(left)
        left_w.setMinimumWidth(400)
        right_w = QWidget(); right_w.setLayout(right)
        right_w.setMinimumWidth(320); right_w.setMaximumWidth(420)

        root.addWidget(left_w, 3)
        root.addWidget(_sep(vertical=True))
        root.addWidget(right_w, 2)

        # Run button + progress at bottom – added to parent via main window later
        self._run_btn_ref  = self.btn_run
        self._prog_ref     = self.progress

        # We'll place run btn via _build_bottom_bar in main window

    # ── TABLE HELPERS ─────────────────────────────────────────
    def _add_row(self, weight=1.0, value=1.0, name=""):
        r = self.table.rowCount(); self.table.insertRow(r)
        self.table.setItem(r, 0, QTableWidgetItem(name or f"Item_{r}"))
        self.table.setItem(r, 1, QTableWidgetItem(str(weight)))
        self.table.setItem(r, 2, QTableWidgetItem(str(value)))

    def _delete_row(self):
        rows = sorted({i.row() for i in self.table.selectedItems()}, reverse=True)
        for r in rows: self.table.removeRow(r)
        if not rows and self.table.rowCount() > 0:
            self.table.removeRow(self.table.rowCount()-1)

    def _clear_table(self):
        self.table.setRowCount(0)
        self.lbl_source.setText("📄  Nguồn: nhập tay / random")

    def _generate_random(self):
        if not BACKEND_OK: return
        n = self.spin_rand_n.value(); sv = self.spin_seed.value()
        seed = sv if sv >= 0 else None
        try:
            prob = generate_random_problem(n=n, seed=seed)
            self.spin_capacity.setValue(prob.capacity)
            self.table.setRowCount(0)
            for item in prob.items: self._add_row(item.weight, item.value, item.name)
            self.lbl_source.setText(
                f"📄  Random — {n} vật | seed={seed} | capacity={prob.capacity:.2f}")
        except Exception as e:
            QMessageBox.warning(self, "Lỗi", str(e))

    def _import_csv(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Chọn file CSV Knapsack", self._last_dir,
            "CSV Files (*.csv);;Text Files (*.txt);;All Files (*)")
        if not path: return
        self._last_dir = os.path.dirname(path)
        res = parse_knapsack_csv(path)
        if res.error:
            QMessageBox.critical(self, "Lỗi đọc CSV", res.error); return
        if res.capacity is None:
            dlg = CapacityDialog(sum(res.weights), self)
            if dlg.exec() != QDialog.DialogCode.Accepted: return
            res.capacity = dlg.value()
        if res.warnings:
            QMessageBox.warning(self, "Cảnh báo", "\n".join(res.warnings[:10]))
        self.spin_capacity.setValue(res.capacity)
        self.table.setRowCount(0)
        for n, w, v in zip(res.names, res.weights, res.values):
            self._add_row(w, v, n)
        fname = os.path.basename(path)
        self.lbl_source.setText(f"📄  {fname} — {len(res.names)} vật | cap={res.capacity}")
        QMessageBox.information(self, "Import thành công",
            f"Đã nạp <b>{len(res.names)}</b> vật phẩm từ <b>{fname}</b>")

    def populate_default(self):
        for n,w,v in [("A",2,3),("B",3,4),("C",4,5),("D",5,8),("E",6,9)]:
            self._add_row(w,v,n)

    def parse_input(self):
        cap = self.spin_capacity.value()
        if cap <= 0: raise ValueError("Sức chứa phải > 0.")
        n = self.table.rowCount()
        if n == 0: raise ValueError("Bảng vật phẩm trống.")
        weights, values, names = [], [], []
        for row in range(n):
            ni = self.table.item(row,0)
            wi = self.table.item(row,1)
            vi = self.table.item(row,2)
            name = (ni.text().strip() if ni else "") or f"Item_{row}"
            try:   w = float(wi.text()) if wi else 0; v = float(vi.text()) if vi else 0
            except: raise ValueError(f"Hàng {row+1}: trọng lượng/giá trị phải là số.")
            if w <= 0: raise ValueError(f"Hàng {row+1}: trọng lượng phải > 0.")
            if v <  0: raise ValueError(f"Hàng {row+1}: giá trị không được âm.")
            weights.append(w); values.append(v); names.append(name)
        return cap, weights, values, names

    def get_algo_list(self) -> List[str]:
        idx = self.combo_algo.currentIndex()
        return {0:["greedy_ratio"], 1:["gbfs"],
                2:["bat"], 3:["greedy_ratio","gbfs","bat"]}.get(idx, ["greedy_ratio"])

    def get_bat_params(self) -> dict:
        return {
            "n_bats":        int(self.bat_n_bats.value()),
            "max_iter":      int(self.bat_max_iter.value()),
            "alpha":         self.bat_alpha.value(),
            "gamma":         self.bat_gamma.value(),
            "mutation_rate": self.bat_mut_rate.value(),
            "f_min":         self.bat_f_min.value(),
            "f_max":         self.bat_f_max.value(),
            "seed":          None,
        }


# ══════════════════════════════════════════════════════════════════
#  ALGORITHM RESULT CARD (used on Result page)
# ══════════════════════════════════════════════════════════════════
class AlgoResultCard(CardFrame):
    def __init__(self, algo_name: str, is_best=False, parent=None):
        color = ALGO_COLORS.get(algo_name, C["teal"])
        super().__init__(parent, accent=color if is_best else None, glow=is_best)
        self.algo_name = algo_name
        self._color    = color
        self._is_best  = is_best
        self._build(is_best)

    def _build(self, is_best):
        v = QVBoxLayout(self); v.setContentsMargins(16,14,16,14); v.setSpacing(8)

        # Header
        hdr = QHBoxLayout(); hdr.setSpacing(8)
        icon = ALGO_ICONS.get(self.algo_name, "🔷")
        icon_lbl = _lbl(icon, 18)
        icon_lbl.setFixedWidth(28)
        hdr.addWidget(icon_lbl)
        name_lbl = _lbl(self.algo_name, 13, self._color, bold=True)
        hdr.addWidget(name_lbl)
        hdr.addStretch()
        if is_best:
            best_badge = _lbl("THUẬT TOÁN TỐT NHẤT", 9, "#0d1117", bold=True)
            best_badge.setStyleSheet(
                f"background:{self._color}; color:#0d1117; border-radius:4px;"
                f"padding:2px 8px; font-size:9px; font-weight:700;")
            hdr.addWidget(best_badge)
        v.addLayout(hdr)

        # Value
        val_lbl = _lbl("Tổng giá trị", 10, C["text2"])
        self.lbl_value = _lbl("—", 28, self._color, bold=True)
        v.addWidget(val_lbl); v.addWidget(self.lbl_value)

        # Stats row
        stats = QHBoxLayout(); stats.setSpacing(12)
        self.stat_weight = StatCard("Trọng lượng", "—", C["text"])
        self.stat_time   = StatCard("Thời gian",   "—", C["text"])
        self.stat_count  = StatCard("Số vật chọn", "—", C["text"])
        for s in [self.stat_weight, self.stat_time, self.stat_count]:
            stats.addWidget(s)
        v.addLayout(stats)

        # Detail link
        self.btn_detail = QPushButton("⊕  Xem chi tiết vật phẩm được chọn")
        self.btn_detail.setStyleSheet(f"""
            QPushButton {{
                background: transparent; color: {self._color};
                border: none; text-align:left; padding: 2px 0;
                font-size:11px;
            }}
            QPushButton:hover {{ text-decoration: underline; }}
        """)
        v.addWidget(self.btn_detail)

    def fill(self, result, problem):
        selected = [i for i,s in enumerate(result.solution) if s==1]
        n_total  = problem.n
        w_pct    = result.weight / problem.capacity * 100

        self.lbl_value.setText(f"{result.value:.4f}")
        self.stat_weight.set_value(f"{result.weight:.4f}\n({w_pct:.0f}%)")
        self.stat_time.set_value(f"{result.time*1000:.3f} ms")
        self.stat_count.set_value(f"{len(selected)} / {n_total}")


# ══════════════════════════════════════════════════════════════════
#  STEP 2: RESULT PAGE
# ══════════════════════════════════════════════════════════════════
class ResultPage(QWidget):
    go_back     = pyqtSignal()
    go_analysis = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._results  = []
        self._problem  = None
        self._build()

    def _build(self):
        root = QVBoxLayout(self); root.setContentsMargins(24,16,24,0); root.setSpacing(12)

        # Header row
        hdr = QHBoxLayout()
        icon = _lbl("🏆", 18); icon.setFixedWidth(28)
        hdr.addWidget(icon)
        title_col = QVBoxLayout(); title_col.setSpacing(2)
        title_col.addWidget(_lbl("KẾT QUẢ THUẬT TOÁN", 14, C["text"], bold=True))
        self.lbl_subtitle = _lbl("Kết quả từng thuật toán trên cùng bộ dữ liệu", 11, C["text2"])
        title_col.addWidget(self.lbl_subtitle)
        hdr.addLayout(title_col); hdr.addStretch()

        # Export buttons
        btn_export = GlowButton("↓  Xuất kết quả", C["teal"]); btn_export.setFixedHeight(32)
        btn_log    = GlowButton("📋  Log chi tiết",  C["text2"]); btn_log.setFixedHeight(32)
        btn_export.clicked.connect(self._export)
        btn_log.clicked.connect(self._show_log)
        hdr.addWidget(btn_export); hdr.addWidget(btn_log)
        root.addLayout(hdr)

        # Algorithm cards container (filled dynamically)
        self.cards_row = QHBoxLayout(); self.cards_row.setSpacing(12)
        root.addLayout(self.cards_row)

        # Detail table section
        detail_lbl = _lbl("⭐  CHI TIẾT KẾT QUẢ THUẬT TOÁN TỐT NHẤT", 12, C["teal"], bold=True)
        root.addWidget(detail_lbl)
        self.lbl_detail_algo = _lbl("Thông tin chi tiết từ thuật toán: —", 11, C["text2"])
        root.addWidget(self.lbl_detail_algo)

        detail_row = QHBoxLayout(); detail_row.setSpacing(12)

        # Left: detail table
        self.tbl_detail = QTableWidget(0, 5)
        self.tbl_detail.setHorizontalHeaderLabels(["#","Tên","Giá trị","Tỉ lệ (v/w)","Được chọn"])
        self.tbl_detail.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.tbl_detail.setAlternatingRowColors(True)
        self.tbl_detail.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.tbl_detail.setMaximumHeight(220)
        detail_row.addWidget(self.tbl_detail, 3)

        # Right: overview card
        overview = CardFrame()
        ov_lay = QVBoxLayout(overview); ov_lay.setContentsMargins(16,12,16,12); ov_lay.setSpacing(8)
        ov_lay.addWidget(_lbl("📊  TỔNG QUAN", 12, C["teal"], bold=True))
        self.ov_stats = QFormLayout()
        self.ov_stats.setSpacing(8)
        self._ov_fields = {}
        for key in ["Sức chứa (capacity)","Tổng trọng lượng","Tổng giá trị đạt được",
                    "Số vật phẩm được chọn","Thời gian chạy","Thời điểm"]:
            lbl = _lbl(key+":", 11, C["text2"])
            val = _lbl("—", 11, C["text"])
            self._ov_fields[key] = val
            self.ov_stats.addRow(lbl, val)
        ov_lay.addLayout(self.ov_stats)
        detail_row.addWidget(overview, 2)
        root.addLayout(detail_row)

    def fill(self, results: List, problem):
        self._results = results; self._problem = problem

        # Clear old cards
        while self.cards_row.count():
            item = self.cards_row.takeAt(0)
            if item.widget(): item.widget().deleteLater()

        if not results: return
        best = max(results, key=lambda r: r.value)

        for r in results:
            card = AlgoResultCard(r.algorithm, is_best=(r==best))
            card.fill(r, problem)
            card.btn_detail.clicked.connect(lambda checked, _r=r: self._fill_detail(_r))
            self.cards_row.addWidget(card)

        # Default to best
        self._fill_detail(best)

        # Overview
        self._ov_fields["Sức chứa (capacity)"].setText(f"{problem.capacity:.4f}")
        self._ov_fields["Tổng trọng lượng"].setText(f"{best.weight:.4f}  ({best.weight/problem.capacity*100:.0f}%)")
        self._ov_fields["Tổng giá trị đạt được"].setText(f"{best.value:.4f}")
        sel = sum(best.solution)
        self._ov_fields["Số vật phẩm được chọn"].setText(f"{sel} / {problem.n}")
        self._ov_fields["Thời gian chạy"].setText(f"{best.time*1000:.3f} ms")
        self._ov_fields["Thời điểm"].setText(datetime.now().strftime("%d/%m/%Y %I:%M %p"))

    def _fill_detail(self, result):
        prob = self._problem
        if not prob: return
        self.lbl_detail_algo.setText(f"Thông tin chi tiết từ thuật toán: {result.algorithm}")
        self.tbl_detail.setRowCount(prob.n)
        for i, item in enumerate(prob.items):
            chosen = result.solution[i] == 1
            vals = [str(i+1), item.name,
                    f"{item.value:.3f}", f"{item.ratio:.4f}",
                    "✅" if chosen else "—"]
            for col, txt in enumerate(vals):
                cell = QTableWidgetItem(txt)
                cell.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                if chosen:
                    cell.setForeground(QColor(C["teal"]))
                self.tbl_detail.setItem(i, col, cell)

    def _export(self):
        if not self._results:
            QMessageBox.information(self,"Chưa có kết quả","Hãy chạy thuật toán trước."); return
        path, _ = QFileDialog.getSaveFileName(self,"Lưu kết quả","knapsack_result.csv",
                                               "CSV (*.csv);;Text (*.txt)")
        if not path: return
        try:
            with open(path,"w",newline="",encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["Algorithm","Value","Weight","Time_ms"])
                for r in self._results:
                    w.writerow([r.algorithm, r.value, r.weight, r.time*1000])
            QMessageBox.information(self,"Đã lưu",f"Kết quả đã lưu vào:\n{path}")
        except Exception as e:
            QMessageBox.critical(self,"Lỗi",str(e))

    def _show_log(self):
        if not self._results:
            QMessageBox.information(self,"Chưa có kết quả","Hãy chạy thuật toán trước."); return
        dlg = QDialog(self); dlg.setWindowTitle("Log chi tiết")
        dlg.resize(500,400); dlg.setStyleSheet(f"background:{C['surface']}; color:{C['text']};")
        v = QVBoxLayout(dlg)
        txt = QTextEdit(); txt.setReadOnly(True)
        txt.setStyleSheet(f"background:{C['card']}; color:{C['text']}; font-family:Consolas; border:none;")
        for r in self._results:
            sel = [self._problem.items[i].name for i,s in enumerate(r.solution) if s==1]
            txt.append(f"[{r.algorithm}]")
            txt.append(f"  Value  = {r.value:.4f}")
            txt.append(f"  Weight = {r.weight:.4f}")
            txt.append(f"  Time   = {r.time*1000:.4f} ms")
            txt.append(f"  Items  = {', '.join(sel)}")
            if hasattr(r,"extra") and r.extra:
                for k,v2 in r.extra.items():
                    if k != "convergence":
                        txt.append(f"  {k} = {v2}")
            txt.append("")
        v.addWidget(txt)
        btn = SolidButton("Đóng", C["teal"]); btn.clicked.connect(dlg.accept)
        v.addWidget(btn)
        dlg.exec()


# ══════════════════════════════════════════════════════════════════
#  STEP 3: ANALYSIS PAGE
# ══════════════════════════════════════════════════════════════════
class AnalysisPage(QWidget):
    go_back   = pyqtSignal()
    go_reset  = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._results = []
        self._problem = None
        self._run_start: Optional[datetime] = None
        self._run_end:   Optional[datetime] = None
        self._build()

    def _build(self):
        root = QVBoxLayout(self); root.setContentsMargins(24,16,24,0); root.setSpacing(12)

        # ── Header ───────────────────────────────────────────
        hdr = QHBoxLayout()
        hdr.addWidget(_lbl("📊  SO SÁNH TỔNG QUAN", 14, C["text"], bold=True))
        hdr.addStretch()
        btn_export = GlowButton("↓  Xuất kết quả", C["teal"]); btn_export.setFixedHeight(32)
        btn_log    = GlowButton("📋  Log chi tiết",  C["text2"]); btn_log.setFixedHeight(32)
        btn_export.clicked.connect(self._export)
        btn_log.clicked.connect(self._show_log)
        hdr.addWidget(btn_export); hdr.addWidget(btn_log)
        root.addLayout(hdr)

        # ── Best algorithm cards ──────────────────────────────
        self.top_cards_row = QHBoxLayout(); self.top_cards_row.setSpacing(12)
        root.addLayout(self.top_cards_row)

        # ── 3 charts + Commentary ─────────────────────────────
        charts_row = QHBoxLayout(); charts_row.setSpacing(12)

        # Bar value chart
        val_card = CardFrame(); val_lay = QVBoxLayout(val_card); val_lay.setContentsMargins(12,10,12,10)
        self.chart_value = BarChart("SO SÁNH GIÁ TRỊ", ylabel="Giá trị", figsize=(3.6,2.8))
        val_lay.addWidget(_lbl("📈  SO SÁNH GIÁ TRỊ", 11, C["text2"], bold=True))
        val_lay.addWidget(self.chart_value)
        val_lay.addWidget(_lbl("Đơn vị: Tổng giá trị", 9, C["text3"]))
        charts_row.addWidget(val_card)

        # HBar time chart
        time_card = CardFrame(); time_lay = QVBoxLayout(time_card); time_lay.setContentsMargins(12,10,12,10)
        self.chart_time = HBarChart(figsize=(3.6,2.8))
        time_lay.addWidget(_lbl("⏱  SO SÁNH THỜI GIAN THỰC THI (ms)", 11, C["text2"], bold=True))
        time_lay.addWidget(self.chart_time)
        charts_row.addWidget(time_card)

        # Convergence chart
        conv_card = CardFrame(); conv_lay = QVBoxLayout(conv_card); conv_lay.setContentsMargins(12,10,12,10)
        self.chart_conv = LineChart(figsize=(3.6,2.8))
        conv_lay.addWidget(_lbl("📉  BIỂU ĐỒ HỘI TỤ – BAT ALGORITHM", 11, C["text2"], bold=True))
        conv_lay.addWidget(self.chart_conv)
        charts_row.addWidget(conv_card)

        root.addLayout(charts_row)

        # ── Bottom: Info + Comment ────────────────────────────
        bottom_row = QHBoxLayout(); bottom_row.setSpacing(12)

        # Problem info card
        info_card = CardFrame(); info_lay = QVBoxLayout(info_card)
        info_lay.setContentsMargins(16,12,16,12); info_lay.setSpacing(6)
        info_lay.addWidget(_lbl("ℹ  THÔNG TIN BÀI TOÁN", 11, C["teal"], bold=True))
        self.info_form = QFormLayout(); self.info_form.setSpacing(6)
        self._info_fields = {}
        for k in ["Sức chứa (capacity)","Số vật phẩm","Thuật toán đã chạy",
                  "Thời gian bắt đầu","Thời gian kết thúc","Tổng thời gian"]:
            lv = _lbl("—", 11, C["text"])
            self._info_fields[k] = lv
            self.info_form.addRow(_lbl(k+":", 11, C["text2"]), lv)
        info_lay.addLayout(self.info_form)
        bottom_row.addWidget(info_card)

        # Comment card
        comment_card = CardFrame(); comment_lay = QVBoxLayout(comment_card)
        comment_lay.setContentsMargins(16,12,16,12); comment_lay.setSpacing(8)
        comment_lay.addWidget(_lbl("ℹ  NHẬN XÉT", 11, C["teal"], bold=True))
        self.lbl_comment = _lbl("—", 12, C["text"], wrap=True)
        comment_lay.addWidget(self.lbl_comment)
        comment_lay.addStretch()
        bottom_row.addWidget(comment_card)

        root.addLayout(bottom_row)

    def fill(self, results: List, problem, run_start=None, run_end=None):
        self._results   = results
        self._problem   = problem
        self._run_start = run_start
        self._run_end   = run_end

        if not results: return
        best = max(results, key=lambda r: r.value)

        # ── Top cards ─────────────────────────────────────────
        while self.top_cards_row.count():
            item = self.top_cards_row.takeAt(0)
            if item.widget(): item.widget().deleteLater()

        for r in results:
            card = AlgoResultCard(r.algorithm, is_best=(r==best))
            card.fill(r, problem)
            card.btn_detail.setVisible(False)
            self.top_cards_row.addWidget(card)

        # ── Charts ────────────────────────────────────────────
        names  = [r.algorithm for r in results]
        values = [r.value for r in results]
        times  = [r.time*1000 for r in results]
        colors = [ALGO_COLORS.get(r.algorithm, C["teal"]) for r in results]

        if MPL:
            self.chart_value.update(names, values, colors)
            self.chart_time.update(names, times, colors)
            self.chart_conv.update(results)

        # ── Info fields ───────────────────────────────────────
        self._info_fields["Sức chứa (capacity)"].setText(f"{problem.capacity:.4f}")
        self._info_fields["Số vật phẩm"].setText(str(problem.n))
        self._info_fields["Thuật toán đã chạy"].setText(
            f"Tất cả ({len(results)} thuật toán)" if len(results)>1
            else results[0].algorithm)
        if run_start:
            self._info_fields["Thời gian bắt đầu"].setText(run_start.strftime("%d/%m/%Y %I:%M %p"))
        if run_end:
            self._info_fields["Thời gian kết thúc"].setText(run_end.strftime("%d/%m/%Y %I:%M %p"))
        total_ms = sum(r.time for r in results) * 1000
        self._info_fields["Tổng thời gian"].setText(f"{total_ms:.3f} ms")

        # ── Comment ───────────────────────────────────────────
        comment = self._generate_comment(results, best, problem)
        self.lbl_comment.setText(comment)

    def _generate_comment(self, results, best, problem) -> str:
        lines = []
        lines.append(f"<b>{best.algorithm}</b> là thuật toán tốt nhất cho bộ dữ liệu hiện tại "
                     f"với tổng giá trị đạt được là <b>{best.value:.4f}</b>.")
        for r in results:
            if r == best: continue
            if abs(r.value - best.value) < 1e-9:
                lines.append(f"<b>{r.algorithm}</b> cho kết quả tương đương về giá trị "
                              f"nhưng thời gian thực thi lâu hơn đáng kể.")
            else:
                diff_pct = (best.value - r.value) / best.value * 100 if best.value else 0
                lines.append(f"<b>{r.algorithm}</b> đạt <b>{r.value:.4f}</b>, "
                              f"thấp hơn {diff_pct:.1f}% so với kết quả tốt nhất.")
        fastest = min(results, key=lambda r: r.time)
        if fastest != best:
            lines.append(f"<b>{fastest.algorithm}</b> có thời gian nhanh nhất "
                         f"({fastest.time*1000:.3f} ms) nhưng cho tổng giá trị thấp hơn.")
        return "<br>".join(lines)

    def _export(self):
        if not self._results:
            QMessageBox.information(self,"","Chưa có kết quả."); return
        path, _ = QFileDialog.getSaveFileName(self,"Lưu","analysis.csv","CSV (*.csv)")
        if not path: return
        try:
            with open(path,"w",newline="",encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["Algorithm","Value","Weight","Time_ms","Items_selected"])
                for r in self._results:
                    sel = sum(r.solution)
                    w.writerow([r.algorithm, r.value, r.weight, r.time*1000, sel])
            QMessageBox.information(self,"Đã lưu",f"Đã lưu:\n{path}")
        except Exception as e:
            QMessageBox.critical(self,"Lỗi",str(e))

    def _show_log(self):
        if not self._results:
            QMessageBox.information(self,"","Chưa có kết quả."); return
        dlg = QDialog(self); dlg.setWindowTitle("Log chi tiết"); dlg.resize(500,400)
        dlg.setStyleSheet(f"background:{C['surface']}; color:{C['text']};")
        v = QVBoxLayout(dlg)
        txt = QTextEdit(); txt.setReadOnly(True)
        txt.setStyleSheet(f"background:{C['card']}; color:{C['text']}; font-family:Consolas; border:none;")
        for r in self._results:
            sel = [self._problem.items[i].name for i,s in enumerate(r.solution) if s==1]
            txt.append(f"=== {r.algorithm} ===")
            txt.append(f"Value  = {r.value:.6f}")
            txt.append(f"Weight = {r.weight:.6f}  (cap={self._problem.capacity:.4f})")
            txt.append(f"Time   = {r.time*1000:.4f} ms")
            txt.append(f"Items  = {', '.join(sel)}")
            txt.append("")
        v.addWidget(txt)
        btn = SolidButton("Đóng", C["teal"]); btn.clicked.connect(dlg.accept)
        v.addWidget(btn)
        dlg.exec()


# ══════════════════════════════════════════════════════════════════
#  MAIN WINDOW
# ══════════════════════════════════════════════════════════════════
class KnapsackApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("KNAPSACK SOLVER  —  Greedy & Bat Algorithm")
        self.setMinimumSize(1100, 720)
        self.resize(1360, 820)

        self._results     = []
        self._problem     = None
        self._worker: Optional[SolverWorker] = None
        self._run_start   = None
        self._run_end     = None

        self._build_ui()

        if not BACKEND_OK:
            QMessageBox.critical(self, "Lỗi import",
                f"Không thể import backend:\n{BACKEND_ERR}\n\n"
                "Đặt core.py, greedy.py, bat.py cùng thư mục với gui.py.")

    # ── BUILD ─────────────────────────────────────────────────
    def _build_ui(self):
        # Root container with solid background
        root_widget = QWidget()
        root_widget.setStyleSheet(f"background:{C['bg']};")
        self.setCentralWidget(root_widget)

        root_v = QVBoxLayout(root_widget)
        root_v.setContentsMargins(0,0,0,0)
        root_v.setSpacing(0)

        # ── Top bar ───────────────────────────────────────────
        topbar = self._build_topbar()
        topbar.setStyleSheet(f"background:{C['surface']}; border-bottom:1px solid {C['border']};")
        root_v.addWidget(topbar)

        # ── Step indicator ────────────────────────────────────
        step_container = QWidget()
        step_container.setStyleSheet(f"background:{C['surface2']}; border-bottom:1px solid {C['border']};")
        step_lay = QHBoxLayout(step_container)
        step_lay.setContentsMargins(60,0,60,0)
        self.step_indicator = StepIndicator()
        step_lay.addWidget(self.step_indicator)
        root_v.addWidget(step_container)

        # ── Pages ─────────────────────────────────────────────
        self.pages = QStackedWidget()
        self.pages.setStyleSheet(f"background:{C['bg']};")

        self.page_import   = ImportPage()
        self.page_result   = ResultPage()
        self.page_analysis = AnalysisPage()

        self.pages.addWidget(self.page_import)
        self.pages.addWidget(self.page_result)
        self.pages.addWidget(self.page_analysis)
        root_v.addWidget(self.pages, 1)

        # ── Bottom navigation bar ─────────────────────────────
        self.bottom_bar = self._build_bottom_bar()
        self.bottom_bar.setStyleSheet(
            f"background:{C['surface']}; border-top:1px solid {C['border']};")
        root_v.addWidget(self.bottom_bar)

        # Progress bar (sits above bottom bar, hidden by default)
        self.progress = QProgressBar(); self.progress.setRange(0,0)
        self.progress.setFixedHeight(4); self.progress.setVisible(False)
        self.progress.setStyleSheet(
            f"QProgressBar{{background:{C['border']};border:none;}}"
            f"QProgressBar::chunk{{background:{C['teal']};border:none;}}")
        root_v.insertWidget(root_v.count()-1, self.progress)

        # Signals
        self.page_import.run_requested.connect(self._run_solver)
        self.page_result.go_back.connect(lambda: self._go_to(0))
        self.page_result.go_analysis.connect(lambda: self._go_to(2))
        self.page_analysis.go_back.connect(lambda: self._go_to(1))
        self.page_analysis.go_reset.connect(lambda: self._go_to(0))

        # Default data
        self.page_import.populate_default()
        self._go_to(0)

    def _build_topbar(self) -> QWidget:
        w = QWidget(); h = QHBoxLayout(w)
        h.setContentsMargins(20,12,20,12); h.setSpacing(12)

        # Logo area
        logo_icon = _lbl("🎒", 22)
        logo_icon.setFixedWidth(32)
        logo_col = QVBoxLayout(); logo_col.setSpacing(0)
        logo_col.addWidget(_lbl("KNAPSACK SOLVER", 15, C["teal"], bold=True))
        logo_col.addWidget(_lbl("Greedy & Bat Algorithm", 10, C["text2"]))
        h.addWidget(logo_icon)
        h.addLayout(logo_col)
        h.addStretch()

        btn_help = GlowButton("?  Hướng dẫn", C["text2"]); btn_help.setFixedHeight(32)
        btn_help.clicked.connect(self._show_help)
        h.addWidget(btn_help)

        # Dark mode toggle (decorative)
        btn_dark = QPushButton("🌙"); btn_dark.setFixedSize(36,36)
        btn_dark.setStyleSheet(f"""QPushButton{{background:{C['surface2']};color:{C['text2']};
            border:1px solid {C['border']};border-radius:18px;font-size:16px;}}
            QPushButton:hover{{border-color:{C['teal']};}}""")
        h.addWidget(btn_dark)
        return w

    def _build_bottom_bar(self) -> QWidget:
        w = QWidget(); h = QHBoxLayout(w)
        h.setContentsMargins(24,10,24,10); h.setSpacing(12)

        self.btn_back = GlowButton("←  Quay lại", C["text2"])
        self.btn_back.setFixedHeight(44)
        self.lbl_back_sub = _lbl("", 10, C["text3"])
        back_col = QVBoxLayout(); back_col.setSpacing(2)
        back_col.addWidget(self.btn_back)
        back_col.addWidget(self.lbl_back_sub)
        self.btn_back.clicked.connect(self._nav_back)

        # Dots
        self.dots = []
        dots_row = QHBoxLayout(); dots_row.setSpacing(6)
        for _ in range(3):
            d = QLabel("●"); d.setFixedWidth(16)
            d.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.dots.append(d); dots_row.addWidget(d)

        self.btn_next = SolidButton("Tiếp tục  →", C["teal"])
        self.btn_next.setFixedHeight(44)
        self.lbl_next_sub = _lbl("", 10,"#0d1117")
        next_col = QVBoxLayout(); next_col.setSpacing(2)
        next_col.addWidget(self.btn_next)
        next_col.addWidget(self.lbl_next_sub)
        self.btn_next.clicked.connect(self._nav_next)

        h.addLayout(back_col)
        h.addStretch()
        h.addLayout(dots_row)
        h.addStretch()
        h.addLayout(next_col)
        return w

    # ── NAVIGATION ────────────────────────────────────────────
    def _go_to(self, page: int):
        self.pages.setCurrentIndex(page)
        self.step_indicator.set_step(page)
        self._update_nav_bar(page)

    def _update_nav_bar(self, page: int):
        # Dots
        for i, d in enumerate(self.dots):
            d.setStyleSheet(
                f"color:{C['teal']}; font-size:12px;" if i==page
                else f"color:{C['border2']}; font-size:10px;")

        # Back button
        if page == 0:
            self.btn_back.setVisible(False)
            self.lbl_back_sub.setText("")
        else:
            self.btn_back.setVisible(True)
            sub = ["","Chỉnh sửa dữ liệu","Xem kết quả"][page]
            self.lbl_back_sub.setText(sub)

        # Next button
        if page == 0:
            self.btn_next.setText("▶   CHẠY THUẬT TOÁN")
            self.btn_next.setStyleSheet(self.btn_next.styleSheet())
            self.lbl_next_sub.setText("Bắt đầu tìm kiếm giải pháp tối ưu")
        elif page == 1:
            self.btn_next.setText("Tiếp tục  →")
            self.lbl_next_sub.setText("Xem phân tích và biểu đồ")
        else:
            self.btn_next.setText("🏁  Kết thúc & Tạo bài toán mới")
            self.lbl_next_sub.setText("Quay về trang nhập dữ liệu")

    def _nav_back(self):
        cur = self.pages.currentIndex()
        if cur > 0: self._go_to(cur - 1)

    def _nav_next(self):
        cur = self.pages.currentIndex()
        if cur == 0:
            self._run_solver()
        elif cur == 1:
            if not self._results:
                QMessageBox.information(self,"","Hãy chạy thuật toán trước."); return
            self._go_to(2)
        else:
            self._go_to(0)

    # ── SOLVER ────────────────────────────────────────────────
    def _run_solver(self):
        if not BACKEND_OK:
            QMessageBox.critical(self,"Lỗi","Backend chưa sẵn sàng."); return
        try:
            cap, weights, values, names = self.page_import.parse_input()
        except ValueError as e:
            QMessageBox.warning(self,"Dữ liệu không hợp lệ",str(e)); return

        self._problem   = create_problem(cap, weights, values, names)
        algorithms      = self.page_import.get_algo_list()
        bat_params      = self.page_import.get_bat_params()
        self._run_start = datetime.now()

        self.progress.setVisible(True)
        self.btn_next.setEnabled(False)

        self._worker = SolverWorker(self._problem, algorithms, bat_params)
        self._worker.finished.connect(self._on_solver_done)
        self._worker.error.connect(self._on_solver_error)
        self._worker.start()

    def _on_solver_done(self, results: List):
        self._run_end = datetime.now()
        self.progress.setVisible(False)
        self.btn_next.setEnabled(True)
        self._results = results

        self.page_result.fill(results, self._problem)
        self.page_analysis.fill(results, self._problem, self._run_start, self._run_end)

        self._go_to(1)

    def _on_solver_error(self, msg: str):
        self.progress.setVisible(False)
        self.btn_next.setEnabled(True)
        QMessageBox.critical(self, "Lỗi khi chạy thuật toán", msg)

    # ── HELP ──────────────────────────────────────────────────
    def _show_help(self):
        dlg = QDialog(self); dlg.setWindowTitle("Hướng dẫn sử dụng")
        dlg.resize(540, 480)
        dlg.setStyleSheet(f"background:{C['surface']}; color:{C['text']};")
        v = QVBoxLayout(dlg); v.setContentsMargins(20,20,20,20); v.setSpacing(12)
        v.addWidget(_lbl("📖  Hướng dẫn sử dụng Knapsack Solver", 14, C["teal"], bold=True))

        txt = QTextEdit(); txt.setReadOnly(True)
        txt.setStyleSheet(f"background:{C['card']}; color:{C['text']}; font-size:12px; border:none; border-radius:8px;")
        txt.setHtml(f"""
        <style>
          body{{color:{C['text']};font-family:'Segoe UI';font-size:13px;line-height:1.6;}}
          h3{{color:{C['teal']};margin:12px 0 4px;}}
          code{{background:{C['surface']};color:{C['teal']};padding:1px 5px;border-radius:3px;}}
          li{{margin:3px 0;}}
        </style>
        <h3>🎯 Tổng quan</h3>
        <p>App giải bài toán 0/1 Knapsack bằng 3 thuật toán: Greedy, GBFS và Bat Algorithm.</p>

        <h3>📋 Bước 1 — Nhập dữ liệu</h3>
        <ul>
          <li>Nhập <b>sức chứa</b> vào ô Capacity.</li>
          <li>Nhập danh sách vật phẩm (Tên, Trọng lượng, Giá trị) vào bảng.</li>
          <li>Hoặc nhấn <b>🎲 Random Data</b> để tạo dữ liệu ngẫu nhiên.</li>
          <li>Hoặc nhấn <b>📂 Import CSV</b> để tải file CSV.</li>
        </ul>

        <h3>📂 Định dạng CSV</h3>
        <pre style="background:{C['surface']};padding:8px;border-radius:6px;font-size:11px;">
capacity,200
name,weight,value
Book,5,10
Laptop,20,100</pre>

        <h3>⚙ Bước 1 — Cài đặt thuật toán</h3>
        <ul>
          <li>Chọn thuật toán từ dropdown (có thể chọn "Tất cả" để so sánh).</li>
          <li>Điều chỉnh tham số Bat Algorithm nếu cần.</li>
        </ul>

        <h3>📊 Bước 2 — Kết quả</h3>
        <p>Hiển thị kết quả từng thuật toán, card thuật toán tốt nhất được highlight xanh.</p>

        <h3>📈 Bước 3 — Phân tích</h3>
        <p>So sánh bằng biểu đồ cột giá trị, thời gian thực thi và đường hội tụ Bat Algorithm.</p>

        <h3>💡 Gợi ý dataset</h3>
        <ul>
          <li><code>complex_adversarial.csv</code> — dataset hay nhất để thấy Bat thắng Greedy.</li>
          <li><code>large_200.csv</code> — test hiệu năng thời gian chạy.</li>
        </ul>
        """)
        v.addWidget(txt)
        btn = SolidButton("Đóng", C["teal"]); btn.clicked.connect(dlg.accept)
        v.addWidget(btn)
        dlg.exec()


# ══════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════════
def main():
    app = QApplication(sys.argv)
    app.setStyleSheet(STYLE)
    app.setApplicationName("Knapsack Solver")

    # High-DPI
    try:
        app.setAttribute(Qt.ApplicationAttribute.AA_UseHighDpiPixmaps)
    except Exception:
        pass

    win = KnapsackApp()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()



