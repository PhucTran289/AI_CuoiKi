"""
gui.py - Giao diện PyQt6 cho bài toán Knapsack
===============================================
Kết nối với core.py, greedy.py, bat.py để giải và so sánh kết quả.

Yêu cầu:
    pip install PyQt6 matplotlib
"""

import sys
import time
from typing import List, Optional

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGridLayout, QLabel, QPushButton, QTableWidget, QTableWidgetItem,
    QComboBox, QSpinBox, QDoubleSpinBox, QGroupBox, QSplitter,
    QHeaderView, QMessageBox, QTabWidget, QTextEdit, QScrollArea,
    QFrame, QSizePolicy, QProgressBar,
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt6.QtGui import QFont, QColor, QPalette, QIcon

# ── Matplotlib cho biểu đồ ────────────────────────────────────
try:
    import matplotlib
    matplotlib.use("QtAgg")
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.figure import Figure
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

# ── Import backend ────────────────────────────────────────────
try:
    from core import (
        create_problem, generate_random_problem,
        KnapsackProblem, KnapsackResult,
        evaluate, is_feasible,
    )
    from greedy import greedy_ratio, greedy_best_first_search
    from bat import bat_algorithm
    BACKEND_OK = True
except ImportError as e:
    BACKEND_OK = False
    BACKEND_ERROR = str(e)


# ════════════════════════════════════════════════════════════════
# STYLE SHEET — giao diện tối hiện đại
# ════════════════════════════════════════════════════════════════

STYLE = """
QMainWindow, QWidget {
    background-color: #1e1e2e;
    color: #cdd6f4;
    font-family: 'Segoe UI', Arial, sans-serif;
    font-size: 13px;
}

QGroupBox {
    border: 1px solid #45475a;
    border-radius: 8px;
    margin-top: 12px;
    padding: 10px;
    font-weight: bold;
    font-size: 13px;
    color: #89b4fa;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 6px;
}

QPushButton {
    background-color: #313244;
    color: #cdd6f4;
    border: 1px solid #45475a;
    border-radius: 6px;
    padding: 7px 18px;
    font-size: 13px;
}
QPushButton:hover   { background-color: #45475a; border-color: #89b4fa; }
QPushButton:pressed { background-color: #181825; }

QPushButton#btn_run {
    background-color: #89b4fa;
    color: #1e1e2e;
    font-weight: bold;
    font-size: 14px;
    border: none;
    padding: 9px 30px;
    border-radius: 8px;
}
QPushButton#btn_run:hover   { background-color: #b4befe; }
QPushButton#btn_run:pressed { background-color: #74c7ec; }

QPushButton#btn_random {
    background-color: #a6e3a1;
    color: #1e1e2e;
    font-weight: bold;
    border: none;
    border-radius: 6px;
}
QPushButton#btn_random:hover { background-color: #94e2d5; }

QPushButton#btn_clear {
    background-color: #f38ba8;
    color: #1e1e2e;
    border: none;
    border-radius: 6px;
}
QPushButton#btn_clear:hover { background-color: #eba0ac; }

QPushButton#btn_add_row {
    background-color: #fab387;
    color: #1e1e2e;
    border: none;
    border-radius: 6px;
}

QTableWidget {
    background-color: #181825;
    alternate-background-color: #1e1e2e;
    gridline-color: #313244;
    border: 1px solid #45475a;
    border-radius: 6px;
    selection-background-color: #45475a;
}
QTableWidget::item { padding: 4px 8px; }
QHeaderView::section {
    background-color: #313244;
    color: #89b4fa;
    border: none;
    border-bottom: 1px solid #45475a;
    padding: 6px 8px;
    font-weight: bold;
}

QComboBox {
    background-color: #313244;
    border: 1px solid #45475a;
    border-radius: 5px;
    padding: 5px 10px;
    color: #cdd6f4;
}
QComboBox:hover { border-color: #89b4fa; }
QComboBox QAbstractItemView {
    background-color: #313244;
    selection-background-color: #45475a;
}
QComboBox::drop-down { border: none; }

QSpinBox, QDoubleSpinBox {
    background-color: #313244;
    border: 1px solid #45475a;
    border-radius: 5px;
    padding: 5px 8px;
    color: #cdd6f4;
}
QSpinBox:hover, QDoubleSpinBox:hover { border-color: #89b4fa; }

QTabWidget::pane {
    border: 1px solid #45475a;
    border-radius: 6px;
    background: #181825;
}
QTabBar::tab {
    background: #313244;
    color: #a6adc8;
    padding: 8px 20px;
    border-top-left-radius: 6px;
    border-top-right-radius: 6px;
    margin-right: 2px;
}
QTabBar::tab:selected { background: #89b4fa; color: #1e1e2e; font-weight: bold; }
QTabBar::tab:hover    { background: #45475a; color: #cdd6f4; }

QTextEdit {
    background-color: #181825;
    color: #cdd6f4;
    border: 1px solid #45475a;
    border-radius: 6px;
    padding: 8px;
    font-family: 'Consolas', monospace;
    font-size: 12px;
}

QScrollBar:vertical {
    background: #1e1e2e;
    width: 10px;
    border-radius: 5px;
}
QScrollBar::handle:vertical {
    background: #45475a;
    border-radius: 5px;
    min-height: 20px;
}
QScrollBar::handle:vertical:hover { background: #89b4fa; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }

QLabel#label_stat {
    background-color: #313244;
    border-radius: 6px;
    padding: 8px 14px;
    color: #cdd6f4;
}
QLabel#label_best {
    background-color: #1e4620;
    border: 1px solid #a6e3a1;
    border-radius: 6px;
    padding: 8px 14px;
    color: #a6e3a1;
    font-weight: bold;
}

QProgressBar {
    border: 1px solid #45475a;
    border-radius: 5px;
    background: #181825;
    height: 8px;
    text-align: center;
}
QProgressBar::chunk { background: #89b4fa; border-radius: 5px; }

QFrame#separator {
    background-color: #45475a;
    max-height: 1px;
}
"""


# ════════════════════════════════════════════════════════════════
# WORKER THREAD — chạy thuật toán trong background
# ════════════════════════════════════════════════════════════════

class SolverWorker(QThread):
    """Chạy thuật toán knapsack trong thread riêng để không block UI."""

    finished = pyqtSignal(list)   # Phát ra list[KnapsackResult]
    error    = pyqtSignal(str)    # Phát ra thông báo lỗi

    def __init__(self, problem: "KnapsackProblem", algorithms: List[str],
                 bat_params: dict):
        super().__init__()
        self.problem    = problem
        self.algorithms = algorithms   # ["greedy_ratio", "gbfs", "bat"]
        self.bat_params = bat_params

    def run(self):
        try:
            results = []
            for algo in self.algorithms:
                if algo == "greedy_ratio":
                    r = greedy_ratio(self.problem)
                elif algo == "gbfs":
                    r = greedy_best_first_search(self.problem)
                elif algo == "bat":
                    r = bat_algorithm(self.problem, **self.bat_params)
                else:
                    continue
                results.append(r)
            self.finished.emit(results)
        except Exception as e:
            self.error.emit(str(e))


# ════════════════════════════════════════════════════════════════
# CHART WIDGET — biểu đồ matplotlib nhúng trong Qt
# ════════════════════════════════════════════════════════════════

class ChartWidget(QWidget):
    """Widget hiển thị 2 biểu đồ: so sánh giá trị và thời gian."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        if not MATPLOTLIB_AVAILABLE:
            lbl = QLabel("⚠ matplotlib chưa được cài đặt.\nChạy: pip install matplotlib")
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            lbl.setStyleSheet("color: #f38ba8; font-size: 14px;")
            layout.addWidget(lbl)
            self.canvas = None
            return

        self.fig = Figure(facecolor="#181825", tight_layout=True)
        self.canvas = FigureCanvas(self.fig)
        layout.addWidget(self.canvas)

        self.ax_value = self.fig.add_subplot(1, 2, 1)
        self.ax_time  = self.fig.add_subplot(1, 2, 2)
        self._style_axes()

    def _style_axes(self):
        for ax, title in [(self.ax_value, "Giá trị đạt được"),
                          (self.ax_time,  "Thời gian chạy (ms)")]:
            ax.set_facecolor("#1e1e2e")
            ax.set_title(title, color="#89b4fa", fontsize=12, pad=10)
            ax.tick_params(colors="#a6adc8")
            for spine in ax.spines.values():
                spine.set_edgecolor("#45475a")

    def update_charts(self, results: List["KnapsackResult"]):
        if not MATPLOTLIB_AVAILABLE or self.canvas is None:
            return

        self.ax_value.clear()
        self.ax_time.clear()
        self._style_axes()

        names  = [r.algorithm for r in results]
        values = [r.value for r in results]
        times  = [r.time * 1000 for r in results]   # ms

        colors = ["#89b4fa", "#a6e3a1", "#fab387", "#f38ba8", "#cba6f7"][:len(results)]

        # Cột giá trị
        bars1 = self.ax_value.bar(names, values, color=colors, width=0.5)
        for bar, v in zip(bars1, values):
            self.ax_value.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height() + max(values) * 0.01,
                f"{v:.2f}", ha="center", va="bottom", color="#cdd6f4", fontsize=10
            )
        self.ax_value.set_ylabel("Tổng giá trị", color="#a6adc8")
        self.ax_value.tick_params(axis="x", rotation=15)

        # Cột thời gian
        bars2 = self.ax_time.bar(names, times, color=colors, width=0.5)
        for bar, t in zip(bars2, times):
            self.ax_time.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height() + max(times) * 0.01,
                f"{t:.3f}", ha="center", va="bottom", color="#cdd6f4", fontsize=10
            )
        self.ax_time.set_ylabel("Thời gian (ms)", color="#a6adc8")
        self.ax_time.tick_params(axis="x", rotation=15)

        self.canvas.draw()

    def draw_convergence(self, results: List["KnapsackResult"]):
        """Vẽ đường hội tụ của Bat Algorithm (nếu có)."""
        if not MATPLOTLIB_AVAILABLE or self.canvas is None:
            return

        bat_results = [r for r in results if hasattr(r, "extra")
                       and "convergence" in (r.extra or {})]
        if not bat_results:
            return

        self.ax_value.clear()
        self.ax_value.set_facecolor("#1e1e2e")
        self.ax_value.set_title("Hội tụ Bat Algorithm", color="#89b4fa", fontsize=12)
        self.ax_value.tick_params(colors="#a6adc8")
        for spine in self.ax_value.spines.values():
            spine.set_edgecolor("#45475a")

        for r in bat_results:
            conv = r.extra["convergence"]
            self.ax_value.plot(conv, color="#89b4fa", linewidth=1.5, label=r.algorithm)
        self.ax_value.set_xlabel("Vòng lặp", color="#a6adc8")
        self.ax_value.set_ylabel("Giá trị tốt nhất", color="#a6adc8")
        self.ax_value.legend(facecolor="#313244", labelcolor="#cdd6f4")

        self.canvas.draw()


# ════════════════════════════════════════════════════════════════
# RESULT PANEL — hiển thị kết quả 1 thuật toán
# ════════════════════════════════════════════════════════════════

class ResultPanel(QGroupBox):
    """Panel nhỏ hiển thị kết quả của 1 thuật toán."""

    def __init__(self, title: str, color: str = "#89b4fa", parent=None):
        super().__init__(title, parent)
        self.accent = color
        self._setup_ui()
        self.setStyleSheet(
            f"QGroupBox {{ color: {color}; border-color: {color}; }}"
        )

    def _setup_ui(self):
        layout = QVBoxLayout(self)

        # Thống kê nhanh
        stats_layout = QGridLayout()
        self.lbl_value  = self._make_stat("Giá trị", "—")
        self.lbl_weight = self._make_stat("Trọng lượng", "—")
        self.lbl_time   = self._make_stat("Thời gian", "—")
        stats_layout.addWidget(self.lbl_value,  0, 0)
        stats_layout.addWidget(self.lbl_weight, 0, 1)
        stats_layout.addWidget(self.lbl_time,   0, 2)
        layout.addLayout(stats_layout)

        # Vật phẩm được chọn
        self.lbl_items = QLabel("Vật phẩm được chọn: —")
        self.lbl_items.setWordWrap(True)
        self.lbl_items.setStyleSheet("color: #a6adc8; padding: 4px;")
        layout.addWidget(self.lbl_items)

    def _make_stat(self, label: str, value: str) -> QLabel:
        lbl = QLabel(f"<b style='color:#a6adc8'>{label}</b><br>{value}")
        lbl.setObjectName("label_stat")
        lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        return lbl

    def update_result(self, result: "KnapsackResult", problem: "KnapsackProblem"):
        selected = [i for i, s in enumerate(result.solution) if s == 1]
        names = [problem.items[i].name for i in selected]

        self.lbl_value.setText(
            f"<b style='color:#a6adc8'>Giá trị</b><br>"
            f"<span style='color:{self.accent};font-size:16px;font-weight:bold'>"
            f"{result.value:.4f}</span>"
        )
        self.lbl_weight.setText(
            f"<b style='color:#a6adc8'>Trọng lượng</b><br>"
            f"<span style='font-size:14px'>{result.weight:.4f}</span>"
        )
        self.lbl_time.setText(
            f"<b style='color:#a6adc8'>Thời gian</b><br>"
            f"<span style='font-size:14px'>{result.time*1000:.3f} ms</span>"
        )
        item_str = ", ".join(names) if names else "(không chọn vật nào)"
        self.lbl_items.setText(f"Vật phẩm được chọn: <b>{item_str}</b>")

    def reset(self):
        for lbl, label in [(self.lbl_value, "Giá trị"),
                           (self.lbl_weight, "Trọng lượng"),
                           (self.lbl_time, "Thời gian")]:
            lbl.setText(f"<b style='color:#a6adc8'>{label}</b><br>—")
        self.lbl_items.setText("Vật phẩm được chọn: —")


# ════════════════════════════════════════════════════════════════
# CỬA SỔ CHÍNH
# ════════════════════════════════════════════════════════════════

class KnapsackApp(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("🎒 Knapsack Solver — Greedy & Bat Algorithm")
        self.setMinimumSize(1200, 750)
        self.resize(1400, 820)

        self._results: List = []
        self._problem: Optional[object] = None
        self._worker: Optional[SolverWorker] = None

        self._build_ui()
        self._populate_default()

        if not BACKEND_OK:
            QMessageBox.critical(self, "Lỗi import",
                f"Không thể import backend:\n{BACKEND_ERROR}\n\n"
                "Đảm bảo core.py, greedy.py, bat.py cùng thư mục với gui.py.")

    # ── Xây dựng UI ──────────────────────────────────────────

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(12, 12, 12, 12)
        main_layout.setSpacing(12)

        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Cột trái: Input + Control
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setSpacing(10)
        left_layout.addWidget(self._build_input_panel())
        left_layout.addWidget(self._build_control_panel())
        left.setMaximumWidth(440)
        left.setMinimumWidth(340)
        splitter.addWidget(left)

        # Cột phải: Tabs (Kết quả / Biểu đồ / Log)
        splitter.addWidget(self._build_right_panel())
        splitter.setStretchFactor(1, 3)

        main_layout.addWidget(splitter)

    # ── Panel nhập dữ liệu ───────────────────────────────────

    def _build_input_panel(self) -> QGroupBox:
        grp = QGroupBox("📦 Dữ liệu đầu vào")
        layout = QVBoxLayout(grp)

        # Capacity
        cap_row = QHBoxLayout()
        cap_row.addWidget(QLabel("Sức chứa (capacity):"))
        self.spin_capacity = QDoubleSpinBox()
        self.spin_capacity.setRange(0.01, 1_000_000)
        self.spin_capacity.setValue(15.0)
        self.spin_capacity.setDecimals(2)
        self.spin_capacity.setSingleStep(1.0)
        cap_row.addWidget(self.spin_capacity)
        layout.addLayout(cap_row)

        # Bảng vật phẩm
        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["Tên", "Trọng lượng", "Giá trị"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.setAlternatingRowColors(True)
        self.table.setMinimumHeight(220)
        layout.addWidget(self.table)

        # Nút thao tác bảng
        btn_row = QHBoxLayout()
        btn_add = QPushButton("➕ Thêm hàng")
        btn_add.setObjectName("btn_add_row")
        btn_del = QPushButton("🗑 Xóa hàng")
        btn_random = QPushButton("🎲 Random Data")
        btn_random.setObjectName("btn_random")
        btn_clear = QPushButton("✖ Xóa tất cả")
        btn_clear.setObjectName("btn_clear")

        btn_add.clicked.connect(self._add_row)
        btn_del.clicked.connect(self._delete_row)
        btn_random.clicked.connect(self._generate_random)
        btn_clear.clicked.connect(self._clear_table)

        btn_row.addWidget(btn_add)
        btn_row.addWidget(btn_del)
        layout.addLayout(btn_row)

        btn_row2 = QHBoxLayout()
        btn_row2.addWidget(btn_random)
        btn_row2.addWidget(btn_clear)
        layout.addLayout(btn_row2)

        # Tùy chỉnh random
        rand_row = QHBoxLayout()
        rand_row.addWidget(QLabel("Số vật ngẫu nhiên:"))
        self.spin_rand_n = QSpinBox()
        self.spin_rand_n.setRange(2, 200)
        self.spin_rand_n.setValue(10)
        rand_row.addWidget(self.spin_rand_n)
        rand_row.addWidget(QLabel("Seed:"))
        self.spin_seed = QSpinBox()
        self.spin_seed.setRange(-1, 99999)
        self.spin_seed.setValue(42)
        self.spin_seed.setSpecialValueText("Random")
        rand_row.addWidget(self.spin_seed)
        layout.addLayout(rand_row)

        return grp

    # ── Panel điều khiển ─────────────────────────────────────

    def _build_control_panel(self) -> QGroupBox:
        grp = QGroupBox("⚙ Cấu hình & Chạy")
        layout = QVBoxLayout(grp)

        # Chọn thuật toán
        algo_row = QHBoxLayout()
        algo_row.addWidget(QLabel("Thuật toán:"))
        self.combo_algo = QComboBox()
        self.combo_algo.addItems([
            "Greedy (Ratio)",
            "Greedy Best-First Search (GBFS)",
            "Bat Algorithm",
            "Tất cả (so sánh)",
        ])
        self.combo_algo.currentIndexChanged.connect(self._on_algo_changed)
        algo_row.addWidget(self.combo_algo)
        layout.addLayout(algo_row)

        # Tham số Bat Algorithm (ẩn/hiện theo lựa chọn)
        self.grp_bat = QGroupBox("Tham số Bat Algorithm")
        self.grp_bat.setStyleSheet("QGroupBox { color: #fab387; border-color: #fab387; }")
        bat_layout = QGridLayout(self.grp_bat)

        def bat_spin(lo, hi, val, step=1, dec=0):
            s = QDoubleSpinBox() if dec > 0 else QSpinBox()
            s.setRange(lo, hi)
            s.setValue(val)
            if dec > 0:
                s.setDecimals(dec)
                s.setSingleStep(step)
            return s

        self.bat_n_bats   = bat_spin(5, 200, 30)
        self.bat_max_iter = bat_spin(50, 5000, 500)
        self.bat_alpha    = bat_spin(0.01, 1.0, 0.9, 0.05, 2)
        self.bat_gamma    = bat_spin(0.01, 5.0, 0.9, 0.1, 2)
        self.bat_mut_rate = bat_spin(0.0, 0.05, 0.02, 0.005, 3)
        self.bat_f_min    = bat_spin(0.0, 10.0, 0.0, 0.5, 1)
        self.bat_f_max    = bat_spin(0.1, 10.0, 2.0, 0.5, 1)

        params = [
            ("Số dơi (n_bats):", self.bat_n_bats),
            ("Vòng lặp (max_iter):", self.bat_max_iter),
            ("Alpha (loudness):", self.bat_alpha),
            ("Gamma (pulse):", self.bat_gamma),
            ("Đột biến (≤5%):", self.bat_mut_rate),
            ("f_min:", self.bat_f_min),
            ("f_max:", self.bat_f_max),
        ]
        for row, (lbl, spin) in enumerate(params):
            bat_layout.addWidget(QLabel(lbl), row, 0)
            bat_layout.addWidget(spin, row, 1)

        layout.addWidget(self.grp_bat)
        self.grp_bat.setVisible(False)

        # Nút chạy
        self.btn_run = QPushButton("▶  CHẠY THUẬT TOÁN")
        self.btn_run.setObjectName("btn_run")
        self.btn_run.clicked.connect(self._run_solver)
        layout.addWidget(self.btn_run)

        # Progress bar (ẩn lúc đầu)
        self.progress = QProgressBar()
        self.progress.setRange(0, 0)   # Indeterminate
        self.progress.setVisible(False)
        layout.addWidget(self.progress)

        return grp

    # ── Panel phải: tabs ─────────────────────────────────────

    def _build_right_panel(self) -> QTabWidget:
        self.tabs = QTabWidget()

        # Tab 1: Kết quả
        self.tabs.addTab(self._build_result_tab(), "📊 Kết quả")
        # Tab 2: Biểu đồ
        self.tabs.addTab(self._build_chart_tab(), "📈 Biểu đồ")
        # Tab 3: So sánh chi tiết
        self.tabs.addTab(self._build_compare_tab(), "⚖ So sánh")
        # Tab 4: Log
        self.tabs.addTab(self._build_log_tab(), "📋 Log")

        return self.tabs

    def _build_result_tab(self) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.setSpacing(10)

        # Banner best result
        self.lbl_banner = QLabel("Chưa có kết quả — hãy nhập dữ liệu và nhấn 'Chạy'.")
        self.lbl_banner.setObjectName("label_stat")
        self.lbl_banner.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_banner.setStyleSheet(
            "QLabel { background: #313244; border-radius: 8px; "
            "padding: 12px; font-size: 14px; color: #a6adc8; }"
        )
        layout.addWidget(self.lbl_banner)

        # Panels kết quả (tối đa 3 thuật toán)
        self.result_panels: List[ResultPanel] = []
        self.panels_row = QHBoxLayout()
        layout.addLayout(self.panels_row)

        # Bảng vật phẩm được chọn (của thuật toán tốt nhất)
        grp_items = QGroupBox("📋 Chi tiết vật phẩm được chọn (kết quả tốt nhất)")
        items_layout = QVBoxLayout(grp_items)
        self.tbl_selected = QTableWidget(0, 5)
        self.tbl_selected.setHorizontalHeaderLabels(
            ["STT", "Tên", "Trọng lượng", "Giá trị", "Tỉ lệ (v/w)"]
        )
        self.tbl_selected.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch
        )
        self.tbl_selected.setAlternatingRowColors(True)
        self.tbl_selected.setEditTriggers(
            QTableWidget.EditTrigger.NoEditTriggers
        )
        items_layout.addWidget(self.tbl_selected)
        layout.addWidget(grp_items)

        return w

    def _build_chart_tab(self) -> QWidget:
        self.chart_widget = ChartWidget()
        return self.chart_widget

    def _build_compare_tab(self) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout(w)

        self.tbl_compare = QTableWidget(0, 5)
        self.tbl_compare.setHorizontalHeaderLabels([
            "Thuật toán", "Giá trị", "Trọng lượng", "Thời gian (ms)", "Ghi chú"
        ])
        self.tbl_compare.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch
        )
        self.tbl_compare.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.tbl_compare.setAlternatingRowColors(True)
        layout.addWidget(self.tbl_compare)

        return w

    def _build_log_tab(self) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout(w)
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        btn_clear_log = QPushButton("Xóa log")
        btn_clear_log.clicked.connect(self.log_text.clear)
        layout.addWidget(self.log_text)
        layout.addWidget(btn_clear_log)
        return w

    # ── Logic UI ─────────────────────────────────────────────

    def _on_algo_changed(self, idx: int):
        """Hiện/ẩn panel tham số Bat khi cần."""
        show_bat = idx in (2, 3)   # "Bat" hoặc "Tất cả"
        self.grp_bat.setVisible(show_bat)

    def _add_row(self, weight: float = 1.0, value: float = 1.0, name: str = ""):
        row = self.table.rowCount()
        self.table.insertRow(row)
        auto_name = name or f"Item_{row}"
        self.table.setItem(row, 0, QTableWidgetItem(auto_name))
        self.table.setItem(row, 1, QTableWidgetItem(str(weight)))
        self.table.setItem(row, 2, QTableWidgetItem(str(value)))

    def _delete_row(self):
        rows = sorted(set(i.row() for i in self.table.selectedItems()), reverse=True)
        for r in rows:
            self.table.removeRow(r)
        if not rows:
            rc = self.table.rowCount()
            if rc > 0:
                self.table.removeRow(rc - 1)

    def _clear_table(self):
        self.table.setRowCount(0)

    def _generate_random(self):
        if not BACKEND_OK:
            return
        n = self.spin_rand_n.value()
        seed_val = self.spin_seed.value()
        seed = seed_val if seed_val >= 0 else None
        try:
            prob = generate_random_problem(n=n, seed=seed)
            self.spin_capacity.setValue(prob.capacity)
            self.table.setRowCount(0)
            for item in prob.items:
                self._add_row(item.weight, item.value, item.name)
            self._log(f"✅ Đã tạo {n} vật phẩm ngẫu nhiên (seed={seed}). "
                      f"Sức chứa = {prob.capacity}")
        except Exception as e:
            QMessageBox.warning(self, "Lỗi", str(e))

    def _populate_default(self):
        """Thêm ví dụ mặc định 5 vật phẩm."""
        data = [
            ("A", 2, 3), ("B", 3, 4), ("C", 4, 5), ("D", 5, 8), ("E", 6, 9)
        ]
        for name, w, v in data:
            self._add_row(w, v, name)

    # ── Validation ───────────────────────────────────────────

    def _parse_input(self):
        """
        Đọc dữ liệu từ bảng và trả về (capacity, weights, values, names).
        Raise ValueError nếu dữ liệu không hợp lệ.
        """
        capacity = self.spin_capacity.value()
        if capacity <= 0:
            raise ValueError("Sức chứa phải > 0.")

        n = self.table.rowCount()
        if n == 0:
            raise ValueError("Bảng vật phẩm đang trống. Hãy thêm ít nhất 1 vật.")

        weights, values, names = [], [], []
        for row in range(n):
            name_item  = self.table.item(row, 0)
            weight_item = self.table.item(row, 1)
            value_item  = self.table.item(row, 2)

            name = name_item.text().strip() if name_item else f"Item_{row}"
            if not name:
                name = f"Item_{row}"

            try:
                w = float(weight_item.text()) if weight_item else 0
                v = float(value_item.text())  if value_item  else 0
            except (ValueError, AttributeError):
                raise ValueError(
                    f"Hàng {row+1}: Trọng lượng và giá trị phải là số thực."
                )
            if w <= 0:
                raise ValueError(f"Hàng {row+1}: Trọng lượng phải > 0.")
            if v < 0:
                raise ValueError(f"Hàng {row+1}: Giá trị không được âm.")

            weights.append(w)
            values.append(v)
            names.append(name)

        return capacity, weights, values, names

    def _get_bat_params(self) -> dict:
        return {
            "n_bats"       : int(self.bat_n_bats.value()),
            "max_iter"     : int(self.bat_max_iter.value()),
            "alpha"        : self.bat_alpha.value(),
            "gamma"        : self.bat_gamma.value(),
            "mutation_rate": self.bat_mut_rate.value(),
            "f_min"        : self.bat_f_min.value(),
            "f_max"        : self.bat_f_max.value(),
            "seed"         : None,
        }

    # ── Chạy thuật toán ──────────────────────────────────────

    def _run_solver(self):
        if not BACKEND_OK:
            QMessageBox.critical(self, "Lỗi", "Backend chưa sẵn sàng.")
            return

        try:
            cap, weights, values, names = self._parse_input()
        except ValueError as e:
            QMessageBox.warning(self, "Dữ liệu không hợp lệ", str(e))
            return

        self._problem = create_problem(cap, weights, values, names)

        idx = self.combo_algo.currentIndex()
        algo_map = {
            0: ["greedy_ratio"],
            1: ["gbfs"],
            2: ["bat"],
            3: ["greedy_ratio", "gbfs", "bat"],
        }
        algorithms = algo_map.get(idx, ["greedy_ratio"])

        bat_params = self._get_bat_params() if idx in (2, 3) else {}

        self._log(f"▶ Bắt đầu chạy: {', '.join(algorithms)} "
                  f"| {self._problem.n} vật | capacity={cap}")

        self.btn_run.setEnabled(False)
        self.progress.setVisible(True)

        self._worker = SolverWorker(self._problem, algorithms, bat_params)
        self._worker.finished.connect(self._on_solver_done)
        self._worker.error.connect(self._on_solver_error)
        self._worker.start()

    def _on_solver_done(self, results: List):
        self.progress.setVisible(False)
        self.btn_run.setEnabled(True)
        self._results = results

        for r in results:
            self._log(
                f"✅ [{r.algorithm}] "
                f"value={r.value:.4f}, weight={r.weight:.4f}, "
                f"time={r.time*1000:.3f}ms"
            )

        self._update_result_tab(results)
        self._update_compare_tab(results)

        if MATPLOTLIB_AVAILABLE:
            self.chart_widget.update_charts(results)
            self.chart_widget.draw_convergence(results)

        # Chuyển sang tab kết quả
        self.tabs.setCurrentIndex(0)

    def _on_solver_error(self, msg: str):
        self.progress.setVisible(False)
        self.btn_run.setEnabled(True)
        self._log(f"❌ Lỗi: {msg}")
        QMessageBox.critical(self, "Lỗi khi chạy", msg)

    # ── Cập nhật UI sau khi có kết quả ──────────────────────

    def _update_result_tab(self, results: List):
        # Xóa panels cũ
        for panel in self.result_panels:
            self.panels_row.removeWidget(panel)
            panel.deleteLater()
        self.result_panels.clear()

        best = max(results, key=lambda r: r.value)
        self.lbl_banner.setText(
            f"🏆 Kết quả tốt nhất: <b>{best.algorithm}</b> — "
            f"Giá trị = <b style='color:#a6e3a1'>{best.value:.4f}</b> "
            f"| Trọng lượng = {best.weight:.4f} "
            f"| Thời gian = {best.time*1000:.3f} ms"
        )
        self.lbl_banner.setStyleSheet(
            "QLabel { background: #1e3a28; border: 1px solid #a6e3a1; "
            "border-radius: 8px; padding: 12px; font-size: 13px; color: #cdd6f4; }"
        )

        colors = ["#89b4fa", "#a6e3a1", "#fab387"]
        for r, color in zip(results, colors):
            panel = ResultPanel(r.algorithm, color)
            panel.update_result(r, self._problem)
            self.panels_row.addWidget(panel)
            self.result_panels.append(panel)

        # Bảng vật phẩm của kết quả tốt nhất
        selected = [i for i, s in enumerate(best.solution) if s == 1]
        self.tbl_selected.setRowCount(len(selected))
        for row, item_idx in enumerate(selected):
            item = self._problem.items[item_idx]
            for col, val in enumerate([row + 1, item.name,
                                        f"{item.weight:.3f}",
                                        f"{item.value:.3f}",
                                        f"{item.ratio:.4f}"]):
                cell = QTableWidgetItem(str(val))
                cell.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                self.tbl_selected.setItem(row, col, cell)

    def _update_compare_tab(self, results: List):
        best_val = max(r.value for r in results)
        self.tbl_compare.setRowCount(len(results))

        for row, r in enumerate(results):
            is_best = (r.value == best_val)
            cells = [
                r.algorithm,
                f"{r.value:.4f}",
                f"{r.weight:.4f}",
                f"{r.time*1000:.4f}",
                "⭐ Tốt nhất" if is_best else "",
            ]
            for col, txt in enumerate(cells):
                item = QTableWidgetItem(txt)
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                if is_best:
                    item.setForeground(QColor("#a6e3a1"))
                self.tbl_compare.setItem(row, col, item)

    # ── Log ──────────────────────────────────────────────────

    def _log(self, msg: str):
        from datetime import datetime
        ts = datetime.now().strftime("%H:%M:%S")
        self.log_text.append(f"<span style='color:#585b70'>[{ts}]</span> {msg}")
        self.tabs.setTabText(3, f"📋 Log ({self.log_text.document().lineCount()})")


# ════════════════════════════════════════════════════════════════
# ENTRY POINT
# ════════════════════════════════════════════════════════════════

def main():
    app = QApplication(sys.argv)
    app.setStyleSheet(STYLE)
    app.setApplicationName("Knapsack Solver")

    win = KnapsackApp()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()

