"""
bat.py - Bat Algorithm cho bài toán 0/1 Knapsack
=================================================
Cài đặt Bat Algorithm (Yang, 2010) điều chỉnh cho không gian nhị phân.

Tham khảo:
    Yang, X.-S. (2010). A new metaheuristic bat-inspired algorithm.
    Nature Inspired Cooperative Strategies for Optimization (NICSO 2010),
    Studies in Computational Intelligence, 284, 65–74.

Ý tưởng cốt lõi:
    - Mỗi "con dơi" = một nghiệm ứng viên (vector nhị phân 0/1).
    - Dơi phát sóng âm (echolocation) để tìm mồi (nghiệm tốt).
    - Tần số, vận tốc, vị trí được cập nhật mỗi vòng lặp.
    - Độ lớn xung (loudness A) giảm dần → khai thác tinh tế hơn.
    - Tỉ lệ phát xung (pulse rate r) tăng dần → tìm kiếm tập trung hơn.
    - Đột biến ngẫu nhiên nhỏ (≤ 5%) để thoát local optima.
"""

import time
import random
import math
from typing import List, Optional, Tuple
from copy import deepcopy

from core import (
    KnapsackProblem,
    KnapsackResult,
    evaluate,
    is_feasible,
    repair_solution,
)


# ──────────────────────────────────────────────────────────────
# 1. Lớp Bat — đại diện cho một con dơi trong quần thể
# ──────────────────────────────────────────────────────────────

class Bat:
    """
    Đại diện cho một con dơi trong Bat Algorithm.

    Attributes:
        position  : Vector nhị phân — nghiệm ứng viên hiện tại
        velocity  : Vector thực — tốc độ thay đổi (dùng để tính xác suất lật bit)
        frequency : Tần số sóng âm hiện tại
        loudness  : Độ lớn xung âm hiện tại (A)
        pulse_rate: Tỉ lệ phát xung hiện tại (r)
        value     : Giá trị mục tiêu của position hiện tại
        weight    : Trọng lượng của position hiện tại
    """

    __slots__ = (
        "position", "velocity", "frequency",
        "loudness", "pulse_rate", "value", "weight"
    )

    def __init__(
        self,
        n: int,
        f_min: float,
        f_max: float,
        A0: float,
        r0: float,
        rng: random.Random,
    ):
        """
        Khởi tạo con dơi ngẫu nhiên.

        Args:
            n   : Số chiều (số vật phẩm)
            f_min, f_max: Dải tần số
            A0  : Độ lớn xung ban đầu
            r0  : Tỉ lệ phát xung ban đầu
            rng : Đối tượng Random (để tái lập kết quả)
        """
        # Vị trí ban đầu: nhị phân ngẫu nhiên
        self.position  = [rng.randint(0, 1) for _ in range(n)]
        # Vận tốc ban đầu: thực, trong [-1, 1]
        self.velocity  = [rng.uniform(-1.0, 1.0) for _ in range(n)]
        self.frequency = rng.uniform(f_min, f_max)
        self.loudness  = A0
        self.pulse_rate = r0
        self.value     = 0.0
        self.weight    = 0.0


# ──────────────────────────────────────────────────────────────
# 2. Các hàm hỗ trợ nội bộ
# ──────────────────────────────────────────────────────────────

def _sigmoid(x: float) -> float:
    """
    Hàm sigmoid — chuyển velocity sang xác suất lật bit.
    Giới hạn đầu vào để tránh overflow.
    """
    x = max(-500.0, min(500.0, x))
    return 1.0 / (1.0 + math.exp(-x))


def _binarize(velocity: List[float], rng: random.Random) -> List[int]:
    """
    Chuyển vector vận tốc thực thành vector nhị phân
    dùng hàm sigmoid làm ngưỡng xác suất.

    position[i] = 1 nếu random() < sigmoid(velocity[i])
    """
    return [1 if rng.random() < _sigmoid(v) else 0 for v in velocity]


def _mutate(solution: List[int], mutation_rate: float, rng: random.Random) -> List[int]:
    """
    Đột biến ngẫu nhiên: lật bit với xác suất mutation_rate.
    Theo yêu cầu: mutation_rate ≤ 5%.

    Args:
        solution     : Nghiệm nhị phân hiện tại
        mutation_rate: Xác suất lật mỗi bit (0.0 → 0.05)
        rng          : Random generator

    Returns:
        Nghiệm mới sau đột biến
    """
    return [
        (1 - bit) if rng.random() < mutation_rate else bit
        for bit in solution
    ]


def _local_search(
    problem: KnapsackProblem,
    solution: List[int],
    best_value: float,
    loudness: float,
    rng: random.Random,
) -> List[int]:
    """
    Tìm kiếm cục bộ quanh nghiệm tốt nhất hiện tại.
    Xáo trộn ngẫu nhiên một số bit tỉ lệ với loudness.

    Được gọi khi: random() < pulse_rate của con dơi.

    Args:
        problem   : Bài toán knapsack
        solution  : Nghiệm tốt nhất hiện tại (global best)
        best_value: Giá trị tốt nhất hiện tại
        loudness  : Độ lớn xung của con dơi (điều chỉnh bước tìm)
        rng       : Random generator

    Returns:
        Nghiệm mới sau tìm kiếm cục bộ
    """
    n = len(solution)
    new_sol = solution[:]

    # Số bit bị xáo = loudness * một số ngẫu nhiên nhỏ (≥ 1)
    num_flip = max(1, int(loudness * rng.uniform(0.1, 0.3) * n))
    positions = rng.sample(range(n), min(num_flip, n))
    for pos in positions:
        new_sol[pos] = 1 - new_sol[pos]

    return new_sol


# ──────────────────────────────────────────────────────────────
# 3. Hàm chính: bat_algorithm
# ──────────────────────────────────────────────────────────────

def bat_algorithm(
    problem: KnapsackProblem,
    n_bats: int = 30,
    max_iter: int = 500,
    f_min: float = 0.0,
    f_max: float = 2.0,
    A0: float = 1.0,
    r0: float = 0.5,
    alpha: float = 0.9,
    gamma: float = 0.9,
    mutation_rate: float = 0.02,   # ≤ 5% theo yêu cầu
    seed: Optional[int] = None,
) -> KnapsackResult:
    """
    Bat Algorithm (BA) cho 0/1 Knapsack Problem.

    Mô tả thuật toán:
    ─────────────────
    1. Khởi tạo quần thể gồm n_bats con dơi ngẫu nhiên.
       Mỗi con dơi có: vị trí (nhị phân), vận tốc (thực),
       tần số, độ lớn xung (A), tỉ lệ phát xung (r).

    2. Đánh giá và sửa nghiệm (repair) vi phạm trọng lượng.

    3. Lặp max_iter vòng:
       a. Cập nhật tần số: f_i = f_min + (f_max - f_min) * β  (β ~ U[0,1])
       b. Cập nhật vận tốc: v_i = v_i + (x_i - x*) * f_i
          (x* = vị trí con dơi tốt nhất)
       c. Cập nhật vị trí: dùng sigmoid(v_i) làm xác suất lật bit
       d. Tìm kiếm cục bộ: nếu rand() < r_i → xáo trộn quanh global best
       e. Đột biến nhỏ (≤ 5%): lật bit ngẫu nhiên với xác suất mutation_rate
       f. Repair nghiệm vi phạm sức chứa
       g. Chấp nhận nghiệm mới nếu tốt hơn và rand() < A_i (greedy + random)
       h. Cập nhật global best
       i. Cập nhật A và r: A *= alpha, r = r0*(1 - exp(-gamma*t))

    Tham số:
    ────────
    Args:
        problem      : Bài toán knapsack cần giải
        n_bats       : Số lượng dơi trong quần thể (default: 30)
        max_iter     : Số vòng lặp tối đa (default: 500)
        f_min        : Tần số tối thiểu (default: 0.0)
        f_max        : Tần số tối đa (default: 2.0)
        A0           : Độ lớn xung ban đầu (default: 1.0)
        r0           : Tỉ lệ phát xung ban đầu (default: 0.5)
        alpha        : Hệ số giảm loudness (default: 0.9)
        gamma        : Hệ số tăng pulse_rate (default: 0.9)
        mutation_rate: Xác suất đột biến mỗi bit (default: 0.02 = 2%, max 5%)
        seed         : Random seed để tái lập kết quả

    Returns:
        KnapsackResult — kết quả chuẩn theo định nghĩa core.py
    """
    # Ràng buộc mutation_rate ≤ 5% theo yêu cầu đề bài
    mutation_rate = min(mutation_rate, 0.05)

    start = time.perf_counter()
    rng   = random.Random(seed)
    n     = problem.n

    # ── Bước 1: Khởi tạo quần thể ────────────────────────────
    bats: List[Bat] = [
        Bat(n, f_min, f_max, A0, r0, rng) for _ in range(n_bats)
    ]

    # Đánh giá & repair nghiệm ban đầu
    for bat in bats:
        bat.position        = repair_solution(problem, bat.position)
        bat.value, bat.weight = evaluate(problem, bat.position)

    # Tìm global best ban đầu
    best_bat   = max(bats, key=lambda b: b.value)
    best_pos   = best_bat.position[:]
    best_value = best_bat.value
    best_weight = best_bat.weight

    # Lịch sử hội tụ (để phân tích sau)
    convergence: List[float] = [best_value]

    # ── Bước 2: Vòng lặp chính ───────────────────────────────
    for t in range(1, max_iter + 1):

        for bat in bats:

            # (a) Cập nhật tần số
            beta           = rng.random()
            bat.frequency  = f_min + (f_max - f_min) * beta

            # (b) Cập nhật vận tốc (không gian thực)
            bat.velocity = [
                bat.velocity[i] + (bat.position[i] - best_pos[i]) * bat.frequency
                for i in range(n)
            ]

            # (c) Tạo vị trí mới từ vận tốc qua sigmoid
            new_pos = _binarize(bat.velocity, rng)

            # (d) Tìm kiếm cục bộ quanh global best
            if rng.random() < bat.pulse_rate:
                new_pos = _local_search(problem, best_pos, best_value,
                                        bat.loudness, rng)

            # (e) Đột biến nhỏ (≤ 5%)
            new_pos = _mutate(new_pos, mutation_rate, rng)

            # (f) Repair nghiệm vi phạm trọng lượng
            new_pos = repair_solution(problem, new_pos)
            new_val, new_wt = evaluate(problem, new_pos)

            # (g) Chấp nhận nghiệm mới:
            #     - Tốt hơn nghiệm hiện tại CỦA CON DƠI này, VÀ
            #     - rand() < A_i (loudness càng nhỏ → chọn lọc càng khắt khe)
            if new_val >= bat.value and rng.random() < bat.loudness:
                bat.position = new_pos
                bat.value    = new_val
                bat.weight   = new_wt

                # (h) Cập nhật global best
                if new_val > best_value:
                    best_pos    = new_pos[:]
                    best_value  = new_val
                    best_weight = new_wt

                # (i) Giảm loudness, tăng pulse_rate (chỉ khi chấp nhận)
                bat.loudness   *= alpha
                bat.pulse_rate  = r0 * (1.0 - math.exp(-gamma * t))

        convergence.append(best_value)

    elapsed = time.perf_counter() - start

    return KnapsackResult(
        solution  = best_pos,
        value     = best_value,
        weight    = best_weight,
        time      = elapsed,
        algorithm = "Bat Algorithm",
        extra     = {
            "n_bats"        : n_bats,
            "max_iter"      : max_iter,
            "f_min"         : f_min,
            "f_max"         : f_max,
            "A0"            : A0,
            "r0"            : r0,
            "alpha"         : alpha,
            "gamma"         : gamma,
            "mutation_rate" : mutation_rate,
            "seed"          : seed,
            "convergence"   : convergence,   # lịch sử giá trị tốt nhất mỗi iter
        }
    )


# ──────────────────────────────────────────────────────────────
# 4. Giao diện gọi thống nhất (tương tự solve_greedy)
# ──────────────────────────────────────────────────────────────

def solve_bat(
    problem: KnapsackProblem,
    n_bats: int = 30,
    max_iter: int = 500,
    f_min: float = 0.0,
    f_max: float = 2.0,
    A0: float = 1.0,
    r0: float = 0.5,
    alpha: float = 0.9,
    gamma: float = 0.9,
    mutation_rate: float = 0.02,
    seed: Optional[int] = None,
) -> KnapsackResult:
    """
    Hàm gọi Bat Algorithm dạng thống nhất.
    Tất cả tham số đều có giá trị mặc định — chỉ cần truyền problem.

    Args:
        problem      : Bài toán knapsack (từ core.py)
        n_bats       : Số dơi (default 30)
        max_iter     : Số vòng lặp (default 500)
        f_min/f_max  : Dải tần số
        A0           : Loudness ban đầu
        r0           : Pulse rate ban đầu
        alpha        : Hệ số giảm A (0 < alpha < 1)
        gamma        : Hệ số tăng r
        mutation_rate: Tỉ lệ đột biến (≤ 0.05)
        seed         : Seed ngẫu nhiên

    Returns:
        KnapsackResult
    """
    return bat_algorithm(
        problem       = problem,
        n_bats        = n_bats,
        max_iter      = max_iter,
        f_min         = f_min,
        f_max         = f_max,
        A0            = A0,
        r0            = r0,
        alpha         = alpha,
        gamma         = gamma,
        mutation_rate = mutation_rate,
        seed          = seed,
    )


# ──────────────────────────────────────────────────────────────
# 5. Kiểm thử
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from core import (
        create_problem, generate_random_problem,
        print_problem, compare_results, print_result
    )
    from greedy import greedy_ratio, greedy_best_first_search

    # ── Ví dụ 1: Bài toán nhỏ 5 vật ──────────────────────────
    print("=" * 55)
    print("  VÍ DỤ 1: Bài toán 5 vật phẩm")
    print("=" * 55)
    p1 = create_problem(
        capacity = 15,
        weights  = [2, 3, 4, 5, 6],
        values   = [3, 4, 5, 8, 9],
        names    = ["A", "B", "C", "D", "E"]
    )
    print_problem(p1)

    r_greedy = greedy_ratio(p1)
    r_gbfs   = greedy_best_first_search(p1)
    r_bat    = solve_bat(p1, n_bats=10, max_iter=100, seed=42)

    compare_results([r_greedy, r_gbfs, r_bat])
    print_result(r_bat, p1)

    # ── Ví dụ 2: Bài toán ngẫu nhiên 20 vật ──────────────────
    print("\n" + "=" * 55)
    print("  VÍ DỤ 2: Bài toán ngẫu nhiên 20 vật (seed=42)")
    print("=" * 55)
    p2 = generate_random_problem(n=20, seed=42)

    r2_greedy = greedy_ratio(p2)
    r2_gbfs   = greedy_best_first_search(p2)
    r2_bat    = solve_bat(p2, n_bats=30, max_iter=300, seed=42)

    compare_results([r2_greedy, r2_gbfs, r2_bat])

    # ── Ví dụ 3: Bài toán lớn hơn 50 vật ─────────────────────
    print("\n" + "=" * 55)
    print("  VÍ DỤ 3: Bài toán ngẫu nhiên 50 vật (seed=7)")
    print("=" * 55)
    p3 = generate_random_problem(n=50, seed=7)

    r3_greedy = greedy_ratio(p3)
    r3_bat    = solve_bat(p3, n_bats=40, max_iter=500, seed=7)

    compare_results([r3_greedy, r3_bat])

    # ── Ví dụ 4: So sánh ảnh hưởng của tham số ───────────────
    print("\n" + "=" * 55)
    print("  VÍ DỤ 4: Thay đổi tham số Bat Algorithm")
    print("=" * 55)
    p4 = generate_random_problem(n=30, seed=123)

    configs = [
        {"n_bats": 10, "max_iter": 200, "label": "Bat (nhỏ - 10 dơi, 200 iter)"},
        {"n_bats": 30, "max_iter": 500, "label": "Bat (vừa - 30 dơi, 500 iter)"},
        {"n_bats": 50, "max_iter": 500, "label": "Bat (lớn - 50 dơi, 500 iter)"},
        {"n_bats": 30, "max_iter": 500, "mutation_rate": 0.05, "label": "Bat (đột biến 5%)"},
    ]

    results_p4 = []
    for cfg in configs:
        label = cfg.pop("label")
        r = solve_bat(p4, seed=42, **cfg)
        r.algorithm = label
        results_p4.append(r)

    compare_results(results_p4)