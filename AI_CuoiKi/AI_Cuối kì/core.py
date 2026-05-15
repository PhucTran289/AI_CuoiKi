"""
core.py - Phần core chung cho bài toán Knapsack
================================================
Định nghĩa cấu trúc dữ liệu, hàm đọc input, kiểm tra nghiệm,
và tính toán kết quả dùng chung cho TẤT CẢ thuật toán.
"""

from dataclasses import dataclass, field
from typing import List, Optional
import random


# ──────────────────────────────────────────────
# 1. Cấu trúc dữ liệu chung
# ──────────────────────────────────────────────

@dataclass
class Item:
    """Một vật phẩm trong bài toán knapsack."""
    index: int          # Chỉ số (0-based)
    name: str           # Tên vật phẩm
    weight: float       # Trọng lượng
    value: float        # Giá trị
    ratio: float = 0.0  # value / weight (tính sau)

    def __post_init__(self):
        self.ratio = self.value / self.weight if self.weight > 0 else 0.0


@dataclass
class KnapsackProblem:
    """Định nghĩa đầy đủ một bài toán knapsack."""
    capacity: float          # Sức chứa tối đa của túi
    items: List[Item]        # Danh sách vật phẩm

    @property
    def n(self) -> int:
        """Số lượng vật phẩm."""
        return len(self.items)


@dataclass
class KnapsackResult:
    """
    Kết quả trả về CHUẨN cho mọi thuật toán.

    """
    solution: List[int]      # Vector nhị phân: solution[i]=1 nếu chọn vật phẩm i
    value: float             # Tổng giá trị đạt được
    weight: float            # Tổng trọng lượng đã dùng
    time: float              # Thời gian chạy (giây)
    algorithm: str = ""      # Tên thuật toán (để phân biệt khi so sánh)
    extra: dict = field(default_factory=dict)  # Thông tin bổ sung tùy thuật toán

    def to_dict(self) -> dict:
        """Chuyển sang dict (dùng cho logging, JSON export, ...)."""
        return {
            "algorithm": self.algorithm,
            "solution": self.solution,
            "value": self.value,
            "weight": self.weight,
            "time": self.time,
            **self.extra,
        }

def __str__(self) -> str:
        selected = [i for i, s in enumerate(self.solution) if s == 1]
        return (
            f"[{self.algorithm}]\n"
            f"  Chọn vật phẩm: {selected}\n"
            f"  Tổng giá trị : {self.value:.4f}\n"
            f"  Tổng trọng lượng: {self.weight:.4f}\n"
            f"  Thời gian    : {self.time:.6f}s"
        )


# ──────────────────────────────────────────────
# 2. Hàm tạo / đọc input
# ──────────────────────────────────────────────

def create_problem(capacity: float, weights: List[float], values: List[float],
                   names: Optional[List[str]] = None) -> KnapsackProblem:
    """
    Tạo bài toán knapsack từ danh sách trọng lượng và giá trị.

    Args:
        capacity: Sức chứa tối đa
        weights : Danh sách trọng lượng
        values  : Danh sách giá trị
        names   : Tên vật phẩm (tùy chọn)

    Returns:
        KnapsackProblem
    """
    if len(weights) != len(values):
        raise ValueError("weights và values phải có cùng độ dài.")
    if names is None:
        names = [f"Item_{i}" for i in range(len(weights))]

    items = [
        Item(index=i, name=names[i], weight=weights[i], value=values[i])
        for i in range(len(weights))
    ]
    return KnapsackProblem(capacity=capacity, items=items)


def generate_random_problem(n: int, max_weight: float = 20.0,
                             max_value: float = 100.0,
                             capacity_ratio: float = 0.5,
                             seed: Optional[int] = None) -> KnapsackProblem:
    """
    Sinh bài toán ngẫu nhiên để kiểm thử.

    Args:
        n             : Số vật phẩm
        max_weight    : Trọng lượng tối đa mỗi vật
        max_value     : Giá trị tối đa mỗi vật
        capacity_ratio: Tỉ lệ sức chứa / tổng trọng lượng
        seed          : Random seed (tái lập kết quả)
    """
    rng = random.Random(seed)
    weights = [round(rng.uniform(1, max_weight), 2) for _ in range(n)]
    values  = [round(rng.uniform(1, max_value), 2)  for _ in range(n)]
    capacity = round(sum(weights) * capacity_ratio, 2)
    return create_problem(capacity, weights, values)


# ──────────────────────────────────────────────
# 3. Hàm kiểm tra / đánh giá nghiệm
# ──────────────────────────────────────────────

def is_feasible(problem: KnapsackProblem, solution: List[int]) -> bool:
    """
    Kiểm tra nghiệm có hợp lệ không (không vượt sức chứa).
    """
    total_weight = sum(
        problem.items[i].weight
        for i in range(problem.n)
        if solution[i] == 1
    )
    return total_weight <= problem.capacity


def evaluate(problem: KnapsackProblem, solution: List[int]) -> tuple[float, float]:
    """
    Tính tổng giá trị và tổng trọng lượng của nghiệm.

    Returns:
        (total_value, total_weight)
    """
    total_value  = 0.0
    total_weight = 0.0
    for i in range(problem.n):
        if solution[i] == 1:
            total_value  += problem.items[i].value
            total_weight += problem.items[i].weight
    return total_value, total_weight


def repair_solution(problem: KnapsackProblem, solution: List[int]) -> List[int]:
    """
    Sửa nghiệm vi phạm trọng lượng bằng cách loại bỏ vật phẩm
    có ratio thấp nhất cho đến khi hợp lệ.
    Dùng cho Bat Algorithm khi cần 'repair' nghiệm nhị phân.
    """
    sol = solution[:]
    _, w = evaluate(problem, sol)

    # Sắp xếp các vật đang được chọn theo ratio tăng dần → loại thấp nhất trước
    selected = sorted(
        [i for i in range(problem.n) if sol[i] == 1],
        key=lambda i: problem.items[i].ratio
    )
    for i in selected:
        if w <= problem.capacity:
            break
        w -= problem.items[i].weight
        sol[i] = 0
    return sol


# ──────────────────────────────────────────────
# 4. Hàm hiển thị kết quả (dùng chung cho CLI)
# ──────────────────────────────────────────────

def print_problem(problem: KnapsackProblem):
    """In thông tin bài toán ra console."""
    print(f"\n{'='*50}")
    print(f"  BÀI TOÁN KNAPSACK")
    print(f"  Sức chứa: {problem.capacity}  |  Số vật: {problem.n}")
    print(f"{'='*50}")
    print(f"  {'#':<4} {'Tên':<12} {'Trọng lượng':>12} {'Giá trị':>10} {'Tỉ lệ':>8}")
    print(f"  {'-'*50}")
    for item in problem.items:
        print(f"  {item.index:<4} {item.name:<12} {item.weight:>12.2f} "
              f"{item.value:>10.2f} {item.ratio:>8.4f}")
    print()


def print_result(result: KnapsackResult, problem: Optional[KnapsackProblem] = None):
    """In kết quả thuật toán ra console."""
    print(result)
    if problem:
        selected = [problem.items[i] for i, s in enumerate(result.solution) if s == 1]
        if selected:
            print("  Chi tiết vật phẩm được chọn:")
            for item in selected:
                print(f"    - {item.name}: weight={item.weight}, value={item.value}")
    print()


def compare_results(results: List[KnapsackResult]):
    """So sánh kết quả từ nhiều thuật toán."""
    print(f"\n{'='*60}")
    print(f"  SO SÁNH KẾT QUẢ CÁC THUẬT TOÁN")
    print(f"{'='*60}")
    print(f"  {'Thuật toán':<25} {'Giá trị':>10} {'Trọng lượng':>12} {'Thời gian':>12}")
    print(f"  {'-'*60}")
    best = max(results, key=lambda r: r.value)
    for r in results:
        marker = " ◀ tốt nhất" if r == best else ""
        print(f"  {r.algorithm:<25} {r.value:>10.4f} {r.weight:>12.4f} "
              f"{r.time:>11.6f}s{marker}")
    print()


# ──────────────────────────────────────────────
# 5. Kiểm thử nhanh module core
# ──────────────────────────────────────────────

if __name__ == "__main__":
    # Ví dụ đơn giản
    problem = create_problem(
        capacity=15,
        weights=[2, 3, 4, 5, 6],
        values =[3, 4, 5, 8, 9],
        names  =["A", "B", "C", "D", "E"]
    )
    print_problem(problem)

    # Kiểm tra hàm evaluate
    sol = [1, 1, 0, 1, 0]
    v, w = evaluate(problem, sol)
    print(f"Nghiệm {sol}: value={v}, weight={w}, feasible={is_feasible(problem, sol)}")

    # Kiểm tra repair
    bad_sol = [1, 1, 1, 1, 1]
    print(f"\nNghiệm vi phạm: {bad_sol}, weight={evaluate(problem, bad_sol)[1]}")
    fixed = repair_solution(problem, bad_sol)
    print(f"Sau repair    : {fixed}, weight={evaluate(problem, fixed)[1]}")
""
{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": None,
   "id": "5e63c0ff",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.13.6"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
