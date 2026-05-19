

import time
import heapq
from typing import List, Optional
from core import (
    KnapsackProblem, KnapsackResult,
    evaluate, is_feasible
)


# ══════════════════════════════════════════════════════════════
# PHẦN 1: GREEDY RATIO (Tham lam theo tỉ lệ value/weight)
# ══════════════════════════════════════════════════════════════

def greedy_ratio(problem: KnapsackProblem) -> KnapsackResult:
    """
    Thuật toán Greedy cổ điển cho 0/1 Knapsack.

    """
    start = time.perf_counter()

    n = problem.n
    solution = [0] * n
    remaining_capacity = problem.capacity

    # Bước 1: Sắp xếp theo tỉ lệ value/weight giảm dần
    # Giữ nguyên chỉ số gốc để map lại solution
    sorted_items = sorted(
        problem.items,
        key=lambda item: item.ratio,
        reverse=True
    )

    # Bước 2: Tham lam — chọn từng vật nếu còn chỗ
    for item in sorted_items:
        if item.weight <= remaining_capacity:
            solution[item.index] = 1
            remaining_capacity -= item.weight

    total_value, total_weight = evaluate(problem, solution)
    elapsed = time.perf_counter() - start

    return KnapsackResult(
        solution=solution,
        value=total_value,
        weight=total_weight,
        time=elapsed,
        algorithm="Greedy (Ratio)",
        extra={"sorted_order": [item.index for item in sorted_items]}
    )


# ══════════════════════════════════════════════════════════════
# PHẦN 2: GREEDY BEST-FIRST SEARCH (GBFS)
# ══════════════════════════════════════════════════════════════

def _upper_bound(problem: KnapsackProblem, depth: int,
                 current_value: float, remaining_cap: float,
                 sorted_indices: List[int]) -> float:
    """
    Tính upper-bound bằng fractional knapsack cho các vật từ 'depth' trở đi.
  Hàm heuristic h(n) của GBFS 
    """
    bound = current_value
    cap   = remaining_cap

    for idx in sorted_indices[depth:]:
        item = problem.items[idx]
        if item.weight <= cap:
            bound += item.value
            cap   -= item.weight
        else:
            # Lấy phần phân số — cho phép để ước lượng lạc quan
            bound += (cap / item.weight) * item.value
            break

    return bound


class _GBFSNode:
    """
    Nút trong cây tìm kiếm của GBFS.
    Mỗi nút = trạng thái sau khi quyết định với vật phẩm thứ 'depth'.
    """
    __slots__ = ("depth", "current_value", "current_weight",
                 "solution", "heuristic")

    def __init__(self, depth, current_value, current_weight, solution, heuristic):
        self.depth          = depth
        self.current_value  = current_value
        self.current_weight = current_weight
        self.solution       = solution
        self.heuristic      = heuristic  # -upper_bound (vì heapq là min-heap)

    # So sánh theo heuristic (min-heap → âm của upper-bound)
    def __lt__(self, other):
        return self.heuristic < other.heuristic


def greedy_best_first_search(problem: KnapsackProblem,
                              max_nodes: int = 100_000) -> KnapsackResult:
    """
    Greedy Best-First Search (GBFS) cho 0/1 Knapsack.

    """
    start = time.perf_counter()

    n = problem.n

    # Sắp xếp vật theo ratio giảm dần — tối ưu hóa heuristic
    sorted_indices = sorted(
        range(n),
        key=lambda i: problem.items[i].ratio,
        reverse=True
    )

    best_value    = 0.0
    best_solution = [0] * n
    nodes_visited = 0

    # Nút gốc: chưa chọn gì, depth=0
    root_bound = _upper_bound(problem, 0, 0.0, problem.capacity, sorted_indices)
    root = _GBFSNode(
        depth=0,
        current_value=0.0,
        current_weight=0.0,
        solution=[0] * n,
        heuristic=-root_bound  # âm để min-heap hoạt động như max
    )

    priority_queue: List[_GBFSNode] = []
    heapq.heappush(priority_queue, root)

    while priority_queue and nodes_visited < max_nodes:
        node = heapq.heappop(priority_queue)
        nodes_visited += 1

        # Pruning: nếu upper-bound ≤ best đã tìm → bỏ qua
        if -node.heuristic <= best_value:
            continue

        # Nếu đã quyết định hết n vật → đây là lá cây
        if node.depth == n:
            if node.current_value > best_value:
                best_value    = node.current_value
                best_solution = node.solution[:]
            continue

        item_idx = sorted_indices[node.depth]
        item     = problem.items[item_idx]

        # ── Nhánh 1: CHỌN vật phẩm này ──────────────────────────
        if node.current_weight + item.weight <= problem.capacity:
            new_sol    = node.solution[:]
            new_sol[item_idx] = 1
            new_value  = node.current_value  + item.value
            new_weight = node.current_weight + item.weight

            # Cập nhật best ngay (greedy: ưu tiên giải pháp tốt hơn hiện tại)
            if new_value > best_value:
                best_value    = new_value
                best_solution = new_sol[:]

            ub = _upper_bound(problem, node.depth + 1,
                               new_value, problem.capacity - new_weight,
                               sorted_indices)
            if ub > best_value:
                heapq.heappush(priority_queue, _GBFSNode(
                    depth=node.depth + 1,
                    current_value=new_value,
                    current_weight=new_weight,
                    solution=new_sol,
                    heuristic=-ub
                ))

        # ── Nhánh 2: BỎ vật phẩm này ────────────────────────────
        ub_skip = _upper_bound(problem, node.depth + 1,
                                node.current_value,
                                problem.capacity - node.current_weight,
                                sorted_indices)
        if ub_skip > best_value:
            heapq.heappush(priority_queue, _GBFSNode(
                depth=node.depth + 1,
                current_value=node.current_value,
                current_weight=node.current_weight,
                solution=node.solution[:],
                heuristic=-ub_skip
            ))

    total_value, total_weight = evaluate(problem, best_solution)
    elapsed = time.perf_counter() - start

    return KnapsackResult(
        solution=best_solution,
        value=total_value,
        weight=total_weight,
        time=elapsed,
        algorithm="Greedy Best-First Search",
        extra={
            "nodes_visited": nodes_visited,
            "pruned": nodes_visited < max_nodes,
            "max_nodes_limit": max_nodes,
        }
    )


# ══════════════════════════════════════════════════════════════
# PHẦN 3: Giao diện gọi thống nhất
# ══════════════════════════════════════════════════════════════

def solve_greedy(problem: KnapsackProblem,
                 method: str = "ratio") -> KnapsackResult:
    """
    Hàm gọi thuật toán Greedy dạng thống nhất.

    """
    method = method.lower().strip()
    if method == "ratio":
        return greedy_ratio(problem)
    elif method in ("gbfs", "best_first", "best-first"):
        return greedy_best_first_search(problem)
    else:
        raise ValueError(f"method không hợp lệ: '{method}'. Chọn 'ratio' hoặc 'gbfs'.")


# ══════════════════════════════════════════════════════════════
# Kiểm thử
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    from core import create_problem, generate_random_problem, print_problem, compare_results

    # ── Ví dụ 1: bài toán nhỏ ─────────────────────────────────
    print(">>> Ví dụ 1: Bài toán 5 vật phẩm")
    p1 = create_problem(
        capacity=15,
        weights=[2, 3, 4, 5, 6],
        values =[3, 4, 5, 8, 9],
        names  =["A", "B", "C", "D", "E"]
    )
    print_problem(p1)

    r_ratio = greedy_ratio(p1)
    r_gbfs  = greedy_best_first_search(p1)
    compare_results([r_ratio, r_gbfs])

    print(r_ratio)
    print()
    print(r_gbfs)

    # ── Ví dụ 2: bài toán ngẫu nhiên 20 vật ──────────────────
    print("\n\n>>> Ví dụ 2: Bài toán ngẫu nhiên 20 vật (seed=42)")
    p2 = generate_random_problem(n=20, seed=42)
    r2_ratio = greedy_ratio(p2)
    r2_gbfs  = greedy_best_first_search(p2)
    compare_results([r2_ratio, r2_gbfs])


