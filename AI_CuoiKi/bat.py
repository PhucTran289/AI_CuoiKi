 
from __future__ import annotations
import math
import random as _random
import time
from copy import deepcopy
from typing import List, Optional
 
try:
    from core import KnapsackProblem, KnapsackResult
except ImportError:
    # Fallback stub để chạy test độc lập
    class KnapsackProblem:
        pass
    class KnapsackResult:
        pass
 
 
# ══════════════════════════════════════════════════════════════════
#  HELPER: V-shaped transfer function 
# ══════════════════════════════════════════════════════════════════
def _v_transfer(v: float) -> float:
    """
    V-shaped transfer: xác suất flip bit.
    - v lớn (xa 0) → flip gần chắc chắn  (exploration)
    - v nhỏ (gần 0) → flip hiếm (exploitation)
    
    """
    return abs(math.tanh(v))
 
 
def _flip_bit(pos: int, v: float, rng: _random.Random) -> int:
   
    return 1 - pos if rng.random() < _v_transfer(v) else pos
 
 
# ══════════════════════════════════════════════════════════════════
#  HELPER: Repair + Evaluate
# ══════════════════════════════════════════════════════════════════
def _evaluate(solution: List[int], problem: KnapsackProblem) -> tuple:
    """Trả về (value, weight)."""
    value  = sum(problem.items[i].value  for i in range(problem.n) if solution[i])
    weight = sum(problem.items[i].weight for i in range(problem.n) if solution[i])
    return value, weight
 
 
def _repair(solution: List[int], problem: KnapsackProblem,
            rng: _random.Random) -> List[int]:
    """
    1. Nếu vượt capacity: xoá item có ratio thấp nhất cho đến khi hợp lệ.
    2. Thử thêm item chưa chọn (sort ratio cao → thấp) nếu còn chỗ.
    """
    sol = solution[:]
    items = problem.items
 
    # Bước 1: remove nếu overweight
    weight = sum(items[i].weight for i in range(problem.n) if sol[i])
    if weight > problem.capacity:
       
        selected = sorted(
            [i for i in range(problem.n) if sol[i]],
            key=lambda i: items[i].ratio   
        )
        for i in selected:
            if weight <= problem.capacity:
                break
            sol[i]  = 0
            weight -= items[i].weight
 
    # Bước 2: thêm nếu còn dư capacity
    remaining = problem.capacity - weight
    not_selected = sorted(
        [i for i in range(problem.n) if not sol[i]],
        key=lambda i: -items[i].ratio  
    )
    for i in not_selected:
        if items[i].weight <= remaining:
            sol[i]     = 1
            remaining -= items[i].weight
 
    return sol
 
 
# ══════════════════════════════════════════════════════════════════
#  HELPER: Khởi tạo quần thể tốt hơn random thuần
# ══════════════════════════════════════════════════════════════════
def _greedy_solution(problem: KnapsackProblem) -> List[int]:
    
    sol     = [0] * problem.n
    remain  = problem.capacity
    order   = sorted(range(problem.n), key=lambda i: -problem.items[i].ratio)
    for i in order:
        if problem.items[i].weight <= remain:
            sol[i]  = 1
            remain -= problem.items[i].weight
    return sol
 
 
def _init_population(n_bats: int, problem: KnapsackProblem,
                     rng: _random.Random) -> List[List[int]]:
   
    population = []
 
    # Nghiệm greedy làm anchor
    g = _greedy_solution(problem)
    population.append(g)
 
    # Phần còn lại: random với density ngẫu nhiên + repair
    for _ in range(n_bats - 1):
      
        density = rng.uniform(0.2, 0.6)
        sol = [1 if rng.random() < density else 0 for _ in range(problem.n)]
        sol = _repair(sol, problem, rng)
        population.append(sol)
 
    return population
 
 
# ══════════════════════════════════════════════════════════════════
#  BAT STRUCT
# ══════════════════════════════════════════════════════════════════
class _Bat:
    __slots__ = ("position", "velocity", "value", "weight",
                 "freq", "loudness", "pulse_rate")
 
    def __init__(self, position: List[int], value: float, weight: float,
                 loudness: float, pulse_rate: float):
        self.position   = position
        self.velocity   = [0.0] * len(position)   
        self.value      = value
        self.weight     = weight
        self.freq       = 0.0
        self.loudness   = loudness
        self.pulse_rate = pulse_rate
 
 
# ══════════════════════════════════════════════════════════════════
#  MAIN: bat_algorithm
# ══════════════════════════════════════════════════════════════════
def bat_algorithm(
    problem: KnapsackProblem,
    n_bats:        int   = 30,
    max_iter:      int   = 500,
    alpha:         float = 0.97,   
    gamma:         float = 0.10,  
    f_min:         float = 0.0,   
    f_max:         float = 2.0,   
    A0:            float = 1.0,    
    r0:            float = 1.0,   
    mutation_rate: float = 0.02,   
    seed:          Optional[int] = None,
) -> "KnapsackResult":
        t_start = time.perf_counter()
    rng     = _random.Random(seed)
    n       = problem.n
 
    # ── Khởi tạo quần thể ─────────────────────────────────────
    init_positions = _init_population(n_bats, problem, rng)
    bats: List[_Bat] = []
    for pos in init_positions:
        val, wt = _evaluate(pos, problem)
        bats.append(_Bat(
            position   = pos[:],
            value      = val,
            weight     = wt,
            loudness   = A0,
            pulse_rate = 0.0,   # r bắt đầu từ 0, tăng dần theo công thức
        ))
 
    # ── Best solution ──────────────────────────────────────────
    best_bat  = max(bats, key=lambda b: b.value)
    best_pos  = best_bat.position[:]
    best_val  = best_bat.value
    best_wt   = best_bat.weight
 
    convergence: List[float] = []   # lưu best_val mỗi iter cho chart
 
    # ══════════════════════════════════════════════════════════
    #  VÒNG LẶP CHÍNH
    # ══════════════════════════════════════════════════════════
    for t in range(1, max_iter + 1):
 
        # ── FIX 2: Cập nhật A và r TRƯỚC, NGOÀI mọi if ───────
        #    (đúng theo Listing 4.2 dòng 4-5)
        A_global = A0 * (alpha ** t)                      # A = alpha^t * A0
        r_global = r0 * (1.0 - math.exp(-gamma * t))     # r tăng dần → 1
 
        for bat in bats:
            bat.loudness   = A_global
            bat.pulse_rate = r_global
 
        # ── Với mỗi dơi ───────────────────────────────────────
        for bat in bats:
 
            # a. Frequency tuning
            beta = rng.random()
            fi   = f_min + beta * (f_max - f_min)
 
            # b. Velocity update (continuous)
            #    v_i(t+1) = v_i(t) + (x_i - x*) * fi
            for j in range(n):
                bat.velocity[j] += (bat.position[j] - best_pos[j]) * fi
 
            # c. Position update bằng V-shaped transfer

            new_pos = [_flip_bit(bat.position[j], bat.velocity[j], rng)
                       for j in range(n)]
 
            # d. Local search quanh best nếu rand > r
           
            if rng.random() > bat.pulse_rate:
             
                new_pos = best_pos[:]
                sigma   = 0.1 * A_global   
                n_flip  = max(1, int(sigma * n))
                flip_idx = rng.sample(range(n), min(n_flip, n))
                for j in flip_idx:
                    new_pos[j] = 1 - new_pos[j]
 
            # e. Repair (đảm bảo feasible + greedy fill)
            new_pos = _repair(new_pos, problem, rng)
 
            # f. Đánh giá
            new_val, new_wt = _evaluate(new_pos, problem)
 
            # g.
            #    Chấp nhận nếu nghiệm MỚI TỐT HƠN và rand > A
          
           
            if new_val >= bat.value and rng.random() > A_global:
                bat.position = new_pos
                bat.value    = new_val
                bat.weight   = new_wt
 
            # h. Cập nhật best toàn cục
            if new_val > best_val:
                best_pos = new_pos[:]
                best_val = new_val
                best_wt  = new_wt
 
        # ── Đột biến nhỏ tránh stagnation ─────────────────────
        # Mỗi vòng lặp, flip ngẫu nhiên 1 bit của 1 dơi random
        if mutation_rate > 0 and rng.random() < mutation_rate * n_bats:
            target = rng.choice(bats)
            j      = rng.randrange(n)
            mut    = target.position[:]
            mut[j] = 1 - mut[j]
            mut    = _repair(mut, problem, rng)
            mv, mw = _evaluate(mut, problem)
            if mv >= target.value:
                target.position = mut
                target.value    = mv
                target.weight   = mw
                if mv > best_val:
                    best_pos = mut[:]
                    best_val = mv
                    best_wt  = mw
 
        convergence.append(float(best_val))
 
    # ── Kết quả ───────────────────────────────────────────────
    elapsed = time.perf_counter() - t_start
 
    result = KnapsackResult(
        solution=best_pos,
        value=best_val,
        weight=best_wt,
        time=elapsed)
    result.algorithm = "Bat Algorithm"
    result.solution  = best_pos
    result.value     = best_val
    result.weight    = best_wt
    result.time      = elapsed
    result.extra     = {
        "convergence":  convergence,
        "n_bats":       n_bats,
        "max_iter":     max_iter,
        "alpha":        alpha,
        "gamma":        gamma,
        "f_min":        f_min,
        "f_max":        f_max,
        "final_A":      A_global,
        "final_r":      r_global,
    }
    return result

