"""Synthetic reward landscapes for validating the search (selected with FAKE_REWARD_LANDSCAPE=0|1|2).

Every landscape has a known optimum of 0.90 (+/- a deterministic 0.01 jitter that breaks ties):
  0 "trend"    : separable, monotone. best finger = 3 servos, g2=10, g1=1, wedged tip.
  1 "coupling" : reversed preferences (2 servos, g2=1, thinner) plus a +0.10 bonus when all active
                 fingers share the same servo count (an interaction the value net must pick up).
  2 "outlier"  : the general trend favours 3-servo/standard-tip fingers (trend optimum 0.80), but an
                 exception family (2 servos, g2>=7, thinner or wedged) earns a bonus growing with the
                 SQUARED fraction of such fingers: r(k of 5) = 0.80, 0.76, 0.75, 0.77, 0.82, 0.90.
                 The global optimum (0.90) sits across a valley from the trend optimum.
"""
import hashlib, os, random

def landscape() -> str:
    return os.environ.get("FAKE_REWARD_LANDSCAPE", "0")

def _decode(code):
    s = len(code); g2 = int(code[-2]) + 1; g1 = int(code[0]) + 1 if s == 3 else None
    return s, g1, g2

def finger_score(code: str, tip: str, L: str = None) -> float:
    L = L or landscape()
    if not code:
        return 0.0
    s, g1, g2 = _decode(code)
    if L == "1":
        sc = 0.4 * (s == 2) + 0.3 * (1.0 - (g2 - 1) / 9.0)
        sc += 0.1 * (1.0 - (g1 - 1) / 9.0) if s == 3 else 0.1
        sc += {"thinner": 0.2, "standard": 0.1, "wedged": 0.0}[tip]
        return sc
    sc = (0.35 if s == 3 else 0.15) + 0.35 * (g2 / 10.0)   # same trend as the original harness landscape
    sc += 0.15 * (1.0 - (g1 - 1) / 9.0) if s == 3 else 0.075
    if L == "2":
        sc += {"standard": 0.15, "thinner": 0.08, "wedged": 0.0}[tip]
    else:
        sc += {"wedged": 0.15, "thinner": 0.08, "standard": 0.0}[tip]
    return sc

def is_exception_finger(code: str, tip: str) -> bool:
    if not code: return False
    s, _, g2 = _decode(code)
    return s == 2 and g2 >= 7 and tip in ("thinner", "wedged")

def in_exception_family(codes, tips) -> bool:
    active = [(c, t) for c, t in zip(codes, tips) if c]
    return bool(active) and all(is_exception_finger(c, t) for c, t in active)

def hand_reward(codes, tips, group: str = "", L: str = None) -> float:
    L = L or landscape()
    active = [(c, t) for c, t in zip(codes, tips) if c]
    if not active:
        return 0.0
    base = sum(finger_score(c, t, L) for c, t in active) / len(active)
    if L == "0":
        r = 0.9 * base
    elif L == "1":
        same = len({len(c) for c, _ in active}) == 1
        r = 0.8 * base + (0.1 if same else 0.0)
    else:
        k = sum(is_exception_finger(c, t) for c, t in active) / len(active)
        r = 0.8 * base + 0.376 * k * k
    h = int(hashlib.sha1((",".join(codes) + "|" + ",".join(tips) + "|" + group).encode()).hexdigest()[:8], 16)
    jitter = ((h % 2001) - 1000) / 1000.0 * 0.01
    return max(0.0, min(r + jitter, 0.999))

OPTIMUM = 0.90   # by construction, for every landscape (before jitter)

def random_design(rng):
    codes, tips = [], []
    for _ in range(5):
        s = rng.choice([2, 3]); g1 = rng.randint(0, 9); g2 = rng.randint(0, 9)
        codes.append(f"{g1}{g2}0" if s == 3 else f"{g2}0"); tips.append(rng.choice(["standard", "thinner", "wedged"]))
    return codes, tips

def random_baseline(L: str, n: int = 20000, seed: int = 0):
    rng = random.Random(seed); vals = sorted(hand_reward(*random_design(rng), "", L) for _ in range(n))
    return {"mean": sum(vals) / n, "p99": vals[int(0.99 * n)], "max": vals[-1]}
