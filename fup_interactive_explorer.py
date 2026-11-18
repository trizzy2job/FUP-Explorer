#!/usr/bin/env python3
"""
Interactive isotropic / anisotropic high-dimensional FUP exponent explorer.

In anisotropic mode the app sweeps all diagonal scaling vectors

    (N_1,...,N_d),   min_base <= N_i <= max_base,

up to simultaneous coordinate permutation.  Thus only nondecreasing scaling
vectors are computed, and all genuinely anisotropic vectors (at least two
distinct entries) are plotted on the same graph.

The app studies ordered digit systems for a diagonal dilation

    S = diag(N_1,...,N_d)

with alphabets

    A, B subset prod_i {0,...,N_i-1}.

At stage k,

    C_k(A) = {a_0 + S a_1 + ... + S^(k-1) a_(k-1) : a_j in A}
             subset prod_i (Z/N_i^k Z).

The unitary restricted Fourier matrix is

    T_k(B,A)[xi,x]
      = (prod_i N_i)^(-k/2)
        exp(-2*pi*i * sum_i x_i xi_i / N_i^k).

To make isotropic and anisotropic plots directly comparable, define the
normalized density exponent

    delta_A = d log|A| / log(prod_i N_i)

and the finite-stage exponent using the effective linear base

    M_eff = (prod_i N_i)^(1/d),
    beta_k = -log ||T_k|| / (k log M_eff).

Then the subcritical / critical condition is

    delta_A + delta_B <= d,

and the Hilbert--Schmidt exponent is always

    beta_HS = d/2 - (delta_A + delta_B)/2.

In isotropic mode N_1=...=N_d=M, these formulas reduce exactly to the
previous app's conventions.

EXACT REDUCTIONS
----------------
1. Ordinary translation reduction.  If A+t remains in the digit box without
   wraparound, C_k(A+t) is an ordinary torus translate of C_k(A), so beta_k
   is unchanged.  The app keeps the representative with coordinatewise
   minimum zero.

2. Pair-swap reduction.  The Fourier kernel is symmetric, so T_k(A,B) is the
   transpose of T_k(B,A); only unordered alphabet pairs are evaluated.

RIGIDITY MARKER
---------------
In isotropic mode, failure to improve the Hilbert--Schmidt exponent is tied to
Euclidean orthogonality of span(A-A) and span(B-B).  For unequal diagonal
scalings the correct persistent rank-one obstruction is stronger: for each
block of coordinates having the same dilation N_i, the projected difference
spans must be orthogonal.  The app calls this the blockwise HS-rigidity
obstruction and can mark those pairs separately.

CSV import/export, hover, click-to-pin, overlap cycling, PNG export, and the
Windows PyInstaller workflow are retained.  Every CSV row records the exact
scaling vector that generated it.
"""

from __future__ import annotations

import csv
import itertools
import math
import os
import queue
import random
import subprocess
import sys
import threading
import traceback
import textwrap
from dataclasses import dataclass, fields, replace
from typing import Iterable, Sequence

import numpy as np

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

import tkinter as tk
from tkinter import filedialog, messagebox, ttk


# ----------------------------- Data structures ----------------------------- #


@dataclass(frozen=True)
class AlphabetRecord:
    bases: tuple[int, ...]
    alphabet: tuple[int, ...]  # indices in lexicographically ordered digit box
    points: np.ndarray         # shape (m,d)
    cardinality: int
    delta: float
    enumeration: str


@dataclass
class Result:
    # ``base`` is retained for backward-compatible CSV import/export.  It is
    # the isotropic base M in isotropic mode and 0 in anisotropic mode.
    base: int
    dimension: int
    stage: int
    card_A: int
    card_B: int
    delta_A: float
    delta_B: float
    delta_bar: float
    beta_hs: float
    beta: float
    beta_gain: float
    norm: float
    orthogonal_digit_spans: bool
    alphabet_A: str
    alphabet_B: str
    enumeration_A: str
    enumeration_B: str
    mode: str = "isotropic"
    bases: str = ""
    product_base: int = 0
    effective_base: float = 0.0
    hs_rigidity_obstruction: bool = False


@dataclass(frozen=True)
class RunConfig:
    mode: str
    dimension: int
    min_base: int
    max_base: int
    anisotropic_bases: tuple[int, ...]
    max_cardinality: int
    stage: int
    max_alphabets_per_size: int
    max_pairs_per_base: int
    max_matrix_entries: int
    seed: int
    exact_alphabets: bool
    exact_pairs: bool
    mark_orthogonal: bool


# ---------------------------- Mathematical core ---------------------------- #


def digit_box(bases: Sequence[int]) -> np.ndarray:
    """Digit vectors in prod_i {0,...,N_i-1}, lexicographically ordered."""
    return np.array(list(itertools.product(*(range(int(N)) for N in bases))), dtype=np.int64)


def bases_string(bases: Sequence[int]) -> str:
    return "x".join(str(int(N)) for N in bases)


def parse_bases_string(text: str) -> tuple[int, ...]:
    cleaned = text.strip().lower().replace("×", "x").replace(",", "x").replace(" ", "")
    parts = [p for p in cleaned.split("x") if p]
    if not parts:
        raise ValueError("Enter anisotropic bases such as 2,3,4.")
    vals = tuple(int(p) for p in parts)
    if any(N < 2 for N in vals):
        raise ValueError("Every anisotropic base N_i must be at least 2.")
    return vals


def canonical_translation_points(points: np.ndarray) -> np.ndarray:
    """Anchor an alphabet by subtracting its coordinatewise minimum."""
    return points - points.min(axis=0, keepdims=True)


def canonical_translation_combo(
    combo: tuple[int, ...],
    box: np.ndarray,
    index_of: dict[tuple[int, ...], int],
) -> tuple[int, ...]:
    pts = box[np.asarray(combo, dtype=int)]
    canon = canonical_translation_points(pts)
    return tuple(sorted(index_of[tuple(int(x) for x in p)] for p in canon))


def is_translation_canonical(points: np.ndarray) -> bool:
    return bool(np.all(points.min(axis=0) == 0))


def cantor_stage(alphabet_points: np.ndarray, bases: Sequence[int], k: int) -> np.ndarray:
    """Construct C_k(A) for S=diag(bases)."""
    bases_arr = np.asarray(tuple(int(N) for N in bases), dtype=np.int64)
    d = alphabet_points.shape[1]
    if bases_arr.shape != (d,):
        raise ValueError("Number of bases must equal the ambient dimension.")
    pts = np.zeros((1, d), dtype=np.int64)
    scale = np.ones(d, dtype=np.int64)
    for _ in range(k):
        pts = (pts[:, None, :] + scale[None, None, :] * alphabet_points[None, :, :]).reshape(-1, d)
        scale = scale * bases_arr
    return pts


def restricted_fourier_matrix(
    C: np.ndarray,
    D: np.ndarray,
    bases: Sequence[int],
    k: int,
) -> np.ndarray:
    """Dense restricted unitary Fourier matrix for a diagonal product group."""
    bases_arr = np.asarray(tuple(int(N) for N in bases), dtype=np.float64)
    moduli = bases_arr ** int(k)
    # sum_i D_i C_i / N_i^k = (D / moduli) C^T
    phases = (D.astype(np.float64) / moduli[None, :]) @ C.astype(np.float64).T
    log_normalization = -0.5 * int(k) * sum(math.log(int(N)) for N in bases)
    normalization = math.exp(log_normalization)
    return normalization * np.exp(-2j * np.pi * phases)


def op_norm_dense(A: np.ndarray) -> float:
    """Largest singular value."""
    values = np.linalg.svd(A, compute_uv=False)
    return float(values[0])


def density_exponent(cardinality: int, bases: Sequence[int]) -> float:
    """d-normalized density exponent: d log m / log(prod_i N_i)."""
    d = len(tuple(bases))
    if cardinality <= 1:
        return 0.0
    return d * math.log(cardinality) / sum(math.log(int(N)) for N in bases)


def beta_from_norm(norm: float, bases: Sequence[int], k: int) -> float:
    """Exponent relative to M_eff^k where M_eff=(prod_i N_i)^(1/d)."""
    d = len(tuple(bases))
    if norm <= 1.0 + 1e-10:
        norm = min(norm, 1.0)
    norm = max(norm, np.finfo(float).tiny)
    denominator = (int(k) / d) * sum(math.log(int(N)) for N in bases)
    return -math.log(norm) / denominator


def alphabet_to_string(points: np.ndarray) -> str:
    return "{" + ", ".join("(" + ",".join(map(str, p)) + ")" for p in points.tolist()) + "}"


def orthogonal_difference_spans(A: np.ndarray, B: np.ndarray) -> bool:
    """Exact Euclidean test: span(A-A) perpendicular to span(B-B)."""
    DA = A - A[0]
    DB = B - B[0]
    return bool(np.all(DA @ DB.T == 0))


def blockwise_rigidity_obstruction(A: np.ndarray, B: np.ndarray, bases: Sequence[int]) -> bool:
    """
    Persistent rank-one obstruction for diagonal dilation.

    Coordinates are grouped by equal N_i.  In each equal-dilation block, the
    projected difference spans must be Euclidean orthogonal.  When all N_i are
    equal this is exactly ordinary Euclidean orthogonality.
    """
    DA = A - A[0]
    DB = B - B[0]
    groups: dict[int, list[int]] = {}
    for i, N in enumerate(bases):
        groups.setdefault(int(N), []).append(i)
    for inds in groups.values():
        if not np.all(DA[:, inds] @ DB[:, inds].T == 0):
            return False
    return True


def generate_alphabets_for_bases(
    bases: tuple[int, ...],
    max_cardinality: int,
    max_alphabets_per_size: int,
    exact: bool,
    rng: random.Random,
    cancel_event: threading.Event,
) -> tuple[list[AlphabetRecord], int]:
    """Generate ordinary-translation-reduced alphabets for one diagonal base vector."""
    d = len(bases)
    box = digit_box(bases)
    q = len(box)
    index_of = {tuple(int(x) for x in p): i for i, p in enumerate(box)}
    card_cap = max(1, min(max_cardinality, q))

    records: list[AlphabetRecord] = []
    removed = 0

    for m in range(1, card_cap + 1):
        if cancel_event.is_set():
            break
        total = math.comb(q, m)
        delta = density_exponent(m, bases)

        if exact or total <= max_alphabets_per_size:
            tag = "exact/translation-reduced"
            for combo in itertools.combinations(range(q), m):
                if cancel_event.is_set():
                    break
                pts = box[np.asarray(combo, dtype=int)]
                if not is_translation_canonical(pts):
                    removed += 1
                    continue
                records.append(AlphabetRecord(bases, combo, pts, m, delta, tag))
        else:
            target = max_alphabets_per_size
            seen: set[tuple[int, ...]] = set()
            attempts = 0
            max_attempts = max(3000, 80 * target)

            while len(seen) < target and attempts < max_attempts:
                if cancel_event.is_set():
                    break
                attempts += 1
                combo = tuple(sorted(rng.sample(range(q), m)))
                canon = canonical_translation_combo(combo, box, index_of)
                if canon in seen:
                    removed += 1
                    continue
                seen.add(canon)

            for combo in sorted(seen):
                pts = box[np.asarray(combo, dtype=int)]
                records.append(
                    AlphabetRecord(bases, combo, pts, m, delta, "sampled/translation-reduced")
                )

    return records, removed


def choose_unordered_pairs(
    alphabets: list[AlphabetRecord],
    d: int,
    max_pairs: int,
    exact: bool,
    rng: random.Random,
    cancel_event: threading.Event,
) -> list[tuple[AlphabetRecord, AlphabetRecord]]:
    """Choose unordered subcritical/critical alphabet pairs."""
    n = len(alphabets)
    upper_count = n * (n + 1) // 2
    if exact or upper_count <= max_pairs:
        out: list[tuple[AlphabetRecord, AlphabetRecord]] = []
        for i, A in enumerate(alphabets):
            if cancel_event.is_set():
                break
            for j in range(i, n):
                B = alphabets[j]
                if A.delta + B.delta <= d + 1e-12:
                    out.append((A, B))
        if exact or len(out) <= max_pairs:
            return out
        rng.shuffle(out)
        return out[:max_pairs]

    seen: set[tuple[int, int]] = set()
    out = []
    attempts = 0
    max_attempts = max(5000, 150 * max_pairs)
    while len(out) < max_pairs and attempts < max_attempts:
        if cancel_event.is_set():
            break
        attempts += 1
        i = rng.randrange(n)
        j = rng.randrange(n)
        if i > j:
            i, j = j, i
        if (i, j) in seen:
            continue
        seen.add((i, j))
        A, B = alphabets[i], alphabets[j]
        if A.delta + B.delta <= d + 1e-12:
            out.append((A, B))
    return out


def evaluate_pair(
    A: AlphabetRecord,
    B: AlphabetRecord,
    d: int,
    k: int,
    max_matrix_entries: int,
    stage_cache: dict[tuple[int, ...], np.ndarray],
    mode: str,
) -> Result | None:
    nA = A.cardinality ** k
    nB = B.cardinality ** k
    if nA * nB > max_matrix_entries:
        return None

    bases = A.bases
    P = math.prod(bases)

    C = stage_cache.get(A.alphabet)
    if C is None:
        C = cantor_stage(A.points, bases, k)
        stage_cache[A.alphabet] = C

    D = stage_cache.get(B.alphabet)
    if D is None:
        D = cantor_stage(B.points, bases, k)
        stage_cache[B.alphabet] = D

    T = restricted_fourier_matrix(C, D, bases, k)
    norm = op_norm_dense(T)
    beta = beta_from_norm(norm, bases, k)

    delta_bar = 0.5 * (A.delta + B.delta)
    beta_hs = 0.5 * (d - A.delta - B.delta)
    isotropic_base = bases[0] if len(set(bases)) == 1 else 0

    return Result(
        base=isotropic_base,
        dimension=d,
        stage=k,
        card_A=A.cardinality,
        card_B=B.cardinality,
        delta_A=A.delta,
        delta_B=B.delta,
        delta_bar=delta_bar,
        beta_hs=beta_hs,
        beta=beta,
        beta_gain=beta - beta_hs,
        norm=norm,
        orthogonal_digit_spans=orthogonal_difference_spans(A.points, B.points),
        alphabet_A=alphabet_to_string(A.points),
        alphabet_B=alphabet_to_string(B.points),
        enumeration_A=A.enumeration,
        enumeration_B=B.enumeration,
        mode=mode,
        bases=bases_string(bases),
        product_base=P,
        effective_base=P ** (1.0 / d),
        hs_rigidity_obstruction=blockwise_rigidity_obstruction(A.points, B.points, bases),
    )


def run_experiment(
    cfg: RunConfig,
    cancel_event: threading.Event,
    emit,
) -> list[Result]:
    rng = random.Random(cfg.seed)
    results: list[Result] = []
    skipped_matrix = 0

    if cfg.mode == "anisotropic":
        # Sweeping only nondecreasing tuples removes the exact redundancy under
        # simultaneous coordinate permutation.  Since every alphabet is
        # enumerated for each scaling vector, permuting coordinates produces
        # the same collection of beta-values.  All-equal tuples are excluded
        # here because they belong to the isotropic mode.
        base_vectors = [
            tuple(v)
            for v in itertools.combinations_with_replacement(
                range(cfg.min_base, cfg.max_base + 1), cfg.dimension
            )
            if len(set(v)) > 1
        ]
        emit(
            "log",
            f"Anisotropic sweep: {len(base_vectors)} inequivalent scaling vectors "
            f"with {cfg.min_base} <= N_i <= {cfg.max_base}, modulo coordinate permutations."
        )
    else:
        base_vectors = [tuple([M] * cfg.dimension) for M in range(cfg.min_base, cfg.max_base + 1)]

    for bases in base_vectors:
        if cancel_event.is_set():
            break
        label = bases_string(bases)
        emit("status", f"Generating translation classes for N=({label}) ...")
        alphabets, removed = generate_alphabets_for_bases(
            bases=bases,
            max_cardinality=cfg.max_cardinality,
            max_alphabets_per_size=cfg.max_alphabets_per_size,
            exact=cfg.exact_alphabets,
            rng=rng,
            cancel_event=cancel_event,
        )
        if cancel_event.is_set():
            break

        pairs = choose_unordered_pairs(
            alphabets=alphabets,
            d=cfg.dimension,
            max_pairs=cfg.max_pairs_per_base,
            exact=cfg.exact_pairs,
            rng=rng,
            cancel_event=cancel_event,
        )

        emit(
            "log",
            f"N=({label}): {len(alphabets)} translation classes; "
            f"{removed} translation/sample duplicates removed; "
            f"{len(pairs)} unordered subcritical/critical pairs."
        )

        stage_cache: dict[tuple[int, ...], np.ndarray] = {}
        for idx, (A, B) in enumerate(pairs, start=1):
            if cancel_event.is_set():
                break
            res = evaluate_pair(
                A=A,
                B=B,
                d=cfg.dimension,
                k=cfg.stage,
                max_matrix_entries=cfg.max_matrix_entries,
                stage_cache=stage_cache,
                mode=cfg.mode,
            )
            if res is None:
                skipped_matrix += 1
            else:
                results.append(res)

            if idx == 1 or idx % 25 == 0 or idx == len(pairs):
                emit("status", f"N=({label}): evaluated {idx}/{len(pairs)} pairs; total points {len(results)}")

    if skipped_matrix:
        emit(
            "log",
            f"Skipped {skipped_matrix} pairs because |C_k||D_k| exceeded "
            f"the matrix-entry limit {cfg.max_matrix_entries:,}."
        )

    return results


# ------------------------------- GUI helpers ------------------------------- #


def _wrapped_alphabet(label: str, alphabet: str, width: int = 78) -> str:
    return textwrap.fill(
        f"{label} = {alphabet}",
        width=width,
        subsequent_indent="    ",
        break_long_words=False,
        break_on_hyphens=False,
    )


def result_tooltip(r: Result) -> str:
    base_text = r.bases if r.bases else (str(r.base) if r.base else "?")
    return (
        f"mode={r.mode}, bases=({base_text}), d={r.dimension}, k={r.stage}\n"
        f"{_wrapped_alphabet('A', r.alphabet_A)}\n"
        f"{_wrapped_alphabet('B', r.alphabet_B)}\n"
        f"|A|={r.card_A}, |B|={r.card_B}\n"
        f"delta_A={r.delta_A:.8f}, delta_B={r.delta_B:.8f}\n"
        f"delta_bar={r.delta_bar:.8f}\n"
        f"beta_k={r.beta:.10f}\n"
        f"beta_HS={r.beta_hs:.10f}\n"
        f"gain={r.beta_gain:+.10f}\n"
        f"||T_k||={r.norm:.12g}\n"
        f"Euclidean orthogonal spans: {'yes' if r.orthogonal_digit_spans else 'no'}\n"
        f"blockwise HS-rigidity obstruction: {'yes' if r.hs_rigidity_obstruction else 'no'}"
    )


def write_csv(results: Sequence[Result], filename: str) -> None:
    names = [f.name for f in fields(Result)]
    with open(filename, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=names)
        writer.writeheader()
        for r in results:
            writer.writerow({name: getattr(r, name) for name in names})


def _parse_bool(value: object) -> bool:
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "t"}:
        return True
    if text in {"0", "false", "no", "n", "f", ""}:
        return False
    raise ValueError(f"Cannot parse boolean value {value!r}.")


def _first_present(row: dict[str, str], *names: str, default=None):
    for name in names:
        if name in row and str(row[name]).strip() != "":
            return row[name]
    return default


def read_results_csv(filename: str) -> list[Result]:
    """Load app CSVs, including older isotropic CSVs, without recomputation."""
    out: list[Result] = []
    with open(filename, "r", newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None:
            raise ValueError("The CSV has no header row.")

        required_groups = {
            "dimension": ("dimension",),
            "stage": ("stage",),
            "delta_A": ("delta_A",),
            "delta_B": ("delta_B",),
            "beta": ("beta", "beta_standard"),
            "alphabet_A": ("alphabet_A",),
            "alphabet_B": ("alphabet_B",),
        }
        missing = [
            label for label, aliases in required_groups.items()
            if not any(a in reader.fieldnames for a in aliases)
        ]
        if missing:
            raise ValueError(
                "CSV is missing required data: " + ", ".join(missing) + ".\n\n"
                "Expected this app's CSV format or the earlier standard-pairs CSV format."
            )

        for line_no, row in enumerate(reader, start=2):
            try:
                dimension = int(_first_present(row, "dimension"))
                stage = int(_first_present(row, "stage"))
                delta_A = float(_first_present(row, "delta_A"))
                delta_B = float(_first_present(row, "delta_B"))
                beta = float(_first_present(row, "beta", "beta_standard"))

                mode = str(_first_present(row, "mode", default="isotropic")).strip().lower()
                base_raw = _first_present(row, "base", default="0")
                base = int(float(base_raw)) if str(base_raw).strip() else 0
                bases_raw = str(_first_present(row, "bases", default="")).strip()
                if bases_raw:
                    bases_tuple = parse_bases_string(bases_raw)
                elif base >= 2:
                    bases_tuple = tuple([base] * dimension)
                    bases_raw = bases_string(bases_tuple)
                    mode = "isotropic"
                else:
                    raise ValueError("An anisotropic CSV row must include the 'bases' column.")

                if len(bases_tuple) != dimension:
                    raise ValueError(
                        f"base vector {bases_tuple} has length {len(bases_tuple)}, expected dimension {dimension}"
                    )
                P = math.prod(bases_tuple)
                effective_base = P ** (1.0 / dimension)

                delta_bar_raw = _first_present(row, "delta_bar", "x_density")
                delta_bar = float(delta_bar_raw) if delta_bar_raw is not None else 0.5 * (delta_A + delta_B)

                beta_hs_raw = _first_present(row, "beta_hs")
                beta_hs = float(beta_hs_raw) if beta_hs_raw is not None else 0.5 * (dimension - delta_A - delta_B)

                beta_gain_raw = _first_present(row, "beta_gain")
                beta_gain = float(beta_gain_raw) if beta_gain_raw is not None else beta - beta_hs

                norm_raw = _first_present(row, "norm", "norm_standard")
                if norm_raw is None:
                    norm = math.exp(-beta * stage * math.log(effective_base))
                else:
                    norm = float(norm_raw)

                card_A_raw = _first_present(row, "card_A")
                card_B_raw = _first_present(row, "card_B")
                card_A = int(card_A_raw) if card_A_raw is not None else -1
                card_B = int(card_B_raw) if card_B_raw is not None else -1

                ortho_raw = _first_present(row, "orthogonal_digit_spans", default="false")
                orthogonal = _parse_bool(ortho_raw)
                rigid_raw = _first_present(row, "hs_rigidity_obstruction", default=str(orthogonal).lower())
                rigid = _parse_bool(rigid_raw)

                out.append(
                    Result(
                        base=base,
                        dimension=dimension,
                        stage=stage,
                        card_A=card_A,
                        card_B=card_B,
                        delta_A=delta_A,
                        delta_B=delta_B,
                        delta_bar=delta_bar,
                        beta_hs=beta_hs,
                        beta=beta,
                        beta_gain=beta_gain,
                        norm=norm,
                        orthogonal_digit_spans=orthogonal,
                        alphabet_A=str(_first_present(row, "alphabet_A", default="")),
                        alphabet_B=str(_first_present(row, "alphabet_B", default="")),
                        enumeration_A=str(_first_present(row, "enumeration_A", default="imported")),
                        enumeration_B=str(_first_present(row, "enumeration_B", default="imported")),
                        mode=mode,
                        bases=bases_string(bases_tuple),
                        product_base=int(_first_present(row, "product_base", default=P)),
                        effective_base=float(_first_present(row, "effective_base", default=effective_base)),
                        hs_rigidity_obstruction=rigid,
                    )
                )
            except Exception as exc:
                raise ValueError(f"Could not parse CSV line {line_no}: {exc}") from exc

    if not out:
        raise ValueError("The CSV contains no data rows.")

    dimensions = {r.dimension for r in out}
    stages = {r.stage for r in out}
    if len(dimensions) != 1:
        raise ValueError(f"Interactive plotting requires one ambient dimension per CSV; found {sorted(dimensions)}.")
    if len(stages) != 1:
        raise ValueError(f"Interactive plotting requires one stage k per CSV; found {sorted(stages)}.")
    return out


class FUPExplorer(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("FUP Exponent Explorer")
        self.geometry("1220x820")
        self.minsize(940, 650)

        self.results: list[Result] = []
        self.scatter_map: dict[object, list[Result]] = {}
        self.worker: threading.Thread | None = None
        self.cancel_event = threading.Event()
        self.msg_queue: queue.Queue = queue.Queue()
        self.pinned_result: Result | None = None
        self.current_plot_cfg: RunConfig | None = None
        self._cycle_signature = None
        self._cycle_index = 0

        self._build_ui()
        self.after(100, self._poll_queue)

    def _build_ui(self) -> None:
        outer = ttk.Frame(self, padding=8)
        outer.pack(fill=tk.BOTH, expand=True)

        controls = ttk.LabelFrame(outer, text="Parameters", padding=8)
        controls.pack(side=tk.TOP, fill=tk.X)

        self.vars: dict[str, tk.Variable] = {
            "mode": tk.StringVar(value="isotropic"),
            "dimension": tk.IntVar(value=2),
            "min_base": tk.IntVar(value=2),
            "max_base": tk.IntVar(value=4),
            "anisotropic_bases": tk.StringVar(value=""),  # retained for legacy CSV import only
            "max_cardinality": tk.IntVar(value=4),
            "stage": tk.IntVar(value=3),
            "max_alphabets": tk.IntVar(value=300),
            "max_pairs": tk.IntVar(value=1200),
            "max_entries": tk.IntVar(value=2_000_000),
            "seed": tk.IntVar(value=7),
            "exact_alphabets": tk.BooleanVar(value=False),
            "exact_pairs": tk.BooleanVar(value=False),
            "mark_orthogonal": tk.BooleanVar(value=True),
        }

        self.param_widgets: dict[str, tk.Widget] = {}

        # First row: mode and geometric parameters.
        block = ttk.Frame(controls)
        block.grid(row=0, column=0, padx=4, pady=2, sticky="w")
        ttk.Label(block, text="Scaling mode").pack(anchor="w")
        mode_box = ttk.Combobox(
            block, textvariable=self.vars["mode"], values=("isotropic", "anisotropic"),
            width=11, state="readonly"
        )
        mode_box.pack(anchor="w")
        mode_box.bind("<<ComboboxSelected>>", lambda _e: self._toggle_mode_controls())

        entries = [
            ("Dimension d", "dimension", 4),
            ("Min scaling", "min_base", 5),
            ("Max scaling N", "max_base", 6),
            ("Max |A|,|B|", "max_cardinality", 5),
            ("Stage k", "stage", 4),
            ("Max alphabets / size", "max_alphabets", 7),
            ("Max pairs / base", "max_pairs", 7),
            ("Max matrix entries", "max_entries", 10),
            ("Random seed", "seed", 6),
        ]
        for col, (label, key, width) in enumerate(entries, start=1):
            sub = ttk.Frame(controls)
            sub.grid(row=0, column=col, padx=4, pady=2, sticky="w")
            ttk.Label(sub, text=label).pack(anchor="w")
            ent = ttk.Entry(sub, textvariable=self.vars[key], width=width)
            ent.pack(anchor="w")
            self.param_widgets[key] = ent

        checks = ttk.Frame(controls)
        checks.grid(row=1, column=0, columnspan=len(entries) + 1, sticky="w", pady=(6, 2))
        ttk.Checkbutton(checks, text="Exact alphabet enumeration", variable=self.vars["exact_alphabets"]).pack(side=tk.LEFT, padx=(0, 12))
        ttk.Checkbutton(checks, text="Exact unordered pairs", variable=self.vars["exact_pairs"]).pack(side=tk.LEFT, padx=(0, 12))
        ttk.Checkbutton(
            checks,
            text="Mark HS-rigidity obstruction pairs",
            variable=self.vars["mark_orthogonal"],
            command=self._redraw_if_results,
        ).pack(side=tk.LEFT, padx=(0, 12))

        buttons = ttk.Frame(controls)
        buttons.grid(row=2, column=0, columnspan=len(entries) + 1, sticky="w", pady=(6, 0))
        self.run_button = ttk.Button(buttons, text="Generate", command=self._start_run)
        self.run_button.pack(side=tk.LEFT, padx=(0, 6))
        self.cancel_button = ttk.Button(buttons, text="Cancel", command=self._cancel_run, state=tk.DISABLED)
        self.cancel_button.pack(side=tk.LEFT, padx=(0, 6))
        self.save_png_button = ttk.Button(buttons, text="Save PNG", command=self._save_png, state=tk.DISABLED)
        self.save_png_button.pack(side=tk.LEFT, padx=(0, 6))
        self.save_csv_button = ttk.Button(buttons, text="Save CSV", command=self._save_csv, state=tk.DISABLED)
        self.save_csv_button.pack(side=tk.LEFT, padx=(0, 6))
        self.import_csv_button = ttk.Button(buttons, text="Import CSV", command=self._import_csv)
        self.import_csv_button.pack(side=tk.LEFT, padx=(0, 6))

        self.progress = ttk.Progressbar(buttons, mode="indeterminate", length=180)
        self.progress.pack(side=tk.LEFT, padx=(12, 8))
        self.status_var = tk.StringVar(value="Ready.")
        ttk.Label(buttons, textvariable=self.status_var).pack(side=tk.LEFT, padx=4)

        note = ttk.Label(
            controls,
            text=("Anisotropic mode sweeps every inequivalent nondecreasing scaling vector "
                  "(N_1,...,N_d) with Min scaling <= N_i <= Max scaling N, excluding all-equal tuples. "
                  "The plot uses M_eff=(N_1...N_d)^(1/d), so the HS line stays beta=d/2-delta_bar."),
        )
        note.grid(row=3, column=0, columnspan=len(entries) + 1, sticky="w", pady=(5, 0))

        middle = ttk.Panedwindow(outer, orient=tk.HORIZONTAL)
        middle.pack(fill=tk.BOTH, expand=True, pady=(8, 0))

        plot_frame = ttk.Frame(middle)
        info_frame = ttk.Frame(middle, width=330)
        middle.add(plot_frame, weight=4)
        middle.add(info_frame, weight=1)

        self.figure = Figure(figsize=(8.5, 6.2), dpi=100)
        self.ax = self.figure.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.figure, master=plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        toolbar = NavigationToolbar2Tk(self.canvas, plot_frame, pack_toolbar=False)
        toolbar.update()
        toolbar.pack(fill=tk.X)

        self.annotation = self.ax.annotate(
            "", xy=(0, 0), xytext=(15, 15), textcoords="offset points",
            bbox=dict(boxstyle="round,pad=0.45", fc="white", ec="0.35", alpha=0.96),
            arrowprops=dict(arrowstyle="->", color="0.35"), fontsize=8.5, zorder=20,
        )
        self.annotation.set_visible(False)
        self.canvas.mpl_connect("motion_notify_event", self._on_hover)
        self.canvas.mpl_connect("button_press_event", self._on_click)

        ttk.Label(info_frame, text="Selected point", font=("TkDefaultFont", 10, "bold")).pack(anchor="w", pady=(0, 5))
        self.detail_text = tk.Text(info_frame, width=44, height=24, wrap=tk.WORD, state=tk.DISABLED)
        self.detail_text.pack(fill=tk.BOTH, expand=True)
        ttk.Label(info_frame, text="Computation log", font=("TkDefaultFont", 10, "bold")).pack(anchor="w", pady=(8, 5))
        self.log_text = tk.Text(info_frame, width=44, height=10, wrap=tk.WORD, state=tk.DISABLED)
        self.log_text.pack(fill=tk.BOTH, expand=False)

        self._toggle_mode_controls()
        self._draw_empty_plot()

    def _toggle_mode_controls(self) -> None:
        # Both modes use the min/max scaling range.  In isotropic mode this
        # means M=min,...,max; in anisotropic mode it means every inequivalent
        # diagonal vector with entries in that range.
        self.param_widgets["min_base"].configure(state=tk.NORMAL)
        self.param_widgets["max_base"].configure(state=tk.NORMAL)

    def _config_from_ui(self) -> RunConfig:
        try:
            mode = str(self.vars["mode"].get()).strip().lower()
            d = int(self.vars["dimension"].get())
            aniso = tuple()
            cfg = RunConfig(
                mode=mode,
                dimension=d,
                min_base=int(self.vars["min_base"].get()),
                max_base=int(self.vars["max_base"].get()),
                anisotropic_bases=aniso,
                max_cardinality=int(self.vars["max_cardinality"].get()),
                stage=int(self.vars["stage"].get()),
                max_alphabets_per_size=int(self.vars["max_alphabets"].get()),
                max_pairs_per_base=int(self.vars["max_pairs"].get()),
                max_matrix_entries=int(self.vars["max_entries"].get()),
                seed=int(self.vars["seed"].get()),
                exact_alphabets=bool(self.vars["exact_alphabets"].get()),
                exact_pairs=bool(self.vars["exact_pairs"].get()),
                mark_orthogonal=bool(self.vars["mark_orthogonal"].get()),
            )
        except Exception as exc:
            raise ValueError(f"Invalid parameter: {exc}") from exc

        if cfg.mode not in {"isotropic", "anisotropic"}:
            raise ValueError("Scaling mode must be isotropic or anisotropic.")
        if cfg.dimension < 1:
            raise ValueError("Dimension must be at least 1.")
        if cfg.min_base < 2 or cfg.max_base < cfg.min_base:
            raise ValueError("Require 2 <= min scaling <= max scaling.")
        if cfg.mode == "anisotropic":
            if cfg.dimension < 2:
                raise ValueError("Anisotropic sweep requires dimension d >= 2.")
            if cfg.min_base == cfg.max_base:
                raise ValueError(
                    "The anisotropic sweep contains no vectors when min scaling = max scaling. "
                    "Increase the maximum scaling or use isotropic mode."
                )
        if cfg.max_cardinality < 1:
            raise ValueError("Maximum cardinality must be at least 1.")
        if cfg.stage < 1:
            raise ValueError("Stage must be at least 1.")
        if cfg.max_alphabets_per_size < 1 or cfg.max_pairs_per_base < 1:
            raise ValueError("Sampling caps must be positive.")
        if cfg.max_matrix_entries < 1:
            raise ValueError("Matrix-entry limit must be positive.")

        if cfg.exact_alphabets:
            # Safe upper bound for both modes.  The all-max tuple is excluded
            # from the anisotropic sweep, but this conservative bound is useful
            # for warning about combinatorial explosion.
            worst_q = cfg.max_base ** cfg.dimension
            worst_m = min(cfg.max_cardinality, worst_q)
            worst = math.comb(worst_q, worst_m)
            if worst > 2_000_000:
                ok = messagebox.askyesno(
                    "Large exact enumeration",
                    f"One requested alphabet layer can contain about {worst:,} subsets before translation reduction.\n\n"
                    "This may take a very long time and use substantial memory. Continue?",
                )
                if not ok:
                    raise RuntimeError("Run cancelled.")
        return cfg

    def _start_run(self) -> None:
        if self.worker and self.worker.is_alive():
            return
        try:
            cfg = self._config_from_ui()
        except RuntimeError:
            return
        except Exception as exc:
            messagebox.showerror("Invalid parameters", str(exc))
            return

        self.results = []
        self.scatter_map.clear()
        self.pinned_result = None
        self.current_plot_cfg = None
        self._set_detail("")
        self._set_log("")
        self.cancel_event.clear()
        self.run_button.configure(state=tk.DISABLED)
        self.cancel_button.configure(state=tk.NORMAL)
        self.save_png_button.configure(state=tk.DISABLED)
        self.save_csv_button.configure(state=tk.DISABLED)
        self.import_csv_button.configure(state=tk.DISABLED)
        self.progress.start(12)
        self.status_var.set("Starting computation ...")

        def emit(kind: str, payload: str) -> None:
            self.msg_queue.put((kind, payload))

        def work() -> None:
            try:
                data = run_experiment(cfg, self.cancel_event, emit)
                self.msg_queue.put(("done", (cfg, data)))
            except Exception:
                self.msg_queue.put(("error", traceback.format_exc()))

        self.worker = threading.Thread(target=work, daemon=True)
        self.worker.start()

    def _cancel_run(self) -> None:
        self.cancel_event.set()
        self.status_var.set("Cancelling after the current SVD ...")

    def _poll_queue(self) -> None:
        try:
            while True:
                kind, payload = self.msg_queue.get_nowait()
                if kind == "status":
                    self.status_var.set(str(payload))
                elif kind == "log":
                    self._append_log(str(payload))
                elif kind == "done":
                    cfg, data = payload
                    self.progress.stop()
                    self.run_button.configure(state=tk.NORMAL)
                    self.cancel_button.configure(state=tk.DISABLED)
                    self.import_csv_button.configure(state=tk.NORMAL)
                    self.results = data
                    if self.cancel_event.is_set():
                        self.status_var.set(f"Cancelled. Displaying {len(data)} completed points.")
                    else:
                        self.status_var.set(f"Finished: {len(data)} plotted pairs.")
                    if data:
                        self.current_plot_cfg = cfg
                        self._plot_results(cfg)
                        self.save_png_button.configure(state=tk.NORMAL)
                        self.save_csv_button.configure(state=tk.NORMAL)
                    else:
                        self._draw_empty_plot("No pairs were evaluated. Relax caps or matrix limits.")
                elif kind == "error":
                    self.progress.stop()
                    self.run_button.configure(state=tk.NORMAL)
                    self.cancel_button.configure(state=tk.DISABLED)
                    self.import_csv_button.configure(state=tk.NORMAL)
                    self.status_var.set("Computation failed.")
                    self._append_log(str(payload))
                    messagebox.showerror("Computation error", "The computation failed. See the log for details.")
        except queue.Empty:
            pass
        self.after(100, self._poll_queue)

    def _draw_empty_plot(self, message: str = "Choose parameters and click Generate.") -> None:
        self.ax.clear()
        self.ax.text(0.5, 0.5, message, ha="center", va="center", transform=self.ax.transAxes)
        self.ax.set_axis_off()
        self.canvas.draw_idle()

    def _plot_results(self, cfg: RunConfig) -> None:
        self.ax.clear()
        self.scatter_map.clear()
        self.pinned_result = None

        signatures = sorted({r.bases or str(r.base) for r in self.results})
        markers = ["o", "s", "^", "D", "v", "P", "X", "<", ">", "h", "*"]

        if cfg.mode == "anisotropic":
            # All scaling vectors share one scatter collection so a large sweep
            # does not create an unreadable legend.  The exact vector N is
            # retained in every Result and displayed in the hover/click tooltip.
            normal = [
                r for r in self.results
                if not (cfg.mark_orthogonal and r.hs_rigidity_obstruction)
            ]
            if normal:
                sc = self.ax.scatter(
                    [r.delta_bar for r in normal],
                    [r.beta for r in normal],
                    s=28, marker="o", alpha=0.72,
                    label=f"anisotropic pairs ({len(signatures)} scaling vectors)",
                    picker=True,
                )
                self.scatter_map[sc] = normal
        else:
            for j, sig in enumerate(signatures):
                normal = [
                    r for r in self.results
                    if (r.bases or str(r.base)) == sig
                    and not (cfg.mark_orthogonal and r.hs_rigidity_obstruction)
                ]
                if normal:
                    rep = normal[0]
                    bt = parse_bases_string(rep.bases)
                    label = f"M={bt[0]}"
                    sc = self.ax.scatter(
                        [r.delta_bar for r in normal],
                        [r.beta for r in normal],
                        s=30,
                        marker=markers[j % len(markers)],
                        alpha=0.78,
                        label=label,
                        picker=True,
                    )
                    self.scatter_map[sc] = normal

        if cfg.mark_orthogonal:
            rigid = [r for r in self.results if r.hs_rigidity_obstruction]
            if rigid:
                sc = self.ax.scatter(
                    [r.delta_bar for r in rigid],
                    [r.beta for r in rigid],
                    s=72,
                    facecolors="none",
                    edgecolors="black",
                    linewidths=1.25,
                    marker="o",
                    label="blockwise HS-rigidity obstruction",
                    zorder=6,
                    picker=True,
                )
                self.scatter_map[sc] = rigid

        xmax = cfg.dimension / 2.0
        xs = np.linspace(0.0, xmax, 400)
        self.ax.plot(
            xs, cfg.dimension / 2.0 - xs, "k-", linewidth=1.6,
            label=r"Hilbert--Schmidt: $\beta=d/2-\bar\delta$"
        )
        # Dyatlov's familiar dashed reference line is retained only in the
        # isotropic mode; it is not asserted as a general anisotropic bound.
        if cfg.mode == "isotropic":
            self.ax.plot(
                xs, (cfg.dimension - xs) / 2.0, "k--", linewidth=1.25,
                label=r"isotropic reference: $\beta=(d-\bar\delta)/2$"
            )

        vals = [r.beta for r in self.results]
        ymin = min(0.0, min(vals) - 0.03)
        ymax = max(cfg.dimension / 2.0, max(vals) + 0.03)
        self.ax.set_xlim(0.0, xmax)
        self.ax.set_ylim(ymin, ymax)
        self.ax.set_xlabel(r"average density $\bar\delta=(\delta_A+\delta_B)/2$")
        self.ax.set_ylabel(r"finite-stage exponent $\beta_k$")
        if cfg.mode == "anisotropic":
            title = (
                f"Anisotropic scaling sweep: {cfg.min_base} <= N_i <= {cfg.max_base}, "
                f"d={cfg.dimension}, k={cfg.stage}; "
                r"$\delta_A+\delta_B\leq d$"
            )
        else:
            title = (
                f"Isotropic restricted Fourier transform: d={cfg.dimension}, k={cfg.stage}; "
                r"$\delta_A+\delta_B\leq d$"
            )
        self.ax.set_title(title)
        self.ax.grid(alpha=0.2)
        self.ax.legend(fontsize=8, loc="best")

        self.annotation = self.ax.annotate(
            "", xy=(0, 0), xytext=(15, 15), textcoords="offset points",
            bbox=dict(boxstyle="round,pad=0.45", fc="white", ec="0.35", alpha=0.96),
            arrowprops=dict(arrowstyle="->", color="0.35"), fontsize=8.5, zorder=20,
        )
        self.annotation.set_visible(False)

        self.figure.tight_layout()
        self.canvas.draw_idle()

    def _hits_at_event(self, event) -> list[tuple[Result, tuple[float, float]]]:
        if event.inaxes != self.ax:
            return []
        hits: list[tuple[Result, tuple[float, float]]] = []
        # Reverse draw order so highlighted points appear first, but retain all
        # coincident/nearby points returned by Matplotlib's hit testing.
        for collection, records in reversed(list(self.scatter_map.items())):
            contains, info = collection.contains(event)
            if not contains:
                continue
            offsets = collection.get_offsets()
            for raw_idx in info.get("ind", []):
                idx = int(raw_idx)
                if 0 <= idx < len(records):
                    xy = tuple(float(x) for x in offsets[idx])
                    hits.append((records[idx], xy))
        return hits

    def _on_hover(self, event) -> None:
        if self.pinned_result is not None:
            return
        hits = self._hits_at_event(event)
        if not hits:
            if self.annotation.get_visible():
                self.annotation.set_visible(False)
                self.canvas.draw_idle()
            return
        r, xy = hits[0]
        suffix = ""
        if len(hits) > 1:
            suffix = f"\n\n{len(hits)} plotted pairs overlap here; click repeatedly to cycle."
        self.annotation.xy = xy
        self.annotation.set_text(result_tooltip(r) + suffix)
        self.annotation.set_visible(True)
        self.canvas.draw_idle()

    def _on_click(self, event) -> None:
        hits = self._hits_at_event(event)
        if not hits:
            if event.inaxes == self.ax:
                self.pinned_result = None
                self._cycle_signature = None
                self._cycle_index = 0
                self.annotation.set_visible(False)
                self._set_detail("")
                self.canvas.draw_idle()
            return

        # A stable signature lets repeated clicks at the same plotted location
        # cycle through coincident alphabet pairs.
        signature = tuple(
            (h[0].bases, h[0].alphabet_A, h[0].alphabet_B, round(h[0].beta, 14))
            for h in hits
        )
        if signature == self._cycle_signature:
            self._cycle_index = (self._cycle_index + 1) % len(hits)
        else:
            self._cycle_signature = signature
            self._cycle_index = 0

        r, xy = hits[self._cycle_index]
        self.pinned_result = r
        text = result_tooltip(r)
        if len(hits) > 1:
            text += f"\n\ncoincident point {self._cycle_index + 1}/{len(hits)}; click again to cycle"
        self.annotation.xy = xy
        self.annotation.set_text(text)
        self.annotation.set_visible(True)
        self._set_detail(text)
        self.canvas.draw_idle()

    def _redraw_if_results(self) -> None:
        if not self.results or self.current_plot_cfg is None:
            return
        cfg = replace(
            self.current_plot_cfg,
            mark_orthogonal=bool(self.vars["mark_orthogonal"].get()),
        )
        self.current_plot_cfg = cfg
        self._plot_results(cfg)

    def _set_detail(self, text: str) -> None:
        self.detail_text.configure(state=tk.NORMAL)
        self.detail_text.delete("1.0", tk.END)
        self.detail_text.insert(tk.END, text)
        self.detail_text.configure(state=tk.DISABLED)

    def _set_log(self, text: str) -> None:
        self.log_text.configure(state=tk.NORMAL)
        self.log_text.delete("1.0", tk.END)
        self.log_text.insert(tk.END, text)
        self.log_text.configure(state=tk.DISABLED)

    def _append_log(self, text: str) -> None:
        self.log_text.configure(state=tk.NORMAL)
        if self.log_text.index("end-1c") != "1.0":
            self.log_text.insert(tk.END, "\n")
        self.log_text.insert(tk.END, text)
        self.log_text.see(tk.END)
        self.log_text.configure(state=tk.DISABLED)

    def _import_csv(self) -> None:
        if self.worker and self.worker.is_alive():
            messagebox.showinfo("Computation running", "Cancel or finish the current computation before importing a CSV.")
            return

        path = filedialog.askopenfilename(
            title="Import FUP data",
            filetypes=[("CSV", "*.csv"), ("All files", "*.*")],
        )
        if not path:
            return

        try:
            data = read_results_csv(path)
        except Exception as exc:
            messagebox.showerror("CSV import error", str(exc))
            return

        d = data[0].dimension
        k = data[0].stage
        max_card = max(max(r.card_A, r.card_B) for r in data)
        modes = {r.mode for r in data}
        mode = "anisotropic" if modes == {"anisotropic"} else "isotropic"

        self.vars["dimension"].set(d)
        self.vars["stage"].set(k)
        self.vars["mode"].set(mode)
        if max_card > 0:
            self.vars["max_cardinality"].set(max_card)

        if mode == "anisotropic":
            sigs = sorted({r.bases for r in data if r.bases})
            parsed = [parse_bases_string(sig) for sig in sigs]
            all_scales = [N for bt in parsed for N in bt]
            min_base = min(all_scales)
            max_base = max(all_scales)
            self.vars["min_base"].set(min_base)
            self.vars["max_base"].set(max_base)
            # Retain one legacy vector internally only for backward-compatible
            # configuration serialization; generated sweeps do not use it.
            aniso = parsed[0] if len(parsed) == 1 else tuple()
        else:
            iso_bases = sorted({r.base for r in data if r.base >= 2})
            if not iso_bases:
                # Recover from a bases string such as 3x3 when importing a newer CSV.
                recovered = []
                for r in data:
                    bt = parse_bases_string(r.bases)
                    if len(set(bt)) == 1:
                        recovered.append(bt[0])
                iso_bases = sorted(set(recovered))
            min_base, max_base = min(iso_bases), max(iso_bases)
            self.vars["min_base"].set(min_base)
            self.vars["max_base"].set(max_base)
            aniso = tuple()

        self._toggle_mode_controls()
        self.results = data
        self.scatter_map.clear()
        self.pinned_result = None
        self._cycle_signature = None
        self._cycle_index = 0
        self._set_detail("")
        self._set_log(
            f"Imported {len(data)} rows from {os.path.basename(path)}.\n"
            "No Fourier matrices were recomputed."
        )

        cfg = RunConfig(
            mode=mode,
            dimension=d,
            min_base=min_base,
            max_base=max_base,
            anisotropic_bases=aniso,
            max_cardinality=max_card if max_card > 0 else 1,
            stage=k,
            max_alphabets_per_size=1,
            max_pairs_per_base=max(1, len(data)),
            max_matrix_entries=1,
            seed=0,
            exact_alphabets=False,
            exact_pairs=False,
            mark_orthogonal=bool(self.vars["mark_orthogonal"].get()),
        )
        self.current_plot_cfg = cfg
        self._plot_results(cfg)
        self.save_png_button.configure(state=tk.NORMAL)
        self.save_csv_button.configure(state=tk.NORMAL)
        self.status_var.set(f"Imported {len(data)} plotted pairs from {os.path.basename(path)}")

    def _export_stem(self) -> str:
        """Return a compact filename stem describing the active run parameters."""
        cfg = self.current_plot_cfg
        if cfg is None:
            return "fup_pairs"

        if cfg.mode == "anisotropic":
            scale_tag = f"Nsweep{cfg.min_base}-{cfg.max_base}"
            mode_tag = "aniso"
        else:
            scale_tag = f"M{cfg.min_base}-{cfg.max_base}"
            mode_tag = "iso"

        parts = [
            "fup_pairs",
            mode_tag,
            f"d{cfg.dimension}",
            scale_tag,
            f"card{cfg.max_cardinality}",
            f"k{cfg.stage}",
        ]
        if cfg.exact_alphabets:
            parts.append("exactA")
        else:
            parts.append(f"Amax{cfg.max_alphabets_per_size}")
        if cfg.exact_pairs:
            parts.append("exactP")
        else:
            parts.append(f"Pmax{cfg.max_pairs_per_base}")
        parts.extend([f"entries{cfg.max_matrix_entries}", f"seed{cfg.seed}"])
        return "_".join(parts)

    def _save_png(self) -> None:
        if not self.results:
            return
        path = filedialog.asksaveasfilename(
            title="Save FUP plot",
            defaultextension=".png",
            filetypes=[("PNG image", "*.png"), ("PDF", "*.pdf"), ("All files", "*.*")],
            initialfile=f"{self._export_stem()}.png",
        )
        if path:
            self.figure.savefig(path, dpi=220, bbox_inches="tight")
            self.status_var.set(f"Saved {os.path.basename(path)}")

    def _save_csv(self) -> None:
        if not self.results:
            return
        path = filedialog.asksaveasfilename(
            title="Save FUP data",
            defaultextension=".csv",
            filetypes=[("CSV", "*.csv"), ("All files", "*.*")],
            initialfile=f"{self._export_stem()}.csv",
        )
        if path:
            write_csv(self.results, path)
            self.status_var.set(f"Saved {os.path.basename(path)}")


def build_exe() -> None:
    """Build a single-file Windows executable using PyInstaller."""
    if sys.platform != "win32":
        raise SystemExit(
            "--build-exe should be run on Windows because PyInstaller builds for the "
            "platform on which it is executed."
        )
    try:
        import PyInstaller  # noqa: F401
    except ImportError:
        raise SystemExit(
            "PyInstaller is not installed. Run:\n\n"
            "    py -m pip install pyinstaller numpy matplotlib\n\n"
            "then rerun:\n\n"
            "    py fup_interactive_explorer_csv.py --build-exe"
        )

    source = os.path.abspath(__file__)
    cmd = [
        sys.executable, "-m", "PyInstaller",
        "--noconfirm", "--clean", "--onefile", "--windowed",
        "--name", "FUP_Explorer",
        "--collect-all", "matplotlib",
        source,
    ]
    print("Building FUP_Explorer.exe ...")
    subprocess.run(cmd, check=True)
    print("\nBuild complete. Look for dist\\FUP_Explorer.exe")


def main() -> None:
    if "--build-exe" in sys.argv[1:]:
        build_exe()
        return
    app = FUPExplorer()
    app.mainloop()


if __name__ == "__main__":
    main()
