"""Independent XPS fitting benchmark engine (not KherveFitting/CasaXPS).

The GL product and iterative Shirley equations are intentionally aligned with
xps_automation_poc.py in LunaKang2/XPS_KherveFitting-practice.
Fitted peak assignments are NOT chemical identification.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import re
import zipfile
from typing import Any

import numpy as np
from scipy.optimize import least_squares

MAX_CSV_BYTES = 20 * 1024 * 1024


def gl(x: np.ndarray, center: float, fwhm: float, height: float, lor_frac: float) -> np.ndarray:
    """Height-normalized product GL (matches the original PoC approximation)."""
    z = (x - center) / (fwhm / 2.0)
    gaussian = np.exp(-np.log(2.0) * z * z)
    lorentzian = 1.0 / (1.0 + z * z)
    return height * (gaussian ** (1.0 - lor_frac)) * (lorentzian ** lor_frac)


def shirley(x: np.ndarray, y: np.ndarray, iterations: int = 100, tolerance: float = 1e-6) -> np.ndarray:
    """Original PoC's iterative Shirley approximation, high/low BE endpoints."""
    order = np.argsort(x)
    xx, yy = x[order], y[order]
    high = float(np.median(yy[-5:]))
    low = float(np.median(yy[:5]))
    b = np.linspace(low, high, len(xx))
    for _ in range(iterations):
        sub = np.maximum(yy - b, 0.0)
        dx = np.diff(xx)
        cum = np.zeros_like(sub)
        cum[1:] = np.cumsum((sub[:-1] + sub[1:]) * dx / 2.0)
        if cum[-1] < 1e-12:
            break
        new = low + (high - low) * cum / cum[-1]
        if np.max(np.abs(new - b)) < tolerance:
            b = new
            break
        b = new
    restored = np.empty_like(b)
    restored[order] = b
    return restored


def _normalize_heading(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.lower())


def _field_name(headers: list[str], required: str) -> str:
    normalized = {_normalize_heading(h): h for h in headers if h is not None}
    candidates = {
        "binding": ("bindingenergyev", "bindingenergy", "bindingenergyevcorrected", "bindingenergyinbev", "beev"),
        "intensity": ("intensitycps", "intensity", "countscps", "countspersecond", "cps"),
    }[required]
    for candidate in candidates:
        if candidate in normalized:
            return normalized[candidate]
    raise ValueError(
        "필수 열을 찾을 수 없습니다. 'Binding Energy (eV)' 및 'Intensity (CPS)' 열이 필요합니다. "
        f"현재 CSV 열: {headers}"
    )


def parse_csv(raw: bytes) -> tuple[np.ndarray, np.ndarray]:
    """Parse instrument CSV, sort ascending by BE and average exact duplicate energies."""
    if len(raw) > MAX_CSV_BYTES:
        raise ValueError("CSV 파일이 20MB를 초과합니다.")
    decoded = None
    for encoding in ("utf-8-sig", "cp949"):
        try:
            decoded = raw.decode(encoding)
            break
        except UnicodeDecodeError:
            pass
    if decoded is None:
        raise ValueError("CSV 인코딩을 읽을 수 없습니다 (UTF-8 또는 CP949 사용).")
    reader = csv.DictReader(io.StringIO(decoded))
    if not reader.fieldnames:
        raise ValueError("CSV에 열 이름(header)이 없습니다.")
    be_col = _field_name(reader.fieldnames, "binding")
    cps_col = _field_name(reader.fieldnames, "intensity")
    xs, ys = [], []
    for row in reader:
        try:
            xv = float(str(row[be_col]).replace(",", "").strip())
            yv = float(str(row[cps_col]).replace(",", "").strip())
            if np.isfinite(xv) and np.isfinite(yv):
                xs.append(xv)
                ys.append(yv)
        except (TypeError, ValueError):
            continue
    if len(xs) < 20:
        raise ValueError("유효한 XPS 데이터가 20행 미만입니다. CSV 값과 열 이름을 확인하세요.")
    x, y = np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)
    unique, inverse, counts = np.unique(x, return_inverse=True, return_counts=True)
    combined_y = np.bincount(inverse, weights=y) / counts
    if len(unique) < 20:
        raise ValueError("고유한 binding energy 데이터가 20개 미만입니다.")
    if unique[-1] - unique[0] <= 0:
        raise ValueError("유효한 binding energy 범위를 찾을 수 없습니다.")
    return unique, combined_y


def zip_csv_names(zip_bytes: bytes) -> list[str]:
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as archive:
        names = [
            info.filename for info in archive.infolist()
            if not info.is_dir() and info.filename.lower().endswith(".csv")
            and 0 < info.file_size <= MAX_CSV_BYTES
        ]
    if not names:
        raise ValueError("ZIP 안에 크기 20MB 이하인 CSV 파일이 없습니다.")
    return sorted(names, key=lambda name: ("f1s" not in name.lower().replace(" ", ""), name.lower()))


def extract_zip_csv(zip_bytes: bytes, name: str) -> bytes:
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as archive:
        info = archive.getinfo(name)
        if info.file_size > MAX_CSV_BYTES:
            raise ValueError("선택한 CSV 파일이 20MB를 초과합니다.")
        raw = archive.read(info)
        if len(raw) > MAX_CSV_BYTES:
            raise ValueError("압축 해제된 CSV 파일이 20MB를 초과합니다.")
    return raw


def _fit_metrics(y: np.ndarray, fitted: np.ndarray, k: int) -> dict[str, float | None]:
    residual = y - fitted
    ss = float(np.dot(residual, residual))
    tss = float(np.dot(y - np.mean(y), y - np.mean(y)))
    n = len(y)
    # Use a floor for log(RSS/n) in near-exact fits.
    rss_for_log = max(ss, np.finfo(float).tiny)
    base = n * np.log(rss_for_log / n)
    aic = base + 2 * k
    return {
        "r2": float(1 - ss / tss) if tss > 1e-12 else None,
        "rmse": float(np.sqrt(ss / n)),
        "rss": ss,
        "aicc": float(aic + 2 * k * (k + 1) / (n - k - 1)) if n > k + 1 else None,
        "bic": float(base + k * np.log(n)),
    }


def _area_in_measured_window(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.sum((y[1:] + y[:-1]) * np.diff(x) / 2.0))


def fit_model(
    x: np.ndarray,
    y: np.ndarray,
    bg: np.ndarray,
    centers: list[float],
    center_window: float = 1.5,
) -> dict[str, Any]:
    """Fit 1 or 2 GL peaks to y with a *fixed* precomputed Shirley background."""
    if not (1 <= len(centers) <= 2):
        raise ValueError("피크 개수는 1 또는 2여야 합니다.")
    if len(x) <= 4 * len(centers) + 1:
        raise ValueError("모델 피팅에 필요한 데이터 포인트가 부족합니다.")
    if center_window <= 0:
        raise ValueError("중심 위치 허용폭은 0보다 커야 합니다.")
    net_max = max(float(np.max(y - bg)), 1.0)
    height_cap = max(3.0 * net_max, 10.0)
    guesses, low, high = [], [], []
    for i, center in enumerate(centers):
        if not x[0] <= center <= x[-1]:
            raise ValueError(
                f"피크 {i+1} 초기 위치 {center:.2f} eV가 데이터 범위 "
                f"({x[0]:.2f}~{x[-1]:.2f} eV) 밖에 있습니다."
            )
        center_min = max(float(x[0]), center - center_window)
        center_max = min(float(x[-1]), center + center_window)
        if center_max - center_min < 0.2:
            raise ValueError("피크 중심이 데이터 범위 끝에 너무 가깝습니다.")
        fractions = (0.85,) if len(centers) == 1 else (0.70, 0.30)
        initial_height = min(net_max * fractions[i], height_cap * 0.99)
        guesses.extend([center, 1.9 if i == 0 else 2.2, initial_height, 0.35 if i == 0 else 0.20])
        low.extend([center_min, 0.6, 0.0, 0.0])
        high.extend([center_max, 4.0, height_cap, 1.0])

    def predict(par: np.ndarray) -> np.ndarray:
        result = bg.copy()
        for i in range(0, len(par), 4):
            result += gl(x, *par[i:i+4])
        return result

    optimized = least_squares(
        lambda p: predict(p) - y,
        np.asarray(guesses, dtype=float),
        bounds=(np.asarray(low), np.asarray(high)),
        max_nfev=15000,
        xtol=1e-10,
        ftol=1e-10,
        gtol=1e-10,
    )
    fitted = predict(optimized.x)
    residual = y - fitted
    peaks = []
    for i in range(len(centers)):
        center, fwhm, height, mix = [float(v) for v in optimized.x[4*i:4*i+4]]
        contribution = gl(x, center, fwhm, height, mix)
        peaks.append({
            "peak_index": i + 1,
            "center_eV": center,
            "fwhm_eV": fwhm,
            "height_CPS": height,
            "mix": mix,
            "area_CPS_eV_in_window": _area_in_measured_window(x, contribution),
            "component": contribution,
        })
    proximity = np.minimum(optimized.x - np.asarray(low), np.asarray(high) - optimized.x)
    relative_margin = proximity / np.maximum(np.asarray(high) - np.asarray(low), 1e-12)
    return {
        "peak_count": len(centers),
        "metrics": _fit_metrics(y, fitted, len(optimized.x)),
        "peaks": peaks,
        "fitted": fitted,
        "residual": residual,
        "success": bool(optimized.success),
        "optimizer_message": str(optimized.message),
        "nfev": int(optimized.nfev),
        "near_any_bound": bool(np.any(relative_margin < 1e-3)),
    }


def fit_comparison(x: np.ndarray, y: np.ndarray, peak1: float, peak2: float, center_window: float) -> tuple[np.ndarray, list[dict[str, Any]]]:
    bg = shirley(x, y)
    one = fit_model(x, y, bg, [peak1], center_window)
    two = fit_model(x, y, bg, [peak1, peak2], center_window)
    return bg, [one, two]


def serializable_report(
    x: np.ndarray, y: np.ndarray, source_name: str, source_bytes: bytes,
    energy_offset: float, peak1: float, peak2: float, center_window: float,
    results: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "method": "Independent SciPy benchmark: fixed iterative Shirley and product GL",
        "not_khervefitting": True,
        "not_chemical_identification": True,
        "energy_note": "Offset is ADDITIONAL to the values already present in the supplied CSV. Check for pre-applied corrections.",
        "source_filename": source_name,
        "source_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "n_points": int(len(x)),
        "binding_energy_range_eV": [float(x[0]), float(x[-1])],
        "additional_energy_offset_eV": float(energy_offset),
        "initial_centers_eV": [float(peak1), float(peak2)],
        "center_half_window_eV": float(center_window),
        "models": [{
            "peak_count": model["peak_count"],
            "optimizer_success": model["success"],
            "optimizer_message": model["optimizer_message"],
            "near_any_bound": model["near_any_bound"],
            "metrics": model["metrics"],
            "peaks": [{k: v for k, v in peak.items() if k != "component"} for peak in model["peaks"]],
        } for model in results],
    }


def make_synthetic_f1s() -> bytes:
    """A labelled *synthetic* preview dataset, never a measured reference."""
    rng = np.random.default_rng(42)
    x = np.linspace(682.0, 692.5, 315)
    baseline = 9000 + 4000 * (x - x[0]) / (x[-1] - x[0])
    y = baseline + gl(x, 685.1, 1.9, 126000, 0.35) + gl(x, 688.5, 2.2, 46000, 0.20)
    y = y + rng.normal(0, 1300, len(y))
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["Binding Energy (eV)", "Intensity (CPS)"])
    writer.writerows(zip(x, y))
    return buf.getvalue().encode("utf-8")
