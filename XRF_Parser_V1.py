# app.py
# Streamlit viewer for .spx files (counts + robust metadata)
# - upload a .spx or use data/sample.spx
# - optional metadata display
# - optional spectrum plot

import re
import json
import base64
import struct
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Iterable

import numpy as np
import pandas as pd
import streamlit as st
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt


# =========================
#   Low-level utilities
# =========================

_NUM = re.compile(r"^[+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?$")

def _repair_spx_text(text: str) -> str:
    """Fix common truncation of <Channels> lines (ensure closing tag)."""
    out = []
    for ln in text.splitlines():
        if "<Channels>" in ln and "</Channels>" not in ln:
            nums = re.findall(r"-?\d+(?:\.\d+)?", ln)
            ln = "<Channels>" + ",".join(nums) + "</Channels>"
        out.append(ln)
    return "\n".join(out)

def _to_num(s: Optional[str]) -> Optional[float]:
    if s is None:
        return None
    s = s.strip()
    if _NUM.match(s):
        try:
            return float(s)
        except Exception:
            return None
    return None

def _localname(tag: str) -> str:
    """Strip any XML namespace from a tag: '{ns}Current' -> 'Current'."""
    return tag.split('}', 1)[-1] if '}' in tag else tag

def _load_spx_root(path: str, encoding: str = "cp1252") -> ET.Element:
    """Robust XML loader with encoding fallbacks + channel repair."""
    for enc in (encoding, "utf-8", "latin-1", "iso-8859-1"):
        try:
            raw = Path(path).read_bytes().decode(enc, errors="replace")
            fixed = _repair_spx_text(raw)
            return ET.fromstring(fixed)
        except Exception:
            pass
    # last resort
    fixed = _repair_spx_text(Path(path).read_bytes().decode("utf-8", errors="ignore"))
    return ET.fromstring(fixed)

def _find_text_anywhere(root: ET.Element, tag: str) -> Optional[str]:
    # exact tag
    for e in root.iter(tag):
        t = (e.text or "").strip()
        if t:
            return t
    # case-insensitive by localname
    tl = tag.lower()
    for e in root.iter():
        if _localname(e.tag).lower() == tl:
            t = (e.text or "").strip()
            if t:
                return t
    return None

def _flatten_leaves(root: ET.Element) -> Dict[str, str]:
    """Flatten all leaf nodes to dict: 'A/B/C' -> text."""
    out: Dict[str, str] = {}
    def rec(e: ET.Element, path: List[str]):
        kids = list(e)
        if kids:
            for k in kids:
                rec(k, path + [k.tag])
        else:
            t = (e.text or "").strip()
            if t:
                out["/".join(path)] = t
    rec(root, [root.tag])
    return out


# =========================
#   Units & parsing
# =========================

_UNIT_NUM = re.compile(r"^\s*([+-]?\d+(?:\.\d+)?)(?:\s*([A-Za-zµμ]+))?\s*$")

def _parse_value_and_unit(s: str) -> Tuple[Optional[float], Optional[str]]:
    s = (s or "").strip()
    m = _UNIT_NUM.match(s)
    if not m:
        try:
            return float(s), None
        except Exception:
            return None, None
    val = float(m.group(1))
    unit = m.group(2)
    return val, unit

def _to_kV(val: Optional[float], unit: Optional[str]) -> Optional[float]:
    if val is None:
        return None
    if not unit:
        return float(val)
    u = unit.lower()
    if u == "kv":
        return float(val)
    if u == "v":
        return float(val) / 1000.0
    if u in ("k", "kvdc"):
        return float(val)
    return float(val)

def _to_uA(val: Optional[float], unit: Optional[str]) -> Optional[float]:
    if val is None:
        return None
    if not unit:
        return float(val)
    u = unit.lower()
    if "ua" in u or "µa" in u or "μa" in u:
        return float(val)
    if "ma" in u:
        return float(val) * 1000.0
    if u == "a":
        return float(val) * 1e6
    return float(val)

def _length_to_mm(val: float, unit: Optional[str]) -> float:
    if unit is None:
        return float(val)
    u = unit.strip().lower()
    if u in ("mm",):
        return float(val)
    if u in ("µm", "μm", "um"):
        return float(val) / 1000.0
    if u in ("cm",):
        return float(val) * 10.0
    if u in ("m",):
        return float(val) * 1000.0
    return float(val)


# =========================
#   Stage extractor (with RTREM fallback)
# =========================

def extract_axes_with_rtrem_fallback(file_path: str) -> Dict[str, Dict[str, Any]]:
    """
    Extract stage axes from .spx.
    1) Prefer Axis0/1/2 (AxisName/AxisPosition/AxisUnit).
    2) Fallback: decode base64 <Data> blobs and locate a (X,Y,Z) float64 triple in mm.
    """
    root = _load_spx_root(file_path)

    # Canonical Axis*
    axes: Dict[str, Dict[str, Any]] = {}
    for elem in root.iter():
        tag_local = _localname(elem.tag)
        if tag_local.startswith("Axis"):
            name = elem.attrib.get("AxisName", tag_local)
            unit = elem.attrib.get("AxisUnit", "")
            pos  = elem.attrib.get("AxisPosition")
            if pos is not None:
                try:
                    pos_val = float(pos)
                except ValueError:
                    pos_val = pos
                axes[name] = {"position": pos_val, "unit": unit}
    if axes:
        return axes

    # Fallback: scan Data blobs
    def find_xyz_from_blob(blob: bytes):
        best = None
        for off in range(0, len(blob) - 24 + 1, 1):
            try:
                x, y, z = struct.unpack("<ddd", blob[off:off+24])
            except struct.error:
                continue
            if all(np.isfinite([x, y, z])) and all(0 <= v <= 1000 for v in (x, y, z)):
                if all(v > 0 for v in (x, y, z)):
                    best = (x, y, z)
        return best

    for node in root.iter():
        if _localname(node.tag) != "Data":
            continue
        b64 = (node.text or "").strip()
        if len(b64) < 8:
            continue
        try:
            blob = base64.b64decode(b64, validate=True)
        except Exception:
            continue
        xyz = find_xyz_from_blob(blob)
        if xyz:
            x, y, z = xyz
            return {
                "X": {"position": float(x), "unit": "mm"},
                "Y": {"position": float(y), "unit": "mm"},
                "Z": {"position": float(z), "unit": "mm"},
            }
    return {}

def _extract_stage_any(path: str) -> Optional[Dict[str, float]]:
    axes = extract_axes_with_rtrem_fallback(path) or {}
    if not axes:
        return None

    def pick(axis_prefix: str) -> Optional[Tuple[float, Optional[str]]]:
        rec = axes.get(axis_prefix)
        if rec is None:
            for k, v in axes.items():
                if k.upper().startswith(axis_prefix.upper()):
                    rec = v
                    break
        if rec is None:
            return None
        return float(rec.get("position")), rec.get("unit")

    out: Dict[str, float] = {}
    for axis in ("X", "Y", "Z"):
        got = pick(axis)
        if got is not None:
            val, unit = got
            out[f"{axis}_mm"] = _length_to_mm(val, unit)
    return out or None


# =========================
#   XML navigation helpers
# =========================

def _first_by_key(leaves: Dict[str, str], tag: str) -> Optional[str]:
    tag = tag.lower()
    for k, v in leaves.items():
        last = _localname(k.split("/")[-1]).lower()
        if last == tag:
            return v
    return None

def _find_classinstance_with_known_type(root: ET.Element, known_type_text: str) -> Optional[ET.Element]:
    def rec(e: ET.Element) -> Optional[ET.Element]:
        if _localname(e.tag) == "ClassInstance":
            for kh in e.iter():
                if _localname(kh.tag) != "TRTKnownHeader":
                    continue
                for t in kh.iter():
                    if _localname(t.tag) == "Type" and (t.text or "").strip() == known_type_text:
                        return e
        for ch in list(e):
            found = rec(ch)
            if found is not None:
                return found
        return None
    return rec(root)

def _flatten_container_excluding_knownheader(ci: Optional[ET.Element]) -> Dict[str, str]:
    if ci is None:
        return {}
    out: Dict[str, str] = {}
    def rec(e: ET.Element, path: List[str]):
        if _localname(e.tag) == "TRTKnownHeader":
            return
        kids = list(e)
        if kids:
            for k in kids:
                rec(k, path + [k.tag])
        else:
            t = (e.text or "").strip()
            if t:
                out["/".join(path)] = t
    for child in list(ci):
        rec(child, [child.tag])
    return out

def _first_text_in_container(ci: Optional[ET.Element], names: Iterable[str], exclude: Iterable[str] = ()) -> Optional[str]:
    if ci is None:
        return None
    inc = {n.lower() for n in names}
    exc = [s.lower() for s in exclude]
    for e in ci.iter():
        name = _localname(e.tag).lower()
        if name in inc and not any(x in name for x in exc):
            t = (e.text or "").strip()
            if t:
                return t
    return None

def _first_text_in_container_fuzzy(ci: Optional[ET.Element], include: Iterable[str], exclude: Iterable[str] = ()) -> Optional[str]:
    if ci is None:
        return None
    inc = [s.lower() for s in include]
    exc = [s.lower() for s in exclude]
    for e in ci.iter():
        name = _localname(e.tag).lower()
        if any(name == s for s in inc) and not any(x in name for x in exc):
            t = (e.text or "").strip()
            if t:
                return t
    for e in ci.iter():
        name = _localname(e.tag).lower()
        if any(s in name for s in inc) and not any(x in name for x in exc):
            t = (e.text or "").strip()
            if t:
                return t
    return None

def _first_text_global_fuzzy(leaves_all: Dict[str, str], include: Iterable[str], exclude: Iterable[str] = ()) -> Optional[str]:
    inc = [s.lower() for s in include]
    exc = [s.lower() for s in exclude]
    for k, v in leaves_all.items():
        last = _localname(k.split("/")[-1]).lower()
        if any(last == s for s in inc) and not any(x in last for x in exc):
            t = (v or "").strip()
            if t:
                return t
    for k, v in leaves_all.items():
        last = _localname(k.split("/")[-1]).lower()
        if any(s in last for s in inc) and not any(x in last for x in exc):
            t = (v or "").strip()
            if t:
                return t
    return None


# =========================
#   Tube window helpers
# =========================

_Z_TO_SYM = {4:"Be", 13:"Al", 22:"Ti", 24:"Cr", 26:"Fe", 27:"Co", 28:"Ni", 29:"Cu",
             42:"Mo", 45:"Rh", 46:"Pd", 47:"Ag", 74:"W", 78:"Pt", 79:"Au"}

def _z_to_symbol(z: Optional[float]) -> Optional[str]:
    if z is None:
        return None
    zi = int(round(z))
    return _Z_TO_SYM.get(zi)

def _tube_window_from_header_map(xrf_map: Dict[str, str]) -> Optional[Dict[str, Any]]:
    items = {k: v for k, v in xrf_map.items() if _localname(k).lower().startswith("tubewindow/")}
    if not items:
        return None
    filter_id = next((v for k, v in items.items() if k.split("/")[-1].lower() == "filterid"), None)
    def gather(last_tag: str) -> List[str]:
        lt = last_tag.lower()
        return [v for k, v in items.items() if k.split("/")[-1].lower() == lt]
    zs, ths, ras = gather("AtomicNumber"), gather("Thickness"), gather("RelativeArea")
    if (len(zs) <= 1) and (len(ths) <= 1) and (len(ras) <= 1):
        Z = _to_num(zs[0]) if zs else None
        T = _to_num(ths[0]) if ths else None
        R = _to_num(ras[0]) if ras else None
        layer = {"Z": int(round(Z)) if Z is not None else None,
                 "element": _z_to_symbol(Z), "thickness_um": T, "relative_area": R}
        return {"filter_id": filter_id, "layers": [layer]}
    n = max(len(zs), len(ths), len(ras))
    layers = []
    for i in range(n):
        Zi = _to_num(zs[i])  if i < len(zs)  else None
        Ti = _to_num(ths[i]) if i < len(ths) else None
        Ri = _to_num(ras[i]) if i < len(ras) else None
        layers.append({"Z": int(round(Zi)) if Zi is not None else None,
                       "element": _z_to_symbol(Zi), "thickness_um": Ti, "relative_area": Ri})
    return {"filter_id": filter_id, "layers": layers}

def _tube_window_from_leaves(leaves: Dict[str, str]) -> Optional[Dict[str, Any]]:
    items = {k: v for k, v in leaves.items() if "/TubeWindow/" in k}
    if not items:
        return None
    filter_id = next((v for k, v in items.items() if k.split("/")[-1].lower() == "filterid"), None)
    def gather(last_tag: str) -> List[str]:
        lt = last_tag.lower()
        return [v for k, v in items.items() if k.split("/")[-1].lower() == lt]
    zs, ths, ras = gather("AtomicNumber"), gather("Thickness"), gather("RelativeArea")
    if (len(zs) <= 1) and (len(ths) <= 1) and (len(ras) <= 1):
        Z = _to_num(zs[0]) if zs else None
        T = _to_num(ths[0]) if ths else None
        R = _to_num(ras[0]) if ras else None
        layer = {"Z": int(round(Z)) if Z is not None else None,
                 "element": _z_to_symbol(Z), "thickness_um": T, "relative_area": R}
        return {"filter_id": filter_id, "layers": [layer]}
    n = max(len(zs), len(ths), len(ras))
    layers = []
    for i in range(n):
        Zi = _to_num(zs[i]) if i < len(zs) else None
        Ti = _to_num(ths[i]) if i < len(ths) else None
        Ri = _to_num(ras[i]) if i < len(ras) else None
        layers.append({"Z": int(round(Zi)) if Zi is not None else None,
                       "element": _z_to_symbol(Zi), "thickness_um": Ti, "relative_area": Ri})
    return {"filter_id": filter_id, "layers": layers}

def _parse_target_from_tubetype(tubetype: Optional[str]) -> Optional[str]:
    if not tubetype:
        return None
    m = re.search(r"\b([A-Z][a-z]?)\b\s*$", tubetype.strip())
    return m.group(1) if m else None


# =========================
#   Instrument & environment
# =========================

def _extract_instrument_and_env(root: ET.Element, calibration_for_embed: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    leaves_all = _flatten_leaves(root)
    xrf_ci = _find_classinstance_with_known_type(root, "RTXrfHeader")
    xrf = _flatten_container_excluding_knownheader(xrf_ci) if xrf_ci is not None else {}

    # Detector (global leaves)
    det: Dict[str, Any] = {}
    sel_raw = _first_by_key(leaves_all, "SelectedDetectors")
    if sel_raw:
        parts = re.split(r"[,\s;:]+", sel_raw.strip())
        ints = [int(p) for p in parts if p.isdigit()]
        if ints:
            det["detectors_used"] = ints
            det["count"] = det.get("count", len(ints))
    dc = _to_num(_first_by_key(leaves_all, "DetectorCount"))
    if dc is not None:
        det["count"] = int(round(dc))
    for k_xml, k_out in [("Technology","technology"),("Serial","serial"),
                         ("Type","type"),("WindowType","window_type")]:
        v = _first_by_key(leaves_all, k_xml)
        if v is not None: det[k_out] = v
    t_det  = _to_num(_first_by_key(leaves_all, "DetectorThickness"))
    t_dead = _to_num(_first_by_key(leaves_all, "SiDeadLayerThickness"))
    if t_det  is not None: det["thickness"] = t_det
    if t_dead is not None: det["si_dead_layer_thickness"] = t_dead
    for k_xml, k_out in [("ZeroPeakPosition","zero_peak_position"),
                         ("ZeroPeakFrequency","zero_peak_frequency"),
                         ("PulseDensity","pulse_density"),
                         ("Amplification","amplification"),
                         ("ShapingTime","shaping_time")]:
        v = _to_num(_first_by_key(leaves_all, k_xml))
        if v is not None: det[k_out] = v
    # Optional detector HV/current
    for key_xml, out_key, conv in (
        ("DetectorVoltage", "bias_voltage_kV", _to_kV),
        ("BiasVoltage",     "bias_voltage_kV", _to_kV),
        ("HV",              "bias_voltage_kV", _to_kV),
        ("DetectorCurrent", "bias_current_uA", _to_uA),
        ("BiasCurrent",     "bias_current_uA", _to_uA),
    ):
        txt = _first_by_key(leaves_all, key_xml)
        if txt and out_key not in det:
            v,u = _parse_value_and_unit(txt)
            if v is not None:
                det[out_key] = conv(v,u)
    if calibration_for_embed:
        det["calibration"] = {
            "channel_count": calibration_for_embed.get("channel_count"),
            "gain_keV_per_ch": calibration_for_embed.get("gain_keV_per_ch"),
            "offset_keV": calibration_for_embed.get("offset_keV"),
        }

    # X-ray tube (prefer container values)
    tube: Dict[str, Any] = {}
    tubetype = (xrf.get("TubeType") if xrf else None) or _first_by_key(leaves_all, "TubeType")
    if tubetype:
        tube["type"] = tubetype
        tgt_sym = _parse_target_from_tubetype(tubetype)
        if tgt_sym: tube.setdefault("target", tgt_sym)
    tube["serial_number"]   = (xrf.get("TubeNumber") if xrf else None)   or _first_by_key(leaves_all, "TubeNumber")  or tube.get("serial_number")
    tube["production_date"] = (xrf.get("TubeProdDate") if xrf else None) or _first_by_key(leaves_all, "TubeProdDate") or tube.get("production_date")
    anode = _to_num((xrf.get("Anode") if xrf else None) or _first_by_key(leaves_all, "Anode"))
    if anode is not None:
        tube["anode_Z"] = int(round(anode))
        tube.setdefault("target", _z_to_symbol(anode))
    vol_txt = (_first_text_in_container(xrf_ci, {"Voltage"})
               or (xrf.get("Voltage") if xrf else None)
               or _first_text_global_fuzzy(leaves_all, include={"Voltage", "TubeVoltage"}))
    if vol_txt is not None:
        v, u = _parse_value_and_unit(vol_txt)
        if v is not None:
            tube["voltage_kV"] = _to_kV(v, u)
    cur_txt = (_first_text_in_container_fuzzy(xrf_ci, include={"Current", "TubeCurrent"}, exclude={"detector", "bias"})
               or (xrf.get("Current") if xrf else None)
               or _first_text_global_fuzzy(leaves_all, include={"Current", "TubeCurrent"}, exclude={"detector", "bias"}))
    if cur_txt is not None:
        v, u = _parse_value_and_unit(cur_txt)
        if v is not None:
            tube["current_uA"] = _to_uA(v, u)
    inc = _to_num((xrf.get("TubeIncidentAngle") if xrf else None) or _first_by_key(leaves_all, "TubeIncidentAngle"))
    if inc is not None: tube["incident_angle_deg"] = inc
    tof = _to_num((xrf.get("TubeTakeOffAngle") if xrf else None) or _first_by_key(leaves_all, "TubeTakeOffAngle"))
    if tof is not None: tube["takeoff_angle_deg"] = tof
    window = _tube_window_from_header_map(xrf) if xrf else None
    if not window:
        window = _tube_window_from_leaves(leaves_all)
    if window:
        tube["window"] = window
    filt_ids: List[str] = []
    if xrf:
        filt_ids = [v for k, v in xrf.items() if k.lower().endswith("excitationfilter/filterid")]
    if not filt_ids:
        filt_ids = [v for k, v in leaves_all.items()
                    if "/ExcitationFilter/" in k and k.split("/")[-1].lower() == "filterid"]
    if filt_ids:
        tube["excitation_filters"] = filt_ids

    instrument = {}
    if tube: instrument["xray_tube"] = tube
    if det:  instrument["detector"] = det

    # Environment
    env: Dict[str, Any] = {}
    ch_type = xrf.get("ChassisType")   or _first_by_key(leaves_all, "ChassisType")
    ch_num  = xrf.get("ChassisNumber") or _first_by_key(leaves_all, "ChassisNumber")
    ch_date = xrf.get("ChassisProdDate") or _first_by_key(leaves_all, "ChassisProdDate")
    if ch_type or ch_num or ch_date:
        env["chassis"] = {"type": ch_type, "number": ch_num, "production_date": ch_date}
    geom: Dict[str, Any] = {}
    def putg(name_xml: str, out_key: str):
        v = _to_num(xrf.get(name_xml) or _first_by_key(leaves_all, name_xml))
        if v is not None:
            geom[out_key] = v
    for name_xml, out_key in [
        ("AzimutAngleAbs","azimuth_angle_abs_deg"),
        ("ExcitationAngle","excitation_angle_deg"),
        ("CollimatorDiameter","collimator_diameter"),
        ("DetectionAngle","detection_angle_deg"),
        ("TiltAngle","tilt_angle_deg"),
        ("DetAzimutAngle","detector_azimuth_angle_deg"),
        ("ExcitationPathLength","excitation_path_length"),
        ("DetectionPathLength","detection_path_length"),
        ("SolidAngleDetection","solid_angle_detection"),
        ("DetSpotSize","detector_spot_size"),
    ]:
        putg(name_xml, out_key)
    if geom:
        env["geometry"] = geom
    atm = xrf.get("Atmosphere") or _first_by_key(leaves_all, "Atmosphere")
    if atm:
        env["atmosphere"] = atm
    press_txt = (xrf.get("ChamberPressure") or _first_by_key(leaves_all, "ChamberPressure")
                 or _first_text_global_fuzzy(leaves_all, include={"ChamberPressure"}))
    if press_txt is not None:
        press = _to_num(press_txt)
        if press is not None:
            env["ChamberPressure_mbar"] = press
    flow_txt = xrf.get("FlowRate") or _first_by_key(leaves_all, "FlowRate")
    if flow_txt is not None:
        flow = _to_num(flow_txt)
        if flow is not None:
            env["flow_rate"] = flow

    return instrument, env


# =========================
#   Top-level parse & helpers
# =========================

def parse_spx_file(path: str, encoding: str = "cp1252") -> Dict[str, Any]:
    root = _load_spx_root(path, encoding=encoding)

    # Spectrum name
    spec_name = None
    for ci in root.iter():
        if _localname(ci.tag) == "ClassInstance":
            if ci.attrib.get("Type") == "TRTSpectrum" and "Name" in ci.attrib:
                spec_name = ci.attrib["Name"]
                break
    spec_name = spec_name or Path(path).stem

    # Acquisition
    acquisition = {
        "real_time_ms": _to_num(_find_text_anywhere(root, "RealTime")),
        "live_time_ms": _to_num(_find_text_anywhere(root, "LifeTime")),
        "dead_time_percent": _to_num(_find_text_anywhere(root, "DeadTime")),
    }

    # Calibration
    channel_count_txt = _find_text_anywhere(root, "ChannelCount")
    calibration = {
        "channel_count": int(float(channel_count_txt)) if channel_count_txt else None,
        "offset_keV": _to_num(_find_text_anywhere(root, "CalibAbs")),
        "gain_keV_per_ch": _to_num(_find_text_anywhere(root, "CalibLin")),
    }

    # Counts
    counts: List[int] = []
    ch_text = _find_text_anywhere(root, "Channels")
    if ch_text:
        for t in ch_text.split(","):
            t = t.strip()
            if t:
                try:
                    counts.append(int(float(t)))
                except Exception:
                    pass

    # Stage
    stage = _extract_stage_any(path) or None

    # Instrument + environment
    instrument, env = _extract_instrument_and_env(root, calibration_for_embed=calibration)

    return {
        "spectrum_name": spec_name,
        "file": str(Path(path).name),
        "acquisition": acquisition,
        "calibration": calibration,
        "instrument": instrument or None,
        "measurement_environment": env or None,
        "stage": stage,
        "counts": counts,
        "counts_len": len(counts),
    }

def energy_axis_keV(rec: Dict[str, Any]) -> np.ndarray:
    cal = rec.get("calibration") or {}
    offset = float(cal.get("offset_keV") or 0.0)
    gain   = float(cal.get("gain_keV_per_ch") or 0.01)
    n = int(rec.get("counts_len") or (len(rec.get("counts") or [])))
    ch = np.arange(n, dtype=float)
    return offset + gain * ch


# =========================
#   Streamlit UI
# =========================

st.set_page_config(page_title="SPX viewer", layout="wide")
st.title("SPX spectrum & metadata viewer")

with st.sidebar:
    st.header("Load file")
    use_default = st.checkbox("Use default sample (data/sample.spx)", value=True)
    uploaded = st.file_uploader("...or upload .spx", type=["spx"])
    default_path = st.text_input("Default sample path", value="data/sample.spx")
    st.caption("If you upload a file, it takes precedence over the default.")

    st.header("Display options")
    show_meta = st.checkbox("Show metadata", value=True)
    show_plot = st.checkbox("Show spectrum plot", value=True)

# Resolve input path
tmp_path = None
if uploaded is not None:
    # Save upload to a temp file for parsing
    tmp_path = Path(st.experimental_get_query_params().get("tmpdir", ["."])[0]) / ("uploaded.spx")
    tmp_path.write_bytes(uploaded.getvalue())
    spx_path = str(tmp_path)
elif use_default:
    spx_path = default_path
else:
    st.info("Please upload a .spx or enable the default sample.")
    st.stop()

# Parse and show
try:
    rec = parse_spx_file(spx_path)
except Exception as e:
    st.error("Failed to parse the .spx file.")
    st.exception(e)
    st.stop()

col1, col2 = st.columns([1,1])

with col1:
    st.subheader("Header")
    st.write(f"**Spectrum:** {rec['spectrum_name']}")
    st.write(f"**File:** {rec['file']}")
    acq = rec.get("acquisition", {}) or {}
    st.write(f"**Acquisition:** real={acq.get('real_time_ms')} ms, live={acq.get('live_time_ms')} ms, dead={acq.get('dead_time_percent')}%")
    st.write("**Stage (mm):**", rec.get("stage"))

if show_meta:
    with col2:
        st.subheader("Metadata")
        inst = rec.get("instrument") or {}
        env  = rec.get("measurement_environment") or {}
        st.markdown("**Instrument → X-ray tube**")
        st.json(inst.get("xray_tube") or {})
        st.markdown("**Instrument → Detector**")
        st.json(inst.get("detector") or {})
        st.markdown("**Measurement environment**")
        st.json(env)

if show_plot:
    counts = np.array(rec.get("counts") or [], dtype=float)
    if counts.size == 0:
        st.warning("No channel counts found in the file.")
    else:
        x = energy_axis_keV(rec)
        fig = plt.figure(figsize=(8,4))
        ax = fig.add_subplot(111)
        ax.plot(x, counts, lw=1)
        ax.set_xlabel("Energy (keV)")
        ax.set_ylabel("Counts")
        ax.set_title(f"Spectrum: {rec['spectrum_name']}")
        ax.grid(True, alpha=0.3)
        st.pyplot(fig, clear_figure=True)

# Footer: raw JSON expander
with st.expander("Raw parsed record (JSON)"):
    st.code(json.dumps(rec, indent=2), language="json")
