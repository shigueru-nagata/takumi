import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple


@dataclass
class BandState:
    index: int
    energy: float
    occupation: float


@dataclass
class KPointBlock:
    index: int
    coords: Tuple[float, float, float]
    weight: Optional[float] = None
    bands: List[BandState] = field(default_factory=list)


@dataclass
class ElectronicStructureSummary:
    num_spins: int
    num_kpoints: int
    num_bands: int
    # Estructura: spin_kpoints[spin][kpoint_index] = KPointBlock
    spin_kpoints: Dict[int, Dict[int, KPointBlock]]
    vbm_energy: Optional[float] = None
    vbm_location: Optional[Tuple[int, int, int]] = None  # (spin, kpt, band)
    cbm_energy: Optional[float] = None
    cbm_location: Optional[Tuple[int, int, int]] = None  # (spin, kpt, band)
    is_direct: Optional[bool] = None
    bandgap: Optional[float] = None


# =========================================================================
# Funciones para el comando 'bands' (inspección del último bloque)
# =========================================================================

def parse_last_eigenvalues(filepath: Path) -> List[BandState]:
    """Extracts eigenvalues and occupations from the last occurrence of 'band No.' in OUTCAR."""
    if not filepath.is_file():
        raise FileNotFoundError(f"File not found: {filepath}")

    with filepath.open("r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    last_band_line_idx = -1
    for idx in range(len(lines) - 1, -1, -1):
        if "band No." in lines[idx]:
            last_band_line_idx = idx
            break

    if last_band_line_idx == -1:
        raise ValueError(f"Could not find any 'band No.' block in {filepath.name}")

    bands: List[BandState] = []
    for line in lines[last_band_line_idx + 1 :]:
        parts = line.split()
        if len(parts) >= 3:
            try:
                b_idx = int(parts[0])
                energy = float(parts[1])
                occ = float(parts[2])
                bands.append(BandState(index=b_idx, energy=energy, occupation=occ))
            except ValueError:
                break
        else:
            break

    return bands


def find_occupation_transition(
    bands: List[BandState], tolerance: float = 0.05
) -> Optional[int]:
    """Finds the index of the last occupied band before transitioning to empty states."""
    if not bands:
        return None

    max_occ = max(b.occupation for b in bands)
    if max_occ <= 0.0:
        return None

    half_occ_threshold = max_occ / 2.0

    for i in range(len(bands) - 1):
        if bands[i].occupation >= half_occ_threshold and bands[i + 1].occupation < half_occ_threshold:
            return i

    for i in range(len(bands) - 1):
        if bands[i].occupation > tolerance and bands[i + 1].occupation <= tolerance:
            return i

    return None


# =========================================================================
# Funciones para el comando 'scan' (análisis global de k-points y espines)
# =========================================================================

def parse_all_kpoints_bands(filepath: Path) -> ElectronicStructureSummary:
    """Parses all spin components, k-points, and bands from the last electronic step in OUTCAR."""
    if not filepath.is_file():
        raise FileNotFoundError(f"File not found: {filepath}")

    with filepath.open("r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    # 1. Determinar si el cálculo es ISPIN=2 buscando en el archivo
    ispin_2 = False
    for line in lines:
        if "ISPIN" in line and "=" in line:
            parts = line.split("=")
            if len(parts) > 1 and parts[1].strip().startswith("2"):
                ispin_2 = True
                break

    # 2. Localizar el inicio del último bloque completo de bandas
    # Si ISPIN=2, el bloque comienza en la última aparición de "spin component 1"
    # Si ISPIN=1, comienza en la última aparición de "k-point     1" antes del final
    target_start = "spin component 1" if ispin_2 else "k-point     1"
    
    last_eval_idx = -1
    for idx in range(len(lines) - 1, -1, -1):
        line = lines[idx]
        if target_start in line and not line.strip().startswith("soft"):
            last_eval_idx = idx
            break

    # Fallback por si el formato de espacios varía
    if last_eval_idx == -1:
        for idx in range(len(lines) - 1, -1, -1):
            if "k-point" in lines[idx] and "1 :" in lines[idx]:
                last_eval_idx = idx
                break

    if last_eval_idx == -1:
        raise ValueError(f"No eigenvalue blocks found in {filepath.name}")

    spin_kpoints: Dict[int, Dict[int, KPointBlock]] = {}
    current_spin = 1
    current_kblock: Optional[KPointBlock] = None
    reading_bands = False

    kpoint_regex = re.compile(
        r"k-point\s+(\d+)\s*:\s+([-\d\.]+)\s+([-\d\.]+)\s+([-\d\.]+)"
    )

    for line in lines[last_eval_idx:]:
        stripped = line.strip()

        # Condición de salida estricta: fin del bloque de autovalores
        if stripped.startswith("soft charge-density") or "General timing and accounting" in stripped:
            break

        # Detectar cambio de canal de espín
        if stripped.startswith("spin component"):
            parts = stripped.split()
            try:
                current_spin = int(parts[-1])
            except ValueError:
                pass
            reading_bands = False
            continue

        # Detectar k-point
        k_match = kpoint_regex.search(stripped)
        if k_match:
            k_idx = int(k_match.group(1))
            kx, ky, kz = (
                float(k_match.group(2)),
                float(k_match.group(3)),
                float(k_match.group(4)),
            )
            current_kblock = KPointBlock(index=k_idx, coords=(kx, ky, kz), weight=None)
            spin_kpoints.setdefault(current_spin, {})[k_idx] = current_kblock
            reading_bands = False
            continue

        if "band No." in stripped:
            reading_bands = True
            continue

        if reading_bands and current_kblock is not None:
            parts = stripped.split()
            if len(parts) >= 3:
                try:
                    b_idx = int(parts[0])
                    b_energy = float(parts[1])
                    b_occ = float(parts[2])
                    current_kblock.bands.append(
                        BandState(index=b_idx, energy=b_energy, occupation=b_occ)
                    )
                except ValueError:
                    reading_bands = False
            else:
                reading_bands = False

    num_spins = len(spin_kpoints)
    num_kpoints = len(spin_kpoints[1]) if 1 in spin_kpoints else 0
    num_bands = (
        len(spin_kpoints[1][1].bands)
        if 1 in spin_kpoints and 1 in spin_kpoints[1]
        else 0
    )

    # Ocupación máxima para definir umbral VBM/CBM
    max_occ = 0.0
    for s in spin_kpoints.values():
        for k in s.values():
            for b in k.bands:
                if b.occupation > max_occ:
                    max_occ = b.occupation

    occ_threshold = max_occ / 2.0 if max_occ > 0 else 0.5

    vbm_energy = -1e9
    vbm_loc = None
    cbm_energy = 1e9
    cbm_loc = None

    for spin_id, kpts in spin_kpoints.items():
        for k_id, kblock in kpts.items():
            for b in kblock.bands:
                if b.occupation >= occ_threshold:
                    if b.energy > vbm_energy:
                        vbm_energy = b.energy
                        vbm_loc = (spin_id, k_id, b.index)
                else:
                    if b.energy < cbm_energy:
                        cbm_energy = b.energy
                        cbm_loc = (spin_id, k_id, b.index)

    gap = None
    is_direct = None
    if vbm_loc and cbm_loc and cbm_energy > vbm_energy:
        gap = cbm_energy - vbm_energy
        is_direct = (vbm_loc[0] == cbm_loc[0]) and (vbm_loc[1] == cbm_loc[1])

    return ElectronicStructureSummary(
        num_spins=num_spins,
        num_kpoints=num_kpoints,
        num_bands=num_bands,
        spin_kpoints=spin_kpoints,
        vbm_energy=vbm_energy if vbm_loc else None,
        vbm_location=vbm_loc,
        cbm_energy=cbm_energy if cbm_loc else None,
        cbm_location=cbm_loc,
        is_direct=is_direct,
        bandgap=gap,
    )