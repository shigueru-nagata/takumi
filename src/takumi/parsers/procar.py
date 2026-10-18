from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class IonProjection:
    ion_index: int
    element: str
    orbitals: Dict[str, float]
    total: float


@dataclass
class BandProjectionSummary:
    spin: int
    kpoint: int
    band: int
    energy: float
    occupation: float
    orbital_headers: List[str]
    ions: List[IonProjection]
    elemental_contributions: Dict[str, float]
    total_band_projection: float


def parse_poscar_species(poscar_path: Path) -> List[str]:
    """Reads atom species and counts from POSCAR/CONTCAR and returns a flat list of element symbols per atom."""
    if not poscar_path.is_file():
        return []

    with poscar_path.open("r", encoding="utf-8", errors="ignore") as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]

    if len(lines) < 7:
        return []

    # Lógica estándar VASP 5+:
    # Línea 6: Especies (ej. B N C)
    # Línea 7: Cantidades (ej. 36 35 1)
    species = lines[5].split()
    counts_str = lines[6].split()

    if species[0].isdigit():
        return []

    try:
        counts = [int(c) for c in counts_str]
    except ValueError:
        return []

    atom_labels = []
    for elem, count in zip(species, counts):
        atom_labels.extend([elem] * count)

    return atom_labels


def parse_procar_band(
    procar_path: Path,
    kpoint_num: int,
    band_num: int,
    spin_channel: int = 1,
    poscar_path: Optional[Path] = None,
) -> BandProjectionSummary:
    """Extracts orbital projections for a specific k-point, band, and spin channel from PROCAR."""
    if not procar_path.is_file():
        raise FileNotFoundError(f"File not found: {procar_path}")

    species_list = parse_poscar_species(poscar_path) if poscar_path else []

    with procar_path.open("r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    current_spin = 1
    target_k_found = False
    target_band_found = False

    energy = 0.0
    occ = 0.0
    total_proj = 0.0
    orbital_headers: List[str] = []
    ions: List[IonProjection] = []
    reading_ions = False

    for line in lines:
        stripped = line.strip()

        if "spin component" in stripped:
            parts = stripped.split()
            try:
                current_spin = int(parts[-1])
            except ValueError:
                pass
            continue

        if current_spin != spin_channel:
            continue

        if stripped.startswith("k-point") and not target_band_found:
            parts = stripped.split()
            try:
                kp_idx = int(parts[1])
                target_k_found = (kp_idx == kpoint_num)
            except (ValueError, IndexError):
                pass
            continue

        if target_k_found and stripped.startswith("band") and not target_band_found:
            parts = stripped.split()
            try:
                b_idx = int(parts[1])
                if b_idx == band_num:
                    target_band_found = True
                    for i, p in enumerate(parts):
                        if p == "energy":
                            energy = float(parts[i + 1])
                        elif p.startswith("occ"):
                            occ = float(parts[i + 1])
            except (ValueError, IndexError):
                pass
            continue

        if target_band_found:
            if stripped.startswith("ion"):
                parts = stripped.split()
                orbital_headers = parts[1:-1]
                reading_ions = True
                continue

            if reading_ions:
                parts = stripped.split()
                if not parts:
                    continue

                if parts[0] == "tot":
                    try:
                        total_proj = float(parts[-1])
                    except ValueError:
                        pass
                    break

                try:
                    ion_idx = int(parts[0])
                    tot_val = float(parts[-1])
                    orb_vals = [float(x) for x in parts[1:-1]]
                    elem = (
                        species_list[ion_idx - 1]
                        if ion_idx - 1 < len(species_list)
                        else "Ion"
                    )

                    orbitals_dict = dict(zip(orbital_headers, orb_vals))
                    ions.append(
                        IonProjection(
                            ion_index=ion_idx,
                            element=elem,
                            orbitals=orbitals_dict,
                            total=tot_val,
                        )
                    )
                except ValueError:
                    continue

    if not target_band_found:
        raise ValueError(
            f"Could not find k-point {kpoint_num}, band {band_num} (Spin {spin_channel}) in {procar_path.name}"
        )

    elemental_totals: Dict[str, float] = {}
    for ion in ions:
        elemental_totals[ion.element] = elemental_totals.get(ion.element, 0.0) + ion.total

    return BandProjectionSummary(
        spin=spin_channel,
        kpoint=kpoint_num,
        band=band_num,
        energy=energy,
        occupation=occ,
        orbital_headers=orbital_headers,
        ions=ions,
        elemental_contributions=elemental_totals,
        total_band_projection=total_proj,
    )

@dataclass
class IonTrackResult:
    spin: int
    kpoint: int
    band: int
    energy: float
    occupation: float
    total_band_weight: float
    orbital_headers: List[str]
    matched_ions: List[IonProjection]


def track_ions_in_procar(
    procar_path: Path,
    target_ions: List[int],
    min_ratio_percent: float = 5.0
) -> List[IonTrackResult]:
    """Scans the entire PROCAR to find specific ions contributing more than a given percentage."""
    if not procar_path.is_file():
        raise FileNotFoundError(f"File not found: {procar_path}")

    target_ions_set = set(target_ions)
    results = []

    current_spin = 1
    current_k = 0
    current_b = 0
    energy = 0.0
    occ = 0.0
    orbital_headers: List[str] = []
    buffered_ions: List[IonProjection] = []
    reading_ions = False

    with procar_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue

            if "spin component" in stripped:
                parts = stripped.split()
                try:
                    current_spin = int(parts[-1])
                except ValueError:
                    pass
                continue

            if stripped.startswith("k-point"):
                parts = stripped.split()
                try:
                    current_k = int(parts[1])
                except (ValueError, IndexError):
                    pass
                continue

            if stripped.startswith("band"):
                parts = stripped.split()
                try:
                    current_b = int(parts[1])
                    for i, p in enumerate(parts):
                        if p == "energy":
                            energy = float(parts[i + 1])
                        elif p.startswith("occ"):
                            occ = float(parts[i + 1])
                except (ValueError, IndexError):
                    pass
                buffered_ions = []
                continue

            if stripped.startswith("ion"):
                parts = stripped.split()
                orbital_headers = parts[1:-1]
                reading_ions = True
                continue

            if reading_ions:
                parts = stripped.split()
                if not parts:
                    continue
                
                if parts[0] == "tot":
                    reading_ions = False
                    try:
                        tot_val = float(parts[-1])
                        valid_ions = []
                        # Solo guardamos los iones que superan el umbral porcentual
                        for ion in buffered_ions:
                            ratio = (ion.total / tot_val * 100) if tot_val > 0 else 0.0
                            if ratio >= min_ratio_percent:
                                valid_ions.append(ion)
                        
                        if valid_ions:
                            results.append(
                                IonTrackResult(
                                    spin=current_spin,
                                    kpoint=current_k,
                                    band=current_b,
                                    energy=energy,
                                    occupation=occ,
                                    total_band_weight=tot_val,
                                    orbital_headers=orbital_headers,
                                    matched_ions=valid_ions
                                )
                            )
                    except ValueError:
                        pass
                    continue
                
                try:
                    ion_idx = int(parts[0])
                    # Si el ion está en nuestra lista objetivo, lo guardamos temporalmente
                    if ion_idx in target_ions_set:
                        tot_val = float(parts[-1])
                        orb_vals = [float(x) for x in parts[1:-1]]
                        buffered_ions.append(
                            IonProjection(
                                ion_index=ion_idx,
                                element="", 
                                orbitals=dict(zip(orbital_headers, orb_vals)),
                                total=tot_val
                            )
                        )
                except ValueError:
                    pass

    return results