from pathlib import Path
from typing import Optional, List
import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from takumi.parsers.outcar import (
    find_occupation_transition,
    parse_all_kpoints_bands,
    parse_last_eigenvalues,
)

from takumi.parsers.procar import (
    parse_procar_band,
    parse_poscar_species,
    track_ions_in_procar
    )

app = typer.Typer(
    add_completion=False,
    rich_markup_mode="rich",
)
console = Console()


@app.callback()
def main():
    """Takumi (匠) - Handcrafted CLI for VASP post-processing and data analysis."""
    pass


@app.command()
def info():
    """Check Takumi environment status."""
    console.print(
        "[bold cyan]Takumi (匠)[/bold cyan] [green]is ready and operational.[/green]"
    )


@app.command(
    epilog="""
[bold underline yellow]Examples:[/bold underline yellow]

[dim]# Inspect default 30 bands and default OUTCAR file:[/dim]

[bold cyan]takumi bands[/bold cyan]

[dim]# Read a specific file and inspect default 30 bands:[/dim]

[bold cyan]takumi bands -f OUTCAR_scf[/bold cyan]

[dim]# Read a specific file and inspect first 100 bands:[/dim]

[bold cyan]takumi bands -f OUTCAR_scf -n 100[/bold cyan]

[dim]# Focus automatically around occupied-to-unoccupied transition (±5 bands) in default OUTCAR file:[/dim]

[bold cyan]takumi bands -w 5[/bold cyan]

[dim]# Focus automatically around occupied-to-unoccupied transition (±5 bands) in specific OUTCAR file:[/dim]

[bold cyan]takumi bands -f OUTCAR_scf -w 5[/bold cyan]
"""
)
def bands(
    file: Path = typer.Option(
        Path("OUTCAR"),
        "--file",
        "-f",
        help="Path to the VASP OUTCAR file. Defaults to 'OUTCAR'.",
        dir_okay=False,
    ),
    lines: int = typer.Option(
        30,
        "--lines",
        "-n",
        help="Number of band levels to display starting from band 1.",
    ),
    window: int = typer.Option(
        0,
        "--window",
        "-w",
        help="If set (>0), centers output around the occupation transition (± N lines).",
    ),
):
    """Inspect band energies, occupations, and the occupied-to-unoccupied transition."""
    if not file.exists():
        console.print(f"[bold red]Error:[/bold red] File '{file}' not found.")
        raise typer.Exit(code=1)

    console.print(f"[dim]Reading band data from: [bold]{file.resolve()}[/bold][/dim]")

    try:
        band_data = parse_last_eigenvalues(file)
    except Exception as exc:
        console.print(f"[bold red]Error parsing file:[/bold red] {exc}")
        raise typer.Exit(code=1)

    if not band_data:
        console.print("[yellow]Warning: No eigenvalue records found.[/yellow]")
        raise typer.Exit()

    # Detect the transition point adaptively
    transition_idx = find_occupation_transition(band_data)
    max_occ = max(b.occupation for b in band_data)

    # Determine slice of bands to display
    if window > 0:
        if transition_idx is None:
            console.print(
                "[yellow]No clear occupation transition detected. Displaying top bands instead.[/yellow]"
            )
            selected_bands = band_data[:lines]
        else:
            start = max(0, transition_idx - window + 1)
            end = min(len(band_data), transition_idx + window + 1)
            selected_bands = band_data[start:end]
    else:
        selected_bands = band_data[:lines]

    table = Table(title=f"Band Occupations ({file.name})", show_header=True)
    table.add_column("Band No.", justify="right", style="cyan")
    table.add_column("Energy (eV)", justify="right", style="white")
    table.add_column("Occupation", justify="right")
    table.add_column("State", justify="left")

    for b in selected_bands:
        is_last_occ = (
            transition_idx is not None and b.index == band_data[transition_idx].index
        )
        is_first_unocc = (
            transition_idx is not None
            and b.index == band_data[transition_idx + 1].index
        )

        if is_last_occ:
            style = "bold green"
            state_label = "[bold green]VBM / HOMO[/bold green]"
        elif is_first_unocc:
            style = "bold yellow"
            state_label = "[bold yellow]CBM / LUMO[/bold yellow]"
        elif b.occupation >= max_occ * 0.95:
            style = "green"
            state_label = "Full"
        elif b.occupation > 0.05:
            style = "cyan"
            state_label = "Partial"
        else:
            style = "dim"
            state_label = "Empty"

        table.add_row(
            f"[{style}]{b.index}[/{style}]",
            f"[{style}]{b.energy:10.4f}[/{style}]",
            f"[{style}]{b.occupation:8.5f}[/{style}]",
            state_label,
        )

    console.print(table)

    if transition_idx is not None:
        vbm = band_data[transition_idx].energy
        cbm = band_data[transition_idx + 1].energy
        gap = cbm - vbm
        console.print(
            f"\n[bold]Detected Gap at k-point:[/bold] [magenta]{gap:.4f} eV[/magenta] "
            f"(Band {band_data[transition_idx].index} -> Band {band_data[transition_idx + 1].index})"
        )

@app.command(
    epilog="""
[bold underline yellow]Examples:[/bold underline yellow]

  [dim]# Scan all k-points with automatic focus around the frontier:[/dim]
  [bold cyan]takumi scan[/bold cyan]

  [dim]# Inspect specific band window using initial and final bands:[/dim]
  [bold cyan]takumi scan -f OUTCAR --bi 140 --bf 147[/bold cyan]
"""
)
def scan(
    file: Path = typer.Option(
        Path("OUTCAR"),
        "--file",
        "-f",
        help="Path to the OUTCAR file.",
        dir_okay=False,
    ),
    band_init: Optional[int] = typer.Option(
        None,
        "--band-init",
        "--bi",
        help="Initial band index to display. Defaults to VBM-3.",
    ),
    band_final: Optional[int] = typer.Option(
        None,
        "--band-final",
        "--bf",
        help="Final band index to display. Defaults to CBM+3.",
    ),
):
    """Scan band energies across all k-points and spin channels to identify global gap character."""
    if not file.exists():
        console.print(f"[bold red]Error:[/bold red] File '{file}' not found.")
        raise typer.Exit(code=1)

    console.print(f"[dim]Analyzing electronic structure in: [bold]{file.resolve()}[/bold][/dim]")

    summary = parse_all_kpoints_bands(file)

    summary_text = (
        f"[bold]Spins:[/bold] {summary.num_spins} | "
        f"[bold]k-points:[/bold] {summary.num_kpoints} | "
        f"[bold]Total Bands:[/bold] {summary.num_bands}\n"
    )

    if summary.vbm_location and summary.cbm_location:
        v_s, v_k, v_b = summary.vbm_location
        c_s, c_k, c_b = summary.cbm_location
        gap_type = (
            "[bold green]Direct[/bold green]"
            if summary.is_direct
            else "[bold yellow]Indirect[/bold yellow]"
        )

        summary_text += (
            f"[bold]VBM:[/bold] [green]{summary.vbm_energy:.4f} eV[/green] (Spin {v_s}, k-point {v_k}, Band {v_b})\n"
            f"[bold]CBM:[/bold] [yellow]{summary.cbm_energy:.4f} eV[/yellow] (Spin {c_s}, k-point {c_k}, Band {c_b})\n"
            f"[bold]Fundamental Gap:[/bold] [magenta]{summary.bandgap:.4f} eV[/magenta] ({gap_type})"
        )
    else:
        summary_text += "[yellow]Could not determine a distinct bandgap (metallic or zero states).[/yellow]"

    console.print(Panel(summary_text, title="Electronic Structure Summary", expand=False))

    b_start = band_init
    b_end = band_final

    if b_start is None or b_end is None:
        if summary.vbm_location and summary.cbm_location:
            ref_vbm = summary.vbm_location[2]
            ref_cbm = summary.cbm_location[2]
            b_start = b_start or max(1, min(ref_vbm, ref_cbm) - 3)
            b_end = b_end or min(summary.num_bands, max(ref_vbm, ref_cbm) + 3)
        else:
            b_start = b_start or 1
            b_end = b_end or min(summary.num_bands, 20)

    table = Table(
        title=f"Band Window [{b_start} - {b_end}] across k-points",
        show_header=True,
    )
    table.add_column("Spin", justify="center", style="cyan")
    table.add_column("k-point", justify="center", style="blue")
    table.add_column("Band", justify="right", style="white")
    table.add_column("Energy (eV)", justify="right")
    table.add_column("Occupation", justify="right")
    table.add_column("State", justify="left")

    # Iterar y construir la tabla con divisores entre k-points
    total_spins = len(summary.spin_kpoints)
    
    for spin_id, kpts in sorted(summary.spin_kpoints.items()):
        total_kpts = len(kpts)
        for k_id, kblock in sorted(kpts.items()):
            # Filtrar las bandas dentro de la ventana seleccionada
            bands_in_window = [b for b in kblock.bands if b_start <= b.index <= b_end]
            
            for idx_b, b in enumerate(bands_in_window):
                is_vbm = summary.vbm_location == (spin_id, k_id, b.index)
                is_cbm = summary.cbm_location == (spin_id, k_id, b.index)

                if is_vbm:
                    style = "bold green"
                    label = "[bold green]VBM (Global)[/bold green]"
                elif is_cbm:
                    style = "bold yellow"
                    label = "[bold yellow]CBM (Global)[/bold yellow]"
                elif b.occupation > 0.5:
                    style = "green"
                    label = "Occupied"
                else:
                    style = "dim"
                    label = "Empty"

                # Si es la última banda del bloque de este k-point, dibujamos la línea divisoria
                is_last_in_kblock = (idx_b == len(bands_in_window) - 1)
                
                # No colocar divisor al final de la última fila de toda la tabla
                is_last_overall = (spin_id == total_spins and k_id == total_kpts and is_last_in_kblock)
                draw_divider = is_last_in_kblock and not is_last_overall

                table.add_row(
                    str(spin_id),
                    str(k_id),
                    f"[{style}]{b.index}[/{style}]",
                    f"[{style}]{b.energy:10.4f}[/{style}]",
                    f"[{style}]{b.occupation:8.5f}[/{style}]",
                    label,
                    end_section=draw_divider,
                )

    console.print(table)

@app.command(
    epilog="""
[bold underline yellow]Examples:[/bold underline yellow]

  [dim]# Inspect band 144 at k-point 1 (defaults to top 10 contributing ions):[/dim]
  [bold cyan]takumi proj -k 1 -b 144[/bold cyan]

  [dim]# Inspect with custom POSCAR and show all ions with contribution > 0.01 (1%):[/dim]
  [bold cyan]takumi proj -p PROCAR -c POSCAR_defect -k 1 -b 144 --threshold 0.01[/bold cyan]

  [dim]# Show all ions without filtering (equivalent to original awk script):[/dim]
  [bold cyan]takumi proj -k 1 -b 144 --all[/bold cyan]
"""
)
def proj(
    kpoint: int = typer.Option(
        ...,
        "--kpoint",
        "-k",
        help="Target k-point index (NUM_KP).",
    ),
    band: int = typer.Option(
        ...,
        "--band",
        "-b",
        help="Target band index (NUM_BAND).",
    ),
    spin: int = typer.Option(
        1,
        "--spin",
        "-s",
        help="Spin channel (1 for spin up or non-polarized, 2 for spin down).",
    ),
    file: Path = typer.Option(
        Path("PROCAR"),
        "--file",
        "-p",
        help="Path to PROCAR file. Defaults to 'PROCAR'.",
        dir_okay=False,
    ),
    poscar: Path = typer.Option(
        Path("POSCAR"),
        "--poscar",
        "-c",
        help="Path to POSCAR/CONTCAR for atom species mapping. Defaults to 'POSCAR'.",
        dir_okay=False,
    ),
    top: int = typer.Option(
        10,
        "--top",
        "-t",
        help="Show only the top N most contributing atoms. Defaults to 10.",
    ),
    threshold: float = typer.Option(
        0.000,
        "--threshold",
        "-m",
        help="Minimum total projection value to display an atom.",
    ),
    show_all: bool = typer.Option(
        False,
        "--all",
        "-a",
        help="Display all ions without filtering.",
    ),
):
    """Analyze orbital and atomic projections for a specific band and k-point (defect localization)."""
    if not file.exists():
        console.print(f"[bold red]Error:[/bold red] PROCAR file '{file}' not found.")
        raise typer.Exit(code=1)

    poscar_path = poscar if poscar.exists() else None
    if not poscar_path:
        console.print(
            "[dim yellow]Warning: POSCAR not found. Atoms will be displayed by index only.[/dim yellow]"
        )

    try:
        summary = parse_procar_band(
            procar_path=file,
            kpoint_num=kpoint,
            band_num=band,
            spin_channel=spin,
            poscar_path=poscar_path,
        )
    except Exception as exc:
        console.print(f"[bold red]Error parsing PROCAR:[/bold red] {exc}")
        raise typer.Exit(code=1)

    # 1. Panel de resumen de la banda y composición química
    elem_text = " | ".join(
        [
            f"[bold cyan]{elem}:[/bold cyan] {val:.3f} ({(val/summary.total_band_projection*100) if summary.total_band_projection > 0 else 0:.3f}%)"
            for elem, val in sorted(
                summary.elemental_contributions.items(),
                key=lambda x: x[1],
                reverse=True,
            )
        ]
    )

    info_text = (
        f"[bold]Spin:[/bold] {summary.spin} | [bold]k-point:[/bold] {summary.kpoint} | "
        f"[bold]Band:[/bold] {summary.band} | [bold]Energy:[/bold] {summary.energy:.4f} eV | "
        f"[bold]Occ:[/bold] {summary.occupation:.4f}\n"
        f"[bold]Elemental Breakdown:[/bold] {elem_text}\n"
        f"[bold]Total Projected Weight:[/bold] {summary.total_band_projection:.3f}"
    )
    console.print(Panel(info_text, title="Band State Localization", expand=False))

    # 2. Filtrado y ordenamiento de iones
    sorted_ions = sorted(summary.ions, key=lambda x: x.total, reverse=True)

    if not show_all:
        filtered_ions = [ion for ion in sorted_ions if ion.total >= threshold][:top]
    else:
        filtered_ions = summary.ions  # Mantiene el orden original por número de átomo

    # 3. Construir la tabla con Rich
    table = Table(
        title=f"Orbital Projections (Showing {len(filtered_ions)} of {len(summary.ions)} atoms)",
        show_header=True,
    )
    table.add_column("Ion #", justify="right", style="cyan")
    table.add_column("Elem", justify="center", style="bold magenta")

    # Columnas orbitales presentes en el PROCAR (s, pz, etc.)
    for h in summary.orbital_headers:
        table.add_column(h, justify="right", style="dim" if h != "pz" else "white")
    table.add_column("Total", justify="right", style="bold green")
    table.add_column("Ratio (%)", justify="right", style="yellow")

    for ion in filtered_ions:
        ratio = (
            (ion.total / summary.total_band_projection * 100)
            if summary.total_band_projection > 0
            else 0.0
        )
        row = [
            str(ion.ion_index),
            ion.element,
        ]
        for h in summary.orbital_headers:
            val = ion.orbitals.get(h, 0.0)
            row.append(f"{val:.3f}" if val > 0.0 else "[dim]0.000[/dim]")

        row.append(f"{ion.total:.3f}")
        row.append(f"{ratio:.3f}%")
        table.add_row(*row)

    console.print(table)

@app.command(
    epilog="""
[bold underline yellow]Examples:[/bold underline yellow]

  [dim]# Track where Carbon atoms contribute more than 5%:[/dim]
  [bold cyan]takumi track -e C[/bold cyan]

  [dim]# Track specific ions (e.g., 71 and 72) with a threshold of 10%:[/dim]
  [bold cyan]takumi track -i 71 -i 72 --threshold 10.0[/bold cyan]
"""
)
def track(
    elem: Optional[str] = typer.Option(None, "--elem", "-e", help="Element symbol to track (e.g., C). Requires POSCAR."),
    ion: Optional[List[int]] = typer.Option(None, "--ion", "-i", help="Specific ion index to track (can be passed multiple times)."),
    threshold: float = typer.Option(5.0, "--threshold", "-t", help="Minimum contribution percentage to display. Defaults to 5.0%."),
    file: Path = typer.Option(Path("PROCAR"), "--file", "-p", help="Path to PROCAR file.", dir_okay=False),
    poscar: Path = typer.Option(Path("POSCAR"), "--poscar", "-c", help="Path to POSCAR/CONTCAR.", dir_okay=False),
):
    """Scan all bands/k-points to find where specific atoms have significant contributions."""
    if not file.exists():
        console.print(f"[bold red]Error:[/bold red] File '{file}' not found.")
        raise typer.Exit(code=1)

    if not elem and not ion:
        console.print("[bold red]Error:[/bold red] You must specify either --elem (-e) or --ion (-i).")
        raise typer.Exit(code=1)

    species = parse_poscar_species(poscar) if poscar.exists() else []
    target_ions = []
    legend_map = {}

    # Generación de la lista de átomos y la leyenda solicitada (e.g. C1: 30)
    if elem:
        if not species:
            console.print("[bold red]Error:[/bold red] POSCAR is required to map elements to atom indices.")
            raise typer.Exit(code=1)
        
        target_ions = [idx + 1 for idx, s in enumerate(species) if s.lower() == elem.lower()]
        if not target_ions:
            console.print(f"[bold yellow]Warning:[/bold yellow] Element '{elem}' not found in POSCAR.")
            raise typer.Exit()
            
        for i, idx in enumerate(target_ions):
            legend_map[idx] = f"{elem.capitalize()}{i+1}"
    else:
        target_ions = ion
        for idx in target_ions:
            el_name = species[idx-1] if (species and idx-1 < len(species)) else "Ion"
            legend_map[idx] = f"{el_name}-{idx}"

    console.print(f"[dim]Scanning [bold]{file.name}[/bold] for {len(target_ions)} target ions (Threshold: >{threshold}%)...[/dim]")
    
    results = track_ions_in_procar(file, target_ions, min_ratio_percent=threshold)

    if not results:
        console.print(f"\n[yellow]No states found where the specified ions contribute > {threshold}%[/yellow]")
        raise typer.Exit()

    # Imprimir la Leyenda (Legend)
    if elem:
        legend_str = ", ".join([f"[bold cyan]{name}[/bold cyan]: {idx}" for idx, name in legend_map.items()])
        console.print(Panel(legend_str, title="Atom Legend", expand=False))

    # Construir la Tabla de Resultados (con 2 decimales para el ratio y 3 para proyecciones)
    table = Table(title=f"States with > {threshold}% contribution", show_header=True)
    table.add_column("Spin", justify="center", style="cyan")
    table.add_column("k-pt", justify="center", style="blue")
    table.add_column("Band", justify="right", style="white")
    table.add_column("Energy (eV)", justify="right")
    table.add_column("Occ", justify="right")
    table.add_column("Atom", justify="center", style="bold magenta")
    
    headers = results[0].orbital_headers
    for h in headers:
        table.add_column(h, justify="right", style="dim" if h != "pz" else "white")
    table.add_column("Total", justify="right", style="bold green")
    table.add_column("Ratio (%)", justify="right", style="yellow")

    for res in results:
        for i, ion_proj in enumerate(res.matched_ions):
            ratio = (ion_proj.total / res.total_band_weight * 100) if res.total_band_weight > 0 else 0.0
            atom_label = legend_map.get(ion_proj.ion_index, str(ion_proj.ion_index))
            
            row = [
                str(res.spin),
                str(res.kpoint),
                str(res.band),
                f"{res.energy:.4f}",
                f"{res.occupation:.4f}",
                atom_label
            ]
            
            for h in headers:
                val = ion_proj.orbitals.get(h, 0.0)
                row.append(f"{val:.3f}" if val > 0.0 else "[dim]0.000[/dim]")
            
            row.append(f"{ion_proj.total:.3f}")
            row.append(f"{ratio:.2f}%")
            
            # Dibujar una línea divisoria al final de cada banda si agrupó varios átomos
            is_last = (i == len(res.matched_ions) - 1)
            table.add_row(*row, end_section=is_last)

    console.print(table)


if __name__ == "__main__":
    app()