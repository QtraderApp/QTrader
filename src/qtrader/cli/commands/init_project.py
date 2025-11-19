"""Initialize a complete QTrader project."""

import shutil
from pathlib import Path

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()


@click.command("init-project")
@click.argument("path", type=click.Path(path_type=Path))
@click.option(
    "--force",
    "-f",
    is_flag=True,
    help="Overwrite existing directory",
)
def init_project_command(path: Path, force: bool) -> None:
    """
    Initialize a complete QTrader project environment.

    Creates a production-ready backtesting system with:
    - Configuration files (qtrader.yaml, data_sources.yaml)
    - Example backtests (buy & hold, SMA crossover)
    - Example strategies (ready to run)
    - Sample data
    - Runner script
    - Custom library scaffold

    This creates a SYSTEM, not just one backtest. You can run multiple
    different backtests within this project.

    \b
    Examples:
        # Create new project
        qtrader init-project my-trading-system

        # Create in current directory
        qtrader init-project .

        # Overwrite existing
        qtrader init-project my-system --force
    """
    # Expand and resolve path
    target_path = path.expanduser().resolve()

    # Check if path exists and prompt if necessary
    if target_path.exists() and not force:
        existing_files = list(target_path.iterdir()) if target_path.is_dir() else [target_path]
        if existing_files and target_path.name != ".":
            console.print(f"[yellow]Directory {target_path} exists and is not empty.[/yellow]")
            if not click.confirm("Continue and potentially overwrite files?"):
                console.print("[yellow]Cancelled[/yellow]")
                return

    # Create base directory
    target_path.mkdir(parents=True, exist_ok=True)

    # Get scaffold directory (shipped with package)
    templates_dir = Path(__file__).parent.parent.parent / "scaffold"

    if not templates_dir.exists():
        console.print("[red]Error: Project scaffold directory not found[/red]")
        console.print(f"Expected location: {templates_dir}")
        console.print("\nThis is a package installation issue. Please reinstall qtrader.")
        return

    created_items: list[str] = []

    # Copy entire project structure
    console.print("\n[cyan]Creating project structure...[/cyan]")

    # Copy config/
    _copy_directory(templates_dir / "config", target_path / "config", created_items)

    # Copy library/
    _copy_directory(templates_dir / "library", target_path / "library", created_items)

    # Copy data/
    _copy_directory(templates_dir / "data", target_path / "data", created_items)

    # Copy examples/
    _copy_directory(templates_dir / "examples", target_path / "examples", created_items)

    # Copy experiments/
    _copy_directory(templates_dir / "experiments", target_path / "experiments", created_items)

    # Copy QTrader README (named QTRADER_README.md to avoid collision with user's README)
    readme_src = templates_dir / "QTRADER_README.md"
    readme_dst = target_path / "QTRADER_README.md"
    if readme_src.exists():
        shutil.copy2(readme_src, readme_dst)
        created_items.append("QTRADER_README.md")

    # Create output directory
    (target_path / "output").mkdir(exist_ok=True)
    created_items.append("output/")

    # Update config to point to local library
    _update_qtrader_config(target_path)

    # Success message with structure
    console.print(
        Panel.fit(
            f"[green]✓[/green] QTrader project initialized at:\n[cyan]{target_path}[/cyan]",
            title="Success",
            border_style="green",
        )
    )

    # Show structure table
    _display_project_structure(target_path)

    # Next steps
    console.print("\n[bold]🚀 Quick Start:[/bold]")
    console.print("1. [cyan]cd " + (str(target_path) if target_path.name != "." else "# (already here)") + "[/cyan]")
    console.print("2. [dim]# Review example strategies in library/strategies/[/dim]")
    console.print("3. [dim]# Check example backtest configs in experiments/[/dim]")
    console.print("4. [cyan]python examples/run_backtest.py experiments/buy_hold.yaml[/cyan]")
    console.print("\n[bold]📚 What You Got:[/bold]")
    console.print("• [green]2 example strategies[/green] (buy & hold, SMA crossover)")
    console.print("• [green]Sample data[/green] (AAPL, limited history)")
    console.print("• [green]Complete configs[/green] (system, data sources)")
    console.print("• [green]Custom library [/green] with examples")
    console.print("\n[dim]See QTRADER_README.md for full documentation[/dim]")


def _copy_directory(src: Path, dst: Path, items_list: list[str]) -> None:
    """Recursively copy directory and track items."""
    if not src.exists():
        return

    dst.mkdir(parents=True, exist_ok=True)

    for item in src.rglob("*"):
        if item.is_file():
            rel_path = item.relative_to(src)
            dest_file = dst / rel_path
            dest_file.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(item, dest_file)

    # Add to items list (just the top level directory)
    rel_to_parent = dst.name
    if dst.parent.name != dst.parent.parent.name:  # Not root
        rel_to_parent = f"{dst.parent.name}/{dst.name}/"
    items_list.append(rel_to_parent)


def _update_qtrader_config(project_path: Path) -> None:
    """Update qtrader.yaml to point to local library."""
    qtrader_config = project_path / "config" / "qtrader.yaml"

    if not qtrader_config.exists():
        return

    content = qtrader_config.read_text()

    # Update custom_libraries to point to local library
    # Match the line and replace entire line
    import re

    # Replace strategies line
    content = re.sub(
        r"  strategies: null.*$",
        '  strategies: "library/strategies"  # Example strategies included',
        content,
        flags=re.MULTILINE,
    )

    # Replace risk_policies line
    content = re.sub(
        r"  risk_policies: null.*$",
        '  risk_policies: "library/risk_policies"  # Example policies included',
        content,
        flags=re.MULTILINE,
    )

    qtrader_config.write_text(content)


def _display_project_structure(project_path: Path) -> None:
    """Display project structure in a nice table."""
    table = Table(title="Project Structure", show_header=True, header_style="bold cyan")
    table.add_column("Directory/File", style="cyan")
    table.add_column("Description", style="dim")

    structure = [
        ("config/", "System and backtest configurations"),
        ("  ├── qtrader.yaml", "System level settings (execution, portfolio)"),
        ("  ├── data_sources.yaml", "Data source definitions"),
        ("library/", "Your custom components"),
        ("  ├── adapters/", "Custom adapters (examples included)"),
        ("  ├── indicators/", "Custom indicators (template included)"),
        ("  ├── strategies/", "Custom strategies (examples and template included)"),
        ("  └── risk_policies/", "Custom risk policies (template included)"),
        ("experiments/", "Experiment configurations"),
        ("  ├── buy_hold.yaml", "Buy and hold experiment"),
        ("  ├── sma.yaml", "SMA crossover experiment"),
        ("  └── template.yaml", "Complete experiment template"),
        ("data/", "Sample market data"),
        ("examples/", "Example scripts and utilities"),
        ("output/", "Backtest results"),
        ("QTRADER_README.md", "QTrader project guide"),
    ]

    for item, desc in structure:
        table.add_row(item, desc)

    console.print(table)
