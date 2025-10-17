#!/usr/bin/env python3
"""
Script pour automatiser l'exécution des benchmarks xtensor avec VTune.

Ce script:
1. Charge l'environnement spack (VTune et xtl)
2. Liste tous les benchmarks disponibles
3. Pour chaque benchmark:
   - Lance VTune avec hotspots collection
   - Sauvegarde l'output du benchmark dans un fichier
   - Génère un rapport VTune summary
"""

import subprocess
import os
import sys
import re
import shutil
from pathlib import Path
from datetime import datetime
import argparse


class VTuneBenchmarkRunner:
    def __init__(self, benchmark_exe, output_dir, spack_path=None):
        """
        Initialise le runner de benchmarks VTune.

        Args:
            benchmark_exe: Chemin vers l'exécutable benchmark_xtensor
            output_dir: Répertoire de sortie pour les résultats
            spack_path: Chemin vers spack setup-env.sh (détection auto si None)
        """
        self.benchmark_exe = Path(benchmark_exe).resolve()
        self.output_dir = Path(output_dir).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Trouver spack
        if spack_path is None:
            spack_path = Path.home() / "spack" / "share" / "spack" / "setup-env.sh"
        self.spack_path = Path(spack_path)

        if not self.spack_path.exists():
            raise FileNotFoundError(f"Spack setup script not found: {self.spack_path}")
        if not self.benchmark_exe.exists():
            raise FileNotFoundError(f"Benchmark executable not found: {self.benchmark_exe}")

    def get_env_command(self):
        """Retourne la commande pour charger l'environnement spack."""
        return f"source {self.spack_path} && spack load intel-oneapi-vtune && spack load xtl@develop"

    def list_benchmarks(self):
        """Liste tous les benchmarks disponibles."""
        print("📋 Listing available benchmarks...")

        cmd = f"{self.get_env_command()} && {self.benchmark_exe} --benchmark_list_tests"

        result = subprocess.run(
            cmd,
            shell=True,
            executable="/bin/bash",
            capture_output=True,
            text=True
        )

        if result.returncode != 0:
            print(f"❌ Error listing benchmarks: {result.stderr}")
            sys.exit(1)

        # Parser les noms de benchmarks (ignorer les messages de warning/info)
        benchmarks = []
        for line in result.stdout.split('\n'):
            line = line.strip()
            # Ignorer les lignes vides et les messages non-benchmark
            # Les benchmarks valides contiennent généralement des caractères alphanumériques
            # et peuvent contenir des caractères comme <, >, _, /, etc.
            if line and not line.startswith('NOT USING'):
                benchmarks.append(line)

        print(f"✅ Found {len(benchmarks)} benchmarks")
        return benchmarks

    def sanitize_filename(self, name):
        """Convertit un nom de benchmark en nom de fichier valide."""
        # Remplacer les caractères problématiques
        name = re.sub(r'[<>:"/\\|?*]', '_', name)
        name = re.sub(r'\s+', '_', name)
        return name

    def run_vtune_on_benchmark(self, benchmark_name, index, total):
        """
        Lance VTune sur un benchmark spécifique.

        Args:
            benchmark_name: Nom du benchmark à exécuter
            index: Index du benchmark (pour affichage)
            total: Nombre total de benchmarks
        """
        print(f"\n[{index}/{total}] 🔬 Running VTune on: {benchmark_name}")

        # Créer un nom de fichier sécurisé
        safe_name = self.sanitize_filename(benchmark_name)

        # Créer les répertoires de sortie
        result_dir = self.output_dir / "vtune_results" / safe_name
        # Supprimer le répertoire s'il existe déjà (VTune refuse d'écraser)
        if result_dir.exists():
            shutil.rmtree(result_dir)
        result_dir.mkdir(parents=True, exist_ok=True)

        output_file = self.output_dir / "benchmark_outputs" / f"{safe_name}.txt"
        output_file.parent.mkdir(parents=True, exist_ok=True)

        report_file = self.output_dir / "vtune_reports" / f"{safe_name}_summary.txt"
        report_file.parent.mkdir(parents=True, exist_ok=True)

        # Préparer la commande VTune
        # Échapper uniquement les caractères spéciaux regex sans utiliser re.escape
        # qui ajoute trop de backslashes
        def escape_for_regex(s):
            """Échappe manuellement les caractères spéciaux pour regex."""
            # Caractères à échapper en regex
            special_chars = r'\.^$*+?{}[]()|\\'
            result = []
            for char in s:
                if char in special_chars:
                    result.append('\\' + char)
                else:
                    result.append(char)
            return ''.join(result)

        escaped_name = escape_for_regex(benchmark_name)
        benchmark_filter = f"^{escaped_name}$"

        vtune_cmd = (
            f"{self.get_env_command()} && "
            f"vtune -collect hotspots "
            f"-result-dir={result_dir} "
            f"-- {self.benchmark_exe} "
            f'--benchmark_filter="{benchmark_filter}" '
            f"> {output_file} 2>&1"
        )

        # Exécuter VTune
        print(f"   ⏳ Running VTune collection...")
        result = subprocess.run(
            vtune_cmd,
            shell=True,
            executable="/bin/bash",
            capture_output=True,
            text=True
        )

        if result.returncode != 0:
            print(f"   ❌ VTune collection failed:")
            print(f"      {result.stderr}")
            return False

        # Générer le rapport summary
        print(f"   📊 Generating VTune summary report...")
        report_cmd = (
            f"{self.get_env_command()} && "
            f"vtune -report summary "
            f"-r {result_dir} "
            f"-report-output {report_file}"
        )

        result = subprocess.run(
            report_cmd,
            shell=True,
            executable="/bin/bash",
            capture_output=True,
            text=True
        )

        if result.returncode != 0:
            print(f"   ⚠️  Warning: Failed to generate report: {result.stderr}")

        print(f"   ✅ Results saved:")
        print(f"      VTune data:   {result_dir}")
        print(f"      Benchmark:    {output_file}")
        print(f"      Report:       {report_file}")

        return True

    def run_all_benchmarks(self, limit=None, pattern=None):
        """
        Lance VTune sur tous les benchmarks.

        Args:
            limit: Limite optionnelle du nombre de benchmarks à exécuter
            pattern: Pattern regex optionnel pour filtrer les benchmarks
        """
        benchmarks = self.list_benchmarks()

        # Filtrer par pattern si spécifié
        if pattern:
            pattern_re = re.compile(pattern)
            benchmarks = [b for b in benchmarks if pattern_re.search(b)]
            print(f"🔍 Filtered to {len(benchmarks)} benchmarks matching '{pattern}'")

        # Limiter le nombre si spécifié
        if limit:
            benchmarks = benchmarks[:limit]
            print(f"⚠️  Limited to first {limit} benchmarks")

        print(f"\n🚀 Starting VTune profiling on {len(benchmarks)} benchmarks...")
        print(f"📁 Output directory: {self.output_dir}\n")

        # Créer un fichier de log global
        log_file = self.output_dir / "vtune_run.log"
        start_time = datetime.now()

        with open(log_file, 'w') as log:
            log.write(f"VTune Benchmark Run\n")
            log.write(f"==================\n\n")
            log.write(f"Start time: {start_time}\n")
            log.write(f"Benchmark executable: {self.benchmark_exe}\n")
            log.write(f"Total benchmarks: {len(benchmarks)}\n\n")

        # Exécuter chaque benchmark
        success_count = 0
        failed_benchmarks = []

        for i, benchmark in enumerate(benchmarks, 1):
            try:
                if self.run_vtune_on_benchmark(benchmark, i, len(benchmarks)):
                    success_count += 1
                else:
                    failed_benchmarks.append(benchmark)
            except Exception as e:
                print(f"   ❌ Unexpected error: {e}")
                failed_benchmarks.append(benchmark)

        # Résumé final
        end_time = datetime.now()
        duration = end_time - start_time

        print(f"\n" + "="*70)
        print(f"📊 SUMMARY")
        print(f"="*70)
        print(f"✅ Successful: {success_count}/{len(benchmarks)}")
        print(f"❌ Failed:     {len(failed_benchmarks)}/{len(benchmarks)}")
        print(f"⏱️  Duration:   {duration}")
        print(f"📁 Results:    {self.output_dir}")

        if failed_benchmarks:
            print(f"\n❌ Failed benchmarks:")
            for b in failed_benchmarks:
                print(f"   - {b}")

        # Mettre à jour le log
        with open(log_file, 'a') as log:
            log.write(f"\nEnd time: {end_time}\n")
            log.write(f"Duration: {duration}\n")
            log.write(f"Successful: {success_count}/{len(benchmarks)}\n")
            log.write(f"Failed: {len(failed_benchmarks)}/{len(benchmarks)}\n")
            if failed_benchmarks:
                log.write(f"\nFailed benchmarks:\n")
                for b in failed_benchmarks:
                    log.write(f"  - {b}\n")

        print(f"\n📝 Log file: {log_file}")
        print("="*70 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Run VTune profiling on xtensor benchmarks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run on all benchmarks
  python run_vtune_benchmarks.py

  # Run on first 10 benchmarks only
  python run_vtune_benchmarks.py --limit 10

  # Run only on benchmarks matching a pattern
  python run_vtune_benchmarks.py --pattern "assign.*double"

  # Custom benchmark executable and output directory
  python run_vtune_benchmarks.py --benchmark ./benchmark_xtensor --output ./my_results
        """
    )

    parser.add_argument(
        '--benchmark',
        default='./benchmark_xtensor',
        help='Path to benchmark_xtensor executable (default: ./benchmark_xtensor)'
    )

    parser.add_argument(
        '--output',
        default='./vtune_profiling_results',
        help='Output directory for results (default: ./vtune_profiling_results)'
    )

    parser.add_argument(
        '--spack',
        default=None,
        help='Path to spack setup-env.sh (default: auto-detect)'
    )

    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='Limit number of benchmarks to run (useful for testing)'
    )

    parser.add_argument(
        '--pattern',
        default=None,
        help='Regex pattern to filter benchmark names'
    )

    args = parser.parse_args()

    try:
        runner = VTuneBenchmarkRunner(
            benchmark_exe=args.benchmark,
            output_dir=args.output,
            spack_path=args.spack
        )

        runner.run_all_benchmarks(limit=args.limit, pattern=args.pattern)

    except Exception as e:
        print(f"❌ Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
