# VTune Profiling Automation for xtensor Benchmarks

Automates Intel VTune hotspots collection on all 181 xtensor benchmarks.

## Usage

```bash
python3 run_vtune_benchmarks.py              # Run all 181 benchmarks (~25-30 min)
python3 run_vtune_benchmarks.py --limit 10   # Test with first 10 benchmarks
python3 run_vtune_benchmarks.py --pattern "math.*"  # Filter by regex pattern
```

## Results Structure

- `vtune_profiling_results/vtune_results/` - Raw VTune data (for GUI analysis)
- `vtune_profiling_results/vtune_reports/` - Summary reports (CPU time, hotspots)
- `vtune_profiling_results/benchmark_outputs/` - Benchmark execution logs

## View Results

```bash
cat vtune_profiling_results/vtune_reports/<benchmark>_summary.txt  # Text report
vtune-gui vtune_profiling_results/vtune_results/<benchmark>/       # Open in VTune GUI
```
