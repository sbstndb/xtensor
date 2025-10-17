# VTune Profiling des Benchmarks xtensor

Ce répertoire contient un script Python pour automatiser le profilage des benchmarks xtensor avec Intel VTune.

## Prérequis

1. **Spack** avec les modules suivants installés:
   - `intel-oneapi-vtune`
   - `xtl@develop`

2. **Python 3** (version 3.6 ou supérieure)

3. **Benchmark compilé** : L'exécutable `benchmark_xtensor` doit être compilé avec les flags d'optimisation:
   ```bash
   -march=native -mtune=native -O3 -g
   ```

## Compilation des benchmarks

Si les benchmarks ne sont pas encore compilés:

```bash
# Charger l'environnement
source ~/spack/share/spack/setup-env.sh
spack load intel-oneapi-vtune
spack load xtl@develop

# Compiler
cd /home/sbstndbs/xtensor_vtune/benchmark
mkdir -p build && cd build
cmake -DCMAKE_CXX_FLAGS="-march=native -mtune=native -O3 -g" \
      -DDOWNLOAD_GBENCHMARK=ON \
      -Dxtensor_DIR=/home/sbstndbs/xtensor_vtune/build ..
make benchmark_xtensor -j16
```

## Utilisation du script

### Exécuter sur tous les benchmarks

```bash
python3 run_vtune_benchmarks.py
```

Cela va:
- Lister tous les benchmarks disponibles (environ 181)
- Exécuter VTune avec hotspots collection sur chacun
- Sauvegarder les résultats dans `./vtune_profiling_results/`

### Limiter le nombre de benchmarks (pour tests)

```bash
# Exécuter seulement les 10 premiers benchmarks
python3 run_vtune_benchmarks.py --limit 10
```

### Filtrer par pattern

```bash
# Exécuter seulement les benchmarks contenant "assign" et "double"
python3 run_vtune_benchmarks.py --pattern "assign.*double"
```

### Options avancées

```bash
python3 run_vtune_benchmarks.py \
    --benchmark ./benchmark_xtensor \
    --output ./mes_resultats \
    --spack ~/spack/share/spack/setup-env.sh \
    --limit 50 \
    --pattern "math|reducer"
```

## Structure des résultats

Après exécution, les résultats sont organisés comme suit:

```
vtune_profiling_results/
├── vtune_results/              # Données brutes VTune (pour analyse approfondie)
│   ├── assign_c_assign_xt__xtensor_double,_2___32/
│   ├── assign_c_assign_xt__xtensor_double,_2___64/
│   └── ...
├── benchmark_outputs/          # Sorties des benchmarks (temps d'exécution, etc.)
│   ├── assign_c_assign_xt__xtensor_double,_2___32.txt
│   ├── assign_c_assign_xt__xtensor_double,_2___64.txt
│   └── ...
├── vtune_reports/             # Rapports summary VTune (hotspots, CPU time, etc.)
│   ├── assign_c_assign_xt__xtensor_double,_2___32_summary.txt
│   ├── assign_c_assign_xt__xtensor_double,_2___64_summary.txt
│   └── ...
└── vtune_run.log              # Log global de l'exécution
```

## Analyse des résultats

### Consulter le rapport summary d'un benchmark

```bash
cat vtune_profiling_results/vtune_reports/assign_c_assign_xt__xtensor_double,_2___32_summary.txt
```

### Générer d'autres types de rapports VTune

Vous pouvez générer des rapports plus détaillés à partir des données brutes:

```bash
source ~/spack/share/spack/setup-env.sh
spack load intel-oneapi-vtune

# Rapport hotspots détaillé
vtune -report hotspots \
      -r vtune_profiling_results/vtune_results/assign_c_assign_xt__xtensor_double,_2___32 \
      -report-output detailed_hotspots.txt

# Rapport au format CSV
vtune -report summary \
      -r vtune_profiling_results/vtune_results/assign_c_assign_xt__xtensor_double,_2___32 \
      -format csv \
      -report-output summary.csv

# Rapport callstacks
vtune -report callstacks \
      -r vtune_profiling_results/vtune_results/assign_c_assign_xt__xtensor_double,_2___32 \
      -report-output callstacks.txt
```

### Ouvrir dans VTune GUI

```bash
vtune-gui vtune_profiling_results/vtune_results/assign_c_assign_xt__xtensor_double,_2___32
```

## Informations collectées

Pour chaque benchmark, VTune collecte:

- **Temps total (Elapsed Time)**: Temps réel d'exécution
- **CPU Time**: Temps CPU utilisé
- **Effective Time**: Temps CPU effectif (hors spin, overhead)
- **Top Hotspots**: Fonctions les plus coûteuses en temps CPU
- **% of CPU Time**: Pourcentage du temps CPU pour chaque fonction

## Exemples de résultats

Exemple de rapport summary:

```
Elapsed Time: 0.828s
    CPU Time: 0.820s
        Effective Time: 0.820s
        Spin Time: 0s
        Overhead Time: 0s
    Total Thread Count: 1

Top Hotspots
Function                                                    Module             CPU Time  % of CPU Time(%)
----------------------------------------------------------  -----------------  --------  ----------------
xt::assign::assign_c_assign<...>                           benchmark_xtensor    0.810s             98.8%
xt::uvector<double, std::allocator<double>>::data          benchmark_xtensor    0.010s              1.2%
```

## Troubleshooting

### "Spack setup script not found"

Vérifiez le chemin vers spack:
```bash
ls ~/spack/share/spack/setup-env.sh
```

Si le chemin est différent, spécifiez-le avec `--spack`:
```bash
python3 run_vtune_benchmarks.py --spack /chemin/vers/setup-env.sh
```

### "Benchmark executable not found"

Assurez-vous que `benchmark_xtensor` est compilé:
```bash
ls -la ./benchmark_xtensor
```

### Warning "Microarchitecture performance insights will not be available"

Ce warning indique que le driver de sampling VTune n'est pas installé. Les résultats hotspots seront toujours collectés, mais certaines métriques avancées de microarchitecture ne seront pas disponibles.

Pour installer le driver (nécessite les droits root):
```bash
source ~/spack/share/spack/setup-env.sh
spack load intel-oneapi-vtune
vtune-install-sampling-driver
```

## Performance

- Environ 181 benchmarks au total
- Chaque benchmark prend environ 8-10 secondes (collection + rapport)
- Temps total estimé pour tous les benchmarks: ~25-30 minutes
- Espace disque requis: environ 700 MB pour tous les benchmarks

## Notes

- Les résultats peuvent varier légèrement entre les exécutions (bruit système)
- Pour des résultats reproductibles, utilisez `cpupower` pour fixer la fréquence CPU
- Les warnings "CPU scaling is enabled" et "ASLR is enabled" sont normaux et attendus
