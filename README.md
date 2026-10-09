# ML interatomic potentials for solid-state ion conductors: training, fine-tuning and ionic transport

Training and fine-tuning of CHGNet and MACE potentials on DFT data for Li- and Na-ion conductors, and
their use for ion migration (NEB) and ionic transport (MD), validated against DFT.

| | what | key result |
|---|---|---|
| [training](training) | CHGNet fine-tuning (LYC, Na oxides) and MACE from scratch vs fine-tuned from MACE-MPA-0 (four conductors), with per-epoch curves and parity plots | fine-tuned MACE reaches 5.2 meV/Å validation force MAE on Li<sub>15</sub>Y<sub>7</sub>Cl<sub>36</sub>; fine-tuning beats training from scratch in all four materials |
| [na2o_neb_image_selection](na2o_neb_image_selection) | which DFT NEB images to fine-tune a foundation model on (17 strategies) | the right images reproduce DFT barriers to 0.013 eV on average; single-path training collapses other barriers |
| [md_studies](md_studies) | Li/Na diffusion and conductivity from MD with 7 foundation potentials, fine-tuned and from-scratch models, compared with AIMD | fine-tuned MACE diffusivities within a factor of 1.41 of AIMD in four conductors |

![fine-tuned vs foundation vs AIMD](figures/md/mace_ft_vs_mpa0_vs_scratch_vs_aimd.png)

## Repository layout

```
├── training/README.md                  # training study (figures + tables)
├── na2o_neb_image_selection/README.md  # NEB image-selection study
├── md_studies/README.md                # MD / ionic transport studies
├── notebooks/                          # 01-07: every number and figure is produced here
├── results/                            # analysis outputs the notebooks read (JSON / NPZ)
├── figures/                            # figures written by the notebooks
├── iondiff/                            # diffusivity / error / Arrhenius analysis package
├── scripts/                            # log parsing, MD/AIMD analysis drivers, parity predictions
├── manifests/                          # which runs each analysis reads (paths are placeholders)
└── tests/                              # validation of iondiff against the original module
```

## Reproducing

The notebooks only read `results/`, so they run anywhere with numpy, scipy, pandas and matplotlib:

```bash
pip install -r requirements.txt
jupyter nbconvert --to notebook --execute --inplace notebooks/*.ipynb
```

The `results/` files were produced from the raw MD/AIMD trajectories and training logs with
`scripts/run_study.py` (diffusion), `scripts/parse_training_logs.py` (training curves) and
`scripts/run_parity.py` (parity predictions), driven by the files in `manifests/`. The raw data are
not included; the manifest paths are placeholders.

## Data

AIMD (DFT) training and reference data: Mo group, University of Maryland. Na<sub>2</sub>O NEB images:
VASP (PBE) climbing-image NEB. The raw trajectories are not included in this repository.

## Method credit

Diffusivities and their statistical errors follow X. He, Y. Zhu, A. Epstein, Y. Mo, "Statistical
variances of diffusional properties from ab initio molecular dynamics simulations",
*npj Comput. Mater.* **4**, 18 (2018). `iondiff` is adapted from the Mo group (University of Maryland)
diffusion module (MIT License), built on [pymatgen](https://pymatgen.org).

## Future work

* Na analogues of known K-ion conductors (K→Na substitution, structure relaxation and screening): a
  separate repository, in preparation.

## License

MIT (see [LICENSE](LICENSE)).
