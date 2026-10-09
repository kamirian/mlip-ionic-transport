"""Write the study manifests used by run_study.py.

Paths are the cluster locations of the original MD runs (read-only inputs). Fit settings
are the ones the original analysis used for every study, so only the documented bugs change:
site distance 3.38 A, fit window from 0.5 a^2 to 0.8 of the trajectory, minimum MSD range 0.5 a^2,
~1000 time intervals.
"""

import json
import os
import sys

SCRATCH = "/path/to/data"
OUT = sys.argv[1] if len(sys.argv) > 1 else "manifests"
os.makedirs(OUT, exist_ok=True)

FIT = {"lower_bound_in_a_square": 0.5, "upper_bound": 0.8,
       "minimum_msd_diff_in_a_square": 0.5, "time_intervals_number": 1000}
ASE_NVT = {"temperature_dir_regex": r"^ase_test_(\d+)K$",
           "segments": ["md_nvt_nhc.traj", "run{n}_md_nvt_nhc.traj"],
           "frame_interval_fs": 200, "equilibration_frames": 100}
MACE_NVT = {"temperature_dir_regex": r"^(\d+)_NVT$",
            "segments": ["results_step_0/nvt_md.traj", "results/nvt_md.traj"],
            "frame_interval_fs": 100, "equilibration_frames": 100}
CHGNET_MD = {"temperature_dir_regex": r"^(\d+)$", "segments": ["LYC.traj"],
             "frame_interval_fs": 100, "equilibration_frames": 100}


def write(name, doc):
    doc.setdefault("fit", FIT)
    with open(os.path.join(OUT, name + ".json"), "w") as fh:
        json.dump(doc, fh, indent=1)


# 1. LYC, 7 foundation potentials, NVT (Nose-Hoover chain), 2 fs, 400 ps per job.
fp = [
    ("MACE-MPA-0 (medium)", "MACE_env/jobs/md/LYC_ase_NVT_correct_T", "MACE_env/foundation_models/imporved_highpressure_accuracy_mace-mpa-0/mace-mpa-0-medium.model"),
    ("CHGNet (pretrained)", "CHGNET_2/jobs/MD/MD_ase_LYC_correct_T", "chgnet CHGNetCalculator() default"),
    ("M3GNet-MP-2021.2.8-PES", "M3GNet_2/jobs/MD/LYC_ase_NVT_MP2021_2_8_pes_correct_T", "M3GNet_2/pretrained_models/M3GNet-MP-2021.2.8-PES/"),
    ("UMA s1.1 (omat)", "Facebook_env/jobs/MD/LYC_ase_NVT_correct_T", "Facebook/Models/uma-s-1p1.pt"),
    ("M3GNet-MatPES-PBE", "M3GNet_2/jobs/MD/LYC_ase_NVT_m3gnet_matpes_PBE_correct_T", "M3GNet_2/pretrained_models/M3GNet-MatPES-PBE-v2025.1-PES/"),
    ("TensorNet-MatPES-PBE", "M3GNet_2/jobs/MD/LYC_ase_NVT_tensornet_matpes_pbe_correct_T", "M3GNet_2/pretrained_models/TensorNet-MatPES-PBE-v2025.1-PES/"),
    ("MACE-MatPES-PBE", "MACE_env/jobs/md/LYC_ase_NVT_matpes_correct_T", "MACE_env/foundation_models/matpes/MACE-matpes-pbe-omat-ft.model"),
]
write("lyc_7fp_nvt", {
    "study": "lyc_7fp_nvt",
    "description": "Li15Y7Cl36 (58 atoms), NVT Nose-Hoover chain, 2 fs, 200,000 steps per job, "
                   "40 temperatures 300-1025 K, continuation jobs stitched per temperature.",
    "specie": "Li", "charge": 1, "site_distance": 3.38,
    "arrhenius_windows": {"all": None, "500-1000K": [500, 1000]},
    "series": [dict(ASE_NVT, label=l, root=f"{SCRATCH}/{r}", model_file=m) for l, r, m in fp]})

# 2. MACE fine-tuned vs MACE-MPA-0 vs from scratch, per material. NVT, 1 fs, 400,000 steps.
mats = {"lyc": ("LYC_NVT", "Li", "Li15Y7Cl36 (58 atoms)",
                "MACE_env/jobs/lyc_nvt/fine_tuning_improved_model/MACE_models/medium_finetuned_initial_release_compiled.model",
                "MACE_env/jobs/Training_NVT_structures/MACE_models/energy_corrected_mace_cpu_stagetwo_compiled.model"),
        "20258": ("20258_NVT", "Na", "Na9Y6Si3P9O42 (69 atoms)",
                  "MACE_env/jobs/20258_nvt_training/fine_tuning_improved_model/MACE_models/improved_20258_finetuned_compiled.model",
                  "MACE_env/jobs/20258_nvt_training/from_scratch_training/MACE_models/2058_mace_cpu_stagetwo_compiled.model"),
        "45802": ("45802_NVT", "Na", "Na5Nb5W11O48 (69 atoms)",
                  "MACE_env/jobs/45802_nvt_training/fine_tuning_improved_model/MACE_models/improved_45802_finetuned_compiled.model",
                  "MACE_env/jobs/45802_nvt_training/from_scratch_training/MACE_models/45802_mace_cpu_stagetwo_compiled.model"),
        "75421": ("75421_NVT", "Na", "Na11Si11Sb16As5O80 (123 atoms)",
                  "MACE_env/jobs/75421_nvt_training/fine_tuning_improved_model/MACE_models/improved_75421_finetuned_compiled.model",
                  "MACE_env/jobs/75421_nvt_training/from_scratch_training/MACE_models/75421_mace_cpu_stagetwo_compiled.model")}
MPA0 = "MACE_env/foundation_models/imporved_highpressure_accuracy_mace-mpa-0/mace-mpa-0-medium.model"
for key, (d, sp, comp, ft, scratch) in mats.items():
    base = f"{SCRATCH}/MACE_env/jobs/md/{d}"
    write(f"mace_{key}_ft_vs_mpa0_vs_scratch", {
        "study": f"mace_{key}_ft_vs_mpa0_vs_scratch",
        "description": f"{comp}: MACE fine-tuned from MACE-MPA-0 vs MACE-MPA-0 vs MACE trained from "
                       "scratch. NVT, 1 fs, 400,000 steps, 29 temperatures 300-1000 K.",
        "specie": sp, "charge": 1, "site_distance": 3.38,
        "arrhenius_windows": {"all": None, "769-1000K": [769, 1000]},
        "series": [dict(MACE_NVT, label="MACE fine-tuned (from MPA-0)", root=f"{base}/nvt_final_finetuned_model", model_file=ft),
                   dict(MACE_NVT, label="MACE-MPA-0 (medium)", root=f"{base}/nvt_final_normal_mace_model", model_file=MPA0),
                   dict(MACE_NVT, label="MACE from scratch", root=f"{base}/nvt_final_simpler_model", model_file=scratch)]})

# 3. Fine-tuned CHGNet on LYC, NVT. Labels describe the actual training data (see fix tracker B6).
DG = f"{SCRATCH}/RCHnet/JOBS/LYC_fiinetune2/data_grabber"
chg = [
    ("CHGNet (pretrained)", "MD_RCHGNet_new", "CHGNet.load() default"),
    ("FT: first 7,980 frames (900 K block), 10 epochs", "MD_50_itter", "Fine_tune/10_epochs/09-26-2024/bestE_epoch9_e1_f35_sNA_mNA.pth.tar"),
    ("FT: first 7,980 frames (900 K block), 200 epochs", "MD_50_itter_200epoch", "Fine_tune/200_epochs/09-27-2024/bestE_epoch197_e1_f15_sNA_mNA.pth.tar"),
    ("FT: first 15,960 frames (900 K block), 10 epochs", "MD_25_itter", "Fine_tune/25_itteration_10_epochs/09-29-2024/bestE_epoch7_e1_f33_sNA_mNA.pth.tar"),
    ("FT: first 15,960 frames (900 K block), 20 epochs", "25_itteration_20epoch", "Fine_tune/25_itteration_20_epochs/11-16-2024/bestE_epoch19_e1_f29_sNA_mNA.pth.tar"),
    ("FT: first 30,693 frames (900 K block), 10 epochs", "MD_13_itter_new", "Fine_tune/13_itteration_10_epochs/09-30-2024/bestE_epoch7_e1_f32_sNA_mNA.pth.tar"),
    ("FT: energy-histogram 8,121 frames, batch 8", "histogram_8000_first_try", "Fine_tune/histogram_8000-first-try/11-19-2024/bestE_epoch9_e1_f31_sNA_mNA.pth.tar"),
    ("FT: energy-histogram 16,279 frames, batch 8", "histogram_16000_first_try", "Fine_tune/histogram_16000-first-try/11-19-2024/bestE_epoch7_e1_f29_sNA_mNA.pth.tar"),
    ("FT: energy-histogram 16,279 frames, batch 16", "histogram_16000_first_try_batch_16", "Fine_tune/different_batch_sizes/histogram_16000-first-try_batch_16/12-19-2024/bestE_epoch9_e1_f30_sNA_mNA.pth.tar"),
    ("FT: energy-histogram 16,279 frames, batch 32", "histogram_16000_first_try_batch_32", "Fine_tune/different_batch_sizes/histogram_16000-first-try_batch_32/12-19-2024/bestE_epoch9_e1_f32_sNA_mNA.pth.tar"),
    ("FT: energy-histogram 30,000 frames, batch 8", "MD_histogram_total", "Fine_tune/histogram_Total/11-13-2024/bestE_epoch9_e1_f27_sNA_mNA.pth.tar"),
    ("FT: energy-histogram 30,222 frames, batch 8", "histogram_total_new", "Fine_tune/histogram_Total_new/11-19-2024/bestE_epoch9_e1_f27_sNA_mNA.pth.tar"),
    ("FT: energy-histogram 30,222 frames, batch 16", "histogram_Total_first_try_batch_16", "Fine_tune/different_batch_sizes/histogram_Total_new_16/12-22-2024/bestE_epoch8_e1_f29_sNA_mNA.pth.tar"),
    ("FT: energy-histogram 30,222 frames, batch 32", "histogram_Total_first_try_batch_32", "Fine_tune/different_batch_sizes/histogram_Total_new_32/12-22-2024/bestE_epoch9_e1_f30_sNA_mNA.pth.tar"),
]
write("chgnet_lyc_finetuned_nvt", {
    "study": "chgnet_lyc_finetuned_nvt",
    "description": "Li15Y7Cl36 (58 atoms), CHGNet MolecularDynamics NVT, 1 fs, 400,000 steps, "
                   "pretrained vs fine-tuned CHGNet models (training data described in each label).",
    "specie": "Li", "charge": 1, "site_distance": 3.38,
    "arrhenius_windows": {"all": None, "500-1000K": [500, 1000]},
    "series": [dict(CHGNET_MD, label=l, root=f"{DG}/MD/{d}", model_file=m if m.startswith("CHGNet") else f"{DG}/{m}")
               for l, d, m in chg]})

# 4. CHGNet NPT on LYC, including the long (target 5 ns) runs.
npt = [
    ("FT 7,980 frames, 10 ep: NPT, target 5 ns", "MD_50_itter_npt_10ns", "Fine_tune/10_epochs/09-26-2024/bestE_epoch9_e1_f35_sNA_mNA.pth.tar"),
    ("FT 7,980 frames, 10 ep: NPT, 400 ps", "MD_50_itter_npt", "Fine_tune/10_epochs/09-26-2024/bestE_epoch9_e1_f35_sNA_mNA.pth.tar"),
    ("FT 7,980 frames, 200 ep: NPT, 400 ps", "MD_50_itter_200epoch_npt", "Fine_tune/200_epochs/09-27-2024/bestE_epoch197_e1_f15_sNA_mNA.pth.tar"),
    ("FT 15,960 frames, 10 ep: NPT, 400 ps", "md_25_itter_npt", "Fine_tune/25_itteration_10_epochs/09-29-2024/bestE_epoch7_e1_f33_sNA_mNA.pth.tar"),
    ("FT 30,693 frames, 10 ep: NPT, 400 ps", "MD_13_itter_npt", "Fine_tune/13_itteration_10_epochs/09-30-2024/bestE_epoch7_e1_f32_sNA_mNA.pth.tar"),
    ("CHGNet (pretrained): NPT, 400 ps", "MD_RCHGNet_npt", "CHGNet.load() default"),
]
write("chgnet_lyc_npt", {
    "study": "chgnet_lyc_npt",
    "description": "Li15Y7Cl36 (58 atoms), CHGNet MolecularDynamics NPT (1 atm), 1 fs; per-frame cells used "
                   "for displacements. The 5 ns-target runs stopped at the 144 h wall time (3.5-6.1 ns).",
    "specie": "Li", "charge": 1, "site_distance": 3.38,
    "arrhenius_windows": {"all": None},
    "series": [dict(CHGNET_MD, label=l, root=f"{DG}/MD/{d}", model_file=m if m.startswith("CHGNet") else f"{DG}/{m}",
                    # the 600 K and 700 K long runs used a 2 fs timestep (frames every 200 fs)
                    **({"frame_interval_overrides_fs": {"600": 200, "700": 200}} if d == "MD_50_itter_npt_10ns" else {}))
               for l, d, m in npt]})

# 5. LGPS supercell-size study, MACE-MPA-0, NVT, 2 fs.
L = f"{SCRATCH}/MACE_env/jobs/md/LGPS_ase_NVT_different_supercells"
write("lgps_supercell_mpa0", {
    "study": "lgps_supercell_mpa0",
    "description": "Li10GeP2S12 supercells (1x1x2 = 100 atoms up to 3x3x3 = 1350 atoms), MACE-MPA-0, "
                   "NVT Nose-Hoover chain, 2 fs. Larger cells did not reach 400 ps (see analyzed_time_ps).",
    "specie": "Li", "charge": 1, "site_distance": 3.38,
    "arrhenius_windows": {"all": None, "500-1000K": [500, 1000]},
    "series": [dict(ASE_NVT, label=f"supercell {s}", root=f"{L}/supercell_{s}", model_file=MPA0)
               for s in ["1x1x2", "1x1x3", "2x2x2", "2x2x3", "3x3x2", "3x3x3"]]})

print("\n".join(sorted(os.listdir(OUT))))

# 6. AIMD references (VASP NVT, 2 fs, every step stored). Same fit settings as the FP MD above so the
#    Arrhenius plots can be overlaid. RUN_0 is the heating run and is skipped; runs are stitched as
#    continuations (each RUN_k starts from RUN_{k-1}'s CONTCAR); every 10th step is used (20 fs).
AIMD = {"temperature_dir_regex": r"^(\d+)$", "reader": "vasp_runs", "run_dir_regex": r"^RUN_(\d+)$",
        "min_run_index": 1, "stride": 10, "frame_interval_fs": 20, "equilibration_frames": 500,
        "segments": []}
write("aimd_lyc", {
    "study": "aimd_lyc",
    "description": "AIMD (VASP, PBE, NVT Nose, 2 fs) of Li15Y7Cl36 (58 atoms) at 500-900 K: the data "
                   "set used to fine-tune / train the CHGNet and MACE LYC models.",
    "specie": "Li", "charge": 1, "site_distance": 3.38,
    "arrhenius_windows": {"all": None},
    "series": [dict(AIMD, label="AIMD (PBE)", root=f"{SCRATCH}/RCHnet/JOBS/LYC_data/second_take/aimd/li2.5",
                    model_file="VASP PBE")]})
write("aimd_na_conductors", {
    "study": "aimd_na_conductors",
    "description": "AIMD (VASP, PBE, NVT Nose-Hoover MDALGO=2, 2 fs) of the three Na-ion oxide conductors at "
                   "800-1300 K: the data sets used to fine-tune / train their MACE models.",
    "specie": "Na", "charge": 1, "site_distance": 3.38,
    "arrhenius_windows": {"all": None},
    "series": [dict(AIMD, label=f"AIMD (PBE) {m}", root=f"{SCRATCH}/MACE_env/jobs/data_sets/{m}/{d}",
                    model_file="VASP PBE")
               for m, d in (("20258", "67_AIMD"), ("45802", "10_AIMD"), ("75421", "12_AIMD"))]})
