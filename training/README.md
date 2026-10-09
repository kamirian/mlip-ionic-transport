# Training and fine-tuning CHGNet and MACE

Per-epoch training/validation curves, parity plots and test errors of every model used in the
other studies. Notebook: [01_training_curves](../notebooks/01_training_curves.ipynb). Curves are
parsed from the original training logs (`scripts/parse_training_logs.py`); parity plots come from
evaluating each saved model on its own train/validation/test frames (`scripts/parity_predictions.py`),
and the recomputed test errors reproduce the logged ones (e.g. MACE Li<sub>15</sub>Y<sub>7</sub>Cl<sub>36</sub>
from scratch: test force RMSE 10.3 meV/Å in both).

## MACE: from scratch vs fine-tuned from MACE-MPA-0

Four conductors, each trained on its own AIMD data (energies and forces): Li<sub>15</sub>Y<sub>7</sub>Cl<sub>36</sub>
(LYC) and three Na-ion oxides (Na<sub>9</sub>Y<sub>6</sub>Si<sub>3</sub>P<sub>9</sub>O<sub>42</sub>,
Na<sub>5</sub>Nb<sub>5</sub>W<sub>11</sub>O<sub>48</sub>, Na<sub>11</sub>Si<sub>11</sub>Sb<sub>16</sub>As<sub>5</sub>O<sub>80</sub>).
From scratch: 128x0e+128x1o, 2 interactions, correlation 3, r<sub>max</sub> 5 Å, 10 epochs, EMA + SWA.
Fine-tuned: MACE-MPA-0 (medium), same data and epochs.

![MACE LYC fine-tuned](../figures/training/parity_mace_lyc_finetuned.png)

![MACE LYC from scratch](../figures/training/parity_mace_lyc_scratch.png)

![MACE scratch vs fine-tuned](../figures/training/mace_scratch_vs_finetuned.png)

| material | validation force MAE, from scratch (meV/Å) | fine-tuned (meV/Å) |
|---|---|---|
| Li<sub>15</sub>Y<sub>7</sub>Cl<sub>36</sub> | 7.6 | 5.2 |
| Na<sub>9</sub>Y<sub>6</sub>Si<sub>3</sub>P<sub>9</sub>O<sub>42</sub> | 23.3 | 16.2 |
| Na<sub>5</sub>Nb<sub>5</sub>W<sub>11</sub>O<sub>48</sub> | 31.1 | 25.4 |
| Na<sub>11</sub>Si<sub>11</sub>Sb<sub>16</sub>As<sub>5</sub>O<sub>80</sub> | 20.5 | 13.6 |

Fine-tuning from the foundation model lowers the force error in every material with the same data
and number of epochs. Parity figures for every MACE run are in [`figures/training`](../figures/training).

## CHGNet fine-tuning on Li<sub>15</sub>Y<sub>7</sub>Cl<sub>36</sub>

Pretrained CHGNet (412,525 parameters, all layers trainable) fine-tuned on 399,000 AIMD frames
(500-900 K, MP2020-corrected energies), comparing two ways of choosing the training frames:
contiguous subsets of the frame table (900 K frames only) and an energy-histogram scheme that draws
frames evenly across the energy distribution of all temperatures.

![CHGNet 200 epochs](../figures/training/parity_chgnet_lyc_7980_200ep.png)

![CHGNet learning curves](../figures/training/chgnet_lyc_force_mae_vs_epoch.png)

| training set | frames | epochs | test force MAE (eV/Å) |
|---|---|---|---|
| contiguous, 900 K | 7,980 / 15,960 / 30,693 | 10 | 0.036 / 0.033 / 0.032 |
| contiguous, 900 K | 7,980 | 200 | 0.015 |
| energy histogram, all T | 8,121 / 16,279 / 30,000 / 30,222 | 10 | 0.031 / 0.029 / 0.026 / 0.027 |
| energy histogram, batch 16 / 32 | 16,279 and 30,222 | 10 | 0.028-0.031 |

* Longer training matters most: 200 epochs on 7,980 frames reaches 0.015 eV/Å.
* At 10 epochs the energy-histogram sets reach 0.026-0.031 eV/Å; batch size (8/16/32) changes the
  error by at most 0.003 eV/Å.

## CHGNet fine-tuning on Na-ion oxides, unseen temperature

NTPO (Na<sub>8</sub>Ti<sub>8</sub>P<sub>8</sub>O<sub>40</sub>) and NSGO (Na<sub>12</sub>Ge<sub>12</sub>Sb<sub>16</sub>P<sub>4</sub>O<sub>80</sub>),
25,000 AIMD frames each, with the entire 1150 K block held out as the test set: test force MAE 0.035
and 0.040 eV/Å on the unseen temperature.

![CHGNet Na oxides](../figures/training/chgnet_na_oxides_force_mae_vs_epoch.png)

## Notes

* Splits are random within correlated AIMD trajectories (except the held-out temperature above), so
  test errors measure interpolation within the sampled temperature range.
* How these models behave in MD, including against AIMD, is in [md_studies](../md_studies).
