# SHMS_Calicration_NN

Neural-network training module extracted from the `AI-ML-R-SIDIS` project for SHMS optics calibration.

> Note: The repository name keeps the original requested spelling: **Calicration**.

## Structure

- `training/` — two-stage training framework (SIMC pretraining + sieve/foil finetuning)

## Quick start

1. Create Python environment (recommended Python 3.10+)
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run tests:

```bash
pytest training/tests -v
```

4. Start pretraining:

```bash
python training/scripts/run_pretrain.py --config training/configs/pretrain_config.yaml --simc-files /path/to/*.root
```

## Source attribution

This repository is split from the `training` part of `AI-ML-R-SIDIS`.
