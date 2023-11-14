#     ____________       ______  _____
#          ______ \  /\ (_____ \(_____)
#        _____   \ \/  \ ____)  ) |
#         ___ |__/ / /\ (____  (  |_
#      ___________/ /  \_|   |_|\___)
#      Doppler Aided Radar Tomography
#

BATCH?=2
DART?=python manage.py
TRAIN?=python train.py
METHOD?=ngpsh
RESOLUTION?=50
TARGET?=$(DATASET)/$(METHOD)

_TGT=results/$(TARGET)
_MAP=results/$(TARGET)/map.h5
_RAD=results/$(TARGET)/rad.h5
_CAM=results/$(TARGET)/cam.h5
_METRICS=results/$(TARGET)/metrics.npz
_SLICES=results/$(TARGET)/$(TARGET).slice.z.mp4
_VIDEO=results/$(TARGET)/$(TARGET).video.mp4


# -- DART Scripts -------------------------------------------------------------

# Train
$(_TGT):
	$(TRAIN) $(METHOD) -p data/$(DATASET) -o results/$(TARGET) $(FLAGS)
# Map
$(_MAP): $(_TGT)
	$(DART) map -p results/$(TARGET) --resolution $(RESOLUTION)
# Map video
$(_SLICES): $(_MAP)
	$(DART) slice -p results/$(TARGET)
# Radar evaluation
$(_RAD): $(_TGT)
	$(DART) evaluate -p results/$(TARGET) -a -b $(BATCH)
# SSIM calculation
$(_METRICS): $(_RAD)
	$(DART) metrics -p results/$(TARGET)
# Camera evaluation
$(_CAM): $(_TGT)
	$(DART) evaluate -p results/$(TARGET) -ac -b $(BATCH)
# Camera/Radar video
$(_VIDEO): $(_RAD) $(_CAM)
	$(DART) video -p results/$(TARGET)


# -- Aliases ------------------------------------------------------------------

.phony: train map slices radar metrics camera video
train: results/$(TARGET)
map: $(_MAP)
slices: $(_SLICES)
radar: $(_RAD)
metrics: $(_METRICS)
camera: $(_CAM)
video: $(_VIDEO)

.phony: experiment ablation
experiment: train slices video metrics
ablation: train radar metrics


# -- Utilities ----------------------------------------------------------------

.phony: typecheck
typecheck:
	python -m mypy train.py
	python -m mypy manage.py

.phony: baselines
baselines:
	$(DART) gt_map -p data/$(DATASET) $(FLAGS)
	$(DART) simulate -p data/$(DATASET)
	$(DART) metrics --baseline -p results/$(TARGET)
	$(DART) metrics --psnr 35 -p results/$(TARGET)
	$(DART) metrics --psnr 30 -p results/$(TARGET)
	$(DART) metrics --psnr 25 -p results/$(TARGET)
