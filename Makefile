ifndef RADIUS
RADIUS = 0.6
endif

ifndef BATCH
BATCH = 32
endif

.phony: eval typecheck eval-all video

eval:
	python evaluate.py -p $(TARGET) -b $(BATCH)
	python examples.py -p $(TARGET)
	python map.py -p $(TARGET) -r $(RADIUS)

video:
	python evaluate.py -p $(TARGET) -a
	python evaluate.py -p $(TARGET) -ac

typecheck:
	python -m mypy train.py
