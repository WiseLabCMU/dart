ifndef RADIUS
RADIUS = 0.6
endif

ifndef BATCH
BATCH = 32
endif

.phony: eval typecheck

eval:
	python evaluate.py -p $(TARGET) -b $(BATCH)
	python examples.py -p $(TARGET)
	python map.py -p $(TARGET) -r $(RADIUS)

typecheck:
	mypy train.py
