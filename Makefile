ifndef RADIUS
RADIUS = 0.6
endif

.phony: eval typecheck

eval:
	python evaluate.py -p $(TARGET)
	python examples.py -p $(TARGET)
	python map.py -p $(TARGET) -r $(RADIUS)

typecheck:
	mypy train.py
