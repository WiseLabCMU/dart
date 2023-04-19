.phony: eval
eval:
	python evaluate.py -p $(TARGET)
	python examples.py -p $(TARGET)
	python map.py -p $(TARGET) -r 0.6