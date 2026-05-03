install:
	pip install -r requirements.txt

train:
	python src/train.py

test:
	pytest src/

lint:
	python -m py_compile src/train.py