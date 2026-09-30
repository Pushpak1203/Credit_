PYTHON ?= python
setup:
	$(PYTHON) -m pip install -r requirements.txt
data:
	$(PYTHON) -m src.credit_sense.data.generate_synthetic
train:
	$(PYTHON) -m src.credit_sense.train
evaluate:
	$(PYTHON) -m src.credit_sense.evaluate
fairness:
	$(PYTHON) -m src.credit_sense.fairness
pilot:
	$(PYTHON) -m src.credit_sense.pilot_simulation
shadow:
	$(PYTHON) -m src.credit_sense.shadow
drift:
	$(PYTHON) -m src.credit_sense.drift
test:
	$(PYTHON) -m pytest -q
api:
	uvicorn api.main:app --reload

dashboard:
	streamlit run app/dashboard.py
all: data train evaluate fairness pilot shadow drift test
