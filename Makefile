.PHONY: install run-api run-ui test lint types eval docker-up
install:
	python -m pip install -e '.[dev]'
run-api:
	uvicorn researchrag.api.app:app --host 0.0.0.0 --port 8000
run-ui:
	streamlit run ui/app.py --server.address 0.0.0.0
test:
	pytest -q
lint:
	ruff check .
	ruff format --check .
types:
	mypy src ui eval
eval:
	python eval/run.py
docker-up:
	docker compose up --build
