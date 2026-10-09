.PHONY: help install lint test demo demo-quick serve docker-build docker-demo docker-api clean

help:            ## show this help
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  %-14s %s\n",$$1,$$2}'

install:         ## install runtime + dev dependencies
	pip install -r requirements-dev.txt

lint:            ## static checks (ruff)
	ruff check .

test:            ## unit + smoke tests
	pytest

demo:            ## full synthetic demo on CPU (a few minutes)
	python demo_synthetic.py --out-dir artifacts

demo-quick:      ## ~1 minute version of the demo
	python demo_synthetic.py --quick --out-dir artifacts

serve:           ## start the API (run `make demo` first)
	uvicorn serve:app --host 0.0.0.0 --port 8000

docker-build:    ## build the CPU image
	docker build -t rl4m4:cpu .

docker-demo:     ## run the demo inside Docker, artifacts land in ./artifacts
	docker compose run --rm demo

docker-api:      ## demo + API in Docker on http://localhost:8000/docs
	docker compose up --build api

clean:
	rm -rf artifacts .pytest_cache .ruff_cache __pycache__ tests/__pycache__
