.PHONY: help install run clean

help:
	@echo "Available commands:"
	@echo "  make install   - Install dependencies from requirements.txt"
	@echo "  make run       - Run the Streamlit app (main.py)"
	@echo "  make clean     - Remove cache and temporary files"
	@echo "  make help      - Show this help message"

install:
	pip install -r requirements.txt

run:
	streamlit run APP/main.py

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	rm -rf .streamlit/cache* 2>/dev/null || true
	rm -rf .pytest_cache 2>/dev/null || true
