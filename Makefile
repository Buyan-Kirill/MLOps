# Makefile
SHELL := /bin/bash
VENV_DIR := venv
PYTHON := $(VENV_DIR)/bin/python3
PIP := $(VENV_DIR)/bin/pip

export VIRTUAL_ENV := $(PWD)/$(VENV_DIR)
export PATH := $(VIRTUAL_ENV)/bin:$(PATH)
export PYTHONPATH := $(PWD)

.PHONY: venv setup lint test_pipeline prepare_data embed train test_embeddings evaluate all clean clean-all help

venv:
	@echo "Создание виртуального окружения в $(VENV_DIR)..."
	@test -d "$(VENV_DIR)" || python3 -m venv "$(VENV_DIR)"
	@$(PIP) install --upgrade pip
	@echo "venv создан."

setup: venv
	@echo "Установка зависимостей..."
	@$(PIP) install -r requirements.txt
	@echo "Зависимости установлены."

lint:
	@$(PIP) install flake8 > /dev/null 2>&1 || true
	@$(VENV_DIR)/bin/flake8 src/ scripts/ tests/ || echo " flake8: пропускаем"

test_pipeline:
	$(PYTHON) -m pytest tests/ -v --tb=short

prepare_data:
	@$(PYTHON) scripts/prepare_data.py --config configs/default.yaml

embed:
	@$(PYTHON) scripts/embed.py --config configs/default.yaml

train:
	@$(PYTHON) scripts/train.py --config configs/default.yaml

encode_embeddings:
	@$(PYTHON) scripts/encode_embeddings.py --config configs/default.yaml

test_embeddings:
	@$(PYTHON) scripts/compare_embeddings_quality.py --config configs/default.yaml

evaluate_recommender:
	@$(PYTHON) scripts/evaluate_recommender.py --config configs/default.yaml

recommend:
	@if [ -z "$(TITLES)" ] || [ -z "$(AUTHORS)" ] || [ -z "$(RATINGS)" ]; then \
		echo " Используйте: make recommend TITLES=\"...\" AUTHORS=\"...\" RATINGS=\"...\""; \
		echo " Пример: make recommend TITLES=\"1984; The Picture of Dorian Gray\" AUTHORS=\"Оруэлл; Oscar Wilde\" RATINGS=\"5; 5\""; \
		exit 1; \
	fi
	$(PYTHON) scripts/recommend.py \
		--config configs/default.yaml \
		--titles "$(TITLES)" \
		--authors "$(AUTHORS)" \
		--ratings "$(RATINGS)"

all: setup prepare_data embed train encode_embeddings test_embeddings evaluate_recommender

clean:
	rm -rf logs/* outputs/* processed_data/*
	find . -name "*.pyc" -delete
	find . -name "__pycache__" -type d -exec rm -rf {} +

clean-all: clean
	rm -rf $(VENV_DIR)

help:
	@echo "Book Recommender — Makefile"
	@echo
	@echo "Основные команды:"
	@echo "  make setup                 — создать venv + установить зависимости"
	@echo "  make prepare_data          — скачать датасеты и подготовить данные"
	@echo "  make embed                 — построить мультимодальные эмбеддинги"
	@echo "  make train                 — обучить контрастивный энкодер эмбедингов книг"
	@echo "  make encode_embeddings     — построить сжатые эмбеддинги книг"
	@echo "  make test_embeddings       — сравнить эмбединги до сжатия и после"
	@echo "  make evaluate_recommender  — подобрать вес популярности книги и оценить качество"
	@echo "  make recommend             — рекомендовать книгу на основе введённой истории"
	@echo "  make all                   — полный пайплайн"
	@echo "  make test_pipeline         — запустить тесты пайплайна"
	@echo "  make clean                 — очистить артефакты (не venv)"
	@echo "  make clean-all             — полная очистка (включая venv)"
	@echo
	@echo "💡 Первый запуск: make setup"