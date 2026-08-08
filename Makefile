.PHONY: help up down ps start stop restart state switch reset rotate test units logs config-opencode

help: ## Show this help
	@echo "Model Dial - Use pool of the fastest capable AI models as single model. Model switched automatically if error."
	@echo ""
	@echo "\033[1mDocker Management:\033[0m"
	@echo "  \033[36mup\033[0m              Start containers (detached)"
	@echo "  \033[36mdown\033[0m            Stop containers"
	@echo "  \033[36mps\033[0m              Show running containers"
	@echo "  \033[36mlogs\033[0m            Stream docker logs (follow mode)"
	@echo ""
	@echo "\033[1mGateway Operations:\033[0m"
	@echo "  \033[36mstart\033[0m           Start gateway"
	@echo "  \033[36mstop\033[0m            Stop gateway"
	@echo "  \033[36mrestart\033[0m         Restart gateway"
	@echo "  \033[36mstate\033[0m           Show gateway state and recent logs (live, updates every 2s)"
	@echo "  \033[36mswitch\033[0m          Switch category (usage: make switch CATEGORY=fast)"
	@echo "  \033[36mreset\033[0m           Remove saved state and restart gateway"
	@echo "  \033[36mrotate\033[0m          Rotate provider (usage: make rotate PROVIDER=openai)"
	@echo ""
	@echo "\033[1mModel Testing:\033[0m"
	@echo "  \033[36mtest\033[0m            Run model tests (scan providers + test models)"
	@echo ""
	@echo "\033[1mIntegrations:\033[0m"
	@echo "  \033[36mconfig-opencode\033[0m  Generate OpenCode provider config from config.json"
	@echo ""
	@echo "\033[1mUnit Tests:\033[0m"
	@echo "  \033[36munits\033[0m           Run Python unit tests (pytest)"
	@echo ""

up: ## Start containers (detached)
	@if [ ! -f docker-compose.yml ]; then \
		echo "docker-compose.yml not found. Copying from docker/docker-compose.example.yml..."; \
		cp docker/docker-compose.example.yml docker-compose.yml; \
	fi
	docker compose up -d

down: ## Stop containers
	docker compose down

ps: ## Show running containers
	docker compose ps

start: ## Start gateway
	docker compose exec -T core ./gateway.sh start

stop: ## Stop gateway
	docker compose exec -T core ./gateway.sh stop

restart: ## Restart gateway
	docker compose exec -T core ./gateway.sh restart

state: ## Show gateway state and recent logs (live, updates every 2s)
	docker compose exec core watch ./gateway.sh state

switch: ## Switch category (usage: make switch CATEGORY=fast)
	@if [ -z "$(CATEGORY)" ]; then \
		echo "Error: CATEGORY argument required. Usage: make switch CATEGORY=fast"; \
		exit 1; \
	fi
	docker compose exec -T core ./gateway.sh switch $(CATEGORY)

reset: ## Remove saved state and restart gateway
	docker compose exec -T core ./gateway.sh reset

rotate: ## Rotate provider (usage: make rotate PROVIDER=openai)
	@if [ -z "$(PROVIDER)" ]; then \
		echo "Error: PROVIDER argument required. Usage: make rotate PROVIDER=openai"; \
		exit 1; \
	fi
	docker compose exec -T core ./gateway.sh rotate $(PROVIDER)

test: ## Run model tests
	docker compose exec -T core ./run.sh

units: ## Run Python unit tests
	python3 -m pytest tests/ -v

logs: ## Stream docker logs (follow mode)
	docker compose logs -f

config-opencode: ## Generate OpenCode provider config from config.json
	python3 tools/generate_opencode_config.py

.DEFAULT_GOAL := help
