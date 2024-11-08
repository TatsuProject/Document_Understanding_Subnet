.PHONY: pull_and_run_miner pull_and_run_validator

# Target to pull and run miner
pull_and_run_miner:
	git pull
	python neurons/miner.py --netuid 236 --subtensor.network test --wallet.name miner --wallet.hotkey default --logging.debug

# Target to pull and run validator
pull_and_run_validator:
	git pull
	python neurons/validator.py --netuid 236 --subtensor.network test --wallet.name validator --wallet.hotkey default --logging.debug
