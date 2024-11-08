.PHONY: pull_and_run_miner pull_and_run_validator pull_and_run_miner_with_pm2 pull_and_run_validator_with_pm2

# Target to pull and run miner
pull_and_run_miner:
	git pull
	python neurons/miner.py --netuid 236 --subtensor.network test --wallet.name miner --wallet.hotkey default --logging.debug

# Target to pull and run validator
pull_and_run_validator:
	git pull
	python neurons/validator.py --netuid 236 --subtensor.network test --wallet.name validator --wallet.hotkey default --logging.debug

# Define the pull_and_run_miner_with_pm2 target
pull_and_run_miner_with_pm2:
	git pull
	pm2 start python --name my_miner -- neurons/miner.py --netuid 236 --subtensor.network test --wallet.name miner --wallet.hotkey default --logging.debug

# Define the pull_and_run_validator_with_pm2 target
pull_and_run_validator_with_pm2:
	git pull
	pm2 start python --name my_validator -- neurons/validator.py --netuid 236 --subtensor.network test --wallet.name validator --wallet.hotkey default --logging.debug
