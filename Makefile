.PHONY: pull_and_run_miner pull_and_run_validator pull_and_run_miner_with_pm2 pull_and_run_validator_with_pm2 install_requirements

# Target to install dependencies
install_requirements:
	pip install -r requirements.txt

# Target to pull and run miner
pull_and_run_miner: install_requirements
	git pull
	python neurons/miner.py --netuid 84 --subtensor.network finney --wallet.name miner --wallet.hotkey default --logging.debug

# Target to pull and run validator
pull_and_run_validator: install_requirements
	git pull
	python neurons/validator.py --netuid 84 --subtensor.network finney --wallet.name validator --wallet.hotkey default --logging.debug

# Define the pull_and_run_miner_with_pm2 target
pull_and_run_miner_with_pm2: install_requirements
	git pull
	pm2 start python --name my_miner -- neurons/miner.py --netuid 84 --subtensor.network finney --wallet.name miner --wallet.hotkey default --logging.debug

# Define the pull_and_run_validator_with_pm2 target
pull_and_run_validator_with_pm2: install_requirements
	git pull
	pm2 start python --name my_validator -- neurons/validator.py --netuid 84 --subtensor.network finney --wallet.name validator --wallet.hotkey default --logging.debug
