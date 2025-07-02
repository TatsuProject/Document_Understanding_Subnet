# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# TODO(developer): Set your name
# Copyright © 2023 <your name>

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.


import copy
import numpy as np
import asyncio
import argparse
import threading
import bittensor as bt

from typing import List, Union
from traceback import print_exception

from template.base.neuron import BaseNeuron
from template.base.utils.weight_utils import (
    process_weights_for_netuid,
    convert_weights_and_uids_for_emit,
)  # TODO: Replace when bittensor switches to numpy
from template.mock import MockDendrite
from template.utils.config import add_validator_args
import wandb
import os
from dotenv import load_dotenv
import time
import json
load_dotenv()


class BaseValidatorNeuron(BaseNeuron):
    """
    Base class for Bittensor validators. Your validator should inherit from this class.
    """

    neuron_type: str = "ValidatorNeuron"

    @classmethod
    def add_args(cls, parser: argparse.ArgumentParser):
        super().add_args(parser)
        add_validator_args(cls, parser)

    def __init__(self, config=None):
        super().__init__(config=config)

        # Save a copy of the hotkeys to local memory.
        self.hotkeys = copy.deepcopy(self.metagraph.hotkeys)

        # Dendrite lets us send messages to other nodes (axons) in the network.
        if self.config.mock:
            self.dendrite = MockDendrite(wallet=self.wallet)
        else:
            self.dendrite = bt.dendrite(wallet=self.wallet)
        bt.logging.info(f"Dendrite: {self.dendrite}")

        # Set up initial scoring weights for validation
        bt.logging.info("Building validation weights.")
        self.scores = np.zeros(self.metagraph.n, dtype=np.float32)
        
        # =============================== in case validator crashes, to save miners' scores ====================
        # self.score_path = "/bittensor-subnet-template/tmp"                          # define the path to save the scores
        # loaded = self.load_scores_from_disk(self.score_path)
        # # If scores weren't loaded, initialize with zeros
        # if not loaded:
        #     bt.logging.info("Initializing scores with zeros")
        # ------------------------------------------------------------------------------------------------------
        
        # Initialize top_miner_history if it doesn't exist
        if not hasattr(self, "top_miner_history"):
            self.top_miner_history = []

        # Init sync with the network. Updates the metagraph.
        self.sync()

        # Serve axon to enable external connections.
        if not self.config.neuron.axon_off:
            self.serve_axon()
        else:
            bt.logging.warning("axon off, not serving ip to chain.")

        # Create asyncio event loop to manage async tasks.
        self.loop = asyncio.get_event_loop()

        # Instantiate runners
        self.should_exit: bool = False
        self.is_running: bool = False
        self.thread: Union[threading.Thread, None] = None
        self.lock = asyncio.Lock()

        self.wandb_run = self.initialize_wandb()

    def initialize_wandb(self):
        try:
            api_key = os.getenv("WANDB_API_KEY", "")
            
            # Skip wandb initialization if no API key
            if not api_key:
                bt.logging.warning("No W&B API key found. Skipping W&B initialization.")
                return None
                
            os.environ["WANDB_API_KEY"] = api_key
            # Disable interaction with terminal
            os.environ["WANDB_SILENT"] = "true"
            
            # Generate unique ID
            run_id = "Tatsu-Validator3"
            
            wandb_run = wandb.init(
                project="docs-insight (subnet 84)",
                group="Tatsu-Validator",
                name=run_id,
                id=run_id,
                resume="allow",
                # Add these settings for smooth running
                settings=wandb.Settings(
                    _disable_stats=True,
                    _disable_meta=True,
                    console="off"
                ),
                # Make the run public by default
                job_type="production",
                anonymous="allow"  # Allows public access without login
            )
            
            return wandb_run
            
        except Exception as e:
            bt.logging.error(f"Error initializing W&B: {e}")
            return None

    def serve_axon(self):
        """Serve axon to enable external connections."""

        bt.logging.info("serving ip to chain...")
        try:
            self.axon = bt.axon(wallet=self.wallet, config=self.config)

            try:
                self.subtensor.serve_axon(
                    netuid=self.config.netuid,
                    axon=self.axon,
                )
                bt.logging.info(
                    f"Running validator {self.axon} on network: {self.config.subtensor.chain_endpoint} with netuid: {self.config.netuid}"
                )
            except Exception as e:
                bt.logging.error(f"Failed to serve Axon with exception: {e}")
                pass

        except Exception as e:
            bt.logging.error(f"Failed to create Axon initialize with exception: {e}")
            pass

    async def concurrent_forward(self):
        coroutines = [
            self.forward() for _ in range(self.config.neuron.num_concurrent_forwards)
        ]
        await asyncio.gather(*coroutines)

    def run(self):
        """
        Initiates and manages the main loop for the miner on the Bittensor network. The main loop handles graceful shutdown on keyboard interrupts and logs unforeseen errors.

        This function performs the following primary tasks:
        1. Check for registration on the Bittensor network.
        2. Continuously forwards queries to the miners on the network, rewarding their responses and updating the scores accordingly.
        3. Periodically resynchronizes with the chain; updating the metagraph with the latest network state and setting weights.

        The essence of the validator's operations is in the forward function, which is called every step. The forward function is responsible for querying the network and scoring the responses.

        Note:
            - The function leverages the global configurations set during the initialization of the miner.
            - The miner's axon serves as its interface to the Bittensor network, handling incoming and outgoing requests.

        Raises:
            KeyboardInterrupt: If the miner is stopped by a manual interruption.
            Exception: For unforeseen errors during the miner's operation, which are logged for diagnosis.
        """

        # Check that validator is registered on the network.
        self.sync()

        bt.logging.info(f"Validator starting at block: {self.block}")

        # This loop maintains the validator's operations until intentionally stopped.
        try:
            while True:
                bt.logging.info(f"step({self.step}) block({self.block})")

                # Run multiple forwards concurrently.
                self.loop.run_until_complete(self.concurrent_forward())

                # Check if we should exit.
                if self.should_exit:
                    break

                # Sync metagraph and potentially set weights.
                self.sync()

                self.step += 1

        # If someone intentionally stops the validator, it'll safely terminate operations.
        except KeyboardInterrupt:
            self.axon.stop()
            bt.logging.success("Validator killed by keyboard interrupt.")
            exit()

        # In case of unforeseen errors, the validator will log the error and continue operations.
        except Exception as err:
            bt.logging.error(f"Error during validation: {str(err)}")
            bt.logging.debug(str(print_exception(type(err), err, err.__traceback__)))

    def run_in_background_thread(self):
        """
        Starts the validator's operations in a background thread upon entering the context.
        This method facilitates the use of the validator in a 'with' statement.
        """
        if not self.is_running:
            bt.logging.debug("Starting validator in background thread.")
            self.should_exit = False
            self.thread = threading.Thread(target=self.run, daemon=True)
            self.thread.start()
            self.is_running = True
            bt.logging.debug("Started")

    def stop_run_thread(self):
        """
        Stops the validator's operations that are running in the background thread.
        """
        if self.is_running:
            bt.logging.debug("Stopping validator in background thread.")
            self.should_exit = True
            self.thread.join(5)
            self.is_running = False
            bt.logging.debug("Stopped")

    def __enter__(self):
        self.run_in_background_thread()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """
        Stops the validator's background operations upon exiting the context.
        This method facilitates the use of the validator in a 'with' statement.

        Args:
            exc_type: The type of the exception that caused the context to be exited.
                      None if the context was exited without an exception.
            exc_value: The instance of the exception that caused the context to be exited.
                       None if the context was exited without an exception.
            traceback: A traceback object encoding the stack trace.
                       None if the context was exited without an exception.
        """
        if self.is_running:
            bt.logging.debug("Stopping validator in background thread.")
            self.should_exit = True
            self.thread.join(5)
            self.is_running = False
            bt.logging.debug("Stopped")

    def set_weights(self):
        """
        Updated weight-setting function:
        - Allocate 0.5 weight to UID 0 (subnet)
        - Distribute remaining 0.5 weight based on existing logic:
          - Proportional rewards if 2+ miners have score >= 0.8
          - Otherwise, winner-take-all logic with gap threshold and burn mechanism
        """
        top_reward_threshold = 0.05
        max_consecutive_rewards = 18
        subnet_weight = 1.0  # Fixed weight for subnet (UID 0)

        if np.isnan(self.scores).any():
            bt.logging.warning("Scores contain NaN values.")

        norm = np.linalg.norm(self.scores, ord=1, axis=0, keepdims=True)
        if np.any(norm == 0) or np.isnan(norm).any():
            norm = np.ones_like(norm)

        raw_weights = self.scores # / norm
        scores_flat = raw_weights.flatten()
        uids = self.metagraph.uids.tolist()

        # Find UIDs with score >= 0.8
        max_threshold = 0.8
        high_score_indices = np.where(scores_flat >= max_threshold)[0]
        high_score_uids = [uids[i] for i in high_score_indices]
        high_scores = scores_flat[high_score_indices]

        if not hasattr(self, "top_miner_history"):
            self.top_miner_history = []

        final_weights = np.zeros(len(uids))
        
        # Always allocate 0.5 to subnet (UID 0) if present
        if 0 in uids:
            final_weights[uids.index(0)] = subnet_weight
            bt.logging.info(f"Allocating {subnet_weight} weight to UID 0 (subnet)")
        else:
            bt.logging.warning("UID 0 (subnet) not in metagraph. Cannot allocate subnet weight.")
            subnet_weight = 0.0  # If UID 0 not present, redistribute all weight

        remaining_weight = 1.0 - subnet_weight  # Remaining weight to distribute (0.5 or 1.0)

        if len(high_score_indices) >= 2:
            # Proportional reward among high performers
            total = np.sum(high_scores)
            for idx, uid in zip(high_score_indices, high_score_uids):
                if uid != 0:  # Skip UID 0 as it already has weight allocated
                    weight = (scores_flat[idx] / total) * remaining_weight
                    final_weights[uids.index(uid)] = weight

            bt.logging.info(f"Multiple high scorers (≥{max_threshold}): {list(zip(high_score_uids, high_scores))}")
            bt.logging.info(f"Distributing {remaining_weight} emission proportionally among high performers.")

        else:
            # === Fallback to Winner-Take-All Logic ===

            # Sort by score descending
            sorted_indices = np.argsort(scores_flat)[::-1]
            top_uid = uids[sorted_indices[0]]
            top_score = scores_flat[sorted_indices[0]]

            if len(sorted_indices) > 1:
                second_score = scores_flat[sorted_indices[1]]
                top_reward_threshold = top_reward_threshold if (max_threshold - second_score) >= top_reward_threshold else max_threshold - second_score
            else:
                second_score = 0.0
                top_reward_threshold = max_threshold

            score_gap = top_score - second_score

            bt.logging.debug(f"Top UID: {top_uid}, Score: {top_score}")
            bt.logging.debug(f"Second Score: {second_score}, Gap: {score_gap:.2%}")

            reward_this_round = False
            send_to_subnet = False

            if score_gap >= top_reward_threshold:
                self.top_miner_history.append(top_uid)
                self.top_miner_history = self.top_miner_history[-(max_consecutive_rewards + 1):]

                if len(self.top_miner_history) >= max_consecutive_rewards:
                    if all(uid == top_uid for uid in self.top_miner_history[-max_consecutive_rewards:]):
                        bt.logging.warning(
                            f"Miner {top_uid} has been top for {max_consecutive_rewards} consecutive tempos. Sending remaining reward to subnet."
                        )
                        self.top_miner_history = []
                        send_to_subnet = True
                    else:
                        reward_this_round = True
                else:
                    reward_this_round = True
            else:
                bt.logging.info("No miner exceeded threshold gap. Remaining emission goes to subnet.")
                send_to_subnet = True

            if reward_this_round:
                final_weights[uids.index(top_uid)] += remaining_weight
                bt.logging.info(f"Rewarding top miner: UID {top_uid} with remaining weight {remaining_weight}")

            elif send_to_subnet:
                if 0 in uids:
                    final_weights[uids.index(0)] += remaining_weight
                    bt.logging.info(f"Sending remaining emission {remaining_weight} to UID 0 (subnet)")
                else:
                    bt.logging.warning("UID 0 (subnet) not in metagraph. Emission skipped.")

        # --- Emit Weights ---
        (
            processed_weight_uids,
            processed_weights,
        ) = process_weights_for_netuid(
            uids=self.metagraph.uids,
            weights=final_weights,
            netuid=self.config.netuid,
            subtensor=self.subtensor,
            metagraph=self.metagraph,
        )

        (
            uint_uids,
            uint_weights,
        ) = convert_weights_and_uids_for_emit(
            uids=processed_weight_uids,
            weights=processed_weights,
        )

        bt.logging.debug(f"Final uint_uids: {uint_uids}")
        bt.logging.debug(f"Final uint_weights: {uint_weights}")

        result, msg = self.subtensor.set_weights(
            wallet=self.wallet,
            netuid=self.config.netuid,
            uids=uint_uids,
            weights=uint_weights,
            wait_for_finalization=False,
            wait_for_inclusion=False,
            version_key=self.spec_version,
        )

        if result is True:
            bt.logging.info("set_weights on chain successfully!")
            bt.logging.info(f"Top miner history: {self.top_miner_history}")
        else:
            bt.logging.error("set_weights failed", msg)


    def resync_metagraph(self):
        """Resyncs the metagraph and updates the hotkeys and moving averages based on the new metagraph."""
        bt.logging.info("resync_metagraph()")

        # Copies state of metagraph before syncing.
        previous_metagraph = copy.deepcopy(self.metagraph)

        # Sync the metagraph.
        self.metagraph.sync(subtensor=self.subtensor)

        # Check if the metagraph axon info has changed.
        if previous_metagraph.axons == self.metagraph.axons:
            return

        bt.logging.info(
            "Metagraph updated, re-syncing hotkeys, dendrite pool and moving averages"
        )
        # Zero out all hotkeys that have been replaced.
        for uid, hotkey in enumerate(self.hotkeys):
            if hotkey != self.metagraph.hotkeys[uid]:
                self.scores[uid] = 0  # hotkey has been replaced

        # Check to see if the metagraph has changed size.
        # If so, we need to add new hotkeys and moving averages.
        if len(self.hotkeys) < len(self.metagraph.hotkeys):
            # Update the size of the moving average scores.
            new_moving_average = np.zeros((self.metagraph.n))
            min_len = min(len(self.hotkeys), len(self.scores))
            new_moving_average[:min_len] = self.scores[:min_len]
            self.scores = new_moving_average

        # Update the hotkeys.
        self.hotkeys = copy.deepcopy(self.metagraph.hotkeys)

    # ============================================= in case validator crashes, miners' score should be stored on disk ===============================
    # def save_scores_to_disk(self, path: str = None):
    #     """
    #     Save the current scores to disk.
        
    #     Args:
    #         path: The directory path where scores should be saved.
    #             If None, uses the default path from config.
    #     """
    #     if path is None:
    #         # Use a default path based on the netuid if not specified
    #         path = os.path.expanduser(f"~/.bittensor/scores/netuid{self.config.netuid}")
        
    #     # Create directory if it doesn't exist
    #     os.makedirs(path, exist_ok=True)
        
    #     # Create a dictionary with the scores and UIDs
    #     scores_data = {
    #         "timestamp": time.time(),
    #         "uids": self.metagraph.uids.tolist(),
    #         "scores": self.scores.tolist(),
    #         "top_miner_history": getattr(self, "top_miner_history", []),
    #     }
        
    #     # Save to file
    #     filename = os.path.join(path, "scores.json")
    #     with open(filename, "w") as f:
    #         json.dump(scores_data, f)
        
    #     bt.logging.info(f"Saved scores to {filename}")


    # def load_scores_from_disk(self, path: str = None) -> bool:
    #     """
    #     Load scores from disk if they exist.
        
    #     Args:
    #         path: The directory path where scores should be loaded from.
    #             If None, uses the default path from config.
        
    #     Returns:
    #         bool: True if scores were loaded successfully, False otherwise.
    #     """
    #     if path is None:
    #         # Use a default path based on the netuid if not specified
    #         path = os.path.expanduser(f"~/.bittensor/scores/netuid{self.config.netuid}")
        
    #     filename = os.path.join(path, "scores.json")
        
    #     # Check if the file exists
    #     if not os.path.exists(filename):
    #         bt.logging.info(f"No saved scores found at {filename}")
    #         return False
        
    #     try:
    #         # Load the file
    #         with open(filename, "r") as f:
    #             scores_data = json.load(f)
            
    #         # Get the saved UIDs and scores
    #         saved_uids = scores_data.get("uids", [])
    #         saved_scores = scores_data.get("scores", [])
            
    #         # Check if the metagraph has been updated since the scores were saved
    #         # If UIDs have changed, we need to map the old scores to the new UIDs
    #         if not np.array_equal(saved_uids, self.metagraph.uids.tolist()):
    #             bt.logging.info("Metagraph UIDs have changed since scores were saved. Mapping old scores to new UIDs.")
                
    #             # Create a mapping from old UIDs to their scores
    #             uid_to_score = {uid: score for uid, score in zip(saved_uids, saved_scores)}
                
    #             # Initialize new scores array
    #             new_scores = np.zeros_like(self.scores)
                
    #             # Map the old scores to the new UIDs
    #             for i, uid in enumerate(self.metagraph.uids.tolist()):
    #                 if uid in uid_to_score:
    #                     new_scores[i] = uid_to_score[uid]
                
    #             self.scores = new_scores
    #         else:
    #             # If UIDs haven't changed, just use the saved scores
    #             self.scores = np.array(saved_scores)
            
    #         # Load top miner history if available
    #         if "top_miner_history" in scores_data:
    #             self.top_miner_history = scores_data["top_miner_history"]
            
    #         timestamp = scores_data.get("timestamp", 0)
    #         time_diff = time.time() - timestamp
    #         hours = time_diff / 3600
            
    #         bt.logging.info(f"Loaded scores from {filename} (saved {hours:.2f} hours ago)")
    #         return True
        
    #     except Exception as e:
    #         bt.logging.warning(f"Failed to load scores: {e}")
    #         return False
    # ---------------------------------------------------------------------------------------------------------------------------------------------------

    def update_scores(self, rewards: np.ndarray, uids: List[int], task_type: str):
        """Performs exponential moving average on the scores based on the rewards received from the miners."""

        # Check if rewards contains NaN values.
        if np.isnan(rewards).any():
            bt.logging.warning(f"NaN values detected in rewards: {rewards}")
            # Replace any NaN values in rewards with 0.
            rewards = np.nan_to_num(rewards, nan=0)

        # Ensure rewards is a numpy array.
        rewards = np.asarray(rewards)

        # Check if `uids` is already a numpy array and copy it to avoid the warning.
        if isinstance(uids, np.ndarray):
            uids_array = uids.copy()
        else:
            uids_array = np.array(uids)

        # Handle edge case: If either rewards or uids_array is empty.
        if rewards.size == 0 or uids_array.size == 0:
            bt.logging.info(f"rewards: {rewards}, uids_array: {uids_array}")
            bt.logging.warning(
                "Either rewards or uids_array is empty. No updates will be performed."
            )
            return

        # Check if sizes of rewards and uids_array match.
        if rewards.size != uids_array.size:
            raise ValueError(
                f"Shape mismatch: rewards array of shape {rewards.shape} "
                f"cannot be broadcast to uids array of shape {uids_array.shape}"
            )

        # Compute forward pass rewards, assumes uids are mutually exclusive.
        # shape: [ metagraph.n ]
        scattered_rewards: np.ndarray = np.zeros_like(self.scores)
        scattered_rewards[uids_array] = rewards
        bt.logging.debug(f"Scattered rewards: {rewards}")

        # Update scores with rewards produced by this step.
        # shape: [ metagraph.n ]
        alpha: float = self.config.neuron.moving_average_alpha
        self.scores: np.ndarray = alpha * scattered_rewards + (1 - alpha) * self.scores
        bt.logging.debug(f"Updated moving avg scores: {self.scores}")
        self.push_logs_to_wandb(scattered_rewards, self.scores, task_type)
        # ============================= in case validator crashes, miners' states should be saved ==============================
        # # Save scores to disk after updating
        # if not hasattr(self, '_score_save_counter'):
        #     self._score_save_counter = 0
        
        # self._score_save_counter += 1
        # # Save scores every 10 updates
        # if self._score_save_counter >= 5:
        #     self.save_scores_to_disk(self.score_path)
        #     self._score_save_counter = 0
        # ----------------------------------------------------------------------------------------------------------------------



    def push_logs_to_wandb(self, scattered_rewards, moving_averages, task_type):
        try:
            # Check if wandb is initialized and running
            if not wandb.run:
                bt.logging.warning("W&B is not initialized. Skipping W&B logging.")
                return

            # Convert to list and round to 3 decimal places
            scattered_rewards = [round(float(r), 3) for r in scattered_rewards.tolist()]
            moving_averages = [round(float(s), 3) for s in moving_averages.tolist()]

            if scattered_rewards and moving_averages:

                # Ensure we have proper formatting for the table data
                scattered_data = [[float(i), float(r)] for i, r in enumerate(scattered_rewards)]
                moving_avg_data = [[float(i), float(s)] for i, s in enumerate(moving_averages)]

                scattered_table = wandb.Table(
                    columns=["UID", "Reward"],
                    data=scattered_data
                )
                
                moving_avg_table = wandb.Table(
                    columns=["UID", "Score"],
                    data=moving_avg_data
                )
                
                # Log the tables
                wandb.log({
                    f"{task_type}_Scattered_Rewards": scattered_table,
                    f"Moving_Avg_Scores": moving_avg_table
                })
                
                bt.logging.info(f"📌 Logs pushed to WandB successfully")
            else:
                bt.logging.info(f"📌 Not pushing logs to WandB, One of the lists is empty")

        except Exception as e:
            bt.logging.error(f"Error pushing logs to W&B for task {task_type}: {e}")


    def save_state(self):
        """Saves the state of the validator to a file."""
        bt.logging.info("Saving validator state.")

        # Save the state of the validator to file.
        np.savez(
            self.config.neuron.full_path + "/state.npz",
            step=self.step,
            scores=self.scores,
            hotkeys=self.hotkeys,
        )

    def load_state(self):
        """Loads the state of the validator from a file."""
        bt.logging.info("Loading validator state.")

        # Load the state of the validator from file.
        state = np.load(self.config.neuron.full_path + "/state.npz")
        self.step = state["step"]
        self.scores = state["scores"]
        self.hotkeys = state["hotkeys"]
