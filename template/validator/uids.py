import random
import bittensor as bt
import numpy as np
from typing import List, Dict, Set, Optional
import time


def check_uid_availability(
    metagraph: "bt.metagraph.Metagraph", uid: int, vpermit_tao_limit: int
) -> bool:
    """Check if uid is available. The UID should be available if it is serving and has less than vpermit_tao_limit stake
    Args:
        metagraph (:obj: bt.metagraph.Metagraph): Metagraph object
        uid (int): uid to be checked
        vpermit_tao_limit (int): Validator permit tao limit
    Returns:
        bool: True if uid is available, False otherwise
    """
    # Filter non serving axons.
    if not metagraph.axons[uid].is_serving:
        return False
    # Filter validator permit > 1024 stake.
    if metagraph.validator_permit[uid]:
        if metagraph.S[uid] > vpermit_tao_limit:
            return False
    # Available otherwise.
    return True


def get_random_uids(
    self, k: int, exclude: List[int] = None
) -> np.ndarray:
    """Returns k available random uids from the metagraph.
    Args:
        k (int): Number of uids to return.
        exclude (List[int]): List of uids to exclude from the random sampling.
    Returns:
        uids (np.ndarray): Randomly sampled available uids.
    Notes:
        If `k` is larger than the number of available `uids`, set `k` to the number of available `uids`.
    """
    candidate_uids = []
    avail_uids = []

    for uid in range(self.metagraph.n.item()):
        uid_is_available = check_uid_availability(
            self.metagraph, uid, self.config.neuron.vpermit_tao_limit
        )
        uid_is_not_excluded = exclude is None or uid not in exclude

        if uid_is_available:
            avail_uids.append(uid)
            if uid_is_not_excluded:
                candidate_uids.append(uid)
    # If k is larger than the number of available uids, set k to the number of available uids.
    k = min(k, len(avail_uids))
    # Check if candidate_uids contain enough for querying, if not grab all avaliable uids
    available_uids = candidate_uids
    if len(candidate_uids) < k:
        available_uids += random.sample(
            [uid for uid in avail_uids if uid not in candidate_uids],
            k - len(candidate_uids),
        )
    uids = np.array(random.sample(available_uids, k))
    return uids


class ActiveMinersManager:
    """Manages the collection of active miners and periodically refreshes the list."""
    
    def __init__(self, neuron, refresh_interval_seconds: int = 600):
        """
        Initialize the active miners manager.
        
        Args:
            neuron (:obj:`bittensor.neuron.Neuron`): The neuron object which contains metagraph and config.
            refresh_interval_seconds (int): Time interval in seconds after which to refresh the list of active miners.
                                          Default is 600 seconds (10 minutes).
        """
        self.neuron = neuron
        self.active_miners: Set[int] = set()
        self.last_refresh_time: float = 0
        self.refresh_interval_seconds: int = refresh_interval_seconds
        self.pending_refresh: bool = True  # Start with a refresh
    
    def should_refresh(self) -> bool:
        """
        Check if we should refresh the list of active miners.
        
        Returns:
            bool: True if we should refresh, False otherwise.
        """
        current_time = time.time()
        time_since_refresh = current_time - self.last_refresh_time
        if time_since_refresh>=self.refresh_interval_seconds:
            self.pending_refresh = True
        return self.pending_refresh or time_since_refresh >= self.refresh_interval_seconds
    
    def update_active_miners(self, uids: List[int], responses: List):
        """
        Update the list of active miners based on the responses received.
        
        Args:
            uids (List[int]): List of UIDs that were queried.
            responses (List): List of responses from the queried UIDs.
        """
        # Clear existing active miners when refreshing
        self.active_miners.clear()
        
        for i, (uid, response) in enumerate(zip(uids, responses)):
            try:
                # Consider a miner active if the response is not empty/None
                if response.is_miner or response.miner_output:  # This should be adjusted based on how "empty" responses look
                    self.active_miners.add(uid)
            except:
                continue
        
        self.pending_refresh = False
        self.last_refresh_time = time.time()
        bt.logging.info(f"Updated active miners list. Found {len(self.active_miners)} active miners.")
    
    def mark_for_refresh(self):
        """Mark that we need to refresh the list of active miners on the next iteration."""
        self.pending_refresh = True
    
    def get_all_subnet_uids(self) -> List[int]:
        """
        Returns all UIDs registered to the subnet from the metagraph.
        
        Returns:
            List[int]: All UIDs registered to the subnet.
        """
        available_uids = []
        
        for uid in range(self.neuron.metagraph.n.item()):
            # Include all UIDs that are registered to the subnet
            available_uids.append(uid)
        
        return available_uids
    
    def get_uids_to_query(self) -> np.ndarray:
        """
        Returns UIDs to query, based on the refresh cycle:
        - During refresh: return all subnet UIDs
        - Between refreshes: return only active miners
        
        Returns:
            np.ndarray: Array of UIDs to query.
        """
        # Check if we need to refresh our list of active miners
        if self.should_refresh():
            bt.logging.info("Refreshing active miners list - querying all subnet UIDs...")
            # If we need to refresh, query all UIDs in the subnet
            all_subnet_uids = self.get_all_subnet_uids()
            self.last_refresh_time = time.time()
            return np.array(all_subnet_uids)
        else:
            # Otherwise, only query the active miners
            active_uids = list(self.active_miners)
            bt.logging.info(f"Using {len(active_uids)} active miners")
            return np.array(active_uids)