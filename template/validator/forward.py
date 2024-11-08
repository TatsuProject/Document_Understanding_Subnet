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

import time
import bittensor as bt

# from template.protocol import Dummy
# from template.validator.reward import get_rewards
# from template.utils.uids import get_random_uids


from template.validator.uids import get_random_uids
from template.protocol import ProfileSynapse
import uuid

from fuzzywuzzy import fuzz
from template.validator.reward import get_rewards
import requests
import base64
import sys
import os


def get_random_image_path():
    url = 'http://3.21.227.102:3000/api/tatsu/random'
    
    # Send a GET request to the API
    response = requests.get(url)
    
    # Check if the request was successful
    if response.status_code == 200:
        # Parse the response JSON
        response_data = response.json()
        
        if response_data.get('status') is True:
            # Access the 'data' section which contains image and labels
            data = response_data.get('data')
            
            if data:
                image_url = data.get('image_url')
                label_data = data.get('data')  # Label information inside 'data' key
                
                if image_url and label_data:
                    # Get the image file name from the URL
                    image_name = os.path.basename(image_url)
                    
                    # Fetch the image (binary format)
                    image_response = requests.get(image_url)
                    if image_response.status_code == 200:
                        image = image_response.content  # Image in binary format
                        
                    # JSON label in dictionary format
                    json_label = label_data  # Already a dictionary

                    bt.logging.info(f"Successfully retrieved image and label.")
                else:
                    bt.logging.info("Error: Could not retrieve image URL or label data.")
                    return None, None
            else:
                bt.logging.info("Error: 'data' field is missing in the response.")
                return None, None
        else:
            bt.logging.info("Error: The request status is False.")
            return None, None
    else:
        print(f"Failed to retrieve data. Status code: {response.status_code}")
        return None, None

    # Generate a UUID for the task
    _id = str(uuid.uuid4())
    image_base64 = base64.b64encode(image).decode('utf-8')

    return json_label, ProfileSynapse(
        task_id=_id,
        task_type="got from api",
        img_path=image_base64,  # Image in binary format
        checkbox_output=[],  # This would be updated later
        score=0  # The score will be calculated by the miner
    )

async def forward(self):
    """
    The forward function is called by the validator every time step.

    It is responsible for querying the network and scoring the responses.

    Args:
        self (:obj:`bittensor.neuron.Neuron`): The neuron object which contains all the necessary state for the validator.

    """
    # TODO(developer): Define how the validator selects a miner to query, how often, etc.
    # get_random_uids is an example method, but you can replace it with your own.
    ground_truth, task = get_random_image_path()
    miner_uids = get_random_uids(self, k=self.config.neuron.sample_size)
    bt.logging.info(f"************ available uids: {miner_uids}")
    start_time = time.time()
    responses = await self.dendrite(
        # Send the query to selected miner axons in the network.
        axons=[self.metagraph.axons[uid] for uid in miner_uids],
        synapse=task,
        timeout=3600,
        # All responses have the deserialize function called on them before returning.
        # You are encouraged to define your own deserialization function.
        deserialize=True,
    )
    end_time = time.time()
    Tt = end_time - start_time
    miner_rewards = get_rewards(self, ground_truth.get("checkboxes", []), responses, Tt)
    self.update_scores(miner_rewards, miner_uids)
    time.sleep(5)
