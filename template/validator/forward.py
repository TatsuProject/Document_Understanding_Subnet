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

from template.validator.uids import get_random_uids
from template.validator.uids import ActiveMinersManager

from template.protocol import ProfileSynapse
import uuid

from fuzzywuzzy import fuzz
from template.validator.reward import get_rewards
import requests
import base64
import sys
import os
from io import BytesIO
import random
from itertools import cycle

script_dir = os.path.dirname(os.path.abspath(__file__))


from template.validator.data_generation.checkbox_generator import GenerateCheckboxTextPair
from template.validator.data_generation.document_generator import GenerateDocument
from template.validator.data_generation.encircled_text_generator import DocumentWithEncircledTextGenerator, DocumentWithEncircledLineGenerator
from template.validator.data_generation.highlighted_text_generator import HighlightedDocumentGenerator

available_tasks = ["checkbox", "doc-class", "doc-parse", "encircle", "highlight"]
task_generator = cycle(available_tasks)


# HEARTBEAT_FILE = f"/heartbeat.txt"  # replace with your actual path

# def update_heartbeat():
#     # Ensure the directory exists
#     os.makedirs(os.path.dirname(HEARTBEAT_FILE), exist_ok=True)

#     # Write the current timestamp to the file
#     with open(HEARTBEAT_FILE, "w") as f:
#         f.write(str(time.time()))

#     bt.logging.info(f"❤️❤️❤️❤️❤️❤️❤️❤️❤️❤️❤️❤️❤️❤️ heart has beaten ❤️❤️❤️❤️❤️❤️❤️❤️❤️❤️❤️❤️❤️ ")


def get_random_image():
    _id = str(uuid.uuid4())
    checkbox_data_generator_object = GenerateCheckboxTextPair("", _id)
    document_generator_object = GenerateDocument("", _id)
    encircled_text_generator = random.choice([DocumentWithEncircledTextGenerator(_id), DocumentWithEncircledLineGenerator(_id)])
    highlighted_text_generator = HighlightedDocumentGenerator(_id)

    finalized_task = next(task_generator)
    bt.logging.info(f"########### sub task type: {finalized_task}")

    if finalized_task in ["doc-class", "doc-parse"]:
        json_label, image = document_generator_object.generate_document()
    elif finalized_task == "checkbox":
        json_label, image = checkbox_data_generator_object.draw_checkbox_text_pairs()
    elif finalized_task == "encircle":
        json_label, image = encircled_text_generator.draw_encircled_text()
    elif finalized_task == "highlight":
        json_label, image = highlighted_text_generator.generate_document_with_highlights()
    buffer = BytesIO()          # Create an in-memory bytes buffer
    image.save(buffer, format="PNG")  # Save the image to the buffer in PNG format
    binary_image = buffer.getvalue()  # Get the binary content of the image
    image_base64 = base64.b64encode(binary_image).decode('utf-8')

    # update_heartbeat()
    return json_label, ProfileSynapse(
        task_id=_id,
        task_type="got from api",
        task_sub_type = finalized_task,
        img_path=image_base64,  # Image in binary format
        miner_output=[],  # This would be updated later
        is_miner=False,
    )

async def forward(self):
    """
    The forward function is called by the validator every time step.

    It is responsible for querying the network and scoring the responses.

    Args:
        self (:obj:`bittensor.neuron.Neuron`): The neuron object which contains all the necessary state for the validator.
    """
    # Initialize active_miners_manager as a class attribute if it doesn't exist
    if not hasattr(self, 'active_miners_manager'):
        self.active_miners_manager = ActiveMinersManager(self, refresh_interval_seconds=600)  # 10 minutes
    
    ground_truth, task = get_random_image()
    
    # Get UIDs to query based on our active miners strategy
    # - All subnet UIDs during refresh
    # - Only active miners between refreshes
    miner_uids = self.active_miners_manager.get_uids_to_query()
    
    bt.logging.info(f"************ querying uids: {miner_uids}")
    start_time = time.time()
    
    responses = await self.dendrite(
        # Send the query to selected miner axons in the network.
        axons=[self.metagraph.axons[uid] for uid in miner_uids],
        synapse=task,
        timeout=150,
        # All responses have the deserialize function called on them before returning.
        # You are encouraged to define your own deserialization function.
        deserialize=True,
    )
    end_time = time.time()
    Tt = end_time - start_time
    
    # If we were refreshing, update our active miners list
    if self.active_miners_manager.pending_refresh:
        self.active_miners_manager.update_active_miners(miner_uids, responses)
    
    # Process responses and assign rewards
    if task.task_sub_type == "checkbox":
        miner_rewards = get_rewards(self, ground_truth.get("checkboxes", []), responses, Tt, task.task_sub_type)
    elif task.task_sub_type == "highlight":
        miner_rewards = get_rewards(self, ground_truth.get("highlights", []), responses, Tt, task.task_sub_type)
    elif task.task_sub_type == "encircle":
        miner_rewards = get_rewards(self, ground_truth.get("encircles", []), responses, Tt, task.task_sub_type)
    elif task.task_sub_type == "doc-class":
        miner_rewards = get_rewards(self, [ground_truth.get("document_class", "")], responses, Tt, task.task_sub_type)
    elif task.task_sub_type == "doc-parse":
        miner_rewards = get_rewards(self, [ground_truth], responses, Tt, task.task_sub_type)
    
    self.update_scores(miner_rewards, miner_uids, task.task_sub_type)
    time.sleep(5)
