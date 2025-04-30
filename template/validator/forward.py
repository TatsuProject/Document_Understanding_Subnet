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

available_tasks = ["checkbox", "doc-class", "doc-parse"]
task_generator = cycle(available_tasks)

def get_random_image():
    _id = str(uuid.uuid4())
    checkbox_data_generator_object = GenerateCheckboxTextPair("", _id)
    document_generator_object = GenerateDocument("", _id)

    finalized_task = next(task_generator)
    bt.logging.info(f"########### sub task type: {finalized_task}")

    if finalized_task in ["doc-class", "doc-parse"]:
        json_label, image = document_generator_object.generate_document()
    elif finalized_task == "checkbox":
        json_label, image = checkbox_data_generator_object.draw_checkbox_text_pairs()
    buffer = BytesIO()          # Create an in-memory bytes buffer
    image.save(buffer, format="PNG")  # Save the image to the buffer in PNG format
    binary_image = buffer.getvalue()  # Get the binary content of the image
    image_base64 = base64.b64encode(binary_image).decode('utf-8')

    return json_label, ProfileSynapse(
        task_id=_id,
        task_type="got from api",
        task_sub_type = finalized_task,
        img_path=image_base64,  # Image in binary format
        miner_output=[],  # This would be updated later
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
    ground_truth, task = get_random_image()
    miner_uids = get_random_uids(self, k=self.config.neuron.sample_size)
    bt.logging.info(f"************ available uids: {miner_uids}")
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
    if task.task_sub_type=="checkbox":
        miner_rewards = get_rewards(self, ground_truth.get("checkboxes", []), responses, Tt, task.task_sub_type)
    elif task.task_sub_type=="doc-class":
        miner_rewards = get_rewards(self, [ground_truth.get("document_class", "")], responses, Tt, task.task_sub_type)
    elif task.task_sub_type=="doc-parse":
        miner_rewards = get_rewards(self, [ground_truth], responses, Tt, task.task_sub_type)
    self.update_scores(miner_rewards, miner_uids, task.task_sub_type)
    time.sleep(5)
