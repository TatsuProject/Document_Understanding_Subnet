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
import asyncio
from template.validator.uids import get_random_uids
from template.protocol import ProfileSynapse
import uuid
from template.validator.reward import get_rewards
import requests
import base64
import os
from io import BytesIO
import random
import numpy as np

from .redis_utils import get_next_task_from_queue, store_results_in_s3, delete_task_by_request_id
from template.validator.data_generation.checkbox_generator import GenerateCheckboxTextPair
from template.validator.data_generation.document_generator import GenerateDocument


def get_random_image():
    _id = str(uuid.uuid4())
    checkbox_data_generator_object = GenerateCheckboxTextPair("", _id)
    document_generator_object = GenerateDocument("", _id)

    available_tasks = ["checkbox", "doc-class", "doc-parse"]

    finalized_task = random.choice(available_tasks)
    bt.logging.info(f"########### sub task type: {finalized_task}")

    if finalized_task in ["doc-class", "doc-parse"]:
        json_label, image = document_generator_object.generate_document()
    elif finalized_task == "checkbox":
        json_label, image = checkbox_data_generator_object.draw_checkbox_text_pairs()

    buffer = BytesIO()          # Create an in-memory bytes buffer
    image.save(buffer, format="PNG")  # Save the image to the buffer in PNG format
    binary_image = buffer.getvalue()  # Get the binary content of the image
    image_base64 = base64.b64encode(binary_image).decode('utf-8')

    return "synthetic", json_label, ProfileSynapse(
        task_id=_id,
        task_type="synthetic",
        task_sub_type = finalized_task,
        img_path=image_base64,  # Image in binary format
        miner_output=[],  # This would be updated later
        score=0  # The score will be calculated by the miner
    )

def get_task_from_redis():
    """
    Validator function that processes tasks from the queue.
    """

    # Get the next task from the queue
    task = get_next_task_from_queue()
    
    if not task:
        bt.logging.info(f"No tasks in the queue..")
        return "", {}, None

    bt.logging.info(f"Task found in the queue.")

    task_sub_type = task.get("task_type", "checkbox")

    return "redis", {}, ProfileSynapse(
        task_id=task["request_id"],
        task_type="redis",
        task_sub_type=task_sub_type,
        img_path=task["image_data"],  # Image in binary format
        miner_output=[],  # This would be updated later
        score=0  # The score will be calculated by the miner
    )


async def forward(self):
    """
    Enhanced forward function to split tasks between top 20% miners and the rest.
    """
    available_miner_uids = get_random_uids(self, k=self.config.neuron.sample_size)
    bt.logging.info(f"************ selected_uid: {available_miner_uids}")

    miner_scores = {}
    miner_uids = []
    # Maintain a record of miner scores (initialize if not already done).
    for each_uid, each_score in enumerate(list(self.scores)):
        if each_uid in available_miner_uids:
            miner_uids.append(each_uid)
            miner_scores[each_uid] = each_score  # All miners start with score 0.

    # Sort miners by performance and get top 20%.
    sorted_miners = sorted(miner_scores.items(), key=lambda x: x[1], reverse=True)
    top_30_percent_count = max(1, int(0.3 * len(sorted_miners)))  # Ensure at least 1 miner.
    top_30_percent_miners = [uid for uid, _ in sorted_miners[:top_30_percent_count]]
    rest_miners = [uid for uid, _ in sorted_miners[top_30_percent_count:]]

    # ======================= parallel =========================================
    # Prepare task assignments
    _, synthetic_ground_truth, synthetic_task = get_random_image()

    miner_task_map = {}
    for uid in top_30_percent_miners:
        task_type, ground_truth, task = get_task_from_redis()
        if task:
            bt.logging.info(f"|``````````````````````````````````````````````|")
            bt.logging.info(f"|                   uid: {uid}                     |")
            bt.logging.info(f"|               task type: redis               |")
            bt.logging.info(f"| task id: {task.task_id}|")
            bt.logging.info(f"|______________________________________________|")
            miner_task_map[uid] = task  # Redis task for top 30% miners
        else:
            bt.logging.info(f"|``````````````````````````````````````````````|")
            bt.logging.info(f"|                   uid: {uid}                     |")
            bt.logging.info(f"|              task type: sythetic             |")
            bt.logging.info(f"| task id: {synthetic_task.task_id}|")
            bt.logging.info(f"|______________________________________________|")
            miner_task_map[uid] = synthetic_task

    for uid in rest_miners:
        bt.logging.info(f"|``````````````````````````````````````````````|")
        bt.logging.info(f"|                   uid: {uid}                     |")
        bt.logging.info(f"|              task type: sythetic             |")
        bt.logging.info(f"| task id: {synthetic_task.task_id}|")
        bt.logging.info(f"|______________________________________________|")
        miner_task_map[uid] = synthetic_task  # Synthetic task for the rest

    # Prepare axons and tasks
    axons = []
    tasks = []
    for uid, assigned_task in miner_task_map.items():
        axons.append(self.metagraph.axons[uid])  # Add axon for the miner
        tasks.append(assigned_task)  # Add corresponding task

    start_time = time.time()
    # Send requests to all miners in a single call


    async def send_request_to_miner(uid, axon, task):
        try:
            response = await self.dendrite(
                axons=[axon],
                synapse=task,
                timeout=3600,
                deserialize=True,
            )
            return response  # Return UID and the response
        except Exception as e:
            bt.logging.error(f"Error while sending request to miner {uid}: {e}")
            return None  # Handle errors gracefully

    tasks = [
        send_request_to_miner(
            uid=uid,
            axon=self.metagraph.axons[uid],  # Axon corresponding to the miner
            task=assigned_task,  # Task assigned to this miner
        )
        for uid, assigned_task in miner_task_map.items()
    ]

    responses = await asyncio.gather(*tasks)
    end_time = time.time()
    Tt = end_time - start_time
    # --------------------------------------------------------------------------

    # Evaluate miners handling Redis tasks.
    avg_top_30_score = 0.2
    avg_top_30_score = np.mean(list(self.scores[uid] for uid in top_30_percent_miners))

    if avg_top_30_score<=0.0:
        avg_top_30_score = np.mean(list(miner_scores[uid] for uid in rest_miners))
    redis_score = avg_top_30_score + (0.05*avg_top_30_score)  # give 5% bonus for redis tasks
 
    miner_rewards = get_rewards(self, synthetic_ground_truth, responses, Tt, redis_score)
    self.update_scores(miner_rewards, miner_uids)

    for each_response in responses:
        if each_response[0].task_type=="redis":
            # Process Redis task responses.
            info_detected = each_response[0].miner_output
            json_file = {
                "task_id": each_response[0].task_id,
                "status": "success",
                "task_type": each_response[0].task_sub_type,
                "response": info_detected
            }
            store_results_in_s3(each_response[0].task_id, json_file)
            delete_task_by_request_id(each_response[0].task_id)

    # Log updated scores.
    time.sleep(2)