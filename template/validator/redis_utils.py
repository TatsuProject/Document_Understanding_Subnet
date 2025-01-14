
from redis import StrictRedis
import json

import boto3
import os
from dotenv import load_dotenv
import bittensor as bt
load_dotenv()

redis_password = os.getenv('REDIS_PASSWORD')  # Load from environment variable
redis_host = os.getenv('REDIS_HOST')  # Load from environment variable
redis_client = StrictRedis(
    host=redis_host, 
    port=6379, 
    db=0, 
    password=redis_password
)

def get_next_task_from_queue():
    """
    Fetch the oldest task from the Redis queue, update its status to 'processing',
    and keep it in the FIFO list.
    """
    try:
        # Check if the queue exists and has elements
        if redis_client.llen("task_queue") > 0:
            # Fetch the oldest task without removing it
            oldest_task_data = redis_client.lindex("task_queue", 0)  # Get the first task
            if oldest_task_data:
                oldest_task = json.loads(oldest_task_data)  # Deserialize the task
                oldest_task["status"] = "processing"  # Update the status

                # Update the task back in the queue
                redis_client.lset("task_queue", 0, json.dumps(oldest_task))
                return oldest_task  # Return the updated task
        return None  # Queue is empty or does not exist
    except Exception as e:
        bt.logging.info(f"An error occurred while updating the oldest task: {e}")
        return None


def store_results_in_s3(request_id, json_file):

    bucket_name = "tatsusubnet"

    s3 = s3_client = boto3.client('s3',
                         aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
                         aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
                         region_name=os.getenv('AWS_DEFAULT_REGION'))
    file_name = f"{request_id}.json"  # Save as task_id.json
    json_data = json.dumps(json_file)

    try:
        s3.put_object(Bucket=bucket_name, Key=file_name, Body=json_data)
        bt.logging.info(f"File {file_name} successfully uploaded to S3 bucket {bucket_name}.")
    except Exception as e:
        bt.logging.info(f"An error occurred: {e}")

def delete_task_by_request_id(request_id):
    """
    Delete a specific task from the Redis queue based on the request_id.
    """
    try:
        tasks = redis_client.lrange("task_queue", 0, -1)  # Get all tasks in the queue
        for task_data in tasks:
            task = json.loads(task_data)
            if task["request_id"] == request_id:
                redis_client.lrem("task_queue", 0, task_data)  # Remove the task from the queue
                bt.logging.info(f"Task with request_id '{request_id}' deleted successfully.")
                return
        bt.logging.info(f"Task with request_id '{request_id}' not found.")
    except Exception as e:
        bt.logging.info(f"An error occurred while deleting the task: {e}")

    