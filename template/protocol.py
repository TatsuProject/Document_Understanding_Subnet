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

import typing
import bittensor as bt
import pydantic
from typing import List

# TODO(developer): Rewrite with your protocol definition.

# This is the protocol for the dummy miner and validator.
# It is a simple request-response protocol where the validator sends a request
# to the miner, and the miner responds with a dummy response.

# ---- miner ----
# Example usage:
#   def dummy( synapse: Dummy ) -> Dummy:
#       synapse.dummy_output = synapse.dummy_input + 1
#       return synapse
#   axon = bt.axon().attach( dummy ).serve(netuid=...).start()

# ---- validator ---
# Example usage:
#   dendrite = bt.dendrite()
#   dummy_output = dendrite.query( Dummy( dummy_input = 1 ) )
#   assert dummy_output == 2


class Dummy(bt.Synapse):
    """
    A simple dummy protocol representation which uses bt.Synapse as its base.
    This protocol helps in handling dummy request and response communication between
    the miner and the validator.

    Attributes:
    - dummy_input: An integer value representing the input request sent by the validator.
    - dummy_output: An optional integer value which, when filled, represents the response from the miner.
    """

    # Required request input, filled by sending dendrite caller.
    dummy_input: int

    # Optional request output, filled by receiving axon.
    dummy_output: typing.Optional[int] = None

    def deserialize(self) -> int:
        """
        Deserialize the dummy output. This method retrieves the response from
        the miner in the form of dummy_output, deserializes it and returns it
        as the output of the dendrite.query() call.

        Returns:
        - int: The deserialized response, which in this case is the value of dummy_output.

        Example:
        Assuming a Dummy instance has a dummy_output value of 5:
        >>> dummy_instance = Dummy(dummy_input=4)
        >>> dummy_instance.dummy_output = 5
        >>> dummy_instance.deserialize()
        5
        """
        return self.dummy_output



class ProfileSynapse(bt.Synapse):
    """
    The ProfileSynapse subclass of the Synapse class encapsulates the functionalities related to Identification Scenarios.

    It specifies seven fields - `id`, `label`, `type`, `options`, `value`, `image`, `answer` - that define the state of the ProfileSynapse object.
    All of the fields except `answer` are read-only fields defined during object initialization, and `answer` is a mutable
    field that can be updated as the scenario progresses.

    The Config inner class specifies that assignment validation should occur on this class (validate_assignment = True),
    meaning value assignments to the instance fields are checked against their defined types for correctness.

    Attributes:
        id (str): A unique identifier for the task. This field is both mandatory and immutable.
        type (str): A string that specifies the type of the task. This field is both mandatory and immutable and can take values "Generated" and "User" only.
        img_path (str): a string that is actually a path to image
        score (float): A string that captures the score to the profile. This field is mutable.


    Methods:
        deserialize() -> "ProfileSynapse": Returns the instance of the current object.


    The `ProfileSynapse` class also overrides the `deserialize` method, returning the
    instance itself when this method is invoked. Additionally, it provides a `Config`
    inner class that enforces the validation of assignments (`validate_assignment = True`).
    """
    task_id: str
    task_type: str
    task_sub_type: str
    img_path: str
    miner_output: list
    is_miner: bool
    class Config:
        """
        Pydantic model configuration class for ProfileSynapse. This class sets validation of attribute assignment as True.
        validate_assignment set to True means the pydantic model will validate attribute assignments on the class.
        """

        validate_assignment = True
        arbitrary_types_allowed=True

    def deserialize(self) -> "ProfileSynapse":
        """
        Returns the instance of the current ProfileSynapse object.

        This method is intended to be potentially overridden by subclasses for custom deserialization logic.
        In the context of the ProfileSynapse class, it simply returns the instance itself. However, for subclasses
        inheriting from this class, it might give a custom implementation for deserialization if need be.

        Returns:
            ProfileSynapse: The current instance of the ProfileSynapse class.
        """
        return self

    task_id: str = pydantic.Field(
        ...,
        title="ID",
        description="A unique identifier for the task.",
        allow_mutation=False,
    )

    task_type: str = pydantic.Field(
        ...,
        title="Type",
        description="where this task is fetched from?",
        allow_mutation=False,
    )

    task_sub_type: str = pydantic.Field(
        ...,
        title="Sub-Type",
        description="what type of task is this?",
        allow_mutation=False,
    )

    img_path: str = pydantic.Field(
        ...,
        title="img_path",
        description="Base64-encoded image data",
        allow_mutation=False,
    )

    miner_output: list = pydantic.Field(
        ...,
        title="miner_output",
        description="this is output",
        allow_mutation=True,
    )

    is_miner: bool = pydantic.Field(
        ...,
        title="is_miner",
        description="whether a uid is active miner or not",
        allow_mutation=True,
    )

    required_hash_fields: List[str] = pydantic.Field(
        ["task_id", "task_type", "task_sub_type", "img_path", "is_miner"],
        title="Required Hash Fields",
        description="A list of fields that are required for the hash.",
        allow_mutation=False,
    )

    def __str__(self) -> str:
        """
        Returns a string representation of the ProfileSynapse object.

        Returns:
            str: A string representation of the ProfileSynapse object.
        """
        return f"ProfileSynapse(id={self.task_id}, img_path={self.img_path}"

    def to_dict(self):
        return {
            "id": self.task_id,
            "img_path": self.img_path,
        }
