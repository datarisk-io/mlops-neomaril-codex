import os
from datetime import datetime, timedelta
from http import HTTPStatus
from time import sleep
from typing import Optional

import requests
from dotenv import load_dotenv

from neomaril_codex.__model_states import ModelExecutionState
from neomaril_codex.__utils import (
    parse_json_to_yaml,
    parse_url,
    refresh_token,
    try_login,
)
from neomaril_codex.exceptions import (
    AuthenticationError,
    ExecutionError,
    GroupError,
    InputError,
    ModelError,
    ServerError,
)
from neomaril_codex.logger_config import get_logger

logger = get_logger()


class BaseNeomaril:
    """
    Super base class to initialize other variables and URLs for other Neomaril classes.
    """

    def __init__(
        self,
        *,
        login: Optional[str] = None,
        password: Optional[str] = None,
        url: Optional[str] = None,
    ) -> None:
        load_dotenv()
        logger.info("Loading .env")

        if url is None:
            url = os.getenv("NEOMARIL_URL")
        if url is None:
            url = "https://neomaril.staging.datarisk.net/"

        self.credentials = (
            login if login else os.getenv("NEOMARIL_USER"),
            password if password else os.getenv("NEOMARIL_PASSWORD"),
        )
        self.base_url = url
        self.base_url = parse_url(self.base_url)

        if self.base_url == "https://neomaril.staging.datarisk.net/":
            logger.info(
                "You are using the test enviroment that will have the data cleaned from time to time. If your model is ready to use change the enviroment to Production"
            )

        self.user_token, self.version = try_login(
            self.credentials[0],
            self.credentials[1],
            self.base_url,
        )
        logger.info("Successfully connected to Neomaril")

    def _logs(
        self,
        *,
        url,
        credentials,
        start: Optional[str] = None,
        end: Optional[str] = None,
        routine: Optional[str] = None,
        type: Optional[str] = None,
    ):
        if not start and not end:
            end = datetime.today().strftime("%d-%m-%Y")
            start = (datetime.today() - timedelta(days=6)).strftime("%d-%m-%Y")

        if not start and end:
            start = (datetime.strptime(end, "%d-%m-%Y") - timedelta(days=6)).strftime(
                "%d-%m-%Y"
            )

        if start and not end:
            end = (datetime.strptime(start, "%d-%m-%Y") + timedelta(days=6)).strftime(
                "%d-%m-%Y"
            )

        query = {"start": start, "end": end}

        if routine:
            assert routine in ["Run", "Host"]
            query["routine"] = routine

        if type:
            assert type in ["Ok", "Error", "Debug", "Warning"]
            query["type"] = type

        response = requests.get(
            url,
            params=query,
            headers={
                "Authorization": "Bearer " + refresh_token(*credentials, self.base_url)
            },
        )

        if response.status_code == 200:
            return response.json()

        if response.status_code == 401:
            logger.error(
                "Login or password are invalid, please check your credentials."
            )
            raise AuthenticationError("Login not authorized.")

        if response.status_code >= 500:
            logger.error("Server is not available. Please, try it later.")
            raise ServerError("Server is not available!")

        logger.error(response.text)
        raise InputError("Bad Input. Client error")


class BaseNeomarilClient(BaseNeomaril):
    """
    Base class for Neomaril client side related classes. This is the class that contains some methods related to Client models administration.
    Mainly related to initialize environment and its variables, but also to generate groups.
    A group is a way to organize models clustering for different users and also to increase security.
    Each group has a unique token that should be used to run the models that belongs to that.

    Attributes
    ----------
    login : str
        Login for authenticating with the client. You can also use the env variable NEOMARIL_USER to set this
    password : str
        Password for authenticating with the client. You can also use the env variable NEOMARIL_PASSWORD to set this
    url : str
        URL to Neomaril Server. Default value is https://neomaril.staging.datarisk.net, use it to test your deployment first before changing to production. You can also use the env variable NEOMARIL_URL to set this

    Raises
    ------
    NotImplementedError
        When the environment is production, because it is not implemented yet

    Example
    -------
    In this example you can see how to create a group and after consult the list of groups that already exists.

    .. code-block:: python

        from neomaril_codex.base import BaseNeomarilClient

        def start_group(password):
            client = BaseNeomarilClient(password)
            isCreated = client.create_group('ex_group', 'Group for example purpose')

            print(client.list_groups())

            return isCreated
    """

    def list_groups(self) -> list:
        """
        List all existing groups.

        Raises
        ------
        ServerError
            Unexpected server error

        Returns
        -------
        list
            List with the groups that exists in the database
        """

        url = f"{self.base_url}/groups"

        token = refresh_token(*self.credentials, self.base_url)

        response = requests.get(url, headers={"Authorization": "Bearer " + token})

        if response.status_code == 200:
            results = response.json()["Results"]
            return results

        formatted_msg = parse_json_to_yaml(response.json())

        if response.status_code == 401:
            logger.error(
                "Login or password are invalid, please check your credentials."
            )
            raise AuthenticationError("Login not authorized.")

        if response.status_code >= 500:
            logger.error("Server is not available. Please, try it later.")
            raise ServerError("Server is not available!")

        logger.error(f"Something went wrong...\n{formatted_msg}")
        raise InputError("Bad Input. Client error")

    def create_group(self, *, name: str, description: str) -> bool:
        """
        Create a group for multiple models of the same final client at the end if it returns TRUE, a message with the token for that group will be returned as a INFO message.
        You should keep this token information to be able to run the model of that group afterward.

        Arguments
        ---------
        name : str
            Name of the group. Must be 32 characters long and with no special characters (some parsing will be made)
        description : str
            Short description of the group

        Raises
        ------
        ServerError
            Unexpected server error

        Returns
        -------
        bool
            Returns True if the group was successfully created and False if not
        """
        data = {"name": name, "description": description}

        url = f"{self.base_url}/groups"
        token = refresh_token(*self.credentials, self.base_url)

        response = requests.post(
            url, data=data, headers={"Authorization": "Bearer " + token}
        )

        if response.status_code == 201:
            t = response.json().get("Token")
            logger.info(
                f"Group '{name}' inserted. Use the token for scoring. Carefully save it as we won't show it again. Token: {t}"
            )
            return HTTPStatus.OK

        formatted_msg = parse_json_to_yaml(response.json())

        if response.status_code == 409:
            logger.info(f"Something went wrong:\n {formatted_msg}")
            raise GroupError("Group already exist, nothing was changed.")

        if response.status_code == 401:
            logger.error(
                "Login or password are invalid, please check your credentials."
            )
            raise AuthenticationError("Login not authorized.")

        if response.status_code >= 500:
            logger.error("Server is not available. Please, try it later.")
            raise ServerError("Server is not available!")

        logger.error(f"Something went wrong:\n {formatted_msg}")
        raise InputError("Bad Input. Client error")

    def refresh_group_token(self, *, name: str, force: bool = False) -> bool:
        """
        Refresh the group token. If the token it's still valid it won't be changed, unless you use parameter force = True.
        At the end a message with the token for that group will be returned as a INFO message.
        You should keep this new token information to be able to run the model of that group afterward.

        Arguments
        ---------
        name : str
            Name of the group to have the token refreshed
        force : str
            Force token expiration even if its still valid (this can make multiple models integrations stop working, so use with care)

        Raises
        ------
        ServerError
            Unexpected server error

        Returns
        -------
        bool
            Returns True if the group was successfully created and False if not.

        Example
        --------
        Suppose that you lost the token to access your group, you can create a new one forcing it with this method as at the example below.

        .. code-block:: python

            from neomaril_codex.base import BaseNeomarilClient

            def update_group_token(model_client, group_name):
                model_client.refresh_group_token('ex_group', True)
                print(client.list_groups())

                return isCreated
        """

        url = f"{self.base_url}/groups/refresh/{name}"
        token = refresh_token(*self.credentials, self.base_url)

        response = requests.get(
            url,
            params={"force": str(force).lower()},
            headers={"Authorization": "Bearer " + token},
        )

        if response.status_code == 201:
            t = response.json()["Token"]
            logger.info(f"Group '{name}' was refreshed. New token: {t}")
            return HTTPStatus.OK

        formatted_msg = parse_json_to_yaml(response.json())

        if response.status_code == 401:
            logger.error(
                "Login or password are invalid, please check your credentials."
            )
            raise AuthenticationError("Login not authorized.")

        if response.status_code >= 500:
            logger.error("Server is not available. Please, try it later.")
            raise ServerError("Server is not available!")

        logger.error(f"Something went wrong...\n{formatted_msg}")
        raise InputError("Bad Input. Client error")


class NeomarilExecution(BaseNeomaril):
    """
    Base class for Neomaril asynchronous model executions. With this class you can visualize the status of an execution and download the results after and execution has finished.

    Attributes
    ----------
    parend_id : str
        Model id (hash) from the model you want to access
    exec_type : str
        Flag that contains which type of execution you use. It can be 'AsyncModel' or 'Training'
    group : str, optional
        Group the model is inserted
    exec_id : str, optional
        Execution id
    login : str
        Login for authenticating with the client. You can also use the env variable NEOMARIL_USER to set this
    password : str
        Password for authenticating with the client. You can also use the env variable NEOMARIL_PASSWORD to set this
    url : str
        URL to Neomaril Server. Default value is https://neomaril.staging.datarisk.net, use it to test your deployment first before changing to production. You can also use the env variable NEOMARIL_URL to set this

    Raises
    ------
    InputError
        Invalid execution type
    ModelError
        If the execution id was not found or wasn't possible to retrieve it

    Example
    -------
    In this example you can see how to get the status of an existing execution and download its results

    .. code-block:: python

        from neomaril_codex.base import NeomarilExecution
        from neomaril_codex.model import NeomarilModelClient

        def get_execution_status(password, data_path):
            client = BaseNeomarilClient(password)
            model = client.create_model('Example notebook Async',
                                'score',
                                data_path+'app.py',
                                data_path+'model.pkl',
                                data_path+'requirements.txt',
                                python_version='3.9',
                                operation="Async",
                                input_type='csv'
                                )

            execution = model.predict(data_path+'input.csv')

            execution.get_status()

            execution.download_result()
    """

    def __init__(
        self,
        *,
        parent_id: str,
        exec_type: str,
        group: Optional[str] = None,
        exec_id: Optional[str] = None,
        login: Optional[str] = None,
        password: Optional[str] = None,
        url: str = None,
        group_token: Optional[str] = None,
    ) -> None:
        super().__init__(login=login, password=password, url=url)
        load_dotenv()
        logger.info("Loading .env")

        self.exec_type = exec_type
        self.exec_id = exec_id
        self.status = ModelExecutionState.Requested
        self.group = group
        self.__token = group_token if group_token else os.getenv("NEOMARIL_GROUP_TOKEN")

        if exec_type == "AsyncModel":
            self.__url_path = "model/async"
        elif exec_type == "Training":
            self.__url_path = "training"
        elif exec_type == "AsyncPreprocessing":
            self.__url_path = "preprocessing/async"
        else:
            raise InputError(
                f"Invalid execution type '{exec_type}'. Valid options are 'AsyncModel' and 'Training'"
            )

        if exec_type == "AsyncPreprocessing":
            # TODO: CHANGEME when add describe execution for preprocessing

            self.execution_data = {}

            self.status = ModelExecutionState.Running

        else:
            url = f"{self.base_url}/{self.__url_path.replace('/async', '')}/describe/{group}/{parent_id}/{exec_id}"
            response = requests.get(
                url,
                headers={
                    "Authorization": "Bearer "
                    + refresh_token(*self.credentials, self.base_url)
                },
            )

            if response.status_code == 401:
                logger.error(
                    "Login or password are invalid, please check your credentials."
                )
                raise AuthenticationError("Login not authorized.")

            if response.status_code == 404:
                logger.error(
                    f'Unable to retrieve execution "{exec_id}"\n{response.text}'
                )
                raise ModelError(f'Execution "{exec_id}" not found.')

            if response.status_code >= 500:
                logger.error("Server is not available. Please, try it later.")
                raise ServerError("Server is not available!")

            self.execution_data = response.json()["Description"]

            self.status = ModelExecutionState[self.execution_data["ExecutionState"]]

    def __repr__(self) -> str:
        return f"""Neomaril{self.exec_type}Execution(exec_id="{self.exec_id}", status="{self.status}")"""

    def __str__(self):
        return f'NEOMARIL {self.exec_type }Execution :{self.exec_id} (Status: {self.status})"'

    def get_status(self) -> dict:
        """
        Gets the status of the related execution.

        Raises
        ------
        ExecutionError
            Execution unavailable

        Returns
        -------
        dict
            Returns the execution status.
        """

        url = f"{self.base_url}/{self.__url_path}/status/{self.group}/{self.exec_id}"

        response = requests.get(
            url, headers={"Authorization": "Bearer " + self.__token}
        )
        if response.status_code not in [200, 410]:
            formatted_msg = parse_json_to_yaml(response.json())
            logger.error(f"Something went wrong...\n{formatted_msg}")
            raise ExecutionError(f'Execution "{self.exec_id}" unavailable')

        result = response.json()

        self.status = ModelExecutionState[result["Status"]]
        self.execution_data["ExecutionState"] = result["Status"]

        return result

    def wait_ready(self) -> None:
        """
        Waits the execution until is no longer running

        Example
        -------
        >>> model.wait_ready()
        """

        self.status = ModelExecutionState[self.get_status()["Status"]]
        while self.status in [
            ModelExecutionState.Requested,
            ModelExecutionState.Running,
        ]:
            sleep(30)
            self.status = ModelExecutionState[self.get_status()["Status"]]
        if self.status == ModelExecutionState.Failed:
            logger.error("Execution failed! Please check the logs")
            raise ExecutionError(
                "Execution failed"
            )  # TODO: how to improve this message?
        logger.info("Execution completed successfully")

    def download_result(
        self, *, path: Optional[str] = "./", filename: Optional[str] = "output.zip"
    ) -> None:
        """
        Gets the output of the execution.

        Arguments
        ---------
        path : str
            Path of the result file. Default value is './'
        filename : str
            Name of the result file. Default value is 'output.zip'

        Raises
        ------
        ExecutionError
            Execution unavailable

        Returns
        -------
        dict
            Returns the path for the result file.
        """
        if self.status in [ModelExecutionState.Running, ModelExecutionState.Requested]:
            self.status = ModelExecutionState[self.get_status()["Status"]]

        if self.exec_type in ["AsyncModel", "AsyncPreprocessing"]:
            token = self.__token
        elif self.exec_type == "Training":
            token = refresh_token(*self.credentials, self.base_url)

        if self.status == ModelExecutionState.Succeeded:
            url = (
                f"{self.base_url}/{self.__url_path}/result/{self.group}/{self.exec_id}"
            )
            response = requests.get(url, headers={"Authorization": "Bearer " + token})
            if response.status_code not in [200, 410]:
                formatted_msg = parse_json_to_yaml(response.json())
                logger.error(f"Something went wrong...\n{formatted_msg}")
                raise ExecutionError(f'Execution "{self.exec_id}" unavailable')

            if not path.endswith("/"):
                filename = "/" + filename

            with open(path + filename, "wb") as f:
                f.write(response.content)

            logger.info(f"Output saved in {path+filename}")
        elif self.status == ModelExecutionState.Failed:
            raise ExecutionError("Execution failed")
        else:
            logger.info(f"Execution not ready. Status is {self.status}")

    def execution_info(self) -> None:
        """Show the execution data in a better format"""
        print(parse_json_to_yaml(self.execution_data))
