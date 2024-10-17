#!/usr/bin/env python
# coding: utf-8

import json
import os
import time
from http import HTTPStatus
from time import sleep
from typing import Optional, Union

import requests

from neomaril_codex.__utils import parse_json_to_yaml, refresh_token
from neomaril_codex.base import BaseNeomaril, BaseNeomarilClient, NeomarilExecution
from neomaril_codex.exceptions import (
    AuthenticationError,
    ExecutionError,
    GroupError,
    InputError,
    PreprocessingError,
    ServerError,
)
from neomaril_codex.logger_config import get_logger

logger = get_logger()


class NeomarilPreprocessing(BaseNeomaril):
    """
    Class to manage Preprocessing scripts deployed inside Neomaril

    Attributes
    ----------
    login : str
        Login for authenticating with the client. You can also use the env variable NEOMARIL_USER to set this
    password : str
        Password for authenticating with the client. You can also use the env variable NEOMARIL_PASSWORD to set this
    preprocessing_id : str
        Preprocessing script id (hash) from the script you want to access
    group : str
        Group the model is inserted. Default is 'datarisk' (public group)
    base_url : str
        URL to Neomaril Server. Default value is https://neomaril.staging.datarisk.net, use it to test your deployment first before changing to production. You can also use the env variable NEOMARIL_URL to set this

    Example
    --------
    Getting a model, testing its healthy and putting it to run the prediction

    .. code-block:: python

        from neomaril_codex.preprocessing import NeomarilPreprocessingClient
        from neomaril_codex.model import NeomarilModelClient

        client = NeomarilPreprocessingClient('123456')

        client.search_preprocessing()

        preprocessing = client.get_preprocessing(preprocessing_id='S72110d87c2a4341a7ef0a0cb35e483699db1df6c5d2450f92573c093c65b062', group='ex_group')

    """

    def __init__(
        self,
        *,
        preprocessing_id: str,
        login: Optional[str] = None,
        password: Optional[str] = None,
        group: str = "datarisk",
        group_token: Optional[str] = None,
        url: str = "https://neomaril.staging.datarisk.net/",
    ) -> None:
        super().__init__(login=login, password=password, url=url)
        self.preprocessing_id = preprocessing_id
        self.group = group
        self.__token = group_token if group_token else os.getenv("NEOMARIL_GROUP_TOKEN")

        url = f"{self.base_url}/preprocessing/describe/{group}/{preprocessing_id}"
        response = requests.get(
            url,
            headers={
                "Authorization": "Bearer "
                + refresh_token(*self.credentials, self.base_url)
            },
        )

        result = response.json()["Description"]
        self.operation = result.get("Operation").lower()

        response = self.__get_status()
        self.status = response.get("Status")

        self.__preprocessing_ready = self.status == "Deployed"

    def __repr__(self) -> str:
        return f"""NeomarilPreprocessing, group="{self.group}", 
                                status="{self.status}",
                                preprocessing_id="{self.preprocessing_id}",
                                operation="{self.operation.title()}",
                                )"""

    def __str__(self):
        return f'NEOMARIL preprocessing (Group: {self.group}, Id: {self.preprocessing_id})"'

    def wait_ready(self):
        """
        Waits the pre processing to be with status 'Deployed'

        Example
        -------
        >>> preprocessing.wait_ready()
        """
        if self.status in ["Ready", "Building"]:
            self.status = self.__get_status()["Status"]
            while self.status == "Building":
                sleep(30)
                self.status = self.__get_status()["Status"]

    def get_logs(
        self,
        *,
        start: Optional[str] = None,
        end: Optional[str] = None,
        routine: Optional[str] = None,
        type: Optional[str] = None,
    ):
        """
        Get the logs

        Parameters
        -----------
        start : str, optional
            Date to start filter. At the format aaaa-mm-dd
        end : str, optional
            Date to end filter. At the format aaaa-mm-dd
        routine : str, optional
            Type of routine beeing executed, can assume values Host or Run
        type : str, optional
            Defines the type of the logs that are going to be filtered, can assume the values Ok, Error, Debug or Warning

        Raises
        ------
        ServerError
            Unexpected server error

        Returns
        -------
        json
            Logs list

        Example
        -------
        >>> preprocessing.get_logs(start='2023-01-31', end='2023-02-24', routine='Run', type='Error')
         {'Results':
            [{'Hash': 'M9c3af308c754ee7b96b2f4a273984414d40a33be90242908f9fc4aa28ba8ec4',
                'RegisteredAt': '2023-01-31T16:06:45.5955220Z',
                'OutputType': 'Error',
                'OutputData': '',
                'Routine': 'Run'}]
         }
        """
        url = f"{self.base_url}/preprocessing/logs/{self.group}/{self.preprocessing_id}"
        return self._logs(
            url=url,
            credentials=self.credentials,
            start=start,
            end=end,
            routine=routine,
            type=type,
        )

    def set_token(self, group_token: str) -> None:
        """
        Saves the group token for this preprocessing instance.

        Arguments
        ---------
        group_token : str
            Token for executing the preprocessing (show when creating a group). You can set this using the NEOMARIL_GROUP_TOKEN env variable

        Example
        -------
        >>> preprocessing.set_token('6cb64889a45a45ea8749881e30c136df')
        """

        self.__token = group_token
        logger.info(f"Token for group {self.group} added.")

    def run(
        self,
        *,
        data: Union[dict, str],
        group_token: Optional[str] = None,
        wait_complete: Optional[bool] = False,
    ) -> Union[dict, NeomarilExecution]:
        """
        Runs a prediction from the current preprocessing.

        Arguments
        ---------
        data : Union[dict, str]
            The same data that is used in the source file.
            If Sync is a dict, the keys that are needed inside this dict are the ones in the `schema` attribute.
            If Async is a string with the file path with the same filename used in the source file.
        group_token : str, optional
            Token for executing the preprocessing (show when creating a group). It can be informed when getting the preprocessing or when running predictions, or using the env variable NEOMARIL_GROUP_TOKEN
        wait_complete: bool, optional
            Boolean that informs if a preprocessing training is completed (True) or not (False). Default value is False

        Raises
        ------
        PreprocessingError
            Pre processing is not available

        Returns
        -------
        Union[dict, NeomarilExecution]
            The return of the scoring function in the source file for Sync preprocessing or the execution class for Async preprocessing.
        """
        if self.__preprocessing_ready:
            if (group_token is not None) | (self.__token is not None):
                url = f"{self.base_url}/preprocessing/{self.operation}/run/{self.group}/{self.preprocessing_id}"
                if self.__token and not group_token:
                    group_token = self.__token
                if group_token and not self.__token:
                    self.__token = group_token
                if self.operation == "sync":
                    preprocessing_input = {"Input": data}

                    req = requests.post(
                        url,
                        data=json.dumps(preprocessing_input),
                        headers={"Authorization": "Bearer " + group_token},
                    )

                    return req.json()

                elif self.operation == "async":
                    files = {
                        "dataset": open(data, "rb"),
                    }

                    req = requests.post(
                        url,
                        files=files,
                        headers={"Authorization": "Bearer " + group_token},
                    )

                    # TODO: Shouldn't both sync and async preprocessing have the same succeeded status code?
                    if req.status_code == 202 or req.status_code == 200:
                        message = req.json()
                        logger.info(message["Message"])
                        exec_id = message["ExecutionId"]
                        run = NeomarilExecution(
                            parent_id=self.preprocessing_id,
                            exec_type="AsyncPreprocessing",
                            exec_id=exec_id,
                            login=self.credentials[0],
                            password=self.credentials[1],
                            url=self.base_url,
                            group=self.group,
                            group_token=group_token,
                        )
                        response = run.get_status()
                        status = response["Status"]
                        if wait_complete:
                            print("Waiting the training run.", end="")
                            while status in ["Running", "Requested"]:
                                sleep(30)
                                print(".", end="", flush=True)
                                response = run.get_status()
                                status = response["Status"]
                        if status == "Failed":
                            formatted_msg = parse_json_to_yaml(response.json())
                            logger.error(f"Something went wrong...\n{formatted_msg}")
                            raise ExecutionError("Training execution failed")
                    else:
                        raise ServerError(req.text)

            else:
                logger.error("Login or password are invalid, please check your credentials.")
                raise GroupError("Group token not informed.")

        return run

    def get_preprocessing_execution(self, exec_id: str) -> NeomarilExecution:
        """
        Get an execution instance for that preprocessing.

        Arguments
        ---------
        exec_id: str
            Execution id

        Raises
        ------
        PreprocessingError
            If the user tries to get an execution from a Sync preprocessing

        Example
        -------
        >>> preprocessing.get_preprocessing_execution('1')
        """
        if self.operation == "async":
            return NeomarilExecution(
                parent_id=self.preprocessing_id,
                exec_type="AsyncPreprocessing",
                exec_id=exec_id,
                login=self.credentials[0],
                password=self.credentials[1],
                url=self.base_url,
                group_token=self.__token,
                group=self.group,
            )
        raise PreprocessingError("Sync pre processing don't have executions")

    def __get_status(self):
        """
        Gets the status of the preprocessing.

        Raises
        -------
        PreprocessingError
            Execution unavailable

        Returns
        -------
        str
            The preprocessing status

        """
        url = (
            f"{self.base_url}/preprocessing/status/{self.group}/{self.preprocessing_id}"
        )
        response = requests.get(
            url,
            headers={
                "Authorization": "Bearer "
                + refresh_token(*self.credentials, self.base_url)
            },
        )
        if response.status_code == 200:
            return response.json()

        formatted_msg = parse_json_to_yaml(response.json())
        logger.error(f"Something went wrong...\n{formatted_msg}")
        raise PreprocessingError("Preprocessing has failed")


class NeomarilPreprocessingClient(BaseNeomarilClient):
    """
    Class for client to access Neomaril and manage Preprocessing scripts

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
    AuthenticationError
        Invalid credentials
    ServerError
        Server unavailable

    Example
    --------
    Example 1: Creation and managing a Synchronous Preprocess script

    .. code-block:: python

        from neomaril_codex.preprocessing import NeomarilPreprocessingClient
        from neomaril_codex.model import NeomarilModelClient

        client = NeomarilPreprocessingClient('123456')
        PATH = './samples/syncPreprocessing/'

        sync_preprocessing = client.create('Teste preprocessing Sync', # model_name
                            'process', # name of the scoring function
                            PATH+'app.py', # Path of the source file
                            PATH+'requirements.txt', # Path of the requirements file,
                            schema=PATH+'schema.json', # Path of the schema file, but it could be a dict (only required for Sync models)
                            # env=PATH+'.env'  #  File for env variables (this will be encrypted in the server)
                            # extra_files=[PATH+'utils.py'], # List with extra files paths that should be uploaded along (they will be all in the same folder)
                            python_version='3.9', # Can be 3.8 to 3.10
                            operation="Sync", # Can be Sync or Async
                            group='datarisk' # Model group (create one using the client)
                            )

        sync_preprocessing.set_token('TOKEN')

        result = sync_preprocessing.run({'variable' : 100})
        result

    Example 2: creation and deployment of an Asynchronous Preprocess script

    .. code-block:: python

        from neomaril_codex.preprocessing import NeomarilPreprocessingClient
        from neomaril_codex.model import NeomarilModelClient

        client = NeomarilPreprocessingClient('123456')
        PATH = './samples/asyncPreprocessing/'

        async_preprocessing = client.create('Teste preprocessing Async', # model_name
                            'process', # name of the scoring function
                            PATH+'app.py', # Path of the source file
                            PATH+'requirements.txt', # Path of the requirements file,
                            # env=PATH+'.env',  #  File for env variables (this will be encrypted in the server)
                            # extra_files=[PATH+'input.csv'], # List with extra files paths that should be uploaded along (they will be all in the same folder)
                            python_version='3.9', # Can be 3.8 to 3.10
                            operation="Async", # Can be Sync or Async
                            group='datarisk', # Model group (create one using the client)
                            input_type='csv'
                            )

        async_preprocessing.set_token('TOKEN')

        execution = async_preprocessing.run(PATH+'input.csv')

        execution.get_status()

        execution.wait_ready()

        execution.download_result()

    Example 3: Using preprocessing with a Synchronous model

    .. code-block:: python

        from neomaril_codex.preprocessing import NeomarilPreprocessingClient
        from neomaril_codex.model import NeomarilModelClient

        # the sync preprocess script configuration presented before
        # ...

        model_client = NeomarilModelClient('123456')

        sync_model = model_client.get_model(group='datarisk', model_id='M3aa182ff161478a97f4d3b2dc0e9b064d5a9e7330174daeb302e01586b9654c')

        sync_model.predict(data=sync_model.schema, preprocessing=sync_preprocessing)

    Example 4: Using preprocessing with an Asynchronous model

    .. code-block:: python

        from neomaril_codex.preprocessing import NeomarilPreprocessingClient
        from neomaril_codex.model import NeomarilModelClient

        # the async preprocess script configuration presented before
        # ...

        async_model = model_client.get_model(group='datarisk', model_id='Maa3449c7f474567b6556614a12039d8bfdad0117fec47b2a4e03fcca90b7e7c')

        PATH = './samples/asyncModel/'

        execution = async_model.predict(PATH+'input.csv', preprocessing=async_preprocessing)
        execution.wait_ready()

        execution.download_result()
    """

    def __get_preprocessing_status(self, *, preprocessing_id: str, group: str) -> dict:
        """
        Gets the status of the preprocessing with the hash equal to `preprocessing_id`

        Parameters
        ----------
        group : str
            Group the preprocessing is inserted
        preprocessing_id : str
            Pre processing id (hash) from the preprocessing being searched

        Raises
        ------
        PreprocessingError
            Pre processing unavailable

        Returns
        -------
        dict
            The preprocessing status and a message if the status is 'Failed'
        """

        url = f"{self.base_url}/preprocessing/status/{group}/{preprocessing_id}"
        response = requests.get(
            url=url,
            headers={
                "Authorization": "Bearer "
                + refresh_token(*self.credentials, self.base_url)
            },
            timeout=60,
        )

        if response.status_code not in [200, 410]:
            formatted_msg = parse_json_to_yaml(response.json())
            logger.error(f"Something went wrong...\n{formatted_msg}")
            raise PreprocessingError(f'Preprocessing "{preprocessing_id}" not found')

        return response.json()

    def get_preprocessing(
        self,
        *,
        preprocessing_id: str,
        group: str = "datarisk",
        group_token: Optional[str] = None,
        wait_for_ready: bool = True,
    ) -> NeomarilPreprocessing:
        """
        Access a preprocessing using its id

        Arguments
        ---------
        preprocessing_id : str
            Pre processing id (hash) that needs to be accessed
        group : str
            Group the preprocessing is inserted. Default is 'datarisk' (public group)
        group_token : str, optional
            Token for executing the preprocessing (show when creating a group). It can be informed when getting the preprocessing or when running predictions, or using the env variable NEOMARIL_GROUP_TOKEN
        wait_for_ready : bool
            If the preprocessing is being deployed, wait for it to be ready instead of failing the request. Defaults to True

        Raises
        ------
        PreprocessingError
            Pre processing unavailable
        ServerError
            Unknown return from server

        Returns
        -------
        NeomarilPreprocessing
            A NeomarilPreprocessing instance with the preprocessing hash from `preprocessing_id`

        Example
        -------
        >>> preprocessing.get_preprocessing(preprocessing_id='M9c3af308c754ee7b96b2f4a273984414d40a33be90242908f9fc4aa28ba8ec4', group='ex_group')
        """
        try:
            response = self.__get_preprocessing_status(
                preprocessing_id=preprocessing_id, group=group
            )
        except KeyError:
            raise PreprocessingError("Preprocessing not found")

        status = response["Status"]

        if status == "Building":
            if wait_for_ready:
                print("Waiting for deploy to be ready.", end="")
                while status == "Building":
                    response = self.__get_preprocessing_status(
                        preprocessing_id=preprocessing_id, group=group
                    )
                    status = response["Status"]
                    print(".", end="", flush=True)
                    sleep(10)
            else:
                logger.info("Returning preprocessing, but preprocessing is not ready.")
                NeomarilPreprocessing(
                    preprocessing_id=preprocessing_id,
                    login=self.credentials[0],
                    password=self.credentials[1],
                    group=group,
                    url=self.base_url,
                    group_token=group_token,
                )

        if status in ["Disabled", "Ready"]:
            raise PreprocessingError(
                f'Preprocessing "{preprocessing_id}" unavailable (disabled or deploy process is incomplete)'
            )
        elif status == "Failed":
            logger.error(str(response["Message"]))
            raise PreprocessingError(
                f'Preprocessing "{preprocessing_id}" deploy failed, so preprocessing is unavailable.'
            )
        elif status == "Deployed":
            logger.info(
                f"Preprocessing {preprocessing_id} its deployed. Fetching preprocessing."
            )
            return NeomarilPreprocessing(
                preprocessing_id=preprocessing_id,
                login=self.credentials[0],
                password=self.credentials[1],
                group=group,
                url=self.base_url,
                group_token=group_token,
            )
        else:
            raise ServerError("Unknown preprocessing status: ", status)

    def search_preprocessing(
        self,
        *,
        name: Optional[str] = None,
        state: Optional[str] = None,
        group: Optional[str] = None,
        only_deployed: bool = False,
    ) -> list:
        """
        Search for preprocessing using the name of the preprocessing

        Arguments
        ---------
        name : str, optional
            Text that it's expected to be on the preprocessing name. It runs similar to a LIKE query on SQL
        state : str, optional
            Text that it's expected to be on the state. It runs similar to a LIKE query on SQL
        group : str, optional
            Text that it's expected to be on the group name. It runs similar to a LIKE query on SQL
        only_deployed : bool, optional
            If it's True, filter only preprocessing ready to be used (status == "Deployed"). Defaults to False

        Raises
        ------
        ServerError
            Unexpected server error

        Returns
        -------
        list
            List with the preprocessing data, it can works like a filter depending on the arguments values
        Example
        -------
        >>> client.search_preprocessing(group='ex_group', only_deployed=True)
        """
        url = f"{self.base_url}/preprocessing/search"

        query = {}

        if name:
            query["name"] = name

        if state:
            query["state"] = state

        if group:
            query["group"] = group

        if only_deployed:
            query["state"] = "Deployed"

        response = requests.get(
            url,
            params=query,
            headers={
                "Authorization": "Bearer "
                + refresh_token(*self.credentials, self.base_url)
            },
        )

        if response.status_code == 200:
            results = response.json()["Results"]
            return results

        formatted_msg = parse_json_to_yaml(response.json())

        if response.status_code == 401:
            logger.error("Login or password are invalid, please check your credentials.")
            raise AuthenticationError("Login not authorized.")

        if response.status_code >= 500:
            logger.error("Server is not available. Please, try it later.")
            raise ServerError("Server is not available!")

        logger.error(f"Something went wrong...\n{formatted_msg}")
        raise PreprocessingError("Could not search the preprocessing script")

    def get_logs(
        self,
        *,
        preprocessing_id,
        start: Optional[str] = None,
        end: Optional[str] = None,
        routine: Optional[str] = None,
        type: Optional[str] = None,
    ):
        """
        Get the logs

        Parameters
        ----------
        preprocessing_id : str
            Pre processing id (hash)
        start : str, optional
            Date to start filter. At the format aaaa-mm-dd
        end : str, optional
            Date to end filter. At the format aaaa-mm-dd
        routine : str, optional
            Type of routine being executed, can assume values 'Host' (for deployment logs) or 'Run' (for execution logs)
        type : str, optional
            Defines the type of the logs that are going to be filtered, can assume the values 'Ok', 'Error', 'Debug' or 'Warning'

        Raises
        ------
        ServerError
            Unexpected server error

        Returns
        -------
        json
            Logs list

        Example
        -------
        >>> preprocessing.get_logs(routine='Run')
         {'Results':
            [{'Hash': 'B4c3af308c3e452e7b96b2f4a273984414d40a33be90242908f9fc4aa28ba8ec4',
                'RegisteredAt': '2023-02-03T16:06:45.5955220Z',
                'OutputType': 'Ok',
                'OutputData': '',
                'Routine': 'Run'}]
         }
        """
        url = f"{self.base_url}/preprocessing/logs/{preprocessing_id}"
        return self._logs(
            url=url,
            credentials=self.credentials,
            start=start,
            end=end,
            routine=routine,
            type=type,
        )

    def __upload_preprocessing(
        self,
        *,
        preprocessing_name: str,
        preprocessing_reference: str,
        source_file: str,
        requirements_file: str,
        schema: Optional[Union[str, dict]] = None,
        group: Optional[str],
        extra_files: Optional[list] = None,
        env: Optional[str] = None,
        python_version: str = "3.8",
        operation: str = "Sync",
        input_type: str = None,
    ) -> str:
        """
        Upload the files to the server

        Arguments
        ---------
        preprocessing_name : str
            The name of the preprocessing, in less than 32 characters
        preprocessing_reference : str
            The name of the scoring function inside the source file
        source_file : str
            Path of the source file. The file must have a scoring function that accepts two parameters: data (data for the request body of the preprocessing) and preprocessing_path (absolute path of where the file is located)
        requirements_file : str
            Path of the requirements file. The packages versions must be fixed eg: pandas==1.0
        schema : Union[str, dict], optional
            Path to a JSON or XML file with a sample of the input for the entrypoint function. A dict with the sample input can be sending as well
        group : str, optional
            Group the preprocessing is inserted. If None the server uses 'datarisk' (public group)
        extra_files : list, optional
            A optional list with additional files paths that should be uploaded. If the scoring function refer to this file they will be on the same folder as the source file
        env : str, optional
            Flag that choose which environment (dev, staging, production) of Neomaril you are using. Default is True
        python_version : str, optional
            Python version for the preprocessing environment. Available versions are 3.8, 3.9, 3.10. Defaults to '3.8'
        operation : str
            Defines which kind operation is being executed (Sync or Async). Default value is Sync
        input_type : str
            The type of the input file that should be 'json', 'csv' or 'parquet'

        Raises
        ------
        InputError
            Some input parameters its invalid

        Returns
        -------
        str
            The new preprocessing id (hash)
        """
        url = f"{self.base_url}/preprocessing/register/{group}"

        file_extesions = {"py": "script.py", "ipynb": "notebook.ipynb"}

        upload_data = [
            (
                "source",
                (file_extesions[source_file.split(".")[-1]], open(source_file, "rb")),
            ),
            ("requirements", ("requirements.txt", open(requirements_file, "rb"))),
        ]

        if operation == "Sync":
            input_type = "json"
        if schema:
            if isinstance(schema, str):
                schema_file = open(schema, "rb")
            elif isinstance(schema, dict):
                schema_file = json.dumps(schema)
            upload_data.append(("schema", (schema.split("/")[-1], schema_file)))
        else:
            raise InputError(
                "Schema file is mandatory for preprocessing, choose a input type from json, parquet or csv"
            )

        if env:
            upload_data.append(("env", (".env", open(env, "r"))))

        if extra_files:
            extra_data = [
                ("extra", (c.split("/")[-1], open(c, "rb"))) for c in extra_files
            ]

            upload_data += extra_data

        form_data = {
            "name": preprocessing_name,
            "script_reference": preprocessing_reference,
            "operation": operation,
            "python_version": "Python" + python_version.replace(".", ""),
        }

        response = requests.post(
            url,
            data=form_data,
            files=upload_data,
            headers={
                "Authorization": "Bearer "
                + refresh_token(*self.credentials, self.base_url)
            },
        )

        if response.status_code == 201:
            data = response.json()
            preprocessing_id = data["Hash"]
            logger.info(
                f'{data["Message"]} - Hash: "{preprocessing_id}" with response {response.text}'
            )
            return preprocessing_id

        formatted_msg = parse_json_to_yaml(response.json())

        if response.status_code == 401:
            logger.error("Login or password are invalid, please check your credentials.")
            raise AuthenticationError("Login not authorized.")

        if response.status_code >= 500:
            logger.error("Server is not available. Please, try it later.")
            raise ServerError("Server is not available!")

        logger.error(f"Something went wrong...\n{formatted_msg}")
        raise InputError("Invalid parameters for preprocessing creation")

    def __host_preprocessing(
        self, *, operation: str, preprocessing_id: str, group: str
    ) -> None:
        """
        Builds the preprocessing execution environment

        Arguments
        ---------
        operation : str
            The preprocessing operation type (Sync or Async)
        preprocessing_id : str
            The uploaded preprocessing id (hash)
        group : str
            Group the preprocessing is inserted. Default is 'datarisk' (public group)

        Raises
        ------
        InputError
            Some input parameters its invalid
        """

        url = (
            f"{self.base_url}/preprocessing/{operation}/host/{group}/{preprocessing_id}"
        )

        response = requests.get(
            url,
            headers={
                "Authorization": "Bearer "
                + refresh_token(*self.credentials, self.base_url)
            },
        )

        if response.status_code == 202:
            logger.info(f"Preprocessing host in process - Hash: {preprocessing_id}")
            return HTTPStatus.OK

        formatted_msg = parse_json_to_yaml(response.json())

        if response.status_code == 401:
            logger.error("Login or password are invalid, please check your credentials.")
            raise AuthenticationError("Login not authorized.")

        if response.status_code >= 500:
            logger.error("Server is not available. Please, try it later.")
            raise ServerError("Server is not available!")

        logger.error(f"Something went wrong...\n{formatted_msg}")
        raise InputError("Invalid parameters for preprocessing creation")

    def create(
        self,
        *,
        preprocessing_name: str,
        preprocessing_reference: str,
        source_file: str,
        requirements_file: str,
        schema: Optional[Union[str, dict]] = None,
        group: str,
        extra_files: Optional[list] = None,
        env: Optional[str] = None,
        python_version: str = "3.8",
        operation="Sync",
        input_type: str = "json|csv|parquet",
        wait_for_ready: bool = True,
    ) -> Union[NeomarilPreprocessing, str]:
        """
        Deploy a new preprocessing to Neomaril.

        Arguments
        ---------
        preprocessing_name : str
            The name of the preprocessing, in less than 32 characters
        preprocessing_reference : str
            The name of the scoring function inside the source file
        source_file : str
            Path of the source file. The file must have a scoring function that accepts two parameters: data (data for the request body of the preprocessing) and preprocessing_path (absolute path of where the file is located)
        requirements_file : str
            Path of the requirements file. The packages versions must be fixed eg: pandas==1.0
        schema : Union[str, dict]
            Path to a JSON or XML file with a sample of the input for the entrypoint function. A dict with the sample input can be sending as well. Mandatory for Sync preprocessing
        group : str
            Group the preprocessing is inserted. Default to 'datarisk' (public group)
        extra_files : list, optional
            A optional list with additional files paths that should be uploaded. If the scoring function refer to this file they will be on the same folder as the source file
        env : str, optional
            Flag that choose which environment (dev, staging, production) of Neomaril you are using. Default is True
        python_version : str, optional
            Python version for the preprocessing environment. Avaliable versions are 3.8, 3.9, 3.10. Defaults to '3.8'
        operation : str
            Defines wich kind operation is beeing executed (Sync or Async). Default value is Sync
        input_type : str
            The type of the input file that should be 'json', 'csv' or 'parquet'
        wait_for_ready : bool, optional
            Wait for preprocessing to be ready and returns a NeomarilPreprocessing instace with the new preprocessing. Defaults to True

        Raises
        ------
        InputError
            Some input parameters its invalid

        Returns
        -------
        Union[NeomarilPreprocessing, str]
            Returns the new preprocessing, if wait_for_ready=True runs the deployment process synchronously. If it's False, returns nothing after sending all the data to server and runs the deployment asynchronously

        Example
        -------
        >>> preprocessing = client.create('Pre processing Example Sync', 'score',  './samples/syncPreprocessing/app.py', './samples/syncPreprocessing/'preprocessing.pkl', './samples/syncPreprocessing/requirements.txt','./samples/syncPreprocessing/schema.json', group=group, operation="Sync")
        """

        if python_version not in ["3.8", "3.9", "3.10"]:
            raise InputError(
                "Invalid python version. Avaliable versions are 3.8, 3.9, 3.10"
            )

        if group:
            group = (
                group.lower()
                .strip()
                .replace(" ", "_")
                .replace(".", "_")
                .replace("-", "_")
            )

            groups = groups = [g.get("Name") for g in self.list_groups()]

            if group not in groups:
                raise GroupError("Group dont exist. Create a group first.")

        else:
            group = "datarisk"
            logger.info("Group not informed, using default 'datarisk' group")

        preprocessing_id = self.__upload_preprocessing(
            preprocessing_name=preprocessing_name,
            preprocessing_reference=preprocessing_reference,
            source_file=source_file,
            requirements_file=requirements_file,
            schema=schema,
            group=group,
            extra_files=extra_files,
            python_version=python_version,
            env=env,
            operation=operation,
            input_type=input_type,
        )

        self.__host_preprocessing(
            operation=operation.lower(), preprocessing_id=preprocessing_id, group=group
        )
        time.sleep(1)
        return self.get_preprocessing(
            preprocessing_id=preprocessing_id,
            group=group,
            wait_for_ready=wait_for_ready,
        )

    def get_execution(
        self, preprocessing_id: str, exec_id: str, group: Optional[str] = None
    ) -> NeomarilExecution:
        """
        Get a execution instace (Async preprocessing only).

        Arguments
        ---------
        preprocessing_id : str
            Pre processing id (hash)
        exec_id : str
            Execution id
        group : str, optional
            Group name, default value is None

        Returns
        -------
        NeomarilExecution
            The new execution

        Example
        -------
        >>> preprocessing.get_execution( preprocessing_id='M9c3af308c754ee7b96b2f4a273984414d40a33be90242908f9fc4aa28ba8ec4', exec_id = '1')
        """
        return self.get_preprocessing(
            preprocessing_id=preprocessing_id, group=group
        ).get_preprocessing_execution(exec_id)
