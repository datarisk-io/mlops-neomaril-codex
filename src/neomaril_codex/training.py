#!/usr/bin/env python
# coding: utf-8

import json
import os
import re
import sys
from contextlib import contextmanager
from http import HTTPStatus
from time import sleep
from typing import Any, List, Optional, Union

import cloudpickle
import numpy as np
import pandas as pd
import requests
from lazy_imports import try_import

from neomaril_codex.__utils import parse_dict_or_file, parse_json_to_yaml, refresh_token
from neomaril_codex.base import BaseNeomaril, BaseNeomarilClient, NeomarilExecution
from neomaril_codex.datasources import NeomarilDataset
from neomaril_codex.exceptions import (
    AuthenticationError,
    ExecutionError,
    GroupError,
    InputError,
    ModelError,
    ServerError,
    TrainingError,
)
from neomaril_codex.logger_config import get_logger
from neomaril_codex.model import NeomarilModel

patt = re.compile(r"(\d+)")
logger = get_logger()


class NeomarilTrainingLogger:
    """A class for logging Neomaril training runs.

    Example
    -------

    .. code-block:: python
        with training.log_train('Teste 1', X, y) as logger:
            pipe.fit(X, y)
            logger.save_model(pipe)

            params = pipe.get_params()
            params.pop('steps')
            params.pop('simpleimputer')
            params.pop('lgbmclassifier')
            logger.save_params(params)

            model_output = pd.DataFrame({"pred": pipe.predict(X), "proba": pipe.predict_proba(X)[:,1]})
            logger.save_model_output(model_output)

            auc = cross_val_score(pipe, X, y, cv=5, scoring="roc_auc")
            f_score = cross_val_score(pipe, X, y, cv=5, scoring="f1")
            logger.save_metric(name='auc', value=auc.mean())
            logger.save_metric(name='f1_score', value=f_score.mean())

            logger.set_python_version('3.10')
    """

    def __init__(
        self,
        *,
        name: str,
        X_train: pd.DataFrame,
        y_train: pd.DataFrame,
        description: Optional[str] = None,
        save_path: Optional[str] = None,
    ):
        """
        Initialize a new NeomarilTrainingLogger.

        Args:
            name: The name of the training run.
            X_train: The training data.
            y_train: The training labels.
        """
        self.name = name
        self.X_train = X_train
        self.y_train = y_train
        self.description = description
        self.model_outputs = None
        self.model = None
        self.metrics = {}
        self.params = {}
        self.requirements = None
        self.python_version = f"{sys.version_info.major}.{sys.version_info.minor}"
        self.extras = []

        if not save_path:
            dir_name = self.name.replace(" ", "_")
            if not os.path.exists(f"./{dir_name}"):
                os.mkdir(f"./{dir_name}")
            self.save_path = f"./{dir_name}"

    def save_model(self, model):
        """
        Save the trained model to the logger.

        Args:
            model: The trained model.
        """
        self.model = model

    def save_metric(self, *, name, value):
        """
        Save a metric to the logger.

        Args:
            name: The name of the metric.
            value: The value of the metric.
        """
        self.metrics[name] = value

    def save_model_output(self, model_output):
        """
        Save the model output to the logger.

        Args:
            model_output: The output of the trained model.
        """
        self.model_outputs = model_output

    def set_python_version(self, version: str):
        """
        Set the Python version used to train the model.

        Args:
            version: The Python version.
        """
        self.python_version = version

    def set_requirements(self, requirements: str):
        """
        Set the project requirements.

        Args:
            requirements: The path of project requirements.
        """
        self.requirements = requirements

    def save_plot(self, *, plot: object, save_filename: str):
        """
        Save plot graphic image to the logger.

        Args:
            plot: A Matplotlib/Plotly/Seaborn graphic object.
            save_filename: A name to save the plot.
        """
        filepath = f"./{save_filename}.png"

        with try_import() as plotly_import:
            import plotly

            if isinstance(plot, plotly.graph_objs.Figure):
                self.save_plotly_plot(plot=plot, filepath=filepath)
                return

        with try_import() as seaborn_import:
            import seaborn as sns

            if isinstance(plot, sns.axisgrid.FacetGrid):
                self.save_seaborn_or_matplotlib_plot(plot=plot, filepath=filepath)
                return

        with try_import() as matplotlib_import:
            import matplotlib.pyplot as plt

            if isinstance(plot, plt.Figure):
                self.save_seaborn_or_matplotlib_plot(plot=plot, filepath=filepath)
                return

        raise ValueError("The plot only accepts plots of Matplotlib/Plotly/Seaborn")

    def save_plotly_plot(self, *, plot, filepath):
        image_data = plot.to_image()
        with open(filepath, "wb") as f:
            f.write(image_data)
        self.add_extra(extra=filepath)

    def save_seaborn_or_matplotlib_plot(self, *, plot, filepath):
        plot.savefig(filepath)
        self.add_extra(extra=filepath)

    def set_extra(self, extra: list):
        """
        Set the extra files list.

        Args:
            extra: A list of paths of the extra file.
        """
        self.extras = extra

    def add_extra(self, *, extra: Union[pd.DataFrame, str], filename: str = None):
        """
        Add an extra file in the extra file list.

        Args:
            extra: A path of an extra file or a list to include in extra file list.
            filename: A filename if the extra it's a DataFrame.
        """
        if isinstance(extra, str):
            if os.path.exists(extra):
                self.extras.append(extra)
            else:
                raise FileNotFoundError("Extra file path not found!")
        elif isinstance(extra, pd.DataFrame):
            if filename:
                self.extras.append(
                    self.__to_parquet(output_filename=filename, input_data=extra)
                )
            else:
                raise InputError("Needs a filename to save the dataframe parquet.")

    def add_requirements(self, filename: str):
        """
        Add requirements file.

        Args:
            filename: The name of output filename to save.
        """
        self.requirements = filename

    def __to_parquet(self, *, output_filename: str, input_data: pd.DataFrame):
        """
        Transform dataframe to parquet.

        Args:
            output_filename: The name of output filename to save.
            input_data: A pandas dataframe to save.
        """
        path = os.path.join(self.save_path, f"{output_filename}.parquet")
        input_data.to_parquet(path)
        return path

    def __to_json(self, output_filename: str, input_data: dict):
        """
        Transform dict to json.

        Args:
            output_filename: The name of output filename to save.
            input_data: A dictionary to save.
        """
        path = os.path.join(self.save_path, f"{output_filename}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(input_data, f)
        return path

    def __to_pickle(self, *, output_filename: str, input_data):
        """
        Transform content to pickle.

        Args:
            output_filename: The name of output filename to save.
            input_data: The content to save.
        """
        path = os.path.join(self.save_path, f"{output_filename}.pkl")
        with open(path, "wb") as f:
            cloudpickle.dump(input_data, f)
        return path

    def _set_params(self):
        missing = self.X_train.isna().sum()
        missing_dict = {
            k + "_missings": v
            for k, v in missing[missing > 0].describe().to_dict().items()
            if k != "count"
        }

        params = {
            "shape": self.X_train.shape,
            "cols_with_missing": len(missing[missing > 0]),
            "missing_distribution": missing_dict,
        }

        try:
            params["pipeline_steps"] = list(self.model.named_steps.keys())
        except Exception:
            params["pipeline_steps"] = [
                str(self.model.__class__).replace("<class '", "").replace("'>", "")
            ]

        if "get_all_params" in dir(self.model):
            hyperparameters = {
                f"hyperparam_{k}": str(v)
                for k, v in self.model.get_all_params().items()
                if k != "task_type"
            }
        elif "get_params" in dir(self.model):
            hyperparameters = {
                "hyperparam_" + k: str(v)
                for k, v in self.model.get_params().items()
                if k not in params["pipeline_steps"] + ["steps", "memory", "verbose"]
            }

            params = {**params, **hyperparameters}

        if len(self.y_train.value_counts()) < 10:
            target_proportion = self.y_train.value_counts() / len(self.y_train)
            target_proportion = target_proportion.to_dict()
            target_proportion = [
                {"target": k, "proportion": v} for k, v in target_proportion.items()
            ]
            params["target_proportion"] = target_proportion
        else:
            params["target_distribution"] = {
                k: v
                for k, v in self.y_train.describe().to_dict().items()
                if k != "count"
            }

        self.params = {**params, **self.params}

    @staticmethod
    def _parse_data_objects(obj: Any) -> pd.DataFrame:
        """
        Tranform data types to dataframe
        """

        if isinstance(obj, pd.Series):
            return obj.to_frame()
        elif isinstance(obj, np.ndarray):
            array_df = pd.DataFrame(obj)
            array_df.columns = [str(c) for c in array_df.columns]
            return array_df
        elif isinstance(obj, pd.DataFrame):
            return obj

    def _processing_logging_inputs(self):
        """
        Processing of everything that be logged.
        """

        self._set_params()
        self.params = self.__to_json("params", self.params)

        self.X_train = self.__to_parquet(
            output_filename="features",
            input_data=self._parse_data_objects(self.X_train),
        )

        self.y_train = self.__to_parquet(
            output_filename="target", input_data=self._parse_data_objects(self.y_train)
        )

        self.model_outputs = self.__to_parquet(
            output_filename="predictions",
            input_data=self._parse_data_objects(self.model_outputs),
        )

        if self.model:
            self.model = self.__to_pickle(
                output_filename="model", input_data=self.model
            )

        if self.metrics:
            self.metrics = self.__to_json("metrics", self.metrics)


class NeomarilTrainingExecution(NeomarilExecution):
    """
    Class to manage trained models.

    Attributes
    ----------
    training_id : str
        Training id (hash) from the experiment you want to access
    group : str
        Group the training is inserted. Default is 'datarisk' (public group)
    exec_id : str
        Executiong id for that especific training run
    login : str
        Login for authenticating with the client. You can also use the env variable NEOMARIL_USER to set this
    password : str
        Password for authenticating with the client. You can also use the env variable NEOMARIL_PASSWORD to set this
    environment : str
        Enviroment of Neomaril you are using.
    run_data : dict
        Metadata from the execution.

    Raises
    ------
    TrainingError
        When the training can't be acessed in the server
    AuthenticationError
        Unvalid credentials

    Example
    -------

    .. code-block:: python

        from neomaril_codex.training import NeomarilTrainingClient
        from neomaril_codex.base import NeomarilExecution

        client = NeomarilTrainingClient('123456')
        client.create_group('ex_group', 'Group for example purpose')
        training = client.create_training_experiment('Training example', 'Classification', 'ex_group')
        print(client.get_training(training.training_id, 'ex_group').training_data)

        data_path = './samples/train/'

        run = training.run_training('First test', data_path+'dados.csv', training_reference='train_model', training_type='Custom', python_version='3.9', requirements_file=data_path+'requirements.txt', wait_complete=True)

        print(run.get_training_execution(run.exec_id))
        print(run.download_result())

        run.promote_model('Teste notebook promoted custom', 'score', data_path+'app.py', data_path+'schema.json',  'csv')
    """

    def __init__(
        self,
        *,
        training_id: str,
        group: str,
        exec_id: str,
        login: Optional[str] = None,
        password: Optional[str] = None,
        url: str = None,
    ) -> None:
        super().__init__(
            parent_id=training_id,
            exec_type="Training",
            exec_id=exec_id,
            login=login,
            password=password,
            url=url,
            group=group,
        )

        self.training_id = training_id
        self.group = group

        self.training_type = self.execution_data["TrainingType"]
        self.name = self.execution_data["RunName"]
        self.run_data = self.execution_data["RunData"]

    def __repr__(self) -> str:
        return f"""Neomaril{self.exec_type}Execution(name="{self.name}",
                                        exec_id="{self.exec_id}", status="{self.status}")"""

    def __upload_model(
        self,
        *,
        model_name: str,
        model_reference: Optional[str] = None,
        source_file: Optional[str] = None,
        schema: Optional[Union[str, dict]] = None,
        extra_files: Optional[list] = None,
        env: Optional[str] = None,
        requirements_file: Optional[str] = None,
        operation: str = "Sync",
        input_type: str = None,
    ) -> str:
        """
        Upload the files to the server

        Arguments
        ---------
        model_name : str
            The name of the model, in less than 32 characters
        model_reference : str, optional
            The name of the scoring function inside the source file
        source_file : str, optional
            Path of the source file. The file must have a scoring function that accepts two parameters: data (data for the request body of the model) and model_path (absolute path of where the file is located)
        schema : Union[str, dict], optional
            Path to a JSON or XML file with a sample of the input for the entrypoint function. A dict with the sample input can be sending as well
        extra_files list, optional
            A optional list with additional files paths that should be uploaded. If the scoring function refer to this file they will be on the same folder as the source file
        requirements_file : str, optional
            Path of the requirements file. This will override the requirements used in training. The packages versions must be fixed eg: pandas==1.0
        env : str, optional
            Flag that choose which environment (dev, staging, production) of Neomaril you are using. Default is None
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
            The new model id (hash)
        """

        url = f"{self.base_url}/training/promote/{self.group}/{self.training_id}/{self.exec_id}"

        form_data = {"name": model_name, "operation": operation, "schema": schema}
        upload_data = []

        if self.training_type != "AutoML":
            form_data["model_reference"] = model_reference
            upload_data = [("source", ("app.py", open(source_file, "rb")))]

            if env:
                upload_data.append(("env", (".env", open(env, "rb"))))
            if requirements_file:
                upload_data.append(
                    (
                        "requirements",
                        ("requirements.txt", open(requirements_file, "rb")),
                    )
                )
            if extra_files:
                extra_data = [
                    ("extra", (c.split("/")[-1], open(c, "rb"))) for c in extra_files
                ]

                upload_data += extra_data

        else:
            input_type = "AutoML"

        schema_extesion = ".json"
        input_type = "json"
        if operation == "Async":
            schema_extesion = schema.split(".")[-1]

            if input_type == "json|csv|parquet":
                raise InputError("Choose a input type from " + input_type)

        upload_data += [
            ("schema", ("schema." + schema_extesion, parse_dict_or_file(schema)))
        ]

        form_data["input_type"] = input_type

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
            model_id = data["ModelHash"]
            logger.info(f'{data["Message"]} - Hash: "{model_id}"')
            return model_id
        else:
            formatted_msg = parse_json_to_yaml(response.json())
            logger.error(f"Upload error:\n{formatted_msg}")
            raise InputError("Invalid parameters for model creation")

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

        url = f"{self.base_url}/training/status/{self.group}/{self.exec_id}"

        response = requests.get(
            url,
            headers={
                "Authorization": "Bearer "
                + refresh_token(*self.credentials, self.base_url)
            },
        )
        if response.status_code not in [200, 410]:
            formatted_msg = parse_json_to_yaml(response.json())
            logger.error(f"Something went wrong...\n{formatted_msg}")
            raise ExecutionError(f'Execution "{self.exec_id}" unavailable')

        result = response.json()

        self.status = result["Status"]
        self.execution_data["ExecutionState"] = result["Status"]
        if self.status == "Succeeded":
            url = f"{self.base_url}/training/describe/{self.group}/{self.training_id}/{self.exec_id}"
            response = requests.get(
                url,
                headers={
                    "Authorization": "Bearer "
                    + refresh_token(*self.credentials, self.base_url)
                },
            )
            self.execution_data = response.json()["Description"]
            self.run_data = self.execution_data["RunData"]
            try:
                del self.run_data["tags"]
            except Exception:
                pass
        return result

    def __host_model(self, *, operation: str, model_id: str) -> None:
        """
        Builds the model execution environment

        Arguments
        ----------
        operation : str
            The model operation type (Sync or Async)
        model_id : str
            The uploaded model id (hash)

        Raises
        ------
        InputError
            Some input parameters its invalid
        """

        url = f"{self.base_url}/model/{operation}/host/{self.group}/{model_id}"
        response = requests.get(
            url,
            headers={
                "Authorization": "Bearer "
                + refresh_token(*self.credentials, self.base_url)
            },
        )
        if response.status_code == 202:
            logger.info(f"Model host in process - Hash: {model_id}")

        formatted_msg = parse_json_to_yaml(response.json())

        if response.status_code == 401:
            logger.error("Login or password are invalid, please check your credentials.")
            raise AuthenticationError("Login not authorized.")

        if response.status_code >= 500:
            logger.error("Server is not available. Please, try it later.")
            raise ServerError("Server is not available!")

        logger.error(f"Something went wrong...\n{formatted_msg}")
        raise InputError("Invalid parameters for model creation")

    def promote_model(
        self,
        *,
        model_name: str,
        operation: str = "Sync",
        schema: Union[str, dict],
        model_reference: Optional[str] = None,
        source_file: Optional[str] = None,
        extra_files: Optional[list] = None,
        requirements_file: Optional[str] = None,
        env: Optional[str] = None,
        input_type: str = None,
    ) -> NeomarilModel:
        """
        Upload models trained inside Neomaril.

        Arguments
        ---------
        model_name : str
            The name of the model, in less than 32 characters
        model_reference : str, optional
            The name of the scoring function inside the source file
        source_file : str, optional
            Path of the source file. The file must have a scoring function that accepts two parameters: data (data for the request body of the model) and model_path (absolute path of where the file is located)
        schema : Union[str, dict], optional
            Path to a JSON or XML file with a sample of the input for the entrypoint function. A dict with the sample input can be sending as well
        extra_files: list, optional
            A optional list with additional files paths that should be uploaded. If the scoring function refer to this file they will be on the same folder as the source file
        requirements_file : str, optional
            Path of the requirements file. This will override the requirements used in trainning. The packages versions must be fixed eg: pandas==1.0
        env : str, optional
            Flag that choose which environment (dev, staging, production) of Neomaril you are using. Default is True
        operation : str
            Defines which kind operation is being executed (Sync or Async). Default value is Sync
        input_type : str
            The type of the input file that should be 'json', 'csv' or 'parquet'

        Raises
        ------
        TrainingError
            The training execution shouldn't be succeeded to be promoted

        Returns
        -------
        NeomarilModel
            The new training model

        Example
        -------
        >>> training = run.promote_model('Teste notebook promoted custom', 'score', './samples/train/app.py', './samples/train/schema.json',  'csv')
        """
        if self.training_type == "Custom" or self.training_type == "External":
            input_validator = (
                model_name and operation and model_reference and source_file and schema
            )
            fields_required = (
                "model_name, operation, model_reference, source_file, schema"
            )
        elif self.training_type == "AutoML":
            input_validator = model_name and operation
            fields_required = "model_name, operation"
        else:
            input_validator = False
            raise InputError("Training type needs be: Custom, AutoML or External.")

        if operation and operation == "Async":
            fields_required += ", input_type (for async)"

        if not input_validator:
            raise InputError(
                f"The parameters {fields_required} it's mandatory on {self.training_type} promote."
            )

        if self.status in ["Running", "Requested"]:
            self.status = self.get_status()["Status"]

        if self.status != "Succeeded":
            raise TrainingError(
                "Training execution must be Succeeded to be promoted, current status is "
                + self.status
            )

        model_id = self.__upload_model(
            model_name=model_name,
            model_reference=model_reference,
            source_file=source_file,
            env=env,
            requirements_file=requirements_file,
            schema=schema,
            extra_files=extra_files,
            operation=operation,
            input_type=input_type,
        )

        if model_id:
            self.__host_model(operation=operation.lower(), model_id=model_id)

            return NeomarilModel(
                model_id=model_id,
                login=self.credentials[0],
                password=self.credentials[1],
                group=self.group,
                url=self.base_url,
            )


class NeomarilTrainingExperiment(BaseNeomaril):
    """
    Class to manage models being trained inside Neomaril

    Attributes
    ----------
    login : str
        Login for authenticating with the client. You can also use the env variable NEOMARIL_USER to set this
    password : str
        Password for authenticating with the client. You can also use the env variable NEOMARIL_PASSWORD to set this
    training_id : str
        Training id (hash) from the experiment you want to access
    group : str
        Group the training is inserted. Default is 'datarisk' (public group)
    environment : str
        Flag that choose which environment of Neomaril you are using. Test your deployment first before changing to production. Default is True
    executions : List[int]
        Ids for the executions in that training


    Raises
    ------
    TrainingError
        When the training can't be accessed in the server
    AuthenticationError
        Invalid credentials

    Example
    -------

    .. code-block:: python

        from neomaril_codex.training import NeomarilTrainingClient
        from neomaril_codex.base import NeomarilExecution

        client = NeomarilTrainingClient('123456')
        client.create_group('ex_group', 'Group for example purpose')
        training = client.create_training_experiment('Training example', 'Classification', 'ex_group')
        print(client.get_training(training.training_id, 'ex_group').training_data)

        data_path = './samples/train/'

        run = run = training.run_training('First test', data_path+'dados.csv', training_reference='train_model', training_type='Custom', python_version='3.9', requirements_file=data_path+'requirements.txt', wait_complete=True)

        print(run.get_training_execution(run.exec_id))
        print(run.download_result())
    """

    def __init__(
        self,
        *,
        training_id: str,
        login: Optional[str] = None,
        password: Optional[str] = None,
        group: str = "datarisk",
        url: str = "https://neomaril.staging.datarisk.net/",
    ) -> None:
        super().__init__(login=login, password=password, url=url)

        self.training_id = training_id
        self.group = group

        url = f"{self.base_url}/training/describe/{self.group}/{self.training_id}"
        response = requests.get(
            url,
            headers={
                "Authorization": "Bearer "
                + refresh_token(*self.credentials, self.base_url)
            },
        )

        if response.status_code == 404:
            raise ModelError(f'Experiment "{training_id}" not found.')

        formatted_msg = parse_json_to_yaml(response.json())

        if response.status_code == 401:
            logger.error("Login or password are invalid, please check your credentials.")
            raise AuthenticationError("Login not authorized.")

        if response.status_code >= 500:
            logger.error("Server is not available. Please, try it later.")
            raise ServerError(f'Unable to retrive experiment "{training_id}"')
        
        if response.status_code != 200:
            logger.error(f"Something went wrong...\n{formatted_msg}")
            raise Exception("Unexpected error.")

        self.training_data = response.json()["Description"]
        self.model_type = self.training_data["ModelType"]
        self.experiment_name = self.training_data["ExperimentName"]
        self.executions = self.training_data["Executions"]

    def __repr__(self) -> str:
        return f"""NeomarilTrainingExperiment(name="{self.experiment_name}", 
                                                        group="{self.group}", 
                                                        training_id="{self.training_id}",
                                                        model_type={str(self.model_type)}
                                                        )"""

    def __str__(self):
        return f'NEOMARIL training experiment "{self.experiment_name} (Group: {self.group}, Id: {self.training_id})"'

    def __upload_training(
        self,
        *,
        run_name: str,
        training_type: str = "External",
        description: Optional[str] = None,
        train_data: Optional[str] = None,
        dataset: Union[str, NeomarilDataset] = None,
        training_reference: Optional[str] = None,
        python_version: str = "3.8",
        conf_dict: Optional[Union[str, dict]] = None,
        source_file: Optional[str] = None,
        requirements_file: Optional[str] = None,
        env: Optional[str] = None,
        X_train=None,
        y_train=None,
        model_outputs=None,
        model_file: Optional[str] = None,
        model_metrics: Optional[Union[str, dict]] = None,
        model_params: Optional[Union[str, dict]] = None,
        model_hash: Optional[str] = None,
        extra_files: Optional[list] = None,
    ) -> str:
        """
        Upload the files to the server

        Arguments
        ---------
        run_name : str
            The name of the model, in less than 32 characters
        train_data : str
            Path of the file with train data
        training_type : str
            Can be Custom, AutoML or External
        description : str, optional
            Description of the experiment
        training_reference : str, optional
            The name of the training function inside the source file. Just used when training_type is Custom
        python_version : str
            Python version for the model environment. Available versions are 3.8, 3.9, 3.10. Defaults to '3.8'. Just used when training_type is Custom
        conf_dict : Union[str, dict], optional
            Path to a JSON file with the AutoML configuration. A dict can be sending as well. Just used when training_type is AutoML
        source_file : str, optional
            Path of the source file. The file must have a training function that accepts one parameter: model_path (absolute path of where the file is located). Just used when training_type is Custom
        requirements_file : str, optional
            Path of the requirements file. The packages versions must be fixed eg: pandas==1.0. Just used when training_type is Custom
        env : str, optional
            .env file to be used in your training environment. This will be encrypted in the server.
        extra_files : list, optional
            A optional list with additional files paths that should be uploaded. If the scoring function refer to this file they will be on the same folder as the source file. Just used when training_type is Custom
        X_train: pd.DataFrame, optional
            The training data.
        y_train : pd.Series, optional
            The training labels.
        model_outputs : pd.DataFrame, optional
            The model outputs.
        model_file : str, optional
            The path to the trained model file.
        model_metrics : Union[str, dict], optional
            The path to a JSON file with the model metrics or a dictionary with the metrics.
        model_params : Union[str, dict], optional
            The path to a JSON file with the model parameters or a dictionary with the parameters.

        Raises
        ------
        InputError
            Some input parameters its invalid

        Returns
        -------
        str
            The new model id (hash)
        """

        url = f"{self.base_url}/training/upload/{self.group}/{self.training_id}"

        upload_data = []
        form_data = {"run_name": run_name, "training_type": training_type}

        if description:
            form_data["description"] = description

        if training_type != "External":
            if train_data:
                upload_data.append(
                    ("train_data", (train_data.split("/")[-1], open(train_data, "rb")))
                )
            elif dataset:
                dataset_hash = (
                    dataset if isinstance(dataset, str) else dataset.dataset_hash
                )
                form_data["dataset_hash"] = dataset_hash

        if training_type == "Custom":
            file_extesions = {"py": "app.py", "ipynb": "notebook.ipynb"}

            upload_data = upload_data + [
                (
                    "source",
                    (
                        file_extesions[source_file.split(".")[-1]],
                        open(source_file, "rb"),
                    ),
                ),
                ("requirements", ("requirements.txt", open(requirements_file, "rb"))),
            ]

            if env:
                upload_data.append(("env", (".env", open(env, "r"))))

            if extra_files:
                extra_data = [
                    ("extra", (c.split("/")[-1], open(c, "rb"))) for c in extra_files
                ]

                upload_data += extra_data

            form_data["training_reference"] = (training_reference,)
            form_data["python_version"] = "Python" + python_version.replace(".", "")

        elif training_type == "AutoML":
            if conf_dict:
                upload_data.append(
                    ("conf_dict", ("conf.json", parse_dict_or_file(conf_dict)))
                )
            else:
                raise InputError("conf_dict is mandatory for AutoML training")

        elif training_type == "External":
            upload_data = []
            if model_hash:
                form_data["model_hash"] = model_hash

            if X_train:
                print(f"X_train:{X_train}")
                upload_data += [("features", ("features.parquet", open(X_train, "rb")))]

            if y_train:
                print(f"y_train:{y_train}")
                upload_data += [("target", ("target.parquet", open(y_train, "rb")))]

            if model_outputs:
                print(f"model_outputs:{model_outputs}")
                upload_data += [
                    ("output", ("predictions.parquet", open(model_outputs, "rb"))),
                ]

            if model_file:
                print(f"model_file:{model_file}")
                upload_data += [("model", open(model_file, "rb"))]

            if model_params:
                print(f"model_params:{model_params}")
                upload_data += [("parameters", open(model_params, "rb"))]

            if model_metrics:
                print(f"model_metrics:{model_metrics}")
                upload_data += [("metrics", open(model_metrics, "rb"))]

            if requirements_file:
                upload_data += [("requirements", open(requirements_file, "rb"))]

            if extra_files:
                extra_data = [("extra", open(path, "rb")) for path in extra_files]
                upload_data += extra_data

            if python_version:
                form_data["python_version"] = "Python" + python_version.replace(".", "")

            if env:
                upload_data.append(("env", (".env", open(env, "r"))))

        token = refresh_token(*self.credentials, self.base_url)
        response = requests.post(
            url,
            data=form_data,
            files=upload_data,
            headers={"Authorization": "Bearer " + token},
        )

        message = parse_json_to_yaml(response.json())
        raw_response = response.text

        if response.status_code == 201:
            logger.info(f"Result\n{message}")
            return re.search(patt, raw_response).group(1)

        if response.status_code == 401:
            logger.error("Login or password are invalid, please check your credentials.")
            raise AuthenticationError("Login not authorized.")

        if response.status_code >= 500:
            logger.error("Server is not available. Please, try it later.")
            raise ServerError("Server is not available!")

        logger.error(f"Something went wrong...\n{message}")
        raise InputError("Bad input for training upload")

    def __execute_training(self, exec_id: str) -> None:
        """
        Builds the model execution environment

        Arguments
        ---------
        exec_id : str
            The uploaded training execution id (hash)

        Raises
        ------
        InputError
            Some input parameters its invalid
        """

        url = f"{self.base_url}/training/execute/{self.group}/{self.training_id}/{exec_id}"
        response = requests.get(
            url,
            headers={
                "Authorization": "Bearer "
                + refresh_token(*self.credentials, self.base_url)
            },
        )
        if response.status_code == 200:
            logger.info(f"Model training starting - Hash: {self.training_id}")
            return HTTPStatus.OK

        formatted_msg = parse_json_to_yaml(response.json())

        if response.status_code == 401:
            logger.error("Login or password are invalid, please check your credentials.")
            raise AuthenticationError("Login not authorized.")

        if response.status_code >= 500:
            logger.error("Server is not available. Please, try it later.")
            raise ServerError("Server is not available!")

        logger.error(f"Something went wrong...\n{formatted_msg}")
        raise InputError("Invalid parameters for training execution")

    def __refresh_execution_list(self):
        url = f"{self.base_url}/training/describe/{self.group}/{self.training_id}"
        response = requests.get(
            url,
            headers={
                "Authorization": "Bearer "
                + refresh_token(*self.credentials, self.base_url)
            },
        )

        if response.status_code == 404:
            raise ModelError(f'Experiment "{self.training_id}" not found.')

        formatted_msg = parse_json_to_yaml(response.json())

        if response.status_code == 401:
            logger.error("Login or password are invalid, please check your credentials.")
            raise AuthenticationError("Login not authorized.")

        if response.status_code >= 500:
            logger.error("Server is not available. Please, try it later.")
            raise ServerError(f'Unable to retrive experiment "{self.training_id}"')
        
        if response.status_code != 200:
            logger.error(f"Something went wrong...\n{formatted_msg}")
            raise Exception("Unexpected error.")


        self.training_data = response.json()["Description"]
        self.executions = [c["Id"] for c in self.training_data["Executions"]]

    def run_training(
        self,
        *,
        run_name: str,
        training_type: str = "External",
        description: Optional[str] = None,
        train_data: Optional[str] = None,
        dataset: Union[str, NeomarilDataset] = None,
        training_reference: Optional[str] = None,
        python_version: str = "3.8",
        conf_dict: Optional[Union[str, dict]] = None,
        source_file: Optional[str] = None,
        requirements_file: Optional[str] = None,
        extra_files: Optional[list] = None,
        env: Optional[str] = None,
        X_train=None,
        y_train=None,
        model_outputs=None,
        model_file: Optional[str] = None,
        model_metrics: Optional[Union[str, dict]] = None,
        model_params: Optional[Union[str, dict]] = None,
        model_hash: Optional[str] = None,
        wait_complete: Optional[bool] = False,
    ) -> Union[dict, NeomarilExecution]:
        """
        Runs a prediction from the current model.

        Arguments
        ---------
        run_name : str
            The name of the model, in less than 32 characters
        train_data : str
            Path of the file with train data.
        training_reference : str, optional
            The name of the training function inside the source file. Just used when training_type is Custom
        training_type : str
            Can be Custom, AutoML or External
        description : str, optional
            Description of the experiment
        python_version : str, optional
            Python version for the model environment. Avaliable versions are 3.8, 3.9, 3.10. Defaults to '3.8'. Just used when training_type is Custom
        conf_dict : Union[str, dict]
            Path to a JSON file with the AutoML configuration. A dict can be sending as well. Just used when training_type is AutoML
        source_file : str, optional
            Path of the source file. The file must have a training function that accepts one parameter: model_path (absolute path of where the file is located). Just used when training_type is Custom
        requirements_file : str
            Path of the requirements file. The packages versions must be fixed eg: pandas==1.0. Just used when training_type is Custom
        env : str, optional
            .env file to be used in your training enviroment. This will be encrypted in the server.
        extra_files : list, optional
            A optional list with additional files paths that should be uploaded. If the scoring function refer to this file they will be on the same folder as the source file. Just used when training_type is Custom
        wait_complete : bool, optional
            Boolean that informs if a model training is completed (True) or not (False). Default value is False

        Raises
        ------
        InputError
            Some input parameters its invalid

        Returns
        -------
        Union[dict, NeomarilExecution]
            The return of the scoring function in the source file for Sync models or the execution class for Async models.

        Example
        -------
        >>> execution = run = training.run_training('First test', data_path+'dados.csv', training_reference='train_model', python_version='3.9', requirements_file=data_path+'requirements.txt', wait_complete=True)
        """
        if training_type == "Custom":
            input_validator = (
                train_data
                and (dataset or train_data)
                and source_file
                and requirements_file
                and run_name
                and training_reference
                and python_version
            )
            fields_required = "train_data, dataset or train_data, source_file, requirements_file, run_name, training_reference, python_version"
        elif training_type == "AutoML":
            input_validator = (
                train_data
                and (dataset or train_data)
                and conf_dict
                and run_name
                and python_version
            )
            fields_required = (
                "train_data, dataset or train_data, conf_dict, run_name, python_version"
            )
        elif training_type == "External":
            input_validator = run_name and python_version and X_train and y_train
            fields_required = "run_name, python_version, X_train, y_train"
        else:
            input_validator = False
            raise InputError("Training type needs be: Custom, AutoML or External.")

        if not input_validator:
            raise InputError(
                f"The parameters {fields_required} it's mandatory on {training_type} training."
            )

        if training_type != "External" and not (train_data or dataset):
            raise InputError(
                "Invalid data input. Run training requires a train_data or dataset"
            )

        if python_version not in ["3.8", "3.9", "3.10"]:
            raise InputError(
                "Invalid python version. Avaliable versions are 3.8, 3.9, 3.10"
            )

        if training_type not in ["Custom", "AutoML", "External"]:
            raise InputError(
                f"Invalid training_type {training_type}. Should be one of the following: Custom, AutoML or External"
            )

        if training_type == "Custom":
            exec_id = self.__upload_training(
                run_name=run_name,
                training_type=training_type,
                description=description,
                train_data=train_data,
                dataset=dataset,
                training_reference=training_reference,
                python_version=python_version,
                source_file=source_file,
                env=env,
                requirements_file=requirements_file,
                extra_files=extra_files,
            )

        elif training_type == "AutoML":
            exec_id = self.__upload_training(
                run_name=run_name,
                training_type=training_type,
                description=description,
                train_data=train_data,
                dataset=dataset,
                conf_dict=conf_dict,
            )

        elif training_type == "External":
            exec_id = self.__upload_training(
                run_name=run_name,
                training_type=training_type,
                description=description,
                python_version=python_version,
                requirements_file=requirements_file,
                extra_files=extra_files,
                X_train=X_train,
                y_train=y_train,
                model_outputs=model_outputs,
                model_file=model_file,
                model_metrics=model_metrics,
                model_params=model_params,
                model_hash=model_hash,
            )

        else:
            raise InputError("Invalid training type")

        if exec_id:
            self.__execute_training(exec_id)
            self.__refresh_execution_list()
            run = NeomarilTrainingExecution(
                training_id=self.training_id,
                group=self.group,
                exec_id=exec_id,
                login=self.credentials[0],
                password=self.credentials[1],
                url=self.base_url,
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
                logger.error(response["Message"])
                raise ExecutionError("Training execution failed")
            else:
                return run

    def get_training_execution(
        self, exec_id: Optional[str] = None
    ) -> NeomarilTrainingExecution:
        """
        Get the execution instance.

        Arguments
        ---------
        exec_id : str, optional
            Execution id. If not informed we get the last execution.

        Returns
        -------
        NeomarilExecution
            The chosen execution
        """
        if not exec_id:
            self.__refresh_execution_list()
            logger.info("Execution id not informed. Getting last execution")
            exec_id = max(self.executions)
        try:
            int(exec_id)
        except Exception:
            InputError(
                "Unvalid execution Id informed or this training dont have a successful execution yet."
            )

        exec = NeomarilTrainingExecution(
            training_id=self.training_id,
            group=self.group,
            exec_id=exec_id,
            login=self.credentials[0],
            password=self.credentials[1],
            url=self.base_url,
        )
        exec.get_status()

        return exec

    def get_all_training_executions(self) -> List[NeomarilTrainingExecution]:
        """
        Get all executions from that experiment.

        Returns
        -------
        List[NeomarilTrainingExecution]
            All executions from that training
        """
        self.__refresh_execution_list()
        return [self.get_training_execution(e) for e in self.executions]

    @contextmanager
    def log_train(
        self,
        *,
        name,
        X_train,
        y_train,
        description: Optional[str] = None,
        save_path: Optional[str] = None,
    ):
        try:
            self.trainer = NeomarilTrainingLogger(
                name=name,
                X_train=X_train,
                y_train=y_train,
                description=description,
                save_path=save_path,
            )
            yield self.trainer

        finally:
            self.trainer._processing_logging_inputs()
            self.run_training(
                run_name=self.trainer.name,
                description=self.trainer.description,
                training_type="External",
                python_version=self.trainer.python_version,
                requirements_file=self.trainer.requirements,
                extra_files=self.trainer.extras,
                X_train=self.trainer.X_train,
                y_train=self.trainer.y_train,
                model_outputs=self.trainer.model_outputs,
                model_file=self.trainer.model,
                model_metrics=self.trainer.metrics,
                model_params=self.trainer.params,
            )


class NeomarilTrainingClient(BaseNeomarilClient):
    """
    Class for client for accessing Neomaril and manage models

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
    -------
    .. code-block:: python

        from neomaril_codex.training import NeomarilTrainingClient

        client = NeomarilTrainingClient('123456')
        client.create_group('ex_group', 'Group for example purpose')
        training = client.create_training_experiment('Training example', 'Classification',  'Custom', 'ex_group')
        print(client.get_training(training.training_id, 'ex_group').training_data)

    """

    def __repr__(self) -> str:
        return f'API version {self.version} - NeomarilTrainingClient(url="{self.base_url}", Token="{self.user_token}")'

    def __str__(self):
        return f"NEOMARIL {self.base_url} Training client:{self.user_token}"

    def get_training(
        self, *, training_id: str, group: str = "datarisk"
    ) -> NeomarilTrainingExperiment:
        """
        Acess a model using its id

        Arguments
        ---------
        training_id : str
            Training id (hash) that needs to be acessed
        group : str
            Group the model is inserted. Default is 'datarisk' (public group)

        Raises
        ------
        TrainingError
            Model unavailable
        ServerError
            Unknown return from server

        Returns
        -------
        NeomarilTrainingExperiment
            A NeomarilTrainingExperiment instance with the training hash from `training_id`

        Example
        -------
        >>> training = get_training('Tfb3274827a24dc39d5b78603f348aee8d3dbfe791574dc4a6681a7e2a6622fa')
        """

        return NeomarilTrainingExperiment(
            training_id=training_id,
            login=self.credentials[0],
            password=self.credentials[1],
            group=group,
            url=self.base_url,
        )

    def __get_repeated_thash(
        self, model_type: str, experiment_name: str, group: str
    ) -> str | None:
        """Look for a previous train experiment.

        Args:
            experiment_name (str): name given to the training, should be not null, case-sensitive, have between 3 and 32 characters,
                                   that could be alphanumeric including accentuation (for example: 'é', à', 'ç','ñ') and space,
                                   without blank spaces and special characters

            model_type (str): type of the model being trained. It can be
                                Classification: for ML algorithms related to classification (predicts discrete class labels) problems;
                                Regression: the ones that will use regression (predict a continuous quantity) algorithms;
                                Unsupervised: for training that will use ML algorithms without supervision.

            group (str): name of the group, previous created, where the training will be inserted

        Raises:
            InputError: some part of the data is incorrect
            AuthenticationError: user has insufficient permissions
            ServerError: server is not available
            Exception: generated exception in case of the response to the request is different from 201

        Returns:
            str | None: THash if it is found, otherwise, None is returned
        """
        url = f"{self.base_url}/training/search"
        response = requests.get(
            url,
            headers={
                "Authorization": "Bearer "
                + refresh_token(*self.credentials, self.base_url)
            },
        )

        formatted_msg = parse_json_to_yaml(response.json())

        if response.status_code == 400:
            logger.error(f"Result\n{formatted_msg}")
            raise InputError("Bad Input")

        if response.status_code == 401:
            logger.error("Login or password are invalid, please check your credentials.")
            raise AuthenticationError("Login not authorized.")

        if response.status_code >= 500:
            logger.error("Server is not available. Please, try it later.")
            raise ServerError("Server is not available!")

        if response.status_code != 200:
            logger.error(f"Something went wrong...\n{formatted_msg}")
            raise Exception("Unexpected error!")

        results = response.json().get("Results")
        for result in results:
            condition = (
                result["ExperimentName"] == experiment_name
                and result["GroupName"] == group
                and result["ModelType"] == model_type
            )
            if condition:
                logger.info("Found experiment with same attributes...")
                return result["TrainingHash"]

    def __create(self, experiment_name: str, model_type: str, group: str) -> str:
        """Creates a train experiment. A train experiment can aggregate multiple training runs (also called executions).
        Each execution can eventually become a deployed model or not.

        Args:
            experiment_name (str): name given to the training, should be not null, case-sensitive, have between 3 and 32 characters,
                                   that could be alphanumeric including accentuation (for example: 'é', à', 'ç','ñ') and space,
                                   without blank spaces and special characters

            model_type (str): type of the model being trained. It can be
                                Classification: for ML algorithms related to classification (predicts discrete class labels) problems;
                                Regression: the ones that will use regression (predict a continuous quantity) algorithms;
                                Unsupervised: for training that will use ML algorithms without supervision.

            group (str): name of the group, previous created, where the training will be inserted

        Raises:
            InputError: some part of the data is incorrect
            AuthenticationError: user has insufficient permissions
            ServerError: server is not available
            Exception: generated exception in case of the response to the request is different from 201

        Returns:
            str: training hash of the experiment
        """
        url = f"{self.base_url}/training/register/{group}"

        data = {"experiment_name": experiment_name, "model_type": model_type}

        response = requests.post(
            url,
            data=data,
            headers={
                "Authorization": "Bearer "
                + refresh_token(*self.credentials, self.base_url)
            },
        )

        formatted_msg = parse_json_to_yaml(response.json())

        if response.status_code == 400:
            logger.error(f"Result\n{formatted_msg}")
            raise InputError("Bad Input")

        if response.status_code == 401:
            logger.error("Login or password are invalid, please check your credentials.")
            raise AuthenticationError("Login not authorized.")

        if response.status_code >= 500:
            logger.error("Server is not available. Please, try it later.")
            raise ServerError("Server is not available!")

        if response.status_code != 201:
            logger.error(f"Something went wrong...\n{formatted_msg}")
            raise Exception("Unexpected error!")

        response_data = response.json()
        logger.info(response_data["Message"])
        training_id = response_data["TrainingHash"]
        return training_id

    def create_training_experiment(
        self,
        *,
        experiment_name: str,
        model_type: str,
        group: str = "datarisk",
        force: bool = False,
    ) -> NeomarilTrainingExperiment:
        """
        Create a new training experiment on Neomaril.

        Arguments
        ---------
        experiment_name : str
            The name of the experiment, in less than 32 characters
        model_type : str
            The name of the scoring function inside the source file.
        group : str
            Group the model is inserted. Default to 'datarisk' (public group)
        force: bool
            Forces to create a new training with the same model_type, experiment_name, group

        Raises
        ------
        InputError
            Some input parameters its invalid
        ServerError
            Unknow internal server error

        Returns
        -------
        NeomarilTrainingExperiment
            A NeomarilTrainingExperiment instance with the training hash from `training_id`

        Example
        -------
        >>> training = client.create_training_experiment('Training example', 'Classification', 'ex_group')
        """

        if group:
            group = (
                group.lower()
                .strip()
                .replace(" ", "_")
                .replace(".", "_")
                .replace("-", "_")
            )

            groups = [g.get("Name") for g in self.list_groups()]

            if group not in groups:
                raise GroupError("Group dont exist. Create a group first.")

        else:
            group = "datarisk"
            logger.info("Group not informed, using default 'datarisk' group")

        if model_type not in ["Classification", "Regression", "Unsupervised"]:
            raise InputError(
                f"Invalid model_type {model_type}. Should be one of the following: Classification, Regression or "
                f"Unsupervised"
            )

        logger.info("Trying to load experiment...")
        training_id = self.__get_repeated_thash(
            model_type=model_type, experiment_name=experiment_name, group=group
        )

        if force or training_id is None:
            msg = (
                "The experiment you're creating has identical name, group, and model type attributes to an existing one. "
                + "Since forced creation is active, we will continue with the process as specified"
                if force
                else "Could not find experiment. Creating a new one..."
            )
            logger.info(msg)
            training_id = self.__create(
                experiment_name=experiment_name, model_type=model_type, group=group
            )

        return NeomarilTrainingExperiment(
            training_id=training_id,
            login=self.credentials[0],
            password=self.credentials[1],
            group=group,
            url=self.base_url,
        )
