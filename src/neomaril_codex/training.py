#!/usr/bin/env python
# coding: utf-8

import os
import sys
import re
import cloudpickle
from typing import Union, Optional, List, Any
from contextlib import contextmanager

import requests
import pandas as pd
import numpy as np
from time import sleep
from neomaril_codex.base import *
from neomaril_codex.model import NeomarilModel
from neomaril_codex.__utils import *
from neomaril_codex.exceptions import *

from lazy_imports import try_import

patt = re.compile(r'(\d+)')

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
    def __init__(self, name : str, X_train : pd.DataFrame, y_train: pd.DataFrame, description:Optional[str]=None, save_path:Optional[str]=None):
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
            dir_name = self.name.replace(' ', '_')
            if not os.path.exists(f'./{dir_name}'):
                os.mkdir(f'./{dir_name}')
            self.save_path = f'./{dir_name}'

    def save_model(self, **kwargs):
        """
        Save the trained model to the logger.

        Args:
            model: The trained model.
        """
        model = check_args(kwargs, ["model"], {})

        self.model = model

    def save_metric(self, **kwargs):
        """
        Save a metric to the logger.

        Args:
            name: The name of the metric.
            value: The value of the metric.
        """
        name, value = check_args(kwargs, ["name", "value"], {})

        self.metrics[name] = value 

    def save_model_output(self, **kwargs):
        """
        Save the model output to the logger.

        Args:
            model_output: The output of the trained model.
        """
        model_output = check_args(kwargs, ["model_output"], {})

        self.model_outputs = model_output

    def set_python_version(self, **kwargs):
        """
        Set the Python version used to train the model.

        Args:
            version: The Python version.
        """
        version = check_args(kwargs, ["version"], {})

        self.python_version = version
    
    def set_requirements(self, **kwargs):
        """
        Set the project requirements.

        Args:
            requirements: The path of project requirements.
        """
        requirements = check_args(kwargs, ["requirements"], {})

        self.requirements = requirements

    def save_plot(self, **kwargs):
        """
        Save plot graphic image to the logger.

        Args:
            plot: A Matplotlib/Plotly/Seaborn graphic object.
            filename: A name to save the plot.
        """
        plot, filename = check_args(kwargs, ["plot", "filename"], {})

        filepath = f'./{filename}.png'

        with try_import() as plotly_import:
            import plotly
            if isinstance(plot, plotly.graph_objs.Figure):
                self.save_plotly_plot(plot, filepath)
                return

        with try_import() as seaborn_import:
            import seaborn as sns 
            if isinstance(plot, sns.axisgrid.FacetGrid):
                self.save_seaborn_or_matplotlib_plot(plot, filepath)
                return

        with try_import() as matplotlib_import:
            import matplotlib.pyplot as plt
            if isinstance(plot, plt.Figure):
                self.save_seaborn_or_matplotlib_plot(plot, filepath)
                return

        raise ValueError("The plot only accepts plots of Matplotlib/Plotly/Seaborn")

    def save_plotly_plot(self, **kwargs):

        #TODO: missing docs

        plot, filename = check_args(kwargs, ["plot", "filename"], {})

        image_data = plot.to_image()
        with open(filepath, 'wb') as f:
            f.write(image_data)
        self.add_extra(extra=filepath)
            
    def save_seaborn_or_matplotlib_plot(self, **kwargs):

        #TODO: missing docs
        
        plot, filename = check_args(kwargs, ["plot", "filename"], {})

        plot.savefig(filepath)
        self.add_extra(extra=filepath)

    def set_extra(self, **kwargs):
        """
        Set the extra files list.

        Args:
            extra: A list of paths of the extra file.
        """
        extras = check_args(kwargs, ["extras"], {})

        self.extras = extra

    def add_extra(self, **kwargs):
        """
        Add an extra file in the extra file list.

        Args:
            extra: A path of an extra file or a list to include in extra file list.
            filename: A filename if the extra it's a DataFrame.
        """
        extra, filename = check_args(kwargs, ["extra"], {"filename": None})

        if isinstance(extra, str):
            if os.path.exists(extra):
                self.extras.append(extra)
            else:
                raise FileNotFoundError('Extra file path not found!')
        elif isinstance(extra, pd.DataFrame):
            if filename:
                self.extras.append(self.__to_parquet(filename, extra))
            else:
                raise InputError('Needs a filename to save the dataframe parquet.')

    def add_requirements(self, **kwargs):
        """
        Add requirements file.

        Args:
            filename: The name of output filename to save.
        """
        filename = check_args(kwargs, ["filename"], {})

        self.requirements = filename

    def __to_parquet(self, **kwargs):
        """
        Transform dataframe to parquet.

        Args:
            output_filename: The name of output filename to save.
            input_data: A pandas dataframe to save.
        """
        output_filename, input_data = check_args(kwargs, ["output_filename", "input_data" ], {})

        path = os.path.join(self.save_path, f'{output_filename}.parquet')
        input_data.to_parquet(path)
        return path

    def __to_json(self, **kwargs):
        """
        Transform dict to json.

        Args:
            output_filename: The name of output filename to save.
            input_data: A dictionary to save.
        """
        output_filename, input_data = check_args(kwargs, ["output_filename", "input_data" ], {})

        path = os.path.join(self.save_path, f'{output_filename}.json')
        with open(path, "w", encoding='utf-8') as f:
            json.dump(input_data, f)
        return path
    
    def __to_pickle(self, **kwargs):
        """
        Transform content to pickle.

        Args:
            output_filename: The name of output filename to save.
            input_data: The content to save.
        """
        output_filename, input_data = check_args(kwargs, ["output_filename", "input_data" ], {})

        path = os.path.join(self.save_path, f'{output_filename}.pkl')
        with open(path, "wb") as f:
            cloudpickle.dump(input_data, f)
        return path

    def _set_params(self):

        missing = self.X_train.isna().sum()
        missing_dict = {
            k+"_missings": v
            for k,v in missing[missing > 0].describe().to_dict().items()
            if k != "count"
        }

        params = {
            'shape': self.X_train.shape,
            'cols_with_missing': len(missing[missing > 0]),
            'missing_distribution': missing_dict
        }

        try:
            params['pipeline_steps'] = list(self.model.named_steps.keys())
        except:
            params['pipeline_steps'] = [
                str(self.model.__class__).replace("<class '", '').replace("'>", '')
            ]

        if 'get_all_params' in dir(self.model):
            hyperparameters = {
                f"hyperparam_{k}": str(v)
                for k, v in self.model.get_all_params().items()
                if k != "task_type"
            }
        elif 'get_params' in dir(self.model):
            hyperparameters = {
                "hyperparam_" + k: str(v)
                for k, v in self.model.get_params().items()
                if k not in params[
                    'pipeline_steps'
                ] + [
                    "steps",
                    'memory',
                    'verbose'
                ]
            }
            
            params = {**params, **hyperparameters}

        if len(self.y_train.value_counts()) < 10:
            target_proportion = self.y_train.value_counts() / len(self.y_train)
            target_proportion = target_proportion.to_dict()
            target_proportion = [ { 'target' : k, 'proportion' : v  } for k, v in target_proportion.items() ]
            params['target_proportion'] = target_proportion
        else:
            params["target_distribution"] = {
                k: v
                for k,v in self.y_train.describe().to_dict().items()
                if k != "count"
            }

        self.params = {**params, **self.params}

    def _parse_data_objects(self, **kwargs) -> pd.DataFrame:
        """
        Tranform data types to dataframe
        """

        obj = check_args(kwargs, ["obj"], {})

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
        self.params = self.__to_json('params', self.params)

        self.X_train = self.__to_parquet('features', self._parse_data_objects(self.X_train))

        self.y_train = self.__to_parquet('target', self._parse_data_objects(self.y_train))

        self.model_outputs = self.__to_parquet('predictions', self._parse_data_objects(self.model_outputs))

        if self.model:
            self.model = self.__to_pickle('model', self.model)

        if self.metrics:
            self.metrics = self.__to_json('metrics', self.metrics)


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

    def __init__(self, training_id:str, group:str, exec_id:str, login:Optional[str]=None, 
                 password:Optional[str]=None, url:str=None) -> None:
        super().__init__(training_id, 'Training', exec_id=exec_id, login=login, password=password, url=url, group=group)
        load_dotenv()
        logger.info('Loading .env')

        self.__credentials = (login if login else os.getenv('NEOMARIL_USER'), password if password else os.getenv('NEOMARIL_PASSWORD'))

        self.training_id = training_id
        self.group = group
    
        self.training_type = self.execution_data['TrainingType']
        self.name = self.execution_data['RunName']
        self.run_data = self.execution_data['RunData']


    def __repr__(self) -> str:
        return f"""Neomaril{self.exec_type}Execution(name="{self.name}",
                                        exec_id="{self.exec_id}", status="{self.status}")"""
    
    def __upload_model(self, model_name:str, model_reference:Optional[str]=None, source_file:Optional[str]=None, 
                                         schema:Optional[Union[str, dict]]=None, extra_files:Optional[list]=None, 
                                         env:Optional[str]=None, requirements_file:Optional[str]=None, operation:str='Sync', 
                                         input_type:str=None) -> str:
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
            Path to a JSON or XML file with a sample of the input for the entrypoint function. A dict with the sample input can be send as well
        extra_files list, optional
            A optional list with additional files paths that should be uploaded. If the scoring function refer to this file they will be on the same folder as the source file
        requirements_file : str, optional
            Path of the requirements file. This will override the requirements used in trainning. The packages versions must be fixed eg: pandas==1.0
        env : str, optional
            Flag that choose which environment (dev, staging, production) of Neomaril you are using. Default is True
        operation : str
            Defines wich kind operation is beeing executed (Sync or Async). Default value is Sync
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

        form_data = {'name': model_name, 'operation': operation}
        upload_data = []

        if self.training_type != 'AutoML':
            form_data['model_reference'] = model_reference
      
            if operation=="Sync":
                input_type = "json"
                schema_extesion = '.json'
            else:
                schema_extesion = schema.split('.')[-1]

                if input_type == 'json|csv|parquet':
                    raise InputError("Choose a input type from "+input_type)


            upload_data = [
                ("source", ('app.py', open(source_file, 'rb'))),
                ("schema", ("schema."+schema_extesion, parse_dict_or_file(schema)))
            ]

            if env:
                upload_data.append(("env", (".env", open(env, 'r'))))
            if requirements_file:
                upload_data.append(("requirements", ("requirements.txt", env)))
            if extra_files:
                extra_data = [('extra', (c.split('/')[-1], open(c, 'rb'))) for c in extra_files]
                
                upload_data += extra_data
        
        else:

            input_type = 'automl'

        form_data['input_type'] = input_type
            
        response = requests.post(url, data=form_data, files=upload_data, headers={'Authorization': 'Bearer ' + refresh_token(*self.__credentials, self.base_url)})
        
        if response.status_code == 201:
            data = response.json()
            model_id = data["ModelHash"]
            logger.info(f'{data["Message"]} - Hash: "{model_id}"')
            return model_id
        else:
            logger.error('Upload error: ' + response.text)
            raise InputError('Invalid parameters for model creation')
        
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

        response = requests.get(url, headers={'Authorization': 'Bearer ' + refresh_token(*self.__credentials, self.base_url)})
        if response.status_code not in [200, 410]:
            logger.error(response.text)
            raise ExecutionError(f'Execution "{self.exec_id}" unavailable')

        result = response.json()

        self.status = result['Status']
        self.execution_data['ExecutionState'] = result['Status']
        if self.status == 'Succeeded':
            url = f"{self.base_url}/training/describe/{self.group}/{self.training_id}/{self.exec_id}"
            response = requests.get(url, headers={'Authorization': 'Bearer ' + refresh_token(*self.__credentials, self.base_url)})
            self.execution_data = response.json()['Description']
            self.run_data = self.execution_data['RunData']
            try:
                del self.run_data['tags']
            except:
                pass
        return result

    def __host_model(self, operation:str, model_id:str) -> None:
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
        if operation == 'sync':
            url = url.replace('7070','7071')
        response = requests.get(url, headers={'Authorization': 'Bearer ' + refresh_token(*self.__credentials, self.base_url)})
        if response.status_code == 202:
            logger.info(f"Model host in process - Hash: {model_id}")
        else:
            logger.error(response.text)
            raise InputError('Invalid parameters for model creation')

    def promote_model(self, model_name:str, model_reference:Optional[str]=None, source_file:Optional[str]=None, 
                                         schema:Optional[Union[str, dict]]=None, extra_files:Optional[list]=None, 
                                         requirements_file:Optional[str]=None, env:Optional[str]=None, operation:str='Sync', 
                                         input_type:str=None)-> NeomarilModel:
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
            Path to a JSON or XML file with a sample of the input for the entrypoint function. A dict with the sample input can be send as well
        extra_files list, optional
            A optional list with additional files paths that should be uploaded. If the scoring function refer to this file they will be on the same folder as the source file
        requirements_file : str, optional
            Path of the requirements file. This will override the requirements used in trainning. The packages versions must be fixed eg: pandas==1.0
        env : str, optional
            Flag that choose which environment (dev, staging, production) of Neomaril you are using. Default is True
        operation : str
            Defines wich kind operation is beeing executed (Sync or Async). Default value is Sync
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
        if self.status in ['Running', 'Requested']:
            self.status = self.get_status()['Status']
        
        if self.status != 'Succeeded':
            raise TrainingError("Training execution must be Succeeded to be promoted, current status is "+self.status)
            
        model_id = self.__upload_model(model_name, model_reference=model_reference, source_file=source_file, env=env, 
                                                                        requirements_file=requirements_file, schema=schema,
                                                                        extra_files=extra_files, operation=operation, 
                                                                        input_type=input_type)
        
        if model_id:
            self.__host_model(operation.lower(), model_id)

            return NeomarilModel(model_id, login=self.__credentials[0], password=self.__credentials[1], 
                                 group=self.group, url=self.base_url)

        
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

        run = run = training.run_training('First test', data_path+'dados.csv', training_reference='train_model', training_type='Custom', python_version='3.9', requirements_file=data_path+'requirements.txt', wait_complete=True)
        
        print(run.get_training_execution(run.exec_id))
        print(run.download_result())
    """

    def __init__(self, **kwargs) -> None:

        training_id, login, password, group, url = check_args(kwargs, ["training_id"], {"login": None, "password": None, "group": "datarisk", "url": "https://neomaril.staging.datarisk.net/"})
        
        super().__init__(url)
        load_dotenv()
        logger.info('Loading .env')

        self.__credentials = (login if login else os.getenv('NEOMARIL_USER'), password if password else os.getenv('NEOMARIL_PASSWORD'))
        self.training_id = training_id
        self.base_url = os.getenv('NEOMARIL_URL') if os.getenv('NEOMARIL_URL') else url
        self.base_url = parse_url(self.base_url)
        self.group = group

        try_login(*self.__credentials, self.base_url)
        
        url = f"{self.base_url}/training/describe/{self.group}/{self.training_id}"
        response = requests.get(url, headers={'Authorization': 'Bearer ' + refresh_token(*self.__credentials, self.base_url)})
    
        if response.status_code == 404:
            raise ModelError(f'Experiment "{training_id}" not found.')
            
        elif response.status_code >= 500:
            raise ModelError(f'Unable to retrive experiment "{training_id}"')
    
        self.training_data = response.json()['Description']
        self.model_type = self.training_data['ModelType']
        self.experiment_name = self.training_data['ExperimentName']
        self.executions = self.training_data['Executions']

    def __repr__(self) -> str:
            return f"""NeomarilTrainingExperiment(name="{self.experiment_name}", 
                                                        group="{self.group}", 
                                                        training_id="{self.training_id}",
                                                        model_type={str(self.model_type)}
                                                        )"""

    def __str__(self):
        return f'NEOMARIL training experiment "{self.experiment_name} (Group: {self.group}, Id: {self.training_id})"'
    
    def __upload_training(self, **kwargs) -> str:
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
            Python version for the model environment. Avaliable versions are 3.7, 3.8, 3.9, 3.10. Defaults to '3.8'. Just used when training_type is Custom
        conf_dict : Union[str, dict], optional
            Path to a JSON file with a the AutoML configuration. A dict can be send as well. Just used when training_type is AutoML
        source_file : str, optional
            Path of the source file. The file must have a training function that accepts one parameter: model_path (absolute path of where the file is located). Just used when training_type is Custom
        requirements_file : str, optional
            Path of the requirements file. The packages versions must be fixed eg: pandas==1.0. Just used when training_type is Custom
        env : str, optional
            .env file to be used in your training enviroment. This will be encrypted in the server.
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
        run_name, training_type, description, train_data, training_reference, python_version, conf_dict, source_file, requirements_file, env, model_file, model_metrics, model_params, model_hash, extra_files = check_args(kwargs, ["run_name"], {"training_type": "External", "description": None, "train_data": None, "training_reference": None, "python_version": "3.8", "conf_dict": None, "source_file": None, "requirements_file": None, "env": None, "model_file": None, "model_metrics": None, "model_params": None, "model_hash": None, "extra_files": None})
        
        url = f"{self.base_url}/training/upload/{self.group}/{self.training_id}"

        upload_data = []
        form_data = {'run_name': run_name, 'training_type': training_type}

        if description:
            form_data['description'] = description

        if training_type != 'External':
            upload_data.append(("train_data", (train_data.split('/')[-1], open(train_data, "rb"))))

        if training_type == 'Custom':
        
            file_extesions = {'py': 'app.py', 'ipynb': "notebook.ipynb"}
        
            upload_data = upload_data + [
                ("source", (file_extesions[source_file.split('.')[-1]], open(source_file, 'rb'))),
                ("requirements", ("requirements.txt", open(requirements_file, 'rb')))
            ]

            if env:
                upload_data.append(("env", (".env", open(env, 'r'))))
         
            if extra_files:
                extra_data = [('extra', (c.split('/')[-1], open(c, 'rb'))) for c in extra_files]
                
                upload_data += extra_data
                
            form_data['training_reference']= training_reference,
            form_data['python_version'] = "Python"+python_version.replace('.', '')
        
        elif training_type == 'AutoML':
                
            if conf_dict:
                upload_data.append(("conf_dict", ("conf.json", parse_dict_or_file(conf_dict))))
            else:
                raise InputError("conf_dict is mandatory for AutoML training")
            
        elif training_type == 'External':
            upload_data = []
            if model_hash:
                form_data['model_hash'] = model_hash

            if X_train:
                print(f'X_train:{X_train}')
                upload_data += [
                    ("features", ('features.parquet', open(X_train, 'rb')))
                ]

            if y_train:
                print(f'y_train:{y_train}')
                upload_data += [
                    ("target", ('target.parquet', open(y_train, 'rb')))
                ]
            
            if model_outputs:
                print(f'model_outputs:{model_outputs}')
                upload_data += [
                    ("output", ('predictions.parquet', open(model_outputs, 'rb'))),
                ]

            if model_file:
                print(f'model_file:{model_file}')
                upload_data += [
                    ("model", open(model_file, 'rb'))
                ]

            if model_params:
                print(f'model_params:{model_params}')
                upload_data += [
                    ("parameters", open(model_params, 'rb'))
                ]

            if model_metrics:
                print(f'model_metrics:{model_metrics}')
                upload_data += [
                    ("metrics", open(model_metrics, 'rb'))
                ]

            if requirements_file:
                upload_data += [("requirements", open(requirements_file, 'rb'))]

            if extra_files:
                extra_data = [('extra', open(path, 'rb')) for path in extra_files]
                upload_data += extra_data
            
            if python_version:
                form_data['python_version'] = "Python"+python_version.replace('.', '')

            if env:
                upload_data.append(("env", (".env", open(env, 'r'))))

        
        token = refresh_token(*self.__credentials, self.base_url)
        response = requests.post(
            url,
            data=form_data,
            files=upload_data, 
            headers={'Authorization': 'Bearer ' + token}
        )

        message = response.text

        if response.status_code == 201:
            logger.info(message)
            return re.search(patt, message).group(1)
        else:
            logger.error(message)
            raise InputError('Bad input for training upload')

    def __execute_training(self, **kwargs) -> None:
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
        exec_id = check_args(kwargs, [], {"exec_id": True})
        
        url = f"{self.base_url}/training/execute/{self.group}/{self.training_id}/{exec_id}"
        response = requests.get(url, headers={'Authorization': 'Bearer ' + refresh_token(*self.__credentials, self.base_url)})
        if response.status_code == 200:
            logger.info(f"Model training starting - Hash: {self.training_id}")
        else:
            logger.error(response.text)
            raise InputError('Invalid parameters for training execution')
        
    def __refresh_execution_list(self):
        url = f"{self.base_url}/training/describe/{self.group}/{self.training_id}"
        response = requests.get(url, headers={'Authorization': 'Bearer ' + refresh_token(*self.__credentials, self.base_url)})
    
        if response.status_code == 404:
            raise ModelError(f'Experiment "{self.training_id}" not found.')
            
        elif response.status_code >= 500:
            raise ModelError(f'Unable to retrive experiment "{self.training_id}"')
    
        self.training_data = response.json()['Description']
        self.executions = [c['Id'] for c in self.training_data['Executions']]

    def run_training(self, **kwargs) -> Union[dict, NeomarilExecution]:
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
            Python version for the model environment. Avaliable versions are 3.7, 3.8, 3.9, 3.10. Defaults to '3.8'. Just used when training_type is Custom
        conf_dict : Union[str, dict]
            Path to a JSON file with a the AutoML configuration. A dict can be send as well. Just used when training_type is AutoML
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
        run_name, training_type, description, train_data, training_reference, python_version, conf_dict, source_file, requirements_file, extra_files, env, X_train, y_train, model_outputs, model_file, model_metrics, model_params, model_hash, wait_complete = check_args(kwargs, ["run_name"], {"training_type": "External", "description": None, "train_data": None, "training_reference": None, "python_version": "3.8", "conf_dict": None, "source_file": None, "requirements_file": None, "extra_files": None, "env": None, "X_train": None, "y_train": None, "model_outputs": None, "model_file": None, "model_metrics": None, "model_params": None, "model_hash": None, "wait_complete": False})

        if python_version not in ['3.7', '3.8', '3.9', '3.10']:
            raise InputError('Invalid python version. Avaliable versions are 3.7, 3.8, 3.9, 3.10')

        if training_type not in ['Custom', 'AutoML', 'External']:
            raise InputError(f'Invalid training_type {training_type}. Should be one of the following: Custom, AutoML or External')

        if training_type == 'Custom':
            exec_id = self.__upload_training(run_name, training_type=training_type, description=description,train_data=train_data, training_reference=training_reference,
                                             python_version=python_version, source_file=source_file, env=env,
                                             requirements_file=requirements_file, extra_files=extra_files)

        elif training_type == 'AutoML':
            exec_id = self.__upload_training(run_name, training_type=training_type, description=description, 
                                             train_data=train_data, conf_dict=conf_dict)

        elif training_type == 'External':
            exec_id = self.__upload_training(run_name, training_type=training_type, description=description,
                                             python_version=python_version, requirements_file=requirements_file, 
                                             extra_files=extra_files, X_train=X_train, y_train=y_train, model_outputs=model_outputs,
                                             model_file=model_file, model_metrics=model_metrics, model_params=model_params,
                                             model_hash=model_hash)

        else:
            raise InputError('Invalid training type')

        if exec_id:
            self.__execute_training(exec_id)
            self.__refresh_execution_list()
            run = NeomarilTrainingExecution(self.training_id, self.group, exec_id, login=self.__credentials[0], 
                                            password=self.__credentials[1], url=self.base_url)
            response = run.get_status()
            status = response['Status']
            if wait_complete:
                print('Wating the training run.', end='')
                while status in ['Running', 'Requested']:
                    sleep(30)
                    print('.', end='', flush=True)
                    response = run.get_status()
                    status = response['Status']
            if status == 'Failed':
                logger.error(response['Message'])
                raise ExecutionError("Training execution failed")
            else:
                return run

    def __call__(self, **kwargs) -> dict:
        data = check_args(kwargs, ["data"], {})

        return self.predict(data)

    def get_training_execution(self, **kwargs) -> NeomarilTrainingExecution:
        """
        Get a execution instace.

        Arguments
        ---------
        exec_id : str, optional
            Execution id. If not informed we get the last execution.

        Returns
        -------
        NeomarilExecution
            The choosen execution
        """
        exec_id = check_args(kwargs, [], {"exec_id": None})

        if not exec_id:
            self.__refresh_execution_list()
            logger.info("Execution id not informed. Getting last execution")
            exec_id = max(self.executions)
        try:
            int(exec_id)
        except:
            InputError("Unvalid execution Id informed or this training dont have a successful execution yet.")

        exec = NeomarilTrainingExecution(self.training_id, self.group, exec_id, login=self.__credentials[0],
                                         password=self.__credentials[1], url=self.base_url)
        exec.get_status()
        
        return exec

    def get_all_training_executions(self) -> List[NeomarilTrainingExecution]:
        """
        Get all executions from that experiment.

        Returns
        -------
        List[NeomarilExecution]
            All executions from that training
        """
        self.__refresh_execution_list()
        return [self.get_training_execution(e) for e in self.executions]
    
    @contextmanager
    def log_train(self, **kwargs):

        name, X_train, y_train, description = check_args(kwargs, ["name", "X_train", "y_train"], {"description": None})
        
        try:

            self.trainer = NeomarilTrainingLogger(name, X_train, y_train, description=description, save_path=save_path)
            yield self.trainer

        finally:
            self.trainer._processing_logging_inputs()
            self.run_training(self.trainer.name, description=self.trainer.description, training_type="External", 
                              python_version=self.trainer.python_version, requirements_file=self.trainer.requirements, 
                              extra_files=self.trainer.extras, X_train=self.trainer.X_train, y_train=self.trainer.y_train,
                              model_outputs=self.trainer.model_outputs, model_file=self.trainer.model, 
                              model_metrics=self.trainer.metrics, model_params=self.trainer.params)
        

class NeomarilTrainingClient(BaseNeomarilClient):
    """
    Class for client for acessing Neomaril and manage models

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
        Unvalid credentials
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
    def __init__(self, **kwargs) -> None:
        """Client for acessing Neomaril and manage models

        Args:
                password (str): Password for authenticating with the client
            url (str): URL for Neomaril server. Test your deployment first before changing to production. Default is https://neomaril.staging.datarisk.net/

        Raises:
                AuthenticationError: Unvalid credentials
                ServerError: Server unavailable
        """
        login, password, url = check_args(kwargs, [], {"login": None, "password": None, "url": 'https://neomaril.staging.datarisk.net/'})

        load_dotenv()
        logger.info('Loading .env')

        self.__credentials = (login if login else os.getenv('NEOMARIL_USER'), password if password else os.getenv('NEOMARIL_PASSWORD'))
        self.base_url = os.getenv('NEOMARIL_URL') if os.getenv('NEOMARIL_URL') else url
        self.base_url = parse_url(self.base_url)

        super().__init__(login=self.__credentials[0], password=self.__credentials[1], url=self.base_url)
            
    def __repr__(self) -> str:
            return f'NeomarilTrainingClient(url="{self.base_url}", version="{self.client_version}")'
        
    def __str__(self):
        return f"NEOMARIL {self.base_url} Training client:{self.client_version}"
        
    
    def get_training(self, **kwargs) -> NeomarilTrainingExperiment:
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
        training_id, group = check_args(kwargs, ["training_id"], {"group": "datarisk"})

        return NeomarilTrainingExperiment(training_id, login=self.__credentials[0], 
                                          password=self.__credentials[1], group=group, url=self.base_url)
    

    def create_training_experiment(self, **kwargs) -> NeomarilTrainingExperiment:
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
        experiment_name, model_type, group = check_args(kwargs, ["experiment_name", "model_type"], {"group": "datarisk"})
        
        if group:
            group = group.lower().strip().replace(" ", "_").replace(".", "_").replace("-", "_")

            groups = [g["Name"] for g in self.list_groups()]

            if group not in groups:

                raise GroupError('Group dont exist. Create a group first.')

        else:
            group = 'datarisk'
            logger.info("Group not informed, using default 'datarisk' group")

        if model_type not in ['Classification', 'Regression', 'Unsupervised']:
            raise InputError(f'Invalid model_type {model_type}. Should be one of the following: Classification, Regression or Unsupervised')

        url = f"{self.base_url}/training/register/{group}"

        data = {'experiment_name': experiment_name, 'model_type': model_type}

        response = requests.post(url, data=data, headers={'Authorization': 'Bearer ' + refresh_token(*self.__credentials, self.base_url)})

        if response.status_code < 300:
            response_data = response.json()
            logger.info(response_data['Message'])
            training_id = response_data['TrainingHash']
        else:
            logger.error(response.text)
            raise ServerError('')

        
        return NeomarilTrainingExperiment(training_id, login=self.__credentials[0], password=self.__credentials[1], 
                                          group=group, url=self.base_url)