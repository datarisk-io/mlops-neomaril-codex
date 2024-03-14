import os
import yaml
import json
from loguru import logger
from typing import Optional
from datetime import datetime

from neomaril_codex.exceptions import *
from neomaril_codex.training import *
from neomaril_codex.model import *

class NeomarilPipeline:
    """
    Class to construct and orchestrates the flow data of the models inside Neomaril.

    Atributtes
    ----------
    login : str
        Login for authenticating with the client. You can also use the env variable NEOMARIL_USER to set this
    password : str
        Password for authenticating with the client. You can also use the env variable NEOMARIL_PASSWORD to set this
    group : str
        Group the model is inserted
    url : str
        URL to Neomaril Server. Default value is https://neomaril.staging.datarisk.net, use it to test your deployment first before changing to production. You can also use the env variable NEOMARIL_URL to set this
    python_version : str
        Python version for the model environment. Avaliable versions are 3.7, 3.8, 3.9, 3.10. Defaults to '3.9'
        
    Example
    --------     

    .. code-block:: python

        from neomaril_codex.pipeline import NeomarilPipeline

        pipeline = NeomarilPipeline.from_config_file('./samples/pipeline.yml')
        pipeline.register_monitoring_config(directory = "./samples/monitoring", preprocess = "preprocess.py", preprocess_function = "score", shap_function = "score", config = "configuration.json", packages = "requirements.txt")
        pipeline.start()
        pipeline.run_monitoring('2', 'Mb29d61da4324a39a8bc2e0946f213b4959643916d354bf39940de2124f1e9d8')
    """
    def __init__(self, group:str, login:Optional[str]=None, password:Optional[str]=None, url:str='https://neomaril.staging.datarisk.net/', python_version:float=3.9) -> None:
        load_dotenv()
        logger.info('Loading .env')

        self.__credentials = (login if login else os.getenv('NEOMARIL_USER'), password if password else os.getenv('NEOMARIL_PASSWORD'))
        self.base_url = os.getenv('NEOMARIL_URL') if os.getenv('NEOMARIL_URL') else url
        self.base_url = parse_url(self.base_url)
        self.group = group
        self.python_version = python_version
        self.train_config = None
        self.deploy_config = None
        self.monitoring_config = None

    def register_train_config(self, **kwargs)-> dict:
        """
        Set the files for configure the training

        Parameters
        ----------
        kwargs : list or dict
            List or dictionary with the necessary files for training
        """
        self.train_config = kwargs

    def register_deploy_config(self, **kwargs) -> dict:
        """
        Set the files for configure the deploy

        Parameters
        ----------
        kwargs : list or dict
            List or dictionary with the necessary files for deploy
        """
        self.deploy_config = kwargs

    def register_monitoring_config(self, **kwargs) -> dict:
        """
        Set the files for configure the monitoring

        Parameters
        ----------
        kwargs : list or dict
            List or dictionary with the necessary files for monitoring
        
        Example
        -------
        >>> pipeline.register_monitoring_config(directory = "./samples/monitoring", preprocess = "preprocess.py", preprocess_function = "score", shap_function = "score", config = "configuration.json", packages = "requirements.txt")
        """
        self.monitoring_config = kwargs

    @staticmethod
    def from_config_file(**kwargs):
        """
        Load the configuration files for orchestrate the model

        Parameters
        ----------
        path : str
            Path of the configuration file, but it could be a dict
        
        Raises
        ------
        PipelineError
            Undefined credentials

        Returns
        -------
        NeomarilPipeline
            The new pipeline 
        
        Example
        --------
        >>> pipeline = NeomarilPipeline.from_config_file('./samples/pipeline-just-model.yml')
        >>> pipeline.register_monitoring_config(directory = "./samples/monitoring", preprocess = "preprocess.py", preprocess_function = "score", shap_function = "score", config = "configuration.json", packages = "requirements.txt")
        >>> pipeline.start()
        """
        path = check_args(kwargs, ["path"], {})

        with open(path, 'rb') as stream:
            try:
                conf=yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)

        load_dotenv()
        logger.info('Loading .env')

        login = os.getenv("NEOMARIL_USER")
        if not login:
            raise PipelineError("When using a config file the environment variable NEOMARIL_USER must be defined")

        password = os.getenv("NEOMARIL_PASSWORD")
        if not password:
            raise PipelineError("When using a config file the environment variable NEOMARIL_PASSWORD must be defined")
        
        url = os.getenv('NEOMARIL_URL', conf.get('url'))

        pipeline = NeomarilPipeline(conf['group'], login=login, password=password, url=url, python_version=conf['python_version'])

        if 'training' in conf.keys():
            pipeline.register_train_config(**conf['training'])

        if 'deploy' in conf.keys():
            pipeline.register_deploy_config(**conf['deploy'])

        if 'monitoring' in conf.keys():
            pipeline.register_monitoring_config(**conf['monitoring'])
              
        return pipeline

    def run_training(self) -> tuple[str, str]:
        """
        Run the training process

        Raises
        ------
        TrainingError
            Training has failed

        Returns
        -------
        tuple[str, str]
            A tuple with the 'training_id' and the 'exec_id' 
        
        Example
        -------
        >>> pipeline.run_training()
        """
        logger.info('Running training')
        client = NeomarilTrainingClient(login=self.__credentials[0], password=self.__credentials[1], url=self.base_url)
        client.create_group(self.group, self.group)

        conf = self.train_config

        training = client.create_training_experiment(conf['experiment_name'], conf['model_type'], conf['training_type'], group=self.group)

        
        PATH = conf['directory']
        run_name = conf.get('run_name', 'Pipeline run '+str(datetime.now()))
        extra_files = conf.get('extra')

        if conf['training_type'] == 'Custom':
            run = training.run_training(run_name, os.path.join(PATH, conf['data']), source_file=os.path.join(PATH, conf['source']),
                                        requirements_file=os.path.join(PATH, conf['packages']), training_reference=conf['train_function'],
                                        extra_files=[os.path.join(PATH,e) for e in extra_files] if extra_files else None,
                                        python_version=str(self.python_version), wait_complete=True)

        elif conf['training_type'] == 'AutoML':
            run = training.run_training(run_name, os.path.join(PATH, conf['data']), os.path.join(PATH, conf['config']), wait_complete=True)

        status = run.get_status()
        if status['Status'] == "Succeeded":
            logger.info('Model training finished')
            return training.training_id, run.exec_id
        else:
            raise TrainingError('Training failed: '+status['Message'])

    def run_deploy(self, training_id:Optional[str]=None) -> str:
        """
        Run the deploy process

        Arguments
        ----------
        training_id : str, optional
            The id for the training process that you want to deploy now

        Raises
        ------
        ModelError
            Deploy has failed

        Returns
        -------
        str
            The new Model id (hash) 
        
        Example
        -------
        >>> pipeline.run_deploy('Mb29d61da4324a39a8bc2e0946f213b4959643916d354bf39940de2124f1e9d8')
        """
        conf = self.deploy_config
        PATH = conf['directory']
        extra_files = conf.get('extra')

        if training_id:
            logger.info('Deploying scorer from training')
            training_run = NeomarilTrainingExecution(training_id[0], self.group, training_id[1], login=self.__credentials[0], 
                                                     password=self.__credentials[1], url=self.base_url)

            model_name = conf.get('name', training_run.execution_data.get('ExperimentName', ''))

            if training_run.execution_data['TrainingType'] == 'Custom':
                model = training_run.promote_model(model_name, model_reference=conf['score_function'], 
                                                    source_file=os.path.join(PATH, conf['source']),
                                                    extra_files=[os.path.join(PATH,e) for e in extra_files] if extra_files else None,
                                                    env=os.path.join(PATH, conf['env']) if conf.get('env') else None,
                                                    schema=os.path.join(PATH, conf['schema']) if conf.get('schema') else None,
                                                    operation=conf['operation'])

            elif training_run.execution_data['TrainingType'] == 'AutoML':
                model = training_run.promote_model(model_name, operation=conf['operation'])

        else:
            logger.info('Deploying scorer')
            client = NeomarilModelClient(login=self.__credentials[0], password=self.__credentials[1], url=self.base_url)
            client.create_group(self.group, self.group)
            
            model = client.create_model(conf.get('name'), conf['score_function'], os.path.join(PATH, conf['source']), 
                                        os.path.join(PATH, conf['model']), os.path.join(PATH, conf['packages']),
                                        extra_files=[os.path.join(PATH,e) for e in extra_files] if extra_files else None,
                                        env=os.path.join(PATH, conf['env']) if conf.get('env') else None,
                                        schema=os.path.join(PATH, conf['schema']) if conf.get('schema') else None,
                                        operation=conf['operation'], input_type=conf['input_type'], group=self.group)

        while model.status == 'Building':
            model.wait_ready()

        if model.status == 'Deployed':
            logger.info('Model deployement finished')
            return model.model_id

        else:
            raise ModelError("Model deployement failed: "+ model.get_logs(routine='Host')[0])

    def run_monitoring(self, training_exec_id:Optional[str]=None, model_id:Optional[str]=None):
        """
        Run the monitoring process

        Arguments
        ----------
        training_exec_id : str, optional
            The id for the training execution process that you want to monitore now
        model_id : str, optional

        Example
        -------
        >>> pipeline.run_monitoring('2', 'Mb29d61da4324a39a8bc2e0946f213b4959643916d354bf39940de2124f1e9d8')
        """
        logger.info('Configuring monitoring')

        conf = self.monitoring_config
        PATH = conf['directory']

        if training_exec_id:
            with open(os.path.join(PATH, conf['config']), 'r+') as f:
                conf_dict = json.load(f)
                f.seek(0)
                conf_dict['TrainData']['NeomarilTrainingExecution'] = training_exec_id
                json.dump(conf_dict, f)
                f.truncate()

        model = NeomarilModel(model_id, login=self.__credentials[0], password=self.__credentials[1], group=self.group, 
                              group_token=os.getenv('NEOMARIL_GROUP_TOKEN'), url=self.base_url)

        model.register_monitoring(conf['preprocess_function'], conf['shap_function'], 
                                    configuration_file=os.path.join(PATH, conf['config']),
                                    preprocess_file=os.path.join(PATH, conf['preprocess']),
                                    requirements_file=(os.path.join(PATH, conf['packages']) if conf.get('packages') else None))
    
    def start(self):
        """
        Start the pipeline for the model orchestration

        Raises
        ------
        PipelineError
            Cannot start pipeline without configuration
        
        Example
        -------
        >>> pipeline = NeomarilPipeline.from_config_file('./samples/pipeline.yml').start()
        """
        if (not self.train_config) and (not self.deploy_config) and (not self.monitoring_config):
            raise PipelineError("Cannot start pipeline without configuration")

        if self.train_config:
            training_id = self.run_training()
        else:
            training_id = None

        if self.deploy_config:
            model_id = self.run_deploy(training_id=training_id)
        else:
            model_id = None

        if self.monitoring_config:
            self.run_monitoring(training_exec_id=(training_id[1] if training_id else None), model_id=model_id)