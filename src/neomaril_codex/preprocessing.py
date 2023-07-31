#!/usr/bin/env python
# coding: utf-8

import io, os
from typing import Union, Optional
from time import sleep
import requests
import json
from neomaril_codex.base import *
from neomaril_codex.utils import *
from neomaril_codex.exceptions import *

class NeomarilPreprocessing(BaseNeomaril):

    def __init__(self, preprocessing_id:str, password:Optional[str]=None, group:str="datarisk", group_token:Optional[str]=None, url:str='https://neomaril.staging.datarisk.net/') -> None:

        load_dotenv()
        logger.info('Loading .env')

        super().__init__()
        self.__credentials = os.getenv('NEOMARIL_TOKEN') if os.getenv('NEOMARIL_TOKEN') else password
        self.preprocessing_id = preprocessing_id
        self.group = group
        self.__token = os.getenv('NEOMARIL_GROUP_TOKEN') if os.getenv('NEOMARIL_GROUP_TOKEN') else group_token
        self.base_url = os.getenv('NEOMARIL_URL') if os.getenv('NEOMARIL_URL') else url
        self.base_url = parse_url(self.base_url)

        try_login(self.__credentials, self.base_url)
        
        url = f"{self.base_url}/preprocessing/list"
        response = requests.get(url, headers={'Authorization': 'Bearer ' + self.__credentials})

        results = response.json()
        for result in results.get('Results'):
            if result.get('Hash') == preprocessing_id:
                self.operation = result.get('Operation')
                
        
        response = self.__get_status()
        self.status = response.get('Status')

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
        if self.status in ['Ready', 'Building']:
            self.status = self.__get_status()['Status']
            while self.status == 'Building':
                sleep(30)
                self.status = self.__get_status()['Status']

    def get_logs(self, start:Optional[str]=None, end:Optional[str]=None, routine:Optional[str]=None, type:Optional[str]=None):
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
        return self._logs(url, self.__credentials, start=start, end=end, routine=routine, type=type)
        
    def set_token(self, group_token:str) -> None:
        """
        Saves the group token for this pre processing instance.

        Arguments
        ---------
        group_token : str
            Token for executing the pre processing (show when creating a group). You can set this using the NEOMARIL_GROUP_TOKEN env variable
        
        Example
        -------
        >>> preprocessing.set_token('6cb64889a45a45ea8749881e30c136df')
        """

        self.__token = group_token
        logger.info(f"Token for group {self.group} added.")

    def run(self, data:Union[dict, str], group_token:Optional[str]=None, wait_complete:Optional[bool]=False) -> Union[dict, NeomarilExecution]:
        """
        Runs a prediction from the current pre processing.

        Arguments
        ---------
        data : Union[dict, str]
            The same data that is used in the source file. 
            If Sync is a dict, the keys that are needed inside this dict are the ones in the `schema` atribute.
            If Async is a string with the file path with the same filename used in the source file. 
        group_token : str, optional
            Token for executing the pre processing (show when creating a group). It can be informed when getting the preprocessing or when running predictions, or using the env variable NEOMARIL_GROUP_TOKEN
        wait_complete: bool, optional
            Boolean that informs if a pre processing training is completed (True) or not (False). Default value is False

        Raises
        ------
        PreprocessingError
            Pre processing is not available

        Returns
        -------
        Union[dict, NeomarilExecution]
            The return of the scoring function in the source file for Sync pre processing or the execution class for Async pre processing.
        """
        if self.__preprocessing_ready:
            if (group_token is not None) | (self.__token is not None):
                url = f"{self.base_url}/preprocessing/{self.operation.lower()}/run/{self.group}/{self.preprocessing_id}"
                if self.__token and not group_token:
                    group_token = self.__token
                if group_token and not self.__token:
                    self.__token = group_token
                if self.operation.lower() == 'sync':
                    url = url.replace('localhost:7070', 'localhost:7071')
                    preprocessing_input = {
                            "Input": data
                    }

                    req = requests.post(url, data=json.dumps(preprocessing_input), headers={'Authorization': 'Bearer ' + group_token})

                    return req.json()

                elif self.operation.lower() == 'async':

                    files = {
                        'dataset': open(data, 'rb'),
                    }

                    req = requests.post(url, files=files, headers={'Authorization': 'Bearer ' + group_token})

                    if req.status_code == 202:
                        message = req.json()
                        logger.info(message['Message'])
                        exec_id = message['ExecutionId']
                        run = NeomarilExecution(self.preprocessing_id, 'AsyncPreprocessing', exec_id=exec_id, password=self.__credentials, 
                                                url=self.base_url, group=self.group, group_token=group_token)
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
                        raise ServerError(req.text)
                
            else:
                raise AuthenticationError("Group token not informed")
        return run
        # else:
        #     url = f"{self.base_url}/preprocessing/describe/{self.group}/{self.preprocessing_id}"
        #     response = requests.get(url, headers={'Authorization': 'Bearer ' + self.__credentials}).json()['Description']
        #     if response['Status'] == "Deployed":
        #         self.preprocessing_data = response
        #         self.status = response['Status']
        #         self.__preprocessing_ready = True
        #         return self.predict(data)
    
        #     else:
        #         raise PreprocessingError('Pre processing is not available to predictions')
            
    def __call__(self, data: dict) -> dict:
        return self.predict(data)
    
    def get_preprocessing_execution(self, exec_id:str) -> None:
        """
        Get a execution instance for that pre processing.

        Arguments
        ---------
        exec_id : str
            Execution id

        Raises
        ------
        PreprocessingError
            If the user tries to get a execution from a Sync pre processing 

        Example
        -------
        >>> preprocessing.get_preprocessing_execution('1')
        """
        if self.operation.lower() == 'async':
            return NeomarilExecution(self.preprocessing_id, 'AsyncPreprocessing', exec_id=exec_id, password=self.__credentials, 
                                     url=self.base_url, group_token=self.__token, group=self.group)
        else:
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
        url = f"{self.base_url}/preprocessing/status/{self.group}/{self.preprocessing_id}"
        response = requests.get(url, headers={'Authorization': 'Bearer ' + self.__credentials})
        if response.status_code < 300:
            return response.json()
        else:
            raise PreprocessingError(response.text)
        
class NeomarilPreprocessingClient(BaseNeomarilClient):

    def __init__(self, password:Optional[str]=None, url:str='https://neomaril.staging.datarisk.net/') -> None:

        super().__init__(password=password, url=url)
        load_dotenv()
        self.__credentials = os.getenv('NEOMARIL_TOKEN') if os.getenv('NEOMARIL_TOKEN') else password

    def __get_preprocessing_status(self, preprocessing_id:str, group:str) -> dict:
        """
        Gets the status of the pre processing with the hash equal to `preprocessing_id`

        Parameters
        ----------
        group : str
            Group the pre processing is inserted
        preprocessing_id : str
            Pre processing id (hash) from the pre processing being searched

        Raises
        ------
        PreprocessingError
            Pre processing unavailable

        Returns
        -------
        dict
            The pre processing status and a message if the status is 'Failed'
        """

        url = f"{self.base_url}/preprocessing/status/{group}/{preprocessing_id}"
        response = requests.get(url, headers={'Authorization': 'Bearer ' + self.__credentials})
        if response.status_code not in [200, 410]:
            raise PreprocessingError(f'Preprocessing "{preprocessing_id}" not found')
        
        return response.json()

    def get_preprocessing(self, preprocessing_id:str, group:str="datarisk", group_token:Optional[str]=None, wait_for_ready:bool=True) -> NeomarilPreprocessing:
        """
        Access a pre processing using its id

        Arguments
        ---------
        preprocessing_id : str
            Pre processing id (hash) that needs to be accessed
        group : str
            Group the pre processing is inserted. Default is 'datarisk' (public group)
        group_token : str, optional
            Token for executing the pre processing (show when creating a group). It can be informed when getting the preprocessing or when running predictions, or using the env variable NEOMARIL_GROUP_TOKEN
        wait_for_ready : bool
            If the pre processing is being deployed, wait for it to be ready instead of failing the request. Defaults to True

        Raises
        ------
        PreprocessingError
            Pre processing unavailable
        ServerError
            Unknown return from server

        Returns
        -------
        NeomarilPreprocessing
            A NeomarilPreprocessing instance with the pre processing hash from `preprocessing_id`
        
        Example
        -------
        >>> preprocessing.get_preprocessing(preprocessing_id='M9c3af308c754ee7b96b2f4a273984414d40a33be90242908f9fc4aa28ba8ec4', group='ex_group')
        """        
        try:
            response = self.__get_preprocessing_status(preprocessing_id, group)
        except KeyError:
            raise PreprocessingError("Preprocessing not found")
        
        status = response['Status']
        
        if status == 'Building':
            if wait_for_ready:
                print('Wating for deploy to be ready.', end='')
                while status == 'Building':
                    response = self.__get_preprocessing_status(preprocessing_id, group)
                    status = response['Status']
                    print('.', end='', flush=True)
                    sleep(10)
            else:
                logger.info("Returning preprocessing, but preprocessing is not ready.")
                NeomarilPreprocessing(preprocessing_id, password=self.__credentials, group=group, url=self.base_url, group_token=group_token)
            
        if status in ['Disabled', 'Ready']:
            raise PreprocessingError(f'Preprocessing "{preprocessing_id}" unavailable (disabled or deploy process is incomplete)')
        elif status == 'Failed':
            logger.error(str(response['Message']))
            raise PreprocessingError(f'Preprocessing "{preprocessing_id}" deploy failed, so preprocessing is unavailable.')
        elif status == 'Deployed': 
            logger.info(f'Preprocessing {preprocessing_id} its deployed. Fetching preprocessing.')
            return NeomarilPreprocessing(preprocessing_id, password=self.__credentials, group=group, url=self.base_url, group_token=group_token)
        else:
            raise ServerError('Unknown preprocessing status: ',status)
        
    def search_preprocessing(self, name:Optional[str]=None, state:Optional[str]=None, 
                                        group:Optional[str]=None, only_deployed:bool=False) -> list:
        """
        Search for pre processing using the name of the pre processing

        Arguments
        ---------
        name : str, optional
            Text that its expected to be on the pre processing name. It runs similar to a LIKE query on SQL
        state : str, optional
            Text that its expected to be on the state. It runs similar to a LIKE query on SQL
        group : str, optional
            Text that its expected to be on the group name. It runs similar to a LIKE query on SQL
        only_deployed : bool, optional
            If its True, filter only pre processing ready to be used (status == "Deployed"). Defaults to False

        Raises
        ------
        ServerError
            Unexpected server error

        Returns
        -------
        list
            List with the pre processing data, it can works like a filter depending on the arguments values
        Example
        -------
        >>> client.search_preprocessing(group='ex_group', only_deployed=True)
        """
        url = f"{self.base_url}/preprocessing/search"

        query = {}

        if name:
            query['name'] = name

        if state:
            query['state'] = state

        if group:
            query['group'] = group

        if only_deployed:
            query['state'] = 'Deployed'

        response = requests.get(url, params=query, headers={'Authorization': 'Bearer ' + self.__credentials})
        
        if response.status_code == 200:
            results = response.json()['Results']
            
            return results
        
        else:
            raise ServerError('Unexpected server error: ', response.text)

    def get_logs(self, preprocessing_id, start:Optional[str]=None, end:Optional[str]=None, routine:Optional[str]=None, type:Optional[str]=None):
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
        return self._logs(url, self.__credentials, start=start, end=end, routine=routine, type=type)

    def __upload_preprocessing(self, preprocessing_name:str, preprocessing_reference:str, source_file:str, 
                            requirements_file:str, schema:Optional[Union[str, dict]]=None, 
                            group:Optional[str]=None, extra_files:Optional[list]=None, env:Optional[str]=None, 
                            python_version:str='3.8', operation:str='Sync', input_type:str=None) -> str:
        """
        Upload the files to the server

        Arguments
        ---------
        preprocessing_name : str
            The name of the pre processing, in less than 32 characters
        preprocessing_reference : str
            The name of the scoring function inside the source file
        source_file : str
            Path of the source file. The file must have a scoring function that accepts two parameters: data (data for the request body of the preprocessing) and preprocessing_path (absolute path of where the file is located)
        requirements_file : str
            Path of the requirements file. The packages versions must be fixed eg: pandas==1.0
        schema : Union[str, dict], optional
            Path to a JSON or XML file with a sample of the input for the entrypoint function. A dict with the sample input can be send as well
        group : str, optional
            Group the pre processing is inserted. If None the server uses 'datarisk' (public group)
        extra_files : list, optional
            A optional list with additional files paths that should be uploaded. If the scoring function refer to this file they will be on the same folder as the source file
        env : str, optional
            Flag that choose which environment (dev, staging, production) of Neomaril you are using. Default is True
        python_version : str, optional
            Python version for the pre processing environment. Avaliable versions are 3.7, 3.8, 3.9, 3.10. Defaults to '3.8'
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
            The new pre processing id (hash)
        """
        
        url = f"{self.base_url}/preprocessing/register/{group}"
        
        file_extesions = {'py': 'script.py', 'ipynb': "notebook.ipynb"}
        
     
        upload_data = [
            ("source", (file_extesions[source_file.split('.')[-1]], open(source_file, "r"))),
            ("requirements", ("requirements.txt", open(requirements_file, "r")))
        ]

        if operation=="Sync":
            input_type = "json"
            if schema:
                if isinstance(schema, str):
                    schema_file = open(schema, "r")
                elif isinstance(schema, dict):
                    schema_file = json.dumps(schema)
                    
                upload_data.append(("schema", ("schema.json", schema_file)))
            else:
                raise InputError("Schema file is mandatory for Sync preprocessing")

        else:
            if input_type == 'json|csv|parquet':
                raise InputError("Choose a input type from "+input_type)

        if env:
            upload_data.append(("env", (".env", env)))
        
        if extra_files:
            extra_data = [('extra', (c.split('/')[-1], open(c, "r"))) for c in extra_files]
            
            upload_data += extra_data
            
        form_data = {'name': preprocessing_name, 'script_reference': preprocessing_reference, 'operation': operation, 'python_version': "Python"+python_version.replace('.', '')}
            
        response = requests.post(url, data=form_data, files=upload_data, headers={'Authorization': 'Bearer ' + self.__credentials})
        
        if response.status_code == 201:
            data = response.json()
            preprocessing_id = data["Hash"]
            logger.info(f'{data["Message"]} - Hash: "{preprocessing_id}"')
            return preprocessing_id
        else:
            logger.error('Upload error: ' + response.text)
            raise InputError('Invalid parameters for preprocessing creation')

    def __host_preprocessing(self, operation:str, preprocessing_id:str, group:str) -> None:
        """
        Builds the preprocessing execution environment

        Arguments
        ---------
        operation : str
            The pre processing operation type (Sync or Async)
        preprocessing_id : str
            The uploaded pre processing id (hash)
        group : str
            Group the pre processing is inserted. Default is 'datarisk' (public group)

        Raises
        ------
        InputError
            Some input parameters its invalid
        """
        
        url = f"{self.base_url}/preprocessing/{operation}/host/{group}/{preprocessing_id}"
        if operation == 'sync':
            url = url.replace('localhost:7070', 'localhost:7071')

        response = requests.get(url, headers={'Authorization': 'Bearer ' + self.__credentials})
        if response.status_code == 202:
            logger.info(f"Preprocessing host in process - Hash: {preprocessing_id}")
        else:
            logger.error(response.text)
            raise InputError('Invalid parameters for preprocessing creation')

    def create(self, preprocessing_name:str, preprocessing_reference:str, source_file:str, 
                                     requirements_file:str, schema:Optional[Union[str, dict]]=None, 
                                     group:str=None, extra_files:Optional[list]=None, env:Optional[str]=None,
                                     python_version:str='3.8', operation='Sync', input_type:str='json|csv|parquet', 
                                     wait_for_ready:bool=True)-> Union[NeomarilPreprocessing, str]:
        """
        Deploy a new preprocessing to Neomaril.

        Arguments
        ---------
        preprocessing_name : str
            The name of the pre processing, in less than 32 characters
        preprocessing_reference : str
            The name of the scoring function inside the source file
        source_file : str
            Path of the source file. The file must have a scoring function that accepts two parameters: data (data for the request body of the preprocessing) and preprocessing_path (absolute path of where the file is located)
        requirements_file : str
            Path of the requirements file. The packages versions must be fixed eg: pandas==1.0
        schema : Union[str, dict]
            Path to a JSON or XML file with a sample of the input for the entrypoint function. A dict with the sample input can be send as well. Mandatory for Sync preprocessing
        group : str
            Group the pre processing is inserted. Default to 'datarisk' (public group)
        extra_files : list, optional
            A optional list with additional files paths that should be uploaded. If the scoring function refer to this file they will be on the same folder as the source file
        env : str, optional
            Flag that choose which environment (dev, staging, production) of Neomaril you are using. Default is True
        python_version : str, optional
            Python version for the pre processing environment. Avaliable versions are 3.7, 3.8, 3.9, 3.10. Defaults to '3.8'
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
            Returns the new preprocessing, if wait_for_ready=True runs the deploy process synchronously. If its False, returns nothing after sending all the data to server and runs the deploy asynchronously
        
        Example
        -------
        >>> preprocessing = client.create('Pre processing Example Sync', 'score',  './samples/syncPreprocessing/app.py', './samples/syncPreprocessing/'preprocessing.pkl', './samples/syncPreprocessing/requirements.txt','./samples/syncPreprocessing/schema.json', group=group, operation="Sync")
        """
        
        if python_version not in ['3.7', '3.8', '3.9', '3.10']:
            raise InputError('Invalid python version. Avaliable versions are 3.7, 3.8, 3.9, 3.10')
        
        if group:
            group = group.lower().strip().replace(" ", "_").replace(".", "_").replace("-", "_")

            groups = [g["Name"] for g in self.list_groups()]

            if group not in groups:

                raise GroupError('Group dont exist. Create a group first.')

        else:
            group = 'datarisk'
            logger.info("Group not informed, using default 'datarisk' group")
        
        preprocessing_id = self.__upload_preprocessing(preprocessing_name, preprocessing_reference, source_file, 
                                                requirements_file, schema=schema, group=group, extra_files=extra_files,
                                                python_version=python_version,env=env, operation=operation,
                                                input_type=input_type)
                
        self.__host_preprocessing(operation.lower(), preprocessing_id, group)
        
        return self.get_preprocessing(preprocessing_id, group, wait_for_ready=wait_for_ready)

    def get_execution(self, preprocessing_id:str, exec_id:str, group:Optional[str]=None) -> NeomarilExecution:
        """
        Get a execution instace (Async pre processing only).

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
        return self.get_preprocessing(preprocessing_id=preprocessing_id, group=group).get_preprocessing_execution(exec_id)
