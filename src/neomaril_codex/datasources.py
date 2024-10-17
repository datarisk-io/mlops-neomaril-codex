import json
from http import HTTPStatus
from typing import Union

import requests

from neomaril_codex.__utils import parse_json_to_yaml, refresh_token
from neomaril_codex.base import BaseNeomaril, BaseNeomarilClient
from neomaril_codex.exceptions import (
    AuthenticationError,
    CredentialError,
    DatasetNotFoundError,
    InputError,
    ServerError,
)
from neomaril_codex.logger_config import get_logger

logger = get_logger()


class NeomarilDataSourceClient(BaseNeomarilClient):
    """
    Class for client for manage datasources

    Attributes
    ----------
    login : str
        Login for authenticating with the client.
        You can also use the env variable NEOMARIL_USER to set this
    password : str
        Password for authenticating with the client.
        You can also use the env variable NEOMARIL_PASSWORD to set this
    url : str
        URL to Neomaril Server.
        Default value is https://neomaril.staging.datarisk.net,
        use it to test your deployment first before changing to production.
        You can also use the env variable NEOMARIL_URL to set this

    Raises
    ----------
    ServerError
        Database produced an unexpected error.
    AuthenticationError
        If user is not in the master group.
    CredentialError
        If the Cloud Credential is Invalid
    """

    def register_datasource(
        self,
        *,
        datasource_name: str,
        provider: str,
        cloud_credentials: Union[dict, str],
        group: str,
    ):
        """
        Register the user cloud credentials to allow Neomaril to use the provider to download the datasource.

        Attributes
        ----------
        group : str
            Name of the group where we will search the datasources.
        datasource_name : str
            Name given previously to the datasource.
        provider : str ("Azure" | "AWS" | "GCP")
        cloud_credentials : str | Union[dict,str]
            Path or dict to a JSON with the credentials to access the provider.

        Returns
        ----------
        NeomarilDataSource
            A NeomarilDataSource object

        Example
        -------
        >>> client.register_datasource(
        >>>     datasource_name='MyDataSourceName',
        >>>     provider='GCP',
        >>>     cloud_credentials='./gcp_credentials.json',
        >>>     group='my_group'
        >>> )
        """

        datasource = NeomarilDataSource(
            datasource_name=datasource_name,
            provider=provider,
            group=group,
            login=self.credentials[0],
            password=self.credentials[1],
            url=self.base_url,
        )

        url = f"{self.base_url}/datasource/register/{group}"

        if isinstance(cloud_credentials, dict):
            credential_path = self.credentials_to_json(cloud_credentials)

            with open(credential_path, encoding="utf-8", mode="w") as credential_file:
                json.dump(datasource.credentials, credential_file)
        else:
            credential_path = cloud_credentials

        form_data = {"name": datasource_name, "provider": provider}

        files = {
            "credentials": (
                cloud_credentials.split("/")[-1],
                open(credential_path, "rb"),
            )
        }
        token = refresh_token(*self.credentials, self.base_url)

        response = requests.post(
            url=url,
            data=form_data,
            files=files,
            headers={"Authorization": "Bearer " + token},
            timeout=60,
        )

        if response.status_code == 200:
            logger.info(response.json().get("Message"))
            return datasource
        elif response.status_code == 400:
            del datasource
            if "Database produced an unexpected error" in response.text:
                raise ServerError("Database produced an unexpected error")
            if "not in the master group" in response.text:
                raise AuthenticationError("User is not in the master group.")
        raise CredentialError("Cloud Credential Error")

    def credentials_to_json(self, input_data: dict):
        """
        Transform dict to json.

        Args:
            input_data (dict): A dictionary to save.
        """
        path = "./credentials.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(input_data, f)
        return path

    def list_datasources(self, *, provider: str, group: str):
        """
        List all datasources of the group with this provider type.

        Attributes
        ----------
        group : str
            Name of the group where we will search the datasources
        provider : str ("Azure" | "AWS" | "GCP")

        Returns
        ----------
        list
            A list of datasources information.

        Example
        -------
        >>> client.list_datasources(provider='GCP', group='my_group')
        """
        url = f"{self.base_url}/datasource/list?group={group}&provider={provider}"

        token = refresh_token(*self.credentials, self.base_url)

        response = requests.get(
            url=url, headers={"Authorization": "Bearer " + token}, timeout=60
        )
        if response.status_code == 200:
            results = response.json().get("Results")
            return results

        formatted_msg = parse_json_to_yaml(response.json())

        if response.status_code == 401:
            logger.error("Login or password are invalid, please check your credentials.")
            raise AuthenticationError("Login not authorized.")

        if response.status_code >= 500:
            logger.error("Server is not available. Please, try it later.")
            raise ServerError("Server is not available!")

        logger.error(f"Something went wrong...\n{formatted_msg}")
        raise InputError("Bad Input. Client error")

    def get_datasource(self, *, datasource_name: str, provider: str, group: str):
        """
        Get a NeomarilDataSource to make datasource operations.

        Attributes
        ----------
        group : str
            Name of the group where we will search the datasources
        datasource_name : str
            Name given previously to the datasource.
        provider : str ("Azure" | "AWS" | "GCP")

        Returns
        ----------
        NeomarilDataSource
            A NeomarilDataSource object

        Example
        -------
        >>> client.get_datasource(datasource_name='MyDataSourceName', provider='GCP', group='my_group')
        """
        datasources = self.list_datasources(provider=provider, group=group)
        for datasource in datasources:
            if datasource_name == datasource.get("Name"):
                return NeomarilDataSource(
                    datasource_name=datasource.get("Name"),
                    provider=datasource.get("Provider"),
                    group=datasource.get("Group"),
                    login=self.credentials[0],
                    password=self.credentials[1],
                    url=self.base_url,
                )
        raise InputError("Datasource not found!")


class NeomarilDataSource(BaseNeomaril):
    """
    Class to operate actions in a datasource.

    Attributes
    ----------
    login : str
        Login for authenticating with the client.
        You can also use the env variable NEOMARIL_USER to set this
    password : str
        Password for authenticating with the client.
        You can also use the env variable NEOMARIL_PASSWORD to set this
    url : str
        URL to Neomaril Server.
        Default value is https://neomaril.staging.datarisk.net,
        use it to test your deployment first before changing to production.
        You can also use the env variable NEOMARIL_URL to set this
    datasource_name : str
        Name given previously to the datasource.
    provider : str
        Providers name, currently, Neomaril supports:
        Azure Blob Storage as "Azure",
        AWS S3 as "AWS",
        Google GCP as "GCP".
    group : str
        Name of the group where we will search the datasources
    """

    def __init__(
        self,
        *,
        datasource_name: str,
        provider: str,
        group: str,
        login: str,
        password: str,
        url: str,
    ) -> None:
        super().__init__(login, password, url)
        self.datasource_name = datasource_name
        self.provider = provider
        self.group = group

    def import_dataset(
        self, *, dataset_uri: str, dataset_name: str, force: bool = False
    ):
        """
        Import a dataset inside a datasource.

        Attributes
        ----------
        dataset_uri : str
            Datasource cloud URI path.
        dataset_name : str
            The dataset defined name
        force : bool
            Optional[boolean]: when it is true it will force the datasource download from the provider.

        Returns
        ----------
        NeomarilDataset
            A NeomarilDataset with the identifier as dataset_hash.

        Raises
        ----------
        InputError
            If any data sent is invalidated on server.

        Example
        -------
        >>> dataset = datasource.import_dataset(
        >>>     dataset_uri='https://storage.cloud.google.com/your-name/file.csv',
        >>>     dataset_name='meudataset'
        >>> )
        """
        form_data = {"uri": dataset_uri, "name": dataset_name}

        force = str(force).lower()

        token = refresh_token(*self.credentials, self.base_url)
        url = f"{self.base_url}/datasource/import/{self.group}/{self.datasource_name}?force={force}"
        response = requests.post(
            url=url,
            data=form_data,
            headers={"Authorization": "Bearer " + token},
            timeout=60,
        )

        if response.status_code == 200:
            dataset_hash = response.json().get("ExternalHash")

            dataset = NeomarilDataset(
                dataset_hash=dataset_hash,
                dataset_name=dataset_name,
                datasource_name=self.datasource_name,
                group=self.group,
                login=self.credentials[0],
                password=self.credentials[1],
                url=self.base_url,
            )
            return dataset

        formatted_msg = parse_json_to_yaml(response.json())

        if response.status_code == 401:
            logger.error("Login or password are invalid, please check your credentials.")
            raise AuthenticationError("Login not authorized.")

        if response.status_code >= 500:
            logger.error("Server is not available. Please, try it later.")
            raise ServerError("Server is not available!")

        logger.error(f"Something went wrong...\n{formatted_msg}")
        raise InputError("Bad Input. Client error")

    def list_datasets(self, *, origin: str = None):
        """
        List datasets from datasources.

        Attributes
        ----------
        origin : Optional[str]
            Can be an EHash or a SHash

        Returns
        ----------
        list
            A list of datasets information.

        Example
        -------
        >>> dataset.list_datasets()
        """
        url = f"{self.base_url}/datasets/list?datasource={self.datasource_name}"
        url += f"origin={origin}" if origin else ""

        token = refresh_token(*self.credentials, self.base_url)

        response = requests.get(
            url=url, headers={"Authorization": "Bearer " + token}, timeout=60
        )

        return response.json().get("Results")

    def delete(self):
        """
        Delete the datasource on neomaril.
        Pay attention when doing this action, it is irreversible!
        Returns
        -------
        None

        Example
        -------
        >>> datasource.delete()
        """
        url = f"{self.base_url}/datasources/{self.group}/{self.datasource_name}"

        token = refresh_token(*self.credentials, self.base_url)
        response = requests.delete(
            url=url, headers={"Authorization": "Bearer " + token}, timeout=60
        )
        logger.info(response.json().get("Message"))

    def get_dataset(self, *, dataset_hash: str, origin: str = None):
        """
        Get a NeomarilDataset to make dataset operations.

        Attributes
        ----------
        dataset_hash : str
            Name given previously to the datasource.
        origin : Optional[str]
            Can be an EHash or a SHash

        Returns
        ----------
        NeomarilDataset
            A NeomarilDataset with the identifier as dataset_hash.

        Raises
        ----------
        DatasetNotFoundError
            When the dataset_hash input was not found

        Example
        ----------
        >>> dataset = datasource.get_dataset(dataset_hash='D589654eb26c4377b0df646e7a5675fa3c7d49575e03400b940dd5363006fc3a')
        """
        datasources = self.list_datasets(origin=origin)

        for datasource in datasources:
            if dataset_hash == datasource.get("Id"):
                return NeomarilDataset(
                    dataset_hash=datasource.get("Id"),
                    dataset_name=datasource.get("Name"),
                    datasource_name=self.datasource_name,
                    group=self.group,
                    login=self.credentials[0],
                    password=self.credentials[1],
                    url=self.base_url,
                )
        raise DatasetNotFoundError("Dataset hash not found!")


class NeomarilDataset(BaseNeomaril):
    """
    Class to operate actions in a dataset.

    Attributes
    ----------
    login : str
        Login for authenticating with the client.
        You can also use the env variable NEOMARIL_USER to set this
    password : str
        Password for authenticating with the client.
        You can also use the env variable NEOMARIL_PASSWORD to set this
    url : str
        URL to Neomaril Server.
        Default value is https://neomaril.staging.datarisk.net,
        use it to test your deployment first before changing to production.
        You can also use the env variable NEOMARIL_URL to set this
    dataset_hash : str
        The hash that identify the dataset
    dataset_name : str
        The dataset defined name
    datasource_name : str
        Name given previously to the datasource.
    group : str
        Name of the group where we will search the datasource
    """

    def __init__(
        self,
        *,
        dataset_hash: str,
        dataset_name: str,
        datasource_name: str,
        group: str,
        login: str,
        password: str,
        url: str,
    ) -> None:
        super().__init__(login, password, url)
        self.group = group
        self.dataset_hash = dataset_hash
        self.dataset_name = dataset_name
        self.datasource_name = datasource_name

    def get_status(self):
        """
        Get dataset status.

        Returns
        ----------
        dict
            when success
            {
                status : 'Succeeded',
                log : ''
            }
            when failed
            {
                "status": "Failed",
                "log": "UnexpectedError\n  \"Azure Request error! Message: Service request failed.\nStatus: 403 (Server failed to authenticate the request. Make sure the value of Authorization header is formed correctly including the signature.)\nErrorCode: AuthenticationFailed\n\nHeaders:\nTransfer-Encoding: chunked\nServer: Microsoft-HTTPAPI/2.0\nx-ms-request-id: xxxxx\nx-ms-error-code: AuthenticationFailed\nDate: Wed, 24 Jan 2024 12:00:36 GMT\n\""
            }

        Raises
        ----------
        DatasetNotFound
            When the dataset was not found

        Example
        ----------
        >>> dataset.get_status()
        """
        url = f"{self.base_url}/datasets/status/{self.group}/{self.dataset_hash}"

        token = refresh_token(*self.credentials, self.base_url)

        response = requests.get(
            url=url, headers={"Authorization": "Bearer " + token}, timeout=60
        )

        if response.status_code == 200:
            status = response.json().get("Status")
            log = response.json().get("Log")

            return {"status": status, "log": log}

        formatted_msg = parse_json_to_yaml(response.json())

        if response.status_code == 401:
            logger.error("Login or password are invalid, please check your credentials.")
            raise AuthenticationError("Login not authorized.")

        if response.status_code >= 500:
            logger.error("Server is not available. Please, try it later.")
            raise ServerError("Server is not available!")

        logger.error(f"Something went wrong...\n{formatted_msg}")
        raise DatasetNotFoundError("Dataset not found")

    def delete(self):
        """
        Delete the dataset on neomaril.
        Pay attention when doing this action, it is irreversible!

        Example
        ----------
        >>> dataset.delete()
        """
        url = f"{self.base_url}/datasets/{self.group}/{self.dataset_hash}"

        token = refresh_token(*self.credentials, self.base_url)
        response = requests.delete(
            url=url, headers={"Authorization": "Bearer " + token}, timeout=60
        )
        if response.status_code == 200:
            logger.info(response.json().get("Message"))
            return HTTPStatus.OK

        formatted_msg = parse_json_to_yaml(response.json())

        if response.status_code == 401:
            logger.error("Login or password are invalid, please check your credentials.")
            raise AuthenticationError("Login not authorized.")

        if response.status_code >= 500:
            logger.error("Server is not available. Please, try it later.")
            raise ServerError("Server is not available!")

        logger.error(f"Something went wrong...\n{formatted_msg}")
        raise DatasetNotFoundError("Dataset not found")
