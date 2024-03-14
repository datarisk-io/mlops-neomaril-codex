import requests, json, io
from cachetools.func import ttl_cache
from neomaril_codex.exceptions import *

def parse_dict_or_file(obj):
    if isinstance(obj, str):
        schema_file = open(obj, 'rb')
    elif isinstance(obj, dict):
        schema_file = io.StringIO()
        json.dump(obj, schema_file).seek(0)

    return schema_file

def parse_url(url):
    if url.endswith('/'):
        url = url[:-1]

    if not url.endswith('/api'):
        url = (url+'/api')
    return url

def try_login(login:str, password:str, base_url:str) -> bool:

    response = requests.get(f"{base_url}/health")

    server_status = response.status_code

    if server_status == 200:
      token = refresh_token(login, password, base_url)
      return token
    elif server_status == 401:
      raise AuthenticationError('Invalid credentials.')

    elif server_status >= 500:
      raise ServerError('Neomaril server unavailable at the moment.')
    
@ttl_cache
def refresh_token(login:str, password:str, base_url:str):
  respose = requests.post(f'{base_url}/login', data={'user': login, 'password': password})
  
  if respose.status_code == 200:
    return respose.json()['Token']
  else:
    raise AuthenticationError(respose.text)

def check_args(params: dict, required: list, def_vals: dict):
  try:
    return [params[p] for p in required] + [params.get(p, def_vals[p]) for p in def_vals.keys()]
  except KeyError as e:
    # INFO: https://peps.python.org/pep-0409/
    raise TypeError(f"missing 1 required argument: '{e.args[0]}'") from None

