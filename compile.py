import sys,yaml,os,importlib,inspect,torch

models = None
with open( sys.argv[1] if len(sys.argv) > 1 else 'api.yaml','r' ) as stream:
    try:
        models = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)
        exit()

assert models != None, 'Oooops!'

class API(dict):
    def __init__(self,**kwargs):
        super(self.__class__,self).__init__(**kwargs)
        for k,v in kwargs.items():setattr(self,k,v)

def resave(api):
    module = importlib.import_module(f'models.{api.endpoint}')
    kls = inspect.getmembers(module, inspect.isclass)[0][1]
    obj = kls()
    state = torch.load(f'bin/{api.endpoint}.pt' )
    obj.load_state_dict( state )
    torch.jit.script( obj ).save( f'bin/{api.endpoint}.jit.pt' )

for api in models:
    print(f'compiling {api.endpoint}...')
    api = API(**api)
    # resave(api)
    os.system(
        f'torch-model-archiver --model-name {api.endpoint} --model-file models/{api.endpoint}.py --version {api.version} --extra-files indexes/{api.endpoint}/index_to_name.json --export-path mar --serialized-file bin/{api.endpoint}.pt --handler {api.handler} --force'
    )