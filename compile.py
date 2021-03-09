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
    api = API(**api)
    print(f'compiling {api.endpoint}...')
    # resave(api)
    exf = f"--extra-files {api.extrafiles}" if 'extrafiles' in api else ''
    os.system(
        f'torch-model-archiver --model-name {api.endpoint} --model-file models/{api.endpoint}.py --version {api.version} {exf} --export-path mar --serialized-file bin/{api.endpoint}.pth --handler {api.handler} --force'
    )