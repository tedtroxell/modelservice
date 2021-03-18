import sys,yaml,os

models = None
with open( sys.argv[1] if len(sys.argv) > 1 else 'models.yaml','r' ) as stream:
    try:
        models = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)
        exit()

assert models != None, 'Oooops!'
for mdl in models:
    print(f'Downloading {mdl["mname"]}...')
    os.system( f'wget {mdl["dl_url"]} -O bin/{mdl["mname"]}.pth' )
    try:os.mkdir(f'indexes/{mdl["mname"]}/')
    except:pass
    if "index" in mdl:os.system( f'wget {mdl["index"]} -O indexes/{mdl["mname"]}/index_to_name.json' )
    if "mdl" in mdl:os.system( f'wget {mdl["mdl"]} -O models/{mdl["mname"]}.py' )


