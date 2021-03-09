
from base import BaseHandler

class HandlerInterface(BaseHandler):

    def __init__(self):
        self.model = None
        self.mapping = None
        self.device = None
        self.initialized = False
        self.context = None
        self.manifest = None
        self.map_location = None
        self.explain = False
        self.target = 0

        self._initialize()

    @classmethod
    def _initialize(cls,*args,**kwargs):raise NotImplementedError(f'specific "_initialize" function not implemented for class {cls.__class__.__name__}')