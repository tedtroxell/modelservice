

from base import BaseHandler


class BertTokenizer(BaseHandler):
    """[summary]

    :param BaseHandler: [description]
    :type BaseHandler: [type]
    :return: [description]
    :rtype: [type]
    """    

    fn = None

    def _initialize(self,context):
        from transformers import AutoTokenizer
        self.fn = AutoTokenizer.from_pretrained('bert-base-uncased')

    def preprocess(self,data): return self.fn(data, return_tensors="pt")['input_ids']