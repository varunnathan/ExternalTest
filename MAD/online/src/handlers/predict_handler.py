from tornado.web import RequestHandler
import sys, logging, json, uuid

sys.path.append("../")
from utility import format_response

sys.path.append("/Users/varunn/Documents/ExternalTest/MAD/offline/src/")
from constants import CLASS_LABELS


class PredictionHandler(RequestHandler):

    def initialize(self, model):
        self.model = model

    def _send_reply(self, response):
        logging.info("Response: {}".format(response))
        self.write(response)

    def get(self):
        params = json.loads(self.request.body)

        if "segment" not in params:
            message = "segment has to be provided in the input json"
            self._send_reply(format_response(message=message))
        elif "request_type" not in params:
            message = "request_type has to be provided in the input json"
            self._send_reply(format_response(message=message))
        else:
            segment = params["segment"]
            request_type = params["request_type"]
            if segment not in ("LT20", "GE20"):
                message = "segment has to be one of ('LT20', 'GE20')"
                self._send_reply(format_response(message=message))
            elif request_type not in ("embeddings", "probability"):
                message = "request_type has to be one of ('embeddings', 'probability')"
                self._send_reply(format_response(message=message))
            else:
                request_id = str(uuid.uuid1())
                logging.info("Segment: %s" % str(segment))
                logging.info("request_type: %s" % str(request_type))

                logging.info("Prediction begins...")
                if segment == 'LT20':
                    model = self.model.model_segLT20
                elif segment == 'GE20':
                    model = self.model.model_segGE20

                if request_type == 'embeddings':
                    kind = params["kind"]
                    mapped_id = int(params["mapped_id"])
                    embedding = self.model.get_embedding(
                        model, segment, kind, mapped_id)
                    response = format_response(embeddings=embedding)
                    response["kind"] = kind
                    response["mapped_id"] = mapped_id
                else:
                    input = {}
                    input['cat_feats'] = params['cat_feats']
                    input['numeric_feats'] = params['numeric_feats']
                    pred_prob = self.model.inference(model, input)
                    pred_class = [CLASS_LABELS[i] for i in
                                  pred_prob.argmax(axis=1)]
                    response = format_response(pred_prob=pred_prob.tolist(),
                                               pred_class=pred_class)

                response["request_id"] = request_id
                response["request_type"] = request_type
                response["segment"] = segment
                self._send_reply(response)
