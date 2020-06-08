from tornado.web import RequestHandler
import sys, logging, json, uuid
sys.path.append("../")
from utility import format_response


class RecommendationHandler(RequestHandler):

    def initialize(self, item_recommendation):
        self.item_recommendation = item_recommendation

    def _send_reply(self, response):
        logging.info("Response: {}".format(response))
        self.write(response)

    def get(self):
        params = json.loads(self.request.body)

        if (("user_id" not in params) or
            (not isinstance(params.get("user_id"), str))):
            message = "a valid user_id string has to be provided"
            self._send_reply(format_response(message=message))

        else:
            user_id = str(params["user_id"])
            if (("n_items_recommended" not in params) or
                (params.get("n_items_recommended") is None)):
                n_items_recommended = 10    # default value
            else:
                n_items_recommended = int(params["n_items_recommended"])

            request_id = str(uuid.uuid1())
            logging.info("User ID: %s" % str(user_id))
            logging.info("Num items to be recommended: %d" % (n_items_recommended))

            logging.info("Recommendation begins...")
            recommended_items, probs = self.item_recommendation.recommend(
                user_id, n_items_recommended)
            probs = [float(x) for x in probs]

            response = format_response(recommended_items=recommended_items,
                                       pred_prob=probs)
            response["request_id"] = request_id
            response["user_id"] = user_id
            response["n_items_recommended"] = n_items_recommended
            self._send_reply(response)
