from tornado.web import Application, RequestHandler
from tornado.options import define, options
from tornado.ioloop import IOLoop
from tornado.httpserver import HTTPServer
import tornado.options

import json, os, logging, time, sys
from datetime import timedelta

from model import Model
from recommendation import Recommendation

sys.path.append("../../offline/src/")
from constants import *

sys.path.append("/Users/varunn/Documents/ExternalTest/MAD/online/src/handlers/")
from health_handler import HealthHandler
from predict_handler import PredictionHandler
from recommend_handler import RecommendationHandler


# GLOBALS
if not os.path.exists(LOG_DIR):
    os.mkdir(LOG_DIR)

define('port', default=PORT, help='run on the given port', type=int)

logging.basicConfig(
    filename=LOG_FILE,
    level=logging.DEBUG
    )
logging.getLogger().addHandler(logging.StreamHandler())
formatter = logging.Formatter(
    '[%(levelname)1.1s %(asctime)s.%(msecs)d '
    '%(module)s:%(lineno)d] %(message)s',
    "%Y-%m-%d %H:%M:%S"
)  # creating own format
for handler in logging.getLogger().handlers:  # setting format for all handlers
    handler.setFormatter(formatter)


def main():
    start_time = time.time()
    tornado.options.parse_command_line()

    logging.info("Instantiate model class")
    model = Model()

    logging.info("Instantiate recommendation class")
    item_recommendation = Recommendation(model=model, n_candidates=3)

    handlers = [
        (r"/health", HealthHandler),
        (r"/api/v1/predict", PredictionHandler, dict(model=model)),
        (r"/api/v1/recommend", RecommendationHandler,
         dict(item_recommendation=item_recommendation))
    ]
    settings = dict(
        autoescape=None,  # tornado 2.1 backward compatibility
        gzip=True,
        xsrf_cookies=False,
        debug=False,
    )

    application = Application(handlers, **settings)
    http_server = HTTPServer(application)
    http_server.bind(options.port)
    logging.debug("starting http server")
    http_server.start()
    end_time = time.time()
    logging.info("Recommendation-Serv [Setup time]: {}".format(
        str(timedelta(seconds=(end_time - start_time)))))
    IOLoop.current().start()


if __name__ == '__main__':
    main()
