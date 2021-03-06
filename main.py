import logging

from flask import Flask
from flask_cors import CORS
from flask_restful import Api, Resource

from engine import IQueryable, LoadModel, process_args

try:
    import googleclouddebugger

    googleclouddebugger.enable(
        module='API server',
        version='1.0.6'
    )
except ImportError as e:
    print('Failed to import Google Cloud Debugger: %s' % str(e))
    pass

app = Flask(__name__)
api = Api(app)
# TODO make CORS more granual
CORS(app)


class Articles(Resource):

    def __init__(self, queryable: IQueryable = LoadModel()) -> None:
        self.queryable = queryable

    # @marshal_with(resource_fields)
    def get(self, query: str):
        articles = self.queryable.find_top_n_articles(query)
        return articles


# fixmeX this new change makes the client code break.
api.add_resource(Articles, '/v1/search/<query>')  # Route_1

if __name__ == '__main__':
    args = process_args()
    logging.info("*** Start running server ***")
    app.run(debug=args.debug, port=args.port, host=args.host)
