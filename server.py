import logging

from flask import Flask
from flask_restful import Api, Resource

from main import IQueryable, LoadModel, process_args

app = Flask(__name__)
api = Api(app)


# resource_fields = {
#     'task':   fields.String,
#     'uri':    fields.Url('todo_ep')
# }

class Articles(Resource):

    def __init__(self, queryable: IQueryable = LoadModel()) -> None:
        self.queryable = queryable

    # @marshal_with(resource_fields)
    def get(self, query: str):
        articles = self.queryable.find_top_n_articles(query)
        return articles


api.add_resource(Articles, '/<query>')  # Route_1

if __name__ == '__main__':
    args = process_args()
    logging.info("*** Start running server ***")
    app.run(debug=args.debug, port=args.port)
