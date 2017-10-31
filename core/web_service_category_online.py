from flask import Flask, jsonify, abort, make_response, request
import time
import json
import pickle
import numpy
from category_recommendation import get_categories, extract_information_bulk
from prediction_pipeline import predict

app = Flask(__name__)

"""
curl -i -H "Accept: application/json" "203.116.23.21:5001/category/v1.1/items{'itemid':556324, 'shopid':67000}"
curl "http://203.116.23.21:5001/category/v1.1/items?itemid=556324&shopid=67000"
curl "http://203.116.23.21:5001/category/v1.1/items?itemid=556324&shopid=67000"
"""

@app.route('/category/v1.1/item', methods=['GET'])
def get_items():
    start_time = time.time()

    itemid = request.args.get('itemid')
    shopid = request.args.get('shopid')
    results, status = get_categories(int(itemid), int(shopid))

    print('time elapsed: %s' % (time.time() - start_time))
    return jsonify(results), status


@app.route('/category/v1.1/list_items', methods=['POST'])
def post_items():
    start_time = time.time()
    if not request.json or not 'items' in request.json:
        abort(400)

    # items is a list of tuples [(itemid, shopid), (itemid, shopid),... (itemid, shopid)]
    items = list()
    for item in request.json['items']:
        items.append(item)

    print '# of items in the list: ', len(items)
    if len(items) < 1:
        return jsonify({}), 301

    df_features = extract_information_bulk(items)
    print 'length of dataframe: ', len(df_features)
    # df_features.to_csv('debug_web_services_before.csv', index=None, encoding='utf-8')

    df_features['category_suggestions'] = df_features.apply(
        lambda row: predict(row['tokened_name'], row['tokened_description']), axis=1)

    df_results = df_features[['itemid', 'shopid', 'category_suggestions']]
    print('time elapsed: %s' % (time.time() - start_time))
    # df_results.to_csv('debug_web_services_after.csv', index=None, encoding='utf-8')
    result = df_results.to_dict(orient='records')

    return jsonify(result), 200
    # return json.dumps(result, ensure_ascii=False, default=default), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, threaded=True)
