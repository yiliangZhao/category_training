import requests
import json
import time

if __name__ == '__main__':
    url = 'http://203.116.138.25:5001/category/v1.1/list_items'
    headers = {'content-type': 'application/json'}
    data = {"items": [[500571, 67000], [556324, 67000], [982869, 67000], [1919083, 67000], [1919522, 67000],
                      [1920312, 67000], [1920598, 67000], [1920740, 67000], [1921333, 67000], [1922152, 67000],
                      [5580304, 1621000]]}
    list_items = list()
    with open('tem_list') as f:
        for line in f:
            itemid, shopid = line[:-1].split(',')
            list_items.append([int(itemid), int(shopid)])
    start_time = time.time()
    response = requests.post(url, data=json.dumps({"items": list_items}), headers=headers)
    result = response.json
    # print response.status_code, response.json()
    print('time elapsed: %s' % (time.time() - start_time))