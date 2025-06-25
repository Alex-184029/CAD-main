import os
import json

def parse_result(logfile, task_type):
    res = None
    if not os.path.join(logfile):
        return res
    if task_type == 'Door':
        res = parse_result_door(logfile)
    elif task_type == 'Window':
        res = parse_result_window(logfile)
    elif task_type == 'Wall':
        res = parse_result_wall(logfile)
    elif task_type == 'Area':
        res = parse_result_area(logfile)
    elif task_type == 'MatchLegend':
        pass
    else:
        print('task_type not supported.')
    return res

def parse_result_door(logfile):
    with open(logfile, 'r', encoding='utf-8') as f:
        data = json.load(f)
    data_ans = dict()
    data_ans['range'] = data['range']
    data_ans['single_arc_doors'] = data['doors']['single_arc_doors']
    data_ans['double_arc_doors'] = data['doors']['double_arc_doors']
    data_ans['slide_doors'] = data['doors']['slide_doors']
    return data_ans

def parse_result_window(logfile):
    with open(logfile, 'r', encoding='utf-8') as f:
        data = json.load(f)
    data_ans = dict()
    data_ans['range'] = data['range']
    data_ans['windows'] = data['windows']
    return data_ans

def parse_result_wall(logfile):
    with open(logfile, 'r', encoding='utf-8') as f:
        data = json.load(f)
    data_ans = dict()
    data_ans['range'] = data['range']
    data_ans['walls'] = data['walls']
    return data_ans

def parse_result_area(logfile):
    with open(logfile, 'r', encoding='utf-8') as f:
        data = json.load(f)
    data_ans = dict()
    data_ans['range'] = data['range']
    data_ans['rooms'] = data['rooms']
    return data_ans

def parse_bill(logfile):
    with open(logfile, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data['bill'] if 'bill' in data else None

