import redis
import argparse, json, time
from settings import *
import sys
sys.path.append("../../offline/src/")
from constants import *


TASK_INP_DCT = {"user-baseline-feats-segGE20",
                "user-baseline-feats-segLT20",
                "item-baseline-feats",
                "latest-epoch-for-user",
                "latest-epoch-for-item",
                "user-brand-mapping-LT20"}


def init_db():
    db = redis.StrictRedis(
        host=DB_HOST,
        port=DB_PORT,
        db=DB_NO)
    return db


def define_categories():

    tasks = [
        "user-baseline-feats-segGE20", "user-baseline-feats-segLT20",
        "item-baseline-feats", "latest-epoch-for-user-segGE20",
        "latest-epoch-for-user-segLT20", "latest-epoch-for-item",
        "user-brand-mapping-LT20", "user2idx-segGE20", "user2idx-segLT20"]

    inp_lsts = [[MAPPED_USER_BASELINE_FEATS_SEGGE20_1_FN,
                 MAPPED_USER_BASELINE_FEATS_SEGGE20_2_FN,
                 MAPPED_USER_BASELINE_FEATS_SEGGE20_3_FN],
                [MAPPED_USER_BASELINE_FEATS_SEGLT20_1_FN,
                 MAPPED_USER_BASELINE_FEATS_SEGLT20_2_FN,
                 MAPPED_USER_BASELINE_FEATS_SEGLT20_3_FN,
                 MAPPED_USER_BASELINE_FEATS_SEGLT20_4_FN,
                 MAPPED_USER_BASELINE_FEATS_SEGLT20_5_FN],
                [MAPPED_ITEM_BASELINE_FEATS_1_FN,
                 MAPPED_ITEM_BASELINE_FEATS_2_FN,
                 MAPPED_ITEM_BASELINE_FEATS_3_FN],
                [USER2EPOCH_SEGGE20_1_FN, USER2EPOCH_SEGGE20_2_FN,
                 USER2EPOCH_SEGGE20_3_FN],
                [USER2EPOCH_SEGLT20_1_FN, USER2EPOCH_SEGLT20_2_FN,
                 USER2EPOCH_SEGLT20_3_FN, USER2EPOCH_SEGLT20_4_FN,
                 USER2EPOCH_SEGLT20_5_FN],
                [ITEM2EPOCH_1_FN, ITEM2EPOCH_2_FN, ITEM2EPOCH_3_FN],
                [USER_BRAND_MAPPING_SEGLT20_1_FN,
                 USER_BRAND_MAPPING_SEGLT20_2_FN,
                 USER_BRAND_MAPPING_SEGLT20_3_FN,
                 USER_BRAND_MAPPING_SEGLT20_4_FN],
                [FINAL_USER2IDX_SEGGE20_FN], [FINAL_USER2IDX_SEGLT20_FN]]

    task_inp_dct = dict(zip(tasks, inp_lsts))

    return task_inp_dct, tasks


def write_list_type_data_to_db(d_type, segment, db, pdtDict, num_iterations=100):

    if segment is None:
        segment = 'item'

    pipe = db.pipeline()
    n = 1
    start = time.time()
    for key in pdtDict.keys():
        if n % 10000 == 0:
            print('num keys completed: %d' % (n-1))
            print('time taken: %0.2f' % (time.time() - start))
        new_key = str(d_type) + '::' + str(segment) + '::' + str(key)
        for value in pdtDict[key]:
            pipe.rpush(new_key, value)
        n = n + 1
        if (n % num_iterations) == 0:
            pipe.execute()
            pipe = db.pipeline()

    print('done')


def write_hashlist_type_data_to_db(d_type, segment, db, pdtDict,
                                   num_iterations=100):
    pipe = db.pipeline()
    start = time.time()
    for key in pdtDict.keys():
        n = 1
        for sub_key in pdtDict[key].keys():
            if n % 10000 == 0:
                print('num keys completed: %d' % (n-1))
                print('time taken: %0.2f' % (time.time() - start))
            new_key = str(d_type) + '::' + str(segment) + '::' + str(key) + '::' + str(sub_key)
            for value in pdtDict[key][sub_key]:
                pipe.rpush(new_key, value)
            n = n + 1
            if (n % num_iterations) == 0:
                pipe.execute()
                pipe = db.pipeline()
    print('done')


def write_scalar_type_data_to_db(d_type, segment, db, pdtDict,
                                 num_iterations=100):
    if segment is None:
        segment = 'item'

    pipe = db.pipeline()
    n = 1
    start = time.time()
    for key in pdtDict.keys():
        if n % 10000 == 0:
            print('num keys completed: %d' % (n-1))
            print('time taken: %0.2f' % (time.time() - start))
        new_key = str(d_type) + '::' + str(segment) + '::' + str(key)
        pipe.set(new_key, pdtDict[key])
        n = n + 1
        if (n % num_iterations) == 0:
            pipe.execute()
            pipe = db.pipeline()

    print('done')


def _helper_for_moving_data(db, task, task_inp_dct):

    inp_fns = task_inp_dct[task]

    for i, inp_fn in enumerate(inp_fns):
        print('read file %s' % (inp_fn))
        df = json.load(open(inp_fn))
        print('\n')
        print('db write begins\n')
        if task == "user-baseline-feats-segGE20":
            write_list_type_data_to_db('baseline', 'GE20', db, df, 100)
        elif task == "user-baseline-feats-segLT20":
            write_list_type_data_to_db('baseline', 'LT20', db, df, 100)
        elif task == "item-baseline-feats":
            write_list_type_data_to_db('baseline', 'item', db, df, 100)
        elif task == "latest-epoch-for-user-segGE20":
            write_scalar_type_data_to_db('epoch', 'GE20', db, df, 100)
        elif task == "latest-epoch-for-user-segLT20":
            write_scalar_type_data_to_db('epoch', 'LT20', db, df, 100)
        elif task == "latest-epoch-for-item":
            write_scalar_type_data_to_db('epoch', 'item', db, df, 100)
        elif task == "user-brand-mapping-LT20":
            write_hashlist_type_data_to_db('brand', 'LT20', db, df, 100)
        elif task == "user2idx-segGE20":
            write_scalar_type_data_to_db('user2idx', 'GE20', db, df, 100)
        elif task == "user2idx-segLT20":
            write_scalar_type_data_to_db('user2idx', 'LT20', db, df, 100)


if __name__ == "__main__":
    print('moving data from json to redis to speed up data retrieval at inference time...\n')

    print('DB Init\n')
    redis_db = init_db()

    parser = argparse.ArgumentParser()
    task_inp_dct, choices = define_categories()
    parser.add_argument('task', choices=choices+["all"],
                        help="task to perform")
    args = parser.parse_args()
    task = args.task

    if task != "all":
        _helper_for_moving_data(redis_db, task, task_inp_dct)
    else:
        for task in choices:
            print('Task: %s' % (task))
            _helper_for_moving_data(redis_db, task, task_inp_dct)
