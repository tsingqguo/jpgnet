import json
import os
def load_file_list_recursion(fpath, result):
    allfilelist = os.listdir(fpath)
    for file in allfilelist:
        filepath = os.path.join(fpath, file)
        if os.path.isdir(filepath):
            load_file_list_recursion(filepath, result)
        else:
            result.append(filepath)
            print(len(result))



def scan(input_path, out_put):
    result_list = []
    load_file_list_recursion(input_path, result_list)
    result_list.sort()

    print(result_list)

    with open(out_put, 'w') as j:
        json.dump(result_list, j)

#20 mask
# scan('J:/cv/DATA/mask/divide/test_60', 'J:/CV/workspace/final/RFR/data/60_mask.txt')

# scan('J:/cv/DATA/mask/divide/test_40', 'J:/CV/workspace/final/RFR/data/40_mask.txt')
# scan('J:/cv/DATA/mask/divide/test_20', 'J:/CV/workspace/final/RFR/data/20_mask.txt')
#
#
#
#

# scan('J:/cv/DATA/dunhuang/train/image', 'J:/CV/workspace/final/RFR/data/train.txt')
# scan('J:/cv/DATA/dunhuang/train/mask', 'J:/CV/workspace/final/RFR/data/dunhuang_mask.txt')


scan('J:/CV/DATA/acm_experient/mask_selected', 'J:/CV/workspace/final/RFR_fusion/data/selected_mask.txt')
scan('J:/CV/DATA/acm_experient/places2_100', 'J:/CV/workspace/final/RFR_fusion/data/places.txt')
