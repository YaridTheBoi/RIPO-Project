WORKSPACE_PATH = 'Tensorflow/workspace'
ANNOTATIONS_PATH = WORKSPACE_PATH + '/annotations'


print("\n!!!REMEMBER TO USE SAME LABELS AS YOU DID WHILE LABELING DATA. IF IT'S AHEAD OF YOU REMEMBER TO USE LABEL NAMES DEFINED NOW!!!\n")
amount = int(input("How many labels do you need: "))

with open(ANNOTATIONS_PATH + '/label_map.pbtxt', 'w') as f:
    for id in range (amount):
        f.write('item{\n')
        f.write('\tname:\'{}\'\n'.format(input("Name of {} label: ".format(id+1))))
        f.write('\tid:{}\n'.format(id+1))
        f.write('}\n')