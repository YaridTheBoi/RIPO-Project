import os
dir = "data/door"

with open(dir+ "/bg.txt", "w" ) as f:
    for filename in os.listdir(dir + "/negative"):
        f.write(os.path.join("negative/", filename) + "\n")
