import os 
import os.path as osp

"""
This is an transdata file to form MNIST val.txt
"""


def main():
    input_dir = "/home/hyl/home/DL_Homework/SAR-Vehicle/MNIST"
    output_dir = osp.join(input_dir, "val.txt")
    input_dir = osp.join(input_dir, "test")
    assert osp.exists(input_dir), "Input dir is not exists!"

    print("input: ", input_dir)

    dirs = os.listdir(input_dir)
    print("dirs: ", dirs)
    assert len(dirs) == 10, "MNIST doesn't have 10 folder!"

    output_txt = open(output_dir, 'w')
    for label in range(len(dirs)):
        files = os.listdir(osp.join(input_dir, dirs[label]))
        cur_dir = osp.join("MNIST/test", dirs[label])
        for file in files:
            file = osp.join(cur_dir, file)
            output_txt.write(file + " " + dirs[label] + "\n")

if __name__ == "__main__":
    main()    