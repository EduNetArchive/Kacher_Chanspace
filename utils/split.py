from tqdm import tqdm
import os
from argparse import ArgumentParser

def split(filename, root):
    frame_num = 0
    dst_file = None

    with open(filename) as src_file:
        for line in tqdm(src_file):
            if not line:
                break

            if line.startswith("ENDMDL"):
                frame_num += 1
                dst_file.write(line)
                dst_file.close()
                dst_file = None
            elif dst_file is None:
                file_path = os.path.join(root, f"{frame_num:0>6}.pdb")
                dst_file = open(file_path, "w")

            if dst_file is not None:
                dst_file.write(line)

if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--filename", "-i", type=str, required=True)
    parser.add_argument("--root", "-o", type=str, required=True)

    args = parser.parse_args()
    split(args.filename, args.root)
