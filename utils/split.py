from tqdm import tqdm
import os
from argparse import ArgumentParser

def split(filename, root, splitting_phrase):
    frame_num = 0
    dst_file = None
    if args.option:
        splitting_phrase=args.option
    else:
        splitting_phrase='ENDMDL'

    with open(filename) as src_file:
        for line in tqdm(src_file):
            if line.startswith("CONNECT"):
                continue

            if not line:
                break

            if line.startswith(splitting_phrase) and dst_file is not None:
                frame_num += 1
                dst_file.write(line)
                dst_file.close()
                dst_file = None
                # if flag:
                #     os.remove(file_path)
                # flag = True
            elif dst_file is None:
                file_path = os.path.join(root, f"{frame_num:0>6}.pdb")
                dst_file = open(file_path, "w")

            if dst_file is not None:
                dst_file.write(line)
                # flag = False

if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--filename", "-i", type=str, required=True)
    parser.add_argument("--root", "-o", type=str, required=True)
    parser.add_argument("--option", "-e", type=str,required=False)

    args = parser.parse_args()
    split(args.filename, args.root, args.option)
