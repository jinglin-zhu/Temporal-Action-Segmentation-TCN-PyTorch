import argparse
from metrics import ValMeter
import os

def read_file(path):
    with open(path, 'r') as f:
        content = f.read()
        f.close()
    return content



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_path",
                        type=str,
                        required=True,
                        help="path to directory of saved predictions")
    args = parser.parse_args()
    ground_truth_path = "./data/groundTruth/"
    file_list = "./data/test.bundle"

    mapping_file = "./data/mapping.txt"
    mapping = {}
    with open(mapping_file, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                mapping[parts[1]] = int(parts[0])

    val_meter = ValMeter()
    list_of_videos = [vid.strip() for vid in read_file(file_list).split('\n') if vid.strip()]

    for vid_name in list_of_videos:

        # Note that if the labels of target or prediction are not integers
        # you should convert them to integers using the './data/mapping.txt' file

        # TODO: read the content of the ground truth frame-wise labels for the current video as a python list of integers
        # remove the file extension
        vid_name_clean = vid_name.split('.')[0]
        
        gt_file = os.path.join(ground_truth_path, vid_name_clean + ".txt")
        with open(gt_file, 'r', encoding='utf-8') as f:
            gt_lines = [line.strip() for line in f if line.strip()]
        
        # compatible with both integer labels and string labels in the ground truth files
        if gt_lines[0].isdigit():
            target = [int(x) for x in gt_lines]
        else:
            target = [mapping[x] for x in gt_lines]

        # TODO: this variable should contain a python list of the predicted frame-wise labels for the current video
        pred_file = os.path.join(args.pred_path, vid_name_clean + ".txt")
        if not os.path.exists(pred_file):
            print(f"Warning: Prediction file not found for {vid_name_clean} at {pred_file}")
            continue

        with open(pred_file, 'r', encoding='utf-8') as f:
            pred_lines = [line.strip() for line in f if line.strip()]

        # compatible with both integer labels and string labels in the prediction files
        if pred_lines[0].isdigit():
            prediction = [int(x) for x in pred_lines]
        else:
            prediction = [mapping[x] for x in pred_lines]

        val_meter.update_stats(target=target, prediction=prediction)


    eval_metrics = val_meter.log_stats()
    print("Evaluation metrics:")
    for metric_name in eval_metrics:
        print(f'{metric_name}: {eval_metrics[metric_name]:.5f}')


if __name__ == '__main__':
    main()
