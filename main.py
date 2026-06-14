import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import VideoDataset, collate_fn
from model import SingleStageTCN, MultiStageTCN, MultiScaleTCN

# path settings
FEATURES_DIR = "./data/features"
GT_DIR = "./data/groundTruth"
MAPPING_FILE = "./data/mapping.txt"
TRAIN_BUNDLE = "./data/train.bundle"
TEST_BUNDLE = "./data/test.bundle"

PRED_DIR_Q1 = "./predictions_q1"
PRED_DIR_Q2 = "./predictions_q2"


def train_q1(device):
    print("=== Start training Question 1 (Single-stage linear expansion TCN model) ===")
    train_dataset = VideoDataset(TRAIN_BUNDLE, FEATURES_DIR, GT_DIR, MAPPING_FILE)
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)

    model = SingleStageTCN(num_layers=10, num_f_maps=64, dim=2048, num_classes=48)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    epochs = 50
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch_idx, (features, targets, _, _) in enumerate(train_loader):
            features = features.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            predictions = model(features)
            loss = criterion(predictions, targets)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        print(f"Epoch [{epoch+1}/{epochs}] - Loss: {epoch_loss/len(train_loader):.4f}")

    # save the weights of the trained model
    torch.save(model.state_dict(), "ss_tcn_linear_dilation.model")
    print("Question 1 model successfully saved to: ss_tcn_linear_dilation.model")


def predict_q1(device, pred_dir):
    print("=== Starting to generate prediction results for the Question 1 test set ===")
    os.makedirs(pred_dir, exist_ok=True)

    # read reverse mapping from ID to action name
    id_to_action = {}
    with open(MAPPING_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                id_to_action[int(parts[0])] = parts[1]

    # set batch size is 1 during testing to save the complete temporal sequence for each video individually
    test_dataset = VideoDataset(TEST_BUNDLE, FEATURES_DIR, GT_DIR, MAPPING_FILE)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

    model = SingleStageTCN(num_layers=10, num_f_maps=64, dim=2048, num_classes=48)
    model.load_state_dict(torch.load("ss_tcn_linear_dilation.model", map_location=device))
    model = model.to(device)
    model.eval()

    with torch.no_grad():
        for features, _, masks, video_names in test_loader:
            features = features.to(device)
            predictions = model(features)  # shape: [1, 48, T]
            
            # take the argmax to get predicted class IDs for each frame
            pred_ids = torch.argmax(predictions, dim=1).squeeze(0)  # [T]
            
            # extract the valid sequence length using the masks (remove padding frames)
            seq_len = int(masks.sum().item())
            pred_ids = pred_ids[:seq_len].cpu().numpy()
            
            # write the predicted action names to a text file for the current video
            video_name = video_names[0]
            pred_file = os.path.join(pred_dir, video_name + ".txt")
            with open(pred_file, "w", encoding="utf-8") as f:
                for pid in pred_ids:
                    action_name = id_to_action[pid]
                    f.write(action_name + "\n")

    print(f"Question 1 prediction complete! All results have been saved to: {pred_dir}")


def train_q2(device):
    print("\n=== Start training Question 2 (4-stage multi-stage TCN model) ===")
    
    train_dataset = VideoDataset(TRAIN_BUNDLE, FEATURES_DIR, GT_DIR, MAPPING_FILE)
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)

    model = MultiStageTCN(num_stages=4, num_layers=10, num_f_maps=64, dim=2048, num_classes=48)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    model.train()

    epochs = 50
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch_idx, (features, targets, _, _) in enumerate(train_loader):
            features = features.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()

            # forward pass through the multi-stage TCN model, which returns a list of predictions from each stage
            predictions_list = model(features)
            
            # calculate the loss as the sum of losses from all stages
            loss = sum(criterion(pred, targets) for pred in predictions_list)
            
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        print(f"Epoch [{epoch+1}/{epochs}] - Loss: {epoch_loss/len(train_loader):.4f}")

    # save the weights of the trained multi-stage TCN model
    torch.save(model.state_dict(), "ms_tcn_concat.model")
    print("Question 2 model successfully saved to: ms_tcn_concat.model")


def predict_q2(device, pred_dir):
    print("\n=== Starting to generate prediction results for the Question 2 test set ===")
    os.makedirs(pred_dir, exist_ok=True)

    id_to_action = {}
    with open(MAPPING_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                id_to_action[int(parts[0])] = parts[1]

    test_dataset = VideoDataset(TEST_BUNDLE, FEATURES_DIR, GT_DIR, MAPPING_FILE)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

    model = MultiStageTCN(num_stages=4, num_layers=10, num_f_maps=64, dim=2048, num_classes=48)
    model.load_state_dict(torch.load("ms_tcn_concat.model", map_location=device))
    model = model.to(device)
    
    model.eval()

    with torch.no_grad():
        for features, _, masks, video_names in test_loader:
            features = features.to(device)
            predictions_list = model(features)
            
            # take the predictions from the last stage as the final predictions for evaluation
            final_predictions = predictions_list[-1]  # shape: [1, 48, T]
            
            pred_ids = torch.argmax(final_predictions, dim=1).squeeze(0)  # [T]
            
            seq_len = int(masks.sum().item())
            pred_ids = pred_ids[:seq_len].cpu().numpy()

            video_name = video_names[0]
            pred_file = os.path.join(pred_dir, video_name + ".txt")
            with open(pred_file, "w", encoding="utf-8") as f:
                for pid in pred_ids:
                    f.write(id_to_action[pid] + "\n")

    print(f"Question 2 prediction complete! All results have been saved to: {pred_dir}")



def train_q3(device):
    print("\n=== Start training Question 3 (Multi-stage TCN + Video-level loss) ===")
    
    train_dataset = VideoDataset(TRAIN_BUNDLE, FEATURES_DIR, GT_DIR, MAPPING_FILE)
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)

    model = MultiStageTCN(num_stages=4, num_layers=10, num_f_maps=64, dim=2048, num_classes=48)
    model = model.to(device)

    # frame-level loss uses CrossEntropyLoss with ignore_index=-100 to ignore padding frames
    criterion_frame = nn.CrossEntropyLoss(ignore_index=-100)
    # video-level loss uses BCEWithLogitsLoss since it's a multi-label classification problem
    criterion_video = nn.BCEWithLogitsLoss()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    model.train()

    epochs = 50
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch_idx, (features, targets, masks, _) in enumerate(train_loader):
            features = features.to(device)
            targets = targets.to(device)
            masks = masks.to(device)  # shape: [B, T]

            optimizer.zero_grad()

            # forward pass through the multi-stage TCN model, which returns a list of predictions from each stage
            predictions_list = model(features)
            
            # calculate the frame-level loss as the sum of losses from all stages
            frame_loss = sum(criterion_frame(pred, targets) for pred in predictions_list)

            # calculate the video-level loss using the predictions from the last stage
            # obtain the final stage predictions: shape [B, 48, T]
            final_logits = predictions_list[-1]
            
            # set padding frames' logits to a very large negative value so that they do not contribute to the max pooling
            # expand masks to shape [B, 1, T] for broadcasting
            masked_logits = final_logits.clone()
            masked_logits = masked_logits.masked_fill(masks.unsqueeze(1) == 0, -1e9)
            
            # max pooling over the temporal dimension to get video-level logits: shape [B, 48]
            video_logits = torch.max(masked_logits, dim=2)[0]

            # construct video-level targets: shape [B, 48]
            video_targets = torch.zeros(targets.shape[0], 48, device=device)
            for b in range(targets.shape[0]):
                unique_classes = torch.unique(targets[b])
                # filter out the -100 class which represents padding frames
                unique_classes = unique_classes[unique_classes != -100]
                video_targets[b, unique_classes] = 1.0

            # calculate the video-level loss
            video_loss = criterion_video(video_logits, video_targets)

            # combine frame-level loss and video-level loss
            loss = frame_loss + video_loss
            
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        print(f"Epoch [{epoch+1}/{epochs}] - Loss: {epoch_loss/len(train_loader):.4f}")

    # save the weights of the trained multi-stage TCN model with video-level loss
    torch.save(model.state_dict(), "ms_tcn_video_loss.model")
    print("Question 3 model successfully saved to: ms_tcn_video_loss.model")


def predict_q3(device, pred_dir):
    print("\n=== Starting to generate prediction results for the Question 3 test set ===")
    os.makedirs(pred_dir, exist_ok=True)

    id_to_action = {}
    with open(MAPPING_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                id_to_action[int(parts[0])] = parts[1]

    test_dataset = VideoDataset(TEST_BUNDLE, FEATURES_DIR, GT_DIR, MAPPING_FILE)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

    model = MultiStageTCN(num_stages=4, num_layers=10, num_f_maps=64, dim=2048, num_classes=48)
    model.load_state_dict(torch.load("ms_tcn_video_loss.model", map_location=device))
    model = model.to(device)
    
    model.eval()

    with torch.no_grad():
        for features, _, masks, video_names in test_loader:
            features = features.to(device)
            predictions_list = model(features)
            final_predictions = predictions_list[-1]  # [1, 48, T]
            
            pred_ids = torch.argmax(final_predictions, dim=1).squeeze(0)  # [T]
            seq_len = int(masks.sum().item())
            pred_ids = pred_ids[:seq_len].cpu().numpy()

            video_name = video_names[0]
            pred_file = os.path.join(pred_dir, video_name + ".txt")
            with open(pred_file, "w", encoding="utf-8") as f:
                for pid in pred_ids:
                    f.write(id_to_action[pid] + "\n")

    print(f"Question 3 prediction complete! All results have been saved to: {pred_dir}")


def train_q4(device):
    print("\n=== Start training Question 4 (Parallel Multi-scale TCN Model) ===")
    
    train_dataset = VideoDataset(TRAIN_BUNDLE, FEATURES_DIR, GT_DIR, MAPPING_FILE)
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)

    model = MultiScaleTCN(num_layers=10, num_f_maps=64, dim=2048, num_classes=48)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    model.train()

    epochs = 50
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch_idx, (features, targets, _, _) in enumerate(train_loader):
            features = features.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()

            logits_final, logits1, logits4, logits8 = model(features)

            # calculate the loss for each branch and the final fused output
            # corresponding targets are aligned using respective downsampling for each branch
            loss_final = criterion(logits_final, targets)
            loss_b1 = criterion(logits1, targets)
            loss_b2 = criterion(logits4, targets[:, ::4])
            loss_b3 = criterion(logits8, targets[:, ::8])

            loss = loss_final + loss_b1 + loss_b2 + loss_b3

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        print(f"Epoch [{epoch+1}/{epochs}] - Loss: {epoch_loss/len(train_loader):.4f}")

    # save the weights
    torch.save(model.state_dict(), "ms_tcn_multi_scale.model")
    print("Question 4 model successfully saved to: ms_tcn_multi_scale.model")


def predict_q4(device, pred_dir):
    print("\n=== Starting to generate prediction results for the Question 4 test set ===")
    os.makedirs(pred_dir, exist_ok=True)

    id_to_action = {}
    with open(MAPPING_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                id_to_action[int(parts[0])] = parts[1]

    test_dataset = VideoDataset(TEST_BUNDLE, FEATURES_DIR, GT_DIR, MAPPING_FILE)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

    model = MultiScaleTCN(num_layers=10, num_f_maps=64, dim=2048, num_classes=48)
    model.load_state_dict(torch.load("ms_tcn_multi_scale.model", map_location=device))
    model = model.to(device)
    
    model.eval()

    with torch.no_grad():
        for features, _, masks, video_names in test_loader:
            features = features.to(device)
            
            logits_final, _, _, _ = model(features)
            
            pred_ids = torch.argmax(logits_final, dim=1).squeeze(0)  # [T]
            seq_len = int(masks.sum().item())
            pred_ids = pred_ids[:seq_len].cpu().numpy()

            video_name = video_names[0]
            pred_file = os.path.join(pred_dir, video_name + ".txt")
            with open(pred_file, "w", encoding="utf-8") as f:
                for pid in pred_ids:
                    f.write(id_to_action[pid] + "\n")

    print(f"Question 4 prediction complete! All results have been saved to: {pred_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Action Segmentation with TCNs")
    parser.add_argument("--task", type=str, required=True, choices=["q1", "q2", "q3", "q4"],
                        help="choose question: 'q1', 'q2', 'q3', 'q4'")
    parser.add_argument("--action", type=str, required=True, choices=["train", "predict"],
                        help="choose operation: 'train'or 'predict'")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Current computational device: {device}")

    pred_dir_q1 = "./predictions_q1"
    pred_dir_q2 = "./predictions_q2"
    pred_dir_q3 = "./predictions_q3"
    pred_dir_q4 = "./predictions_q4"

    if args.task == "q1":
        if args.action == "train":
            train_q1(device)
        elif args.action == "predict":
            predict_q1(device, pred_dir_q1)
            
    elif args.task == "q2":
        if args.action == "train":
            train_q2(device)
        elif args.action == "predict":
            predict_q2(device, pred_dir_q2)

    elif args.task == "q3":
        if args.action == "train":
            train_q3(device)
        elif args.action == "predict":
            predict_q3(device, pred_dir_q3)

    elif args.task == "q4":
        if args.action == "train":
            train_q4(device)
        elif args.action == "predict":
            predict_q4(device, pred_dir_q4)