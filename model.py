import torch
import torch.nn as nn
import torch.nn.functional as F

class DilatedResidualLayer(nn.Module):
    def __init__(self, dilation, in_channels, out_channels):
        super().__init__()
        # kernel size=3, padding=dilation to maintain the same temporal dimension
        self.dilated_conv = nn.Conv1d(in_channels, out_channels, kernel_size=3, 
                                      padding=dilation, dilation=dilation)
        self.conv_1x1 = nn.Conv1d(out_channels, out_channels, kernel_size=1)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        out = F.relu(self.dilated_conv(x))
        out = self.conv_1x1(out)
        out = self.dropout(out)
        return x + out  # residual connection

class SingleStageTCN(nn.Module):
    def __init__(self, num_layers=10, num_f_maps=64, dim=2048, num_classes=48):
        super().__init__()
        # dimension adjustment layer: map input dimension (2048) to num_f_maps (64)
        self.first_conv = nn.Conv1d(dim, num_f_maps, kernel_size=1)
        
        # 10 dilated residual layers with linear increasing dilation factors (1, 2, 3, ..., 10)
        self.layers = nn.ModuleList([
            DilatedResidualLayer(dilation=i+1, in_channels=num_f_maps, out_channels=num_f_maps)
            for i in range(num_layers)
        ])
        
        # final 1x1 convolution to map to num_classes (48)
        self.final_conv = nn.Conv1d(num_f_maps, num_classes, kernel_size=1)

    def forward(self, x, return_features=False):
        out = self.first_conv(x)
        for layer in self.layers:
            out = layer(out)
        
        # features before the final classification layer
        features = out 
        logits = self.final_conv(features)
        
        if return_features:
            return logits, features
        return logits
    
class MultiStageTCN(nn.Module):
    def __init__(self, num_stages=4, num_layers=10, num_f_maps=64, dim=2048, num_classes=48):
        super().__init__()
        self.stages = nn.ModuleList()
        
        # first stage: input dimension is 2048, output dimension is 48
        self.stages.append(SingleStageTCN(num_layers, num_f_maps, dim, num_classes))
        
        # subsequent stages: input dimension is 2048 + 48 (concatenated), output dimension is 48
        for _ in range(num_stages - 1):
            self.stages.append(SingleStageTCN(num_layers, num_f_maps, dim + num_classes, num_classes))

    def forward(self, x):
        outputs = []
        
        # first stage
        out = self.stages[0](x)
        outputs.append(out)
        
        # subsequent stages: concatenate the softmax probabilities from the previous stage with the original input
        for s in range(1, len(self.stages)):
            prob = F.softmax(out, dim=1)
            concat_input = torch.cat([prob, x], dim=1)
            out = self.stages[s](concat_input)
            outputs.append(out)
            
        return outputs
    
class MultiScaleTCN(nn.Module):
    def __init__(self, num_layers=10, num_f_maps=64, dim=2048, num_classes=48):
        super().__init__()
        # three branches for different temporal resolutions (1x, 4x, 8x)
        self.branch1 = SingleStageTCN(num_layers, num_f_maps, dim, num_classes)
        self.branch2 = SingleStageTCN(num_layers, num_f_maps, dim, num_classes)
        self.branch3 = SingleStageTCN(num_layers, num_f_maps, dim, num_classes)
        
        # global feature fusion layer: 1x1 convolution to fuse features from three branches
        self.final_classifier = nn.Conv1d(num_f_maps, num_classes, kernel_size=1)

    def forward(self, x):
        # x: [B, 2048, T]
        T = x.shape[2]
        
        # Branch 1: original temporal resolution
        logits1, f1 = self.branch1(x, return_features=True)  # f1 shape: [B, 64, T]
        
        # Branch 2: 4x downsampling, selecting slices at indices: 0, 4, 8, ...
        x_down4 = x[:, :, ::4]
        logits4, f4 = self.branch2(x_down4, return_features=True)  # f4 shape: [B, 64, T_4]
        
        # Branch 3: 8x downsampling, selecting slices at indices: 0, 8, 16, ...
        x_down8 = x[:, :, ::8]
        logits8, f8 = self.branch3(x_down8, return_features=True)  # f8 shape: [B, 64, T_8]
        
        # interpolate intermediate features to the original temporal resolution T
        f4_up = F.interpolate(f4, size=T, mode='linear', align_corners=False)
        f8_up = F.interpolate(f8, size=T, mode='linear', align_corners=False)
        
        # average the features from three branches
        f_avg = (f1 + f4_up + f8_up) / 3.0
        
        # predict final classification output from the fused features
        logits_final = self.final_classifier(f_avg)
        
        # return final logits and intermediate logits from each branch for analysis
        return logits_final, logits1, logits4, logits8