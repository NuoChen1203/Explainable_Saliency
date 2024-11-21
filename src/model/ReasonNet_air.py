import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import clip
from torch.nn import Sequential as Seq, Linear as Lin, ReLU
import torchvision.ops as ops
import os
import json
from PIL import Image
from sklearn.metrics import auc
from clip import clip
# pip install git+https://github.com/openai/CLIP.git

class VisualEncoder(nn.Module):
    """
    Visual Encoder module that processes the image features using a dilated ResNet50 backbone
    and an adapter layer to produce feature maps.
    """

    def __init__(self):
        super(VisualEncoder, self).__init__()

        # Load the CLIP model and preprocess
        clip_model, preprocess = clip.load("RN50", device="cuda" if torch.cuda.is_available() else "cpu")
        clip_model = clip_model.float()
        self.clip_model = clip_model

        # Use the visual part of the CLIP model
        self.backbone = clip_model.visual
        # Convert the ResNet backbone to a dilated version
        self.dilate_resnet(self.backbone)
        # Remove the final pooling layer from the backbone
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])

        # Adapter layer to adjust the channel dimensions
        self.adapter = nn.Conv2d(in_channels=2048, out_channels=1024, kernel_size=3, stride=1, padding=1)

    def dilate_resnet(self, resnet):
        """ 
        Convert standard ResNet50 into a dilated one by adjusting the strides and dilations.
        This expands the receptive field without additional downsampling.
        """
        # Adjust the stride and dilation for layer3 and layer4
        resnet.layer3[0].conv1.stride = 1
        resnet.layer3[0].downsample[0].stride = 1
        resnet.layer4[0].conv1.stride = 1
        resnet.layer4[0].downsample[0].stride = 1

        # Replace downsampling with average pooling
        resnet.layer3[0].avgpool = nn.AvgPool2d(1)
        resnet.layer4[0].avgpool = nn.AvgPool2d(1)
        resnet.layer3[0].downsample[0] = nn.AvgPool2d(1)
        resnet.layer4[0].downsample[0] = nn.AvgPool2d(1)

        # Apply dilation and padding to convolutional layers
        for block in resnet.layer3:
            block.conv2.dilation = 2
            block.conv2.padding = 2

        for block in resnet.layer4:
            block.conv2.dilation = 4
            block.conv2.padding = 4

    def forward(self, image, probe=False):
        """
        Forward pass to compute the visual features from the input image.
        """
        features = self.backbone(image)
        features = self.adapter(features).relu()
        return features

class ContexutalAttention(nn.Module):
    """
    Contextual Attention module that computes attention weights for semantic prototypes
    based on the context, using global average pooling and a convolutional layer.
    """

    def __init__(self, channels, reduction=16):
        super(ContexutalAttention, self).__init__()
        self.channels = channels
        self.reduction = reduction

        # Convolutional layer to compute attention maps
        self.conv = nn.Conv2d(channels, 1, kernel_size=1)
        self.conv_aux = nn.Conv2d(channels, 1, kernel_size=1)  # Unused auxiliary convolution
        self.gap = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        """
        Forward pass to compute the attention weights for the input features.
        """
        attention_map = self.conv(x)  # (N, 1, H, W)
        attention_weights = self.gap(attention_map)  # (N, 1, 1, 1)
        return x, attention_weights

class ReasonNet(nn.Module):
    """
    ReasonNet model that integrates visual features, textual content,
    and attention mechanisms to perform reasoning tasks.
    """

    def __init__(self):
        super(ReasonNet, self).__init__()

        # Visual encoder to extract features from images
        self.visual_encoder = VisualEncoder()

        # CLIP tokenizer
        self.clip_tokenizer = clip.tokenize

        # Linear layer for output (unused in the code snippet)
        self.out_linear = nn.Linear(1024, 1)

        # Background convolutional layers to process features
        self.background_conv = nn.Sequential(
            nn.Conv2d(1024, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 1, 3, 1, 1),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        # Convolutional layer to compute weights
        self.weight_conv = nn.Conv2d(1024, 1, 1, bias=False)
        self.gap = nn.AdaptiveAvgPool2d(1)

        # Transformation layers for segmentation masks
        self.segmentation_transform = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=1, kernel_size=3, padding=1),
            # nn.Sigmoid()
        )

        # Second transformation layers for segmentation masks
        self.segmentation_transform2 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=1, kernel_size=3, padding=1),
            # nn.Sigmoid()
        )

        # Dimensionality settings
        d_model = 512
        self.d_model = d_model

        d = 256
        self.d_k = d
        self.d_v = d

        # Load question JSON files
        self.question_json = json.load(open(os.path.join("./data/air_data/", "val_balanced_questions.json")))

        # Load clean dictionaries for training and validation
        self.clean_dict_train = json.load(open('./data/air_data/sal99_2_sim_sal_cleantrain.json'))
        self.clean_dict_val = json.load(open('./data/air_data/sal99_2_sim_sal_cleanval.json'))
        self.clean_dict = {**self.clean_dict_train, **self.clean_dict_val}

        # Initialize the Contextual Attention module
        self.contextual_attention = ContexutalAttention(channels=1024, reduction=8)

        # Load text content (semantic proposals and explanations)
        with open('./data/air_data/sal91_sim_noqa_val.json', 'r') as f:
            text_content_val = json.load(f)
        with open('./data/air_data/sal91_sim_noqa.json', 'r') as f:
            text_content = json.load(f)

        self.text_content = {**text_content, **text_content_val}

        # Process text content to extract matches (semantic proposals and explanations)
        from tqdm import tqdm
        import re
        for key in tqdm(self.text_content.keys()):
            result = []
            for i in range(len(self.text_content[key]["matches"])):
                data_str = self.text_content[key]["matches"][i].replace('\n', '')
                matches = re.findall(r'"object": "(.*?)",\s*"reason": "(.*?)"', data_str)
                # Convert to a list of dictionaries
                result += [{"object": obj, "reason": reason} for obj, reason in matches]
            self.text_content[key]["matches"] = result

    def compute_score(self, input_map, salMap, fixMap):
        """
        Compute the NSS (Normalized Scanpath Saliency) score for the input saliency map.
        """
        input_map = input_map.cpu().detach().numpy()
        fixMap = fixMap.cpu().detach().numpy()

        with np.errstate(divide="ignore", invalid="ignore"):
            input_map = (input_map - np.mean(input_map)) / np.std(input_map)
            return np.sum(input_map * fixMap) / np.count_nonzero(fixMap)

    def cal_metrics(self, semantic_prototypes, weights, fixmap):
        """
        Calculate evaluation metrics using the semantic prototypes and weights.
        """
        # Compute the weighted sum of semantic prototypes
        weighted_sum = torch.sum(semantic_prototypes * weights.view(len(weights), 1, 1, 1).to(semantic_prototypes.device), dim=0, keepdim=True)
        output_map = self.weight_conv(weighted_sum).sigmoid()

        # Upsample the output to match the fixmap size
        output_map = F.interpolate(output_map, (240, 320))

        # Compute the NSS score
        nss_score = self.compute_score(output_map, fixmap)
        return nss_score

    def forward(self, img_feat, que, gt_op=None, ss_rate=2, ans=None, bbox_list=None, epoch=10, need_faith_metric=False, fixmap=None, topk=3, **kwargs):
        """
        Forward pass for the ReasonNet model.
        """
        scene_graph_list = ss_rate
        # Extract visual features using the visual encoder
        img_feat = self.visual_encoder(img_feat)  # torch.Size([batch_size, 1024, 30, 40])

        faith_loss = []

        clipseg_path = "./data/air_data/segmask_air"
        visualization_list = [[], [], [], [], [], [], [], []]
        clip_features = []
        semantic_features_list = []

        visualization_output_list = []
        visualization_final_list = []
        faith_metric_list = {}
        # Process each sample in the batch
        for sample_idx, sample_tmp in enumerate(scene_graph_list):
            img_feat_sample = img_feat[sample_idx:sample_idx+1]
            visualization_list[0].append(img_feat_sample)

            semantic_features = []


            for i in range(sample_tmp[0]):
                current_vis = [{}, []]

                qid = sample_tmp[1]
                # Load the segmentation mask for the semantic proposal
                clip_seg_mask = Image.open(os.path.join(clipseg_path, f"{qid}_{i}.png")).convert('L')
                # Convert to tensor
                clip_seg_mask = torch.from_numpy(np.array(clip_seg_mask) / 255).to(img_feat.device, img_feat.dtype)
                clip_seg_mask = F.interpolate(clip_seg_mask.unsqueeze(0).unsqueeze(0), (30, 40))

                current_vis[0]["init"] = clip_seg_mask.sigmoid()
                # Transform the segmentation mask
                clip_seg_mask = self.segmentation_transform(clip_seg_mask)
                clip_seg_mask_sig = clip_seg_mask.sigmoid()
                # Multiply with the image features to get the semantic prototype
                foreground_feature = clip_seg_mask_sig * img_feat_sample

                semantic_features.append(foreground_feature)

                current_vis[0]["feat"] = foreground_feature
                current_vis[0]["mask"] = clip_seg_mask_sig


            qid = sample_tmp[1]
            imageId = self.question_json[qid]['imageId']
            clip_seg_mask_list = []

            name_list = []

            # Apply Contextual Attention to get attention weights
            semantic_features_corrected, weights = self.contextual_attention(torch.stack(semantic_features, dim=0).squeeze(1))
            weights = weights.reshape(-1)



            # Progressive Top-K Reduction strategy
            weights = weights.reshape(-1)

            initial_TopK = 20
            final_TopK = topk
            total_epochs = 3

            current_TopK = int(initial_TopK - (initial_TopK - final_TopK) * epoch / (total_epochs - 1))
            current_TopK = max(current_TopK, final_TopK)

            top_num = min(current_TopK, len(semantic_features_corrected))

            _, top_indices = torch.topk(weights, top_num)
            selection_mask = torch.zeros_like(weights)
            selection_mask[top_indices] = 1.0

            selected_weights = weights * selection_mask
            reweighted_weights = selected_weights

            semantic_features_corrected = semantic_features_corrected[reweighted_weights != 0, ...]
            reweighted_weights = reweighted_weights.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            weights = reweighted_weights[reweighted_weights != 0].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

            weights = F.softmax(weights, dim=0)

            if need_faith_metric:
                # Compute initial output and score
                output = torch.sum(semantic_features_corrected * weights, dim=0, keepdim=True)
                output = self.weight_conv(output).sigmoid()
                output = F.interpolate(output, (240, 320))
                initial_score = self.compute_score(output, gt_op[sample_idx][0, :, :], fixMap=fixmap[sample_idx][0, :, :])  # Initial score

                # Compute scores after masking
                sorted_weights, indices = torch.sort(weights.reshape(-1), descending=True)
                cumulative_ratios = torch.cumsum(sorted_weights, dim=0) / sorted_weights.sum()
                relative_scores = []
                aopc_diffs = []
                lodds_scores = []

                cumulative_ratios = torch.cat((torch.tensor([0]).to(cumulative_ratios.device),
                                               cumulative_ratios))

                for i, ratio in enumerate(cumulative_ratios):
                    masked_weights = weights.clone().reshape(-1)
                    masked_weights[indices[:i]] = 0

                    output = torch.sum(semantic_features_corrected * masked_weights.view(*masked_weights.shape, 1, 1, 1), dim=0, keepdim=True)
                    output = self.weight_conv(output).sigmoid()
                    output = F.interpolate(output, (240, 320))
                    score = self.compute_score(output, gt_op[sample_idx][0, :, :], fixMap=fixmap[sample_idx][0, :, :])  # Score after masking

                    if math.isnan(score):
                        score = 0
                    relative_score = (score / initial_score)
                    relative_scores.append(relative_score)
                    aopc_diffs.append(initial_score - score)

                    lodds_score = torch.tensor(score / initial_score)
                    lodds_score = torch.clamp(lodds_score, min=1e-7)
                    lodds_score = torch.log(lodds_score)
                    lodds_scores.append(lodds_score)

                relative_scores = torch.tensor(relative_scores)

                if len(relative_scores) > 1:
                    # Compute AUC, AOPC, and LODDS metrics
                    auc_score = auc(cumulative_ratios.squeeze().cpu().numpy(), relative_scores)
                    if "auc" not in faith_metric_list:
                        faith_metric_list["auc"] = []
                    faith_metric_list["auc"].append(auc_score)
                    if "aopc" not in faith_metric_list:
                        faith_metric_list["aopc"] = []
                    faith_metric_list["aopc"].append(np.mean(aopc_diffs))
                    if "lodds" not in faith_metric_list:
                        faith_metric_list["lodds"] = []
                    faith_metric_list["lodds"].append(np.mean(lodds_scores))

            reweighted_weights_vis = reweighted_weights.clone()
            reweighted_weights_vis[reweighted_weights != 0] = F.softmax(reweighted_weights_vis[reweighted_weights != 0], dim=0)
            reweighted_weights_vis = reweighted_weights_vis.squeeze(1).squeeze(1).squeeze(1)

            # Compute the weighted sum of semantic features
            semantic_features_corrected = torch.sum(semantic_features_corrected * weights, dim=0, keepdim=True)

            semantic_features_list.append(semantic_features_corrected)


        semantic_features_list = torch.stack(semantic_features_list, dim=0).squeeze(1)

        pooled_features_list = self.weight_conv(semantic_features_list)
        pooled_features_list = pooled_features_list.sigmoid()

        ret_corr = F.interpolate(pooled_features_list, (240, 320))
        ret_incorr = None
        if len(faith_loss) == 0:
            ret_incorr = torch.tensor(0.0)
        else:
            ret_incorr = torch.stack(faith_loss).mean()

        if need_faith_metric:
            faith_metric_list = [faith_metric_list["auc"],
                                 faith_metric_list["aopc"], faith_metric_list["lodds"]]

        ret_incorr = [ret_incorr, faith_metric_list]

        return ret_corr, ret_incorr, [tmp[2] for tmp in scene_graph_list]
