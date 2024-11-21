
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import clip
import torch
from torch.nn import Sequential as Seq, Linear as Lin, ReLU
import torchvision.ops as ops
import torch
from torch.nn import Sequential as Seq, Linear as Lin, ReLU
import os
import json
# PIL
from PIL import Image
from sklearn.metrics import auc



from clip import clip
# pip install git+https://github.com/openai/CLIP.git


class VisualEncoder(nn.Module):

    def __init__(self):
        super(VisualEncoder, self).__init__()

        model, preprocess = clip.load("RN50", device="cuda" if torch.cuda.is_available() else "cpu")
        model= model.float()
        self.model=model
        self.dilated_backbone = model.visual
        self.dilate_resnet(self.dilated_backbone)
        self.dilated_backbone = nn.Sequential(*list(
                                self.dilated_backbone.children())[:-1])
        self.adapter = nn.Conv2d(in_channels=2048, out_channels=1024, kernel_size=3, stride=1, padding=1)



    def dilate_resnet(self, resnet):
        """ Converting standard ResNet50 into a dilated one.
        """

        a=0

        resnet.layer3[0].conv1.stride = 1
        resnet.layer3[0].downsample[0].stride = 1
        resnet.layer4[0].conv1.stride = 1
        resnet.layer4[0].downsample[0].stride = 1

        # resnet.layer3[0].downsample=None
        # resnet.layer4[0].downsample=None
        resnet.layer3[0].avgpool=nn.AvgPool2d(1)
        resnet.layer4[0].avgpool=nn.AvgPool2d(1)
        resnet.layer3[0].downsample[0] = nn.AvgPool2d(1)
        resnet.layer4[0].downsample[0] = nn.AvgPool2d(1)


        a=0
        for block in resnet.layer3:
            block.conv2.dilation = 2
            block.conv2.padding = 2

        for block in resnet.layer4:
            block.conv2.dilation = 4
            block.conv2.padding = 4

            

    def forward(self, image, probe=False):


        x = self.dilated_backbone(image)
        # x = x.detach()
        x = self.adapter(x).relu()
        # x = L2_Norm(x)

        return x


class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.channels = channels
        self.reduction = reduction
        


        self.conv = nn.Conv2d(channels, 1, kernel_size=1)
        self.conv2 = nn.Conv2d(channels, 1, kernel_size=1)
        self.gap = nn.AdaptiveAvgPool2d(1)
    def forward(self, x):
        w=x             # (N, C, H, W)
        w=self.conv(w)  # (N, 1, H, W)
        w=self.gap(w)   # (N, 1, 1, 1)

        return x, w





class ReasonNet(nn.Module):

    def __init__(self):
        super(ReasonNet, self).__init__()




        self.visual_encoder = VisualEncoder()


     

        self.clip_tokenizer = clip.tokenize

        self.out_linear = nn.Linear(1024,1)


        self.background_conv=nn.Sequential(
            nn.Conv2d(1024, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 1, 3, 1, 1),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )


        self.weight_roi2 = nn.Conv2d(1024, 1, 1, bias=False)
        self.gap=nn.AdaptiveAvgPool2d(1)


        self.clipseg_trans=nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=1, kernel_size=3, padding=1),
            # nn.Sigmoid()
        )   

        self.clipseg_trans2=nn.Sequential(# kernel_size变大
            nn.Conv2d(in_channels=1, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=1, kernel_size=3, padding=1),
            # nn.Sigmoid()
        )   


        d_model = 512 
        self.d_model = d_model

        d=512
        d=256
        # d=128
        # d=64
        self.d_k = d  
        self.d_v = d 



        self.question_json = json.load(open(os.path.join("./data/air_data/","val"+'_balanced_questions.json')))

        self.clean_dict_train = json.load(open('./data/air_data/sal99_2_sim_sal_cleantrain.json'))
        self.clean_dict_val = json.load(open('./data/air_data/sal99_2_sim_sal_cleanval.json'))
        self.clean_dict= {**self.clean_dict_train, **self.clean_dict_val}


        self.Dynamic=ChannelAttention(channels=1024, reduction=8)

        with open('./data/air_data/sal91_sim_noqa_val.json', 'r') as f:
            text_content_val=json.load(f)
        with open('./data/air_data/sal91_sim_noqa.json', 'r') as f:
            text_content=json.load(f)

        text_content={**text_content, **text_content_val}

        from tqdm import tqdm
        import re
        for key in tqdm(text_content.keys()):
            result = []
            for i in range(len(text_content[key]["matches"])) :
                data_str=text_content[key]["matches"][i].replace('\n','')
                matches = re.findall(r'"object": "(.*?)",\s*"reason": "(.*?)"', data_str)
                # Convert to a list of dictionaries
                result += [{"object": obj, "reason": reason} for obj, reason in matches]
            a=0
            text_content[key]["matches"]=result
        self.text_content=text_content
        a=0
        a=0


  


    def compute_score(self, input, salMap, fixMap):

        input=input.cpu().detach().numpy()
        # salMap=salMap.cpu().detach().numpy()
        fixMap=fixMap.cpu().detach().numpy()


        with np.errstate(divide="ignore", invalid="ignore"):
            input = (input-np.mean(input))/np.std(input)
        
            return np.sum(input * fixMap)/np.count_nonzero(fixMap)


    
    def cal_metrics(self, semantic_prototypes, weights, fixmap):

        weighted_sum = torch.sum(semantic_prototypes * weights.view(len(weights),1,1,1).to(semantic_prototypes.device), dim=0, keepdim=True)
        output = self.weight_roi2(weighted_sum).sigmoid()
        
 
        output = F.interpolate(output, (240, 320))

        nss_score = self.compute_score(output, fixmap)
        return nss_score






    def forward(self,img_feat,que,gt_op=None,ss_rate=2, ans=None, bbox_list=None, epoch=10,  need_faith_metric=False, fixmap=None, topk=3, **kwargs):
        cur_scene_graph_list=ss_rate
        img_feat=self.visual_encoder(img_feat)#torch.Size([150, 2048, 30, 40])

        faith_loss=[]
        

        clipseg_path = "./data/air_data/segmask_air"
        vis_list=[[],[],[],[],[],[],[],[]]
        clip_feat=[]
        semantic_feats_list=[]



        Vis_out_list=[]
        Vis_final_list=[]
        faith_metric_list={}
        # img_feat
        for sample_idx, sample_tmp in enumerate(cur_scene_graph_list):
            # print(sample_idx)
            # clip_seg_mask_list=[]
            img_feat_tmp=img_feat[sample_idx:sample_idx+1]
            # img_feat_tmp=self.visual_encoder(img_feat_tmp)
            vis_list[0].append(img_feat_tmp)
            a=0


            semantic_feats=[]

            
            Vis_out={}
            vis_tmp=[]
            vis_tmp2=[]
            vis_tmp3=[]
            vis_fg=[]


            Vis_final=[]#[currnode, childnode_list]
            for i in range(sample_tmp[0]):
                curr_vis_tmp=[{},[]]

                qid=sample_tmp[1]
                clip_seg_mask=Image.open(os.path.join(clipseg_path, str(qid)+'_'+str(i)+'.png')).convert('L')
                # to tensor
                clip_seg_mask = torch.from_numpy(np.array(clip_seg_mask)/255).to(img_feat.device, img_feat.dtype)
                clip_seg_mask = F.interpolate(clip_seg_mask.unsqueeze(0).unsqueeze(0), (30, 40))

                vis_tmp2.append(clip_seg_mask.sigmoid())
                curr_vis_tmp[0]["init"]=clip_seg_mask.sigmoid()
                clip_seg_mask=self.clipseg_trans(clip_seg_mask)
                # clip_seg_mask=clip_seg_mask.sigmoid()*img_feat_tmp
                
                clip_seg_mask_sig=clip_seg_mask.sigmoid()
                fgfeat=clip_seg_mask_sig*img_feat_tmp

                semantic_feats.append(fgfeat)
                vis_tmp.append(clip_seg_mask_sig)
                vis_tmp3.append(fgfeat)
                curr_vis_tmp[0]["feat"]=fgfeat
                curr_vis_tmp[0]["mask"]=clip_seg_mask_sig
                vis_fg.append("f")
                Vis_final.append(curr_vis_tmp)



            qid=sample_tmp[1]
            imageId = self.question_json[qid]['imageId']
            clip_seg_mask_list=[]

            name_list=[]


            semantic_feats_corr, w=self.Dynamic(torch.stack(semantic_feats, dim=0).squeeze(1))
            # semantic_feats_corr = torch.sum(semantic_feats_corr * w, dim=0, keepdim=True)

            # Apply threshold to select weights greater than 0.15
            w=w.reshape(-1)

            Vis_out["fgw"]=w.clone()
            Vis_out["fgw_incorr"]=torch.ones_like(w)*-1
            
            Vis_out["fg"]=vis_tmp
            Vis_out["fg_init"]=vis_tmp2
            Vis_out["fg_feat"]=vis_tmp3
            Vis_out["qid"]=qid
            Vis_out["imageId"]=imageId
            Vis_out["name"]=name_list
            Vis_out["vis_fb"]=vis_fg

            Vis_out_list.append(Vis_out)


                



            w=w.reshape(-1)

            initial_TopK=20
            final_TopK=topk
            total_epochs=3

            current_TopK = int(initial_TopK - (initial_TopK - final_TopK) * epoch / (total_epochs - 1))
            current_TopK = max(current_TopK, final_TopK)    

            

            top_num= min(current_TopK, len(semantic_feats_corr))

            _, top_indices = torch.topk(w, top_num)
            selection_mask = torch.zeros_like(w)
            selection_mask[top_indices] = 1.0  

            selected_w = w * selection_mask
            reweighted_w = selected_w 

            semantic_feats_corr=semantic_feats_corr[reweighted_w!=0,...]
            reweighted_w=reweighted_w.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            w=reweighted_w[reweighted_w!=0].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

            w=F.softmax(w,dim=0)


            if need_faith_metric:
                output = torch.sum(semantic_feats_corr * w, dim=0, keepdim=True)
                output = self.weight_roi2(output).sigmoid()
                output = F.interpolate(output, (240, 320))
                initial_score = self.compute_score(output, gt_op[sample_idx][0,:,:], fixMap=fixmap[sample_idx][0,:,:])  # 初始分数

                sorted_w, indices = torch.sort(w.reshape(-1), descending=True)  
                cumulative_ratios = torch.cumsum(sorted_w, dim=0) / sorted_w.sum()  
                relative_scores = []  
                aopc_diffs = []
                lodds_scores = []

                cumulative_ratios= torch.cat((torch.tensor([0]).to(cumulative_ratios.device), 
                                              cumulative_ratios))

                for i, ratio in enumerate(cumulative_ratios):
                    masked_w = w.clone().reshape(-1)
                    masked_w[indices[:i]] = 0  

                    output = torch.sum(semantic_feats_corr * masked_w.view(*masked_w.shape, 1, 1, 1), dim=0, keepdim=True)
                    output = self.weight_roi2(output).sigmoid()
                    output = F.interpolate(output, (240, 320))
                    score = self.compute_score(output, gt_op[sample_idx][0,:,:], fixMap=fixmap[sample_idx][0,:,:])  # 遮蔽后的分数

                    if math.isnan(score):
                        # a=0
                        score=0
                    relative_score = (score / initial_score)
                    relative_scores.append(relative_score)
                    aopc_diffs.append(initial_score-score)

                    lodds_score = torch.tensor(score / initial_score)
                    lodds_score = torch.clamp(lodds_score, min=1e-7)
                    lodds_score = torch.log(lodds_score)
                    lodds_scores.append(lodds_score)


            
                relative_scores = torch.tensor(relative_scores)

                if len(relative_scores) > 1:
    
                    auc_score = auc(cumulative_ratios.squeeze().cpu().numpy(), relative_scores)
                    if "auc" not in faith_metric_list:
                        faith_metric_list["auc"]=[]
                    faith_metric_list["auc"].append(auc_score)
                    if "aopc" not in faith_metric_list:
                        faith_metric_list["aopc"]=[]
                    faith_metric_list["aopc"].append(np.mean(aopc_diffs))
                    if "lodds" not in faith_metric_list:
                        faith_metric_list["lodds"]=[]
                    faith_metric_list["lodds"].append(np.mean(lodds_scores))


            reweighted_w_vis=reweighted_w.clone()
            reweighted_w_vis[reweighted_w!=0]=F.softmax(reweighted_w_vis[reweighted_w!=0],dim=0)
            reweighted_w_vis=reweighted_w_vis.squeeze(1).squeeze(1).squeeze(1)

      
            semantic_feats_corr = torch.sum(semantic_feats_corr * w, dim=0, keepdim=True)

            semantic_feats_list.append(semantic_feats_corr)


            Vis_final_list.append(Vis_final)

        semantic_feats_list = torch.stack(semantic_feats_list, dim=0).squeeze(1)
        a=0

        pooled_features_list=self.weight_roi2(semantic_feats_list)

        pooled_features_list = pooled_features_list.sigmoid()



        ret_corr=F.interpolate(pooled_features_list, (240, 320))
        ret_incorr=None
        if len(faith_loss)==0:
            ret_incorr=torch.tensor(0.0)
        else:
            ret_incorr=torch.stack(faith_loss).mean()

        if need_faith_metric:

            faith_metric_list=[faith_metric_list["auc"], 
                            faith_metric_list["aopc"], faith_metric_list["lodds"]]
            
        ret_incorr=[ret_incorr, faith_metric_list]

        return ret_corr, ret_incorr, [tmp[2] for tmp in cur_scene_graph_list]
     