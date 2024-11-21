# from tkinter import N
import numpy as np
# import random
import os
# import time
# import operator
import torch
import torch.utils.data as data
import json
import gc
import cv2
from PIL import Image
from torchvision import transforms
# from skimage import io
from skimage.transform import resize
# import clip

class Batch_generator(data.Dataset):
    def __init__(self,mode='train'):



        agg_att_path="./data/air_data/AiR-D/aggregated_maps/fixmaps"
        self.agg_att_path=agg_att_path


        self.mode = mode

        prep_dir="./data/air_data/"
        que_dir="./data/air_data/"
        self.semantic = json.load(open(os.path.join(prep_dir,'semantic_mask_my.json')))
        self.question = json.load(open(os.path.join(que_dir,"val"+'_balanced_questions.json')))


        # self.nb_embedding = len(self.word2idx.keys())

        self.height=240
        self.width=320


        self.Q, self.Img, self.answer, self.Qid = self.init_data()



        self.transform = transforms.Compose([
                                transforms.Resize((self.height, self.width)),
                                transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                ])






    def init_data(self,):
        question = []
        answer = []
        imgid = []
        Qid = []


 
        json_tmp=None
        if self.mode == 'train':
            with open('./data/air_data/fixations/AiR_fixations_train.json', 'r') as f:
                json_tmp=json.load(f)
        elif self.mode == 'val':
            with open('./data/air_data/fixations/AiR_fixations_validation.json', 'r') as f:
                json_tmp=json.load(f)
        elif self.mode == 'test':
            with open('./data/air_data/fixations/AiR_fixations_test.json', 'r') as f:
                json_tmp=json.load(f)
        else:
            raise ValueError('Invalid mode')
        



        qid_json_list=[q_tmp['question_id'] for q_tmp in json_tmp]
        qid_json_list=list(set(qid_json_list))
        imgid_json_list=[q_tmp['image_id'][:-4] for q_tmp in json_tmp]
        imgid_json_list=list(set(imgid_json_list))
        a=0


        self.fix={}
        for fixation in json_tmp:
            key=fixation["imageId"]
            pos_x = np.array(fixation["X"]).astype(np.float32)
            pos_y = np.array(fixation["Y"]).astype(np.float32)
            pos_x = pos_x/fixation["width"]*self.width
            pos_y = pos_y/fixation["height"]*self.height

            pos_x = np.round(pos_x).astype(np.int32)
            pos_y = np.round(pos_y).astype(np.int32)

            pos_x[pos_x==self.width]=self.width-1
            pos_y[pos_y==self.height]=self.height-1

            tmp=np.zeros([self.height,self.width])
            tmp[pos_y,pos_x]=1

            if key not in self.fix.keys():
                self.fix[key]=[]
            self.fix[key].append(tmp)

            a=0

        for key in self.fix.keys():
            self.fix[key]=np.asarray(self.fix[key]).astype(np.float32).sum(axis=0)
            self.fix[key][self.fix[key]!=0]=1
            a=0



        gene_dict_llava=[]


        paired_info={}

        # for qid in qid_json_list:\
        from tqdm import tqdm
        for qid in tqdm(qid_json_list):
            if qid not in self.question.keys():
                print(qid,'qid not in self.question.keys()')
                continue

            cur_img = self.question[qid]['imageId']


            imgid.append(cur_img)
            Qid.append(qid)
            question.append(self.question[qid]['question'])
            answer.append(self.question[qid]['answer'])
            
            img_path = os.path.join("./data/air_data/stimuli",
                            str(cur_img)+'.jpg')
            tmp={}
            tmp['question_id']=qid
            tmp['image_id']=cur_img
            tmp['question']=self.question[qid]['question']
            tmp['answer']=self.question[qid]['answer']
            tmp['img_path']=img_path
            gene_dict_llava.append(tmp)
        

        with open('./data/air_data/sal91_sim_noqa.json', 'r') as f:
            text_content=json.load(f)
        if self.mode == 'val':
            with open('./data/air_data/sal91_sim_noqa_val.json', 'r') as f:
                text_content=json.load(f)


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


        a=0
        self.text_content=text_content


        self.paired_info=paired_info


        return question, imgid, answer, Qid


    def load_sal(self, agg_att_path, index):
        agg_att_path=os.path.join(agg_att_path,str(self.Qid[index])+'.png')
        # raise NotImplementedError(agg_att_path)
        cur_anno = cv2.imread(agg_att_path).astype('float32')
        cur_anno = cur_anno[:,:,0]
        cur_anno = cv2.resize(cur_anno,(320,240))
        if cur_anno.max() <= 0:
            raise NotImplementedError(agg_att_path)
        cur_anno /= cur_anno.max()
        agg_att = torch.from_numpy(cur_anno)
        cur_fix = agg_att.clone()
        cur_fix[cur_fix>0.1] = 1
        cur_fix[cur_fix <= 0.1] = 0

        return agg_att, cur_fix



    def __getitem__(self,index):

        question = self.Q[index]
        answer = self.answer[index]
        img_id = self.Img[index]
        qid = self.Qid[index]

        answer=answer+"###"+question

        prompts=[]
        for tmp_item in self.text_content[str(qid)]["matches"]:
            prompts.append(tmp_item["object"])

        answer=str(prompts)+answer
       
        img_path = os.path.join("./data/air_data/stimuli",
                                    str(img_id)+'.jpg')
        img = Image.open(img_path).convert('RGB')
        #size
        img_width, img_height = img.size
        if self.transform is not None:
            img = self.transform(img)
 
        ori_scene_graph = json.load(open('./data/air_data/sal_scene_graph.json'))
        cur_scene_graph = ori_scene_graph[str(img_id)]

        cur_scene_graph=[len(self.text_content[str(qid)]["matches"]), qid, qid in self.paired_info.keys()]
      
        semantic = self.semantic[self.Qid[index]]
       
        sg_mask=[]
        # instruction
        for cur in semantic:
            target_obj=[cur_[0] for cur_ in cur[1]]
            target_obj_name=[cur_[2] for cur_ in cur[1]]
            a=0
            
            sg_mask.append(cur[0]+" "+" and ".join(target_obj_name))

            a=0
      
        while len(sg_mask)<4:
            sg_mask.append(["The photo of saliency objects."])
        if len(sg_mask)>4:
            sg_mask=sg_mask[:4]
        a=0

        sg_mask=sg_mask+[question]
    
        for i in range(len(sg_mask)):
            # sg_mask[i]=clip.tokenize(sg_mask[i])
            sg_mask[i]=0
        question=sg_mask



        agg_att_path=self.agg_att_path
        agg_att_path_corr=self.agg_att_path
        agg_att_path_incorr=self.agg_att_path
        
        if qid not in self.paired_info.keys():
            agg_att_path_corr=agg_att_path
            agg_att_path_incorr=agg_att_path

        agg_att_corr, cur_fix_corr = self.load_sal(agg_att_path_corr, index)
        agg_att_incorr, cur_fix_incorr = self.load_sal(agg_att_path_incorr, index)

        cur_fix=self.fix[img_id]
        cur_fix = torch.from_numpy(cur_fix)
        cur_fix_corr = cur_fix.clone()
        cur_fix_incorr = cur_fix.clone()

          

        agg_att = torch.stack([agg_att_corr, agg_att_incorr], dim=0)
        cur_fix = torch.stack([cur_fix_corr, cur_fix_incorr], dim=0)


        bbox=[0]

        if self.mode == 'train':


            
            instr=[cur[0]+" "+ " and ".join([mytmp[2] for mytmp in cur[1]]) for cur in semantic]
            answer=answer+"###"+"###".join(instr)

            img_height, img_width = 240, 320
            a=0
            op=np.array(0)
            att_mask=np.array(0)

            
            return img, question, answer, op, att_mask, agg_att, cur_fix, cur_scene_graph, bbox
        else:

            return img, question, answer, agg_att, cur_fix, cur_scene_graph, bbox

    def __len__(self,):
        return len(self.Img)
