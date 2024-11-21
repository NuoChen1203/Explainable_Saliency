import numpy as np
import os
import torch
import torch.utils.data as data
import json
import cv2
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
import re

class Batch_generator(data.Dataset):
    """
    Dataset loader for visual reasoning tasks.
    Loads images, questions, and attention maps for training/validation/testing.
    """
    def __init__(self, mode='train'):
        # Initialize paths and data
        self.agg_att_path = "./data/air_data/AiR-D/aggregated_maps/fixmaps"
        self.mode = mode
        
        # Load semantic masks and questions
        prep_dir = "./data/air_data/"
        self.semantic = json.load(open(os.path.join(prep_dir, 'semantic_mask_my.json')))
        self.question = json.load(open(os.path.join(prep_dir, "val" + '_balanced_questions.json')))
        
        # Set image dimensions
        self.height = 240
        self.width = 320
        
        # Initialize data
        self.Q, self.Img, self.answer, self.Qid = self.init_data()
        
        # Define image transformations
        self.transform = transforms.Compose([
            transforms.Resize((self.height, self.width)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def init_data(self):
        """Initialize dataset with questions, images, answers and QIDs"""
        question, answer, imgid, Qid = [], [], [], []
        
        # Load fixation data based on mode
        json_tmp = None
        if self.mode == 'train':
            json_tmp = json.load(open('./data/air_data/fixations/AiR_fixations_train.json'))
        elif self.mode == 'val':
            json_tmp = json.load(open('./data/air_data/fixations/AiR_fixations_validation.json'))
        elif self.mode == 'test':
            json_tmp = json.load(open('./data/air_data/fixations/AiR_fixations_test.json'))
        else:
            raise ValueError('Invalid mode')
        
        # Get unique question and image IDs
        qid_json_list = list(set([q_tmp['question_id'] for q_tmp in json_tmp]))
        
        # Process fixation data
        self.fix = {}
        for fixation in json_tmp:
            key = fixation["imageId"]
            # Convert fixation coordinates to image dimensions
            pos_x = np.array(fixation["X"]).astype(np.float32)
            pos_y = np.array(fixation["Y"]).astype(np.float32)
            pos_x = pos_x/fixation["width"]*self.width
            pos_y = pos_y/fixation["height"]*self.height
            
            pos_x = np.round(pos_x).astype(np.int32)
            pos_y = np.round(pos_y).astype(np.int32)
            
            # Clip to image boundaries
            pos_x[pos_x==self.width] = self.width-1
            pos_y[pos_y==self.height] = self.height-1
            
            # Create fixation map
            tmp = np.zeros([self.height, self.width])
            tmp[pos_y, pos_x] = 1
            
            if key not in self.fix.keys():
                self.fix[key] = []
            self.fix[key].append(tmp)
            
        # Aggregate fixation maps
        for key in self.fix.keys():
            self.fix[key] = np.asarray(self.fix[key]).astype(np.float32).sum(axis=0)
            self.fix[key][self.fix[key]!=0] = 1
        
        # Load and process text content
        text_content_path = './data/air_data/sal91_sim_noqa_val.json' if self.mode == 'val' else './data/air_data/sal91_sim_noqa.json'
        text_content = json.load(open(text_content_path))
        
        # Process text matches
        for key in tqdm(text_content.keys()):
            result = []
            for match in text_content[key]["matches"]:
                data_str = match.replace('\n', '')
                matches = re.findall(r'"object": "(.*?)",\s*"reason": "(.*?)"', data_str)
                result.extend([{"object": obj, "reason": reason} for obj, reason in matches])
            text_content[key]["matches"] = result
        
        self.text_content = text_content
        self.paired_info = {}
        
        # Process each question
        for qid in tqdm(qid_json_list):
            if qid not in self.question.keys():
                print(qid, 'qid not in self.question.keys()')
                continue
                
            cur_img = self.question[qid]['imageId']
            imgid.append(cur_img)
            Qid.append(qid)
            question.append(self.question[qid]['question'])
            answer.append(self.question[qid]['answer'])
            
        return question, imgid, answer, Qid

    def load_sal(self, agg_att_path, index):
        """Load and process saliency maps"""
        agg_att_path = os.path.join(agg_att_path, str(self.Qid[index])+'.png')
        cur_anno = cv2.imread(agg_att_path).astype('float32')
        cur_anno = cur_anno[:,:,0]
        cur_anno = cv2.resize(cur_anno, (320, 240))
        if cur_anno.max() <= 0:
            raise NotImplementedError(agg_att_path)
        cur_anno /= cur_anno.max()
        agg_att = torch.from_numpy(cur_anno)
        cur_fix = agg_att.clone()
        cur_fix[cur_fix>0.1] = 1
        cur_fix[cur_fix <= 0.1] = 0
        return agg_att, cur_fix

    def __getitem__(self, index):
        """Get a single item from the dataset"""
        # Load basic information
        question = self.Q[index]
        answer = self.answer[index]
        img_id = self.Img[index]
        qid = self.Qid[index]
        
        # Process answer and question
        answer = answer + "###" + question
        prompts = [item["object"] for item in self.text_content[str(qid)]["matches"]]
        answer = str(prompts) + answer
        
        # Load and transform image
        img_path = os.path.join("./data/air_data/stimuli", str(img_id)+'.jpg')
        img = Image.open(img_path).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
            
        # Get scene graph
        cur_scene_graph = [len(self.text_content[str(qid)]["matches"]), qid, qid in self.paired_info.keys()]
        
        # Process semantic information
        semantic = self.semantic[self.Qid[index]]
        sg_mask = []
        for cur in semantic:
            target_obj_name = [cur_[2] for cur_ in cur[1]]
            sg_mask.append(cur[0] + " " + " and ".join(target_obj_name))
            
        # Pad or trim sg_mask
        while len(sg_mask) < 4:
            sg_mask.append(["The photo of saliency objects."])
        if len(sg_mask) > 4:
            sg_mask = sg_mask[:4]
            
        sg_mask = sg_mask + [question]
        question = [0 for _ in sg_mask]  # Replace with tokenization if needed
        
        # Load saliency maps
        agg_att_corr, cur_fix_corr = self.load_sal(self.agg_att_path, index)
        agg_att_incorr, cur_fix_incorr = self.load_sal(self.agg_att_path, index)
        
        # Process fixation maps
        cur_fix = self.fix[img_id]
        cur_fix = torch.from_numpy(cur_fix)
        cur_fix_corr = cur_fix.clone()
        cur_fix_incorr = cur_fix.clone()
        
        agg_att = torch.stack([agg_att_corr, agg_att_incorr], dim=0)
        cur_fix = torch.stack([cur_fix_corr, cur_fix_incorr], dim=0)
        
        if self.mode == 'train':
            # Add instructions for training mode
            instr = [cur[0]+" "+ " and ".join([mytmp[2] for mytmp in cur[1]]) for cur in semantic]
            answer = answer + "###" + "###".join(instr)
            return img, question, answer, np.array(0), np.array(0), agg_att, cur_fix, cur_scene_graph, [0]
        else:
            return img, question, answer, agg_att, cur_fix, cur_scene_graph, [0]

    def __len__(self):
        """Return the size of the dataset"""
        return len(self.Img)
