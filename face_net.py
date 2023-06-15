from facenet_pytorch import MTCNN, InceptionResnetV1
from torchreid.data.transforms import build_transforms
from PIL import Image
import torchreid
import torch
from torchreid import metrics
import numpy as np



class FaceNet:

    def __init__(self, image_size = 160):
        self.use_gpu = torch.cuda.is_available()
        self.mtcnn = MTCNN(image_size=image_size, keep_all=True)
        self.resnet = InceptionResnetV1(pretrained='vggface2').eval()
        if self.use_gpu:
            self.mtcnn=self.mtcnn.cuda()
            self.resnet=self.resnet.cuda()
        self.dist_metric = 'euclidean'
        _, self.transform_te = build_transforms(
            height=256, width=128,
            random_erase=False,
            color_jitter=False,
            color_aug=False
        )

    def _extract_features(self, input,x):
        self.mtcnn.eval()
        self.resnet.eval()
        # print("input:",input.dtype)
        try:
            img_cropped = self.mtcnn(input, save_path="results/face/output"+str(x)+".jpg")
        except:
            img_cropped = None
        # print("*img_cropped",img_cropped)
        if img_cropped != None:
            img_embedding = self.resnet(img_cropped)
            return img_embedding.detach()
        
        return None
    def _features(self, imgs):
        f = []
        x=0
        for img in imgs:
            img = np.array(Image.fromarray(img.astype('uint8')).convert('RGB'))
            features = self._extract_features(img,x)
            x+=1
            if features is not None:
                try:
                    features = features.data.cpu()  # tensor shape=1x2048
                    f.append(features)
                except AttributeError:
                    return None
        print("length of features: ",len(f))
        try:
            f = torch.cat(f, 0)
        except:
            print("***In exception part")
            f = None
        return f
    # def _features(self, imgs):
    #     f = []
    #     x=0
    #     for img in imgs:
    #         img = np.array(Image.fromarray(img.astype('uint8')).convert('RGB'))
    #         # img = self.transform_te(img)
    #         # img = torch.unsqueeze(img, 0)
    #         # if self.use_gpu:
    #         #     img = img.cuda()
    #         # print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$",img.shape)
            
    #         features = self._extract_features(img,x)
    #         x+=1
    #         # if features == None :
    #         #     continue
    #         features = features.data.cpu()  # tensor shape=1x2048
    #         if features is not None:
    #             f.append(features)
    #         if len(f) == 0:
    #             return None
    #         f = torch.cat(f, 0)
    #     return f
        #     f.append(features)
        # try : 
        #     f = torch.cat(f, 0)
        # except:
        #     f = None
        # return f

    def compute_distance(self, qf, gf):
        distmat = metrics.compute_distance_matrix(qf, gf, self.dist_metric)
        # print(distmat.shape)
        return distmat.numpy()