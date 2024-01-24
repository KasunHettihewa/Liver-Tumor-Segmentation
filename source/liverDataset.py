import glob
import os
import numpy as np
import cv2

cv2.setNumThreads(0)
import torch
import torch.utils.data as data
from tqdm import tqdm



class DatasetSort(data.Dataset):
    def __init__(self, datasetRoot, setName, usedIndices, isTraining=True, liver=False, tumor=False, dataAug=False,
                 transform=None, LiTS=True):

        self.mean = torch.from_numpy(np.array([0.485, 0.456, 0.406])).float()
        self.std = torch.from_numpy(np.array([0.229, 0.224, 0.225])).float()
        self.isTraining = isTraining
        self.liver = liver
        self.tumor = tumor
        self.dataAug = dataAug
        self.transform = transform

        if LiTS==True:
            sort_root = 'E:/Datasets/Liver Dataset/LiTS17/Data_split/'
        else:
            sort_root = 'E:/Datasets/Liver Dataset/3D-IRCADb-1/Data_split/'

        patientList = glob.glob(sort_root + setName + '/*')

        numList = []

        for i in range(len(patientList)):
            _, num = os.path.basename(patientList[i]).split("_")
            numList.append(int(num))

        numList.sort()

        self.imageList = []
        self.labelList = []

        self.OriginalSliceCount = 0

        dataProgressbar = tqdm(usedIndices, total=len(usedIndices), dynamic_ncols=True)
        for idx in dataProgressbar:
            patientImagePath = datasetRoot + '/1.Original/train/liver_' + str(numList[idx]) + '/'
            patientImageFiles = sorted(glob.glob(patientImagePath + '*.png'))

            patientLabelPath = datasetRoot + '/1.Original/' + 'labels' + '/' + 'liver_' + str(numList[idx]) + '/'
            patientLabelsFiles = sorted(glob.glob(patientLabelPath + '*.png'))

            Flip_H_Path = datasetRoot + '/2.FlipH/'
            Flip_V_Path = datasetRoot + '/3.FlipV/'
            Rotate_Path = datasetRoot + '/4.Rotate90/'

            if len(patientImageFiles) != len(patientLabelsFiles):
                raise ValueError("Slice and label not match. Images: {}, Labels: {}".format(len(patientImageFiles),
                                                                                            len(patientLabelsFiles)))

            sortedimageList = []
            sortedlabelList = []

            Flip_H_imageList = []
            Flip_H_labelList = []

            Flip_V_imageList = []
            Flip_V_labelList = []

            Rotate_imageList = []
            Rotate_labelList = []


            for i, sample in enumerate(patientLabelsFiles):
                label = cv2.imread(sample)
                label_max_val = label.max()
                u, counts = np.unique(label, return_counts=True)
                SliceName, ext = os.path.splitext(os.path.basename(sample))

                if self.dataAug is False:
                    if self.liver == True and self.tumor == False and len(u) > 1:
                        sortedimageList.append(patientImageFiles[i])
                        sortedlabelList.append(patientLabelsFiles[i])

                    elif self.liver == False and self.tumor == True and (len(u) == 3 or label_max_val==255):
                        sortedimageList.append(patientImageFiles[i])
                        sortedlabelList.append(patientLabelsFiles[i])

                    elif self.liver == True and self.tumor == True and (len(u) == 3 or label_max_val==255):
                        sortedimageList.append(patientImageFiles[i])
                        sortedlabelList.append(patientLabelsFiles[i])

                    else:
                        sortedimageList = sortedimageList
                        sortedlabelList = sortedlabelList

                if self.dataAug is True:
                    if self.liver == True and self.tumor == False and len(u) > 1:
                        sortedimageList.append(patientImageFiles[i])
                        sortedlabelList.append(patientLabelsFiles[i])

                        Flip_H_imageList.append(Flip_H_Path + 'train/liver_' + str(idx) + '/' + SliceName + '.png')
                        Flip_H_labelList.append(Flip_H_Path + 'labels/liver_' + str(idx) + '/' + SliceName + '.png')

                        Flip_V_imageList.append(Flip_V_Path + 'train/liver_' + str(idx) + '/' + SliceName + '.png')
                        Flip_V_labelList.append(Flip_V_Path + 'labels/liver_' + str(idx) + '/' + SliceName + '.png')

                        Rotate_imageList.append(Rotate_Path + 'train/liver_' + str(idx) + '/' + SliceName + '.png')
                        Rotate_labelList.append(Rotate_Path + 'labels/liver_' + str(idx) + '/' + SliceName + '.png')

                    elif self.liver == False and self.tumor == True and len(u) == 3:
                        sortedimageList.append(patientImageFiles[i])
                        sortedlabelList.append(patientLabelsFiles[i])

                        Flip_H_imageList.append(Flip_H_Path + 'train/liver_' + str(idx) + '/' + SliceName + '.png')
                        Flip_H_labelList.append(Flip_H_Path + 'labels/liver_' + str(idx) + '/' + SliceName + '.png')

                        Flip_V_imageList.append(Flip_V_Path + 'train/liver_' + str(idx) + '/' + SliceName + '.png')
                        Flip_V_labelList.append(Flip_V_Path + 'labels/liver_' + str(idx) + '/' + SliceName + '.png')

                        Rotate_imageList.append(Rotate_Path + 'train/liver_' + str(idx) + '/' + SliceName + '.png')
                        Rotate_labelList.append(Rotate_Path + 'labels/liver_' + str(idx) + '/' + SliceName + '.png')

                    elif self.liver == True and self.tumor == True and len(u) == 3:
                        sortedimageList.append(patientImageFiles[i])
                        sortedlabelList.append(patientLabelsFiles[i])

                        Flip_H_imageList.append(Flip_H_Path + 'train/liver_' + str(idx) + '/' + SliceName + '.png')
                        Flip_H_labelList.append(Flip_H_Path + 'labels/liver_' + str(idx) + '/' + SliceName + '.png')

                        Flip_V_imageList.append(Flip_V_Path + 'train/liver_' + str(idx) + '/' + SliceName + '.png')
                        Flip_V_labelList.append(Flip_V_Path + 'labels/liver_' + str(idx) + '/' + SliceName + '.png')

                        Rotate_imageList.append(Rotate_Path + 'train/liver_' + str(idx) + '/' + SliceName + '.png')
                        Rotate_labelList.append(Rotate_Path + 'labels/liver_' + str(idx) + '/' + SliceName + '.png')

                    else:
                        sortedimageList = sortedimageList
                        sortedlabelList = sortedlabelList

            self.imageList = self.imageList + sortedimageList + Flip_H_imageList + Flip_V_imageList + Rotate_imageList
            self.labelList = self.labelList + sortedlabelList + Flip_H_labelList + Flip_V_labelList + Rotate_labelList

            self.OriginalSliceCount += len(sortedimageList)



            dataProgressbar.set_description("Data loading ---> Total Original slice images: {}, Total slice images: {} "
                                            .format(self.OriginalSliceCount, len(self.imageList)))

        #print('Total slice images', len(self.imageList))

    def __len__(self):
        return len(self.imageList)

    def __getitem__(self, idx):

        sliceImageOriginal = cv2.imread(self.imageList[idx])
        sliceLabelOriginal = cv2.imread(self.labelList[idx], cv2.IMREAD_GRAYSCALE)

        if self.transform is not None:
            transformed = self.transform(image=sliceImageOriginal, mask=sliceLabelOriginal)
            sliceImageOriginal = transformed['image']
            sliceLabelOriginal = transformed['mask']

        # Normalize by imagenet mean and std
        sliceImage = cv2.cvtColor(sliceImageOriginal, cv2.COLOR_BGR2RGB)
        sliceImage = sliceImage / 255.0
        sliceImage = torch.from_numpy(sliceImage).float()  # swap axis from H,W,C --> C,H,W
        sliceImage = (sliceImage - self.mean) / self.std
        sliceImage = sliceImage.permute(2, 0, 1)

        sliceLabel = sliceLabelOriginal.copy()


        if self.liver == True and self.tumor == False:
            sliceLabel[np.where(sliceLabel == 127)] = 1  # ==> Liver area
            sliceLabel[np.where(sliceLabel == 255)] = 1  # ==> Tumor area

        if self.liver == False and self.tumor == True:
            sliceLabel[np.where(sliceLabel == 127)] = 0  # ==> Background area
            sliceLabel[np.where(sliceLabel == 255)] = 2  # ==> Tumor area

        if self.liver == True and self.tumor == True:
            sliceLabel[np.where(sliceLabel == 127)] = 1  # ==> gray liver to label 1
            sliceLabel[np.where(sliceLabel == 255)] = 2  # ==> white tumor to label 2

        sliceLabel = torch.from_numpy(sliceLabel).long()
        if self.isTraining:
            return sliceImage, sliceLabel
        else:
            return sliceImage, sliceLabel, sliceImageOriginal, sliceLabelOriginal, self.imageList[idx]




class DatasetSortTumor(data.Dataset):
    def __init__(self, datasetRoot, setName, usedIndices, isTraining=True, liver=False, tumor=False, transform=None):

        self.mean = torch.from_numpy(np.array([0.485, 0.456, 0.406])).float()
        self.std = torch.from_numpy(np.array([0.229, 0.224, 0.225])).float()
        self.isTraining = isTraining
        self.liver = liver
        self.tumor = tumor

        self.transform = transform

        sort_root = 'E:/Datasets/Liver Dataset/LiTS17/Concatenated Data (Tumor)/'
        patientList = glob.glob(sort_root + setName + '/*')

        numList = []

        for i in range(len(patientList)):
            _, num = os.path.basename(patientList[i]).split("_")
            numList.append(int(num))

        numList.sort()

        self.imageList = []
        self.labelList = []

        self.OriginalSliceCount = 0

        dataProgressbar = tqdm(usedIndices, total=len(usedIndices), dynamic_ncols=True)
        for idx in dataProgressbar:
            patientImagePath = sort_root + setName + '/liver_' + str(numList[idx]) + '/'
            patientImageFiles = sorted(glob.glob(patientImagePath + '*.jpg'))

            sortedimageList = []
            sortedlabelList = []


            for i, sample in enumerate(patientImageFiles):

                SliceName, ext = os.path.splitext(os.path.basename(sample))

                image_path = datasetRoot + '/1.Original/train/liver_' + str(numList[idx]) + '/' + SliceName + '.png'
                label_path = datasetRoot + '/1.Original/labels/liver_' + str(numList[idx]) + '/' + SliceName + '.png'

                sortedimageList.append(image_path)
                sortedlabelList.append(label_path)



            self.imageList = self.imageList + sortedimageList
            self.labelList = self.labelList + sortedlabelList

            self.OriginalSliceCount += len(sortedimageList)

            if len(self.imageList) != len(self.labelList):
                raise ValueError("Slice and label not match. Images: {}, Labels: {}".format(len(self.imageList),
                                                                                            len(self.labelList)))



            dataProgressbar.set_description("Data loading ---> Total Original slice images: {}, Total slice images: {} "
                                            .format(self.OriginalSliceCount, len(self.imageList)))

        #print('Total slice images', len(self.imageList))

    def __len__(self):
        return len(self.imageList)

    def __getitem__(self, idx):

        sliceImageOriginal = cv2.imread(self.imageList[idx])
        sliceLabelOriginal = cv2.imread(self.labelList[idx], cv2.IMREAD_GRAYSCALE)


        # # # Resize data to 224x224
        # sliceImageOriginal = cv2.resize(sliceImageOriginal, (224,224), interpolation = cv2.INTER_NEAREST)
        # sliceLabelOriginal = cv2.resize(sliceLabelOriginal, (224,224), interpolation = cv2.INTER_NEAREST)


        if self.transform is not None:
            transformed = self.transform(image=sliceImageOriginal, mask=sliceLabelOriginal)
            sliceImageOriginal = transformed['image']
            sliceLabelOriginal = transformed['mask']

        # Normalize by imagenet mean and std
        sliceImage = cv2.cvtColor(sliceImageOriginal, cv2.COLOR_BGR2RGB)
        sliceImage = sliceImage / 255.0
        sliceImage = torch.from_numpy(sliceImage).float()  # swap axis from H,W,C --> C,H,W
        sliceImage = (sliceImage - self.mean) / self.std
        sliceImage = sliceImage.permute(2, 0, 1)

        sliceLabel = sliceLabelOriginal.copy()


        if self.liver == True and self.tumor == False:
            sliceLabel[np.where(sliceLabel == 127)] = 1  # ==> Liver area
            sliceLabel[np.where(sliceLabel == 255)] = 1  # ==> Tumor area

        if self.liver == False and self.tumor == True:
            sliceLabel[np.where(sliceLabel == 127)] = 0  # ==> Background area
            sliceLabel[np.where(sliceLabel == 255)] = 2  # ==> Tumor area

        if self.liver == True and self.tumor == True:
            sliceLabel[np.where(sliceLabel == 127)] = 1  # ==> gray liver to label 1
            sliceLabel[np.where(sliceLabel == 255)] = 2  # ==> white tumor to label 2

        sliceLabel = torch.from_numpy(sliceLabel).long()
        if self.isTraining:
            return sliceImage, sliceLabel
        else:
            return sliceImage, sliceLabel, sliceImageOriginal, sliceLabelOriginal, self.imageList[idx]



if __name__ == '__main__':
    import albumentations as A

    trainTransform = A.Compose([
        A.Flip(p=0.5),
        # A.RandomBrightnessContrast(p=0.2),
    ])



    o = liverDataset(datasetRoot="dataset/liver", setName="train", isTraining=False, usedIndices=list(range(0, 105)),
                     transform=trainTransform)
    sliceImage, sliceLabel, sliceImageOriginal, sliceLabelOriginal, imagePath = o.__getitem__(50)

    print(sliceImage.shape, sliceLabel.shape)

    print(sliceImageOriginal.shape, sliceLabelOriginal.shape, imagePath)
    cv2.imwrite('pic.jpg', sliceImageOriginal)
    cv2.imwrite('lab.jpg', sliceLabelOriginal)