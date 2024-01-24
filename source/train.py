import os
import cv2
import numpy as np
import pandas as pd
import segmentation_models_pytorch as smp

import albumentations as A
import albumentations.augmentations.geometric.rotate as R

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from liverDataset import liverDataset, livertumorDataset, DatasetAug, DatasetSort, DatasetSortTumor
from utils import AverageMeter
from evaluationMetrics import Dice_score, DiceLoss, BinaryDiceLoss

from torch.autograd import Variable

from torch.optim.lr_scheduler import StepLR, ExponentialLR
from warmup_scheduler import GradualWarmupScheduler
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
torch.manual_seed(1)
torch.cuda.manual_seed(1)


from models.res_unet_model import ResUNet
from models.ablationModels import AbModel_1, AbModel_2, AbModel_3, AbModel_4, AbModel_5, AbModel_6, AbModel_7
from DefED_NetModel.DefEDNet import DefED_Net

import timeit

def train(model,name, train_Dataloader, test_Dataloader, lr, totalEpoch):
    lr = lr
    totalEpoch = totalEpoch
    testEveryXEpoch = 1
    outputEveryXEpoch = 3

    trainDatasetLoader = train_Dataloader
    testDatasetLoader = test_Dataloader


    expName = name


    outputWeightDirectory = modelRoot + expName + '/weights/'
    outputEvalSampleDirectory = modelRoot + expName + '/samples/'
    os.makedirs(outputEvalSampleDirectory, exist_ok=True)
    os.makedirs(outputWeightDirectory, exist_ok=True)

    writerDirectory = modelRoot + expName + '/trainingSummary/outputData.xlsx'


    net = model
    net.cuda()

    # criterion = nn.CrossEntropyLoss().cuda()
    criterion = DiceLoss().cuda()
    # criterion = BinaryDiceLoss().cuda()


    optimizer = optim.Adam(net.parameters(), lr=lr)



    # scheduler_warmup is chained with schduler_steplr
    scheduler_steplr = StepLR(optimizer, step_size=30, gamma=0.1)
    # scheduler_warmup = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=3, after_scheduler=scheduler_steplr)
    # optimizer.zero_grad()
    # optimizer.step()

    scaler = torch.cuda.amp.GradScaler()
    trainLossAverager = AverageMeter()
    trainDiceAverage = AverageMeter()
    bestPixelAccuracy = 0
    bestDice = 0

    output = []

    for currentEpoch in range(totalEpoch):
        scheduler_steplr.step()
        # scheduler_warmup.step(currentEpoch)

        trainProgressbar = tqdm(enumerate(trainDatasetLoader), total=len(trainDatasetLoader), dynamic_ncols=True)
        for trainBatchIdx, trainBatchData in trainProgressbar:
            net.train()
            with torch.cuda.amp.autocast():  # Use mixed precision to acclerate training speed
                sliceImages, sliceLabels = trainBatchData
                sliceImages, sliceLabels = sliceImages.cuda(non_blocking=True), sliceLabels.cuda(non_blocking=True)

                trainPreds = net(sliceImages)

                loss = criterion(trainPreds, sliceLabels)

                trainDice = Dice_score(trainPreds, sliceLabels)


            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)

            # scheduler.step()


            scaler.update()
            trainLossAverager.update(loss.item())
            trainDiceAverage.update(trainDice.item())

            currentLR = round(optimizer.param_groups[0]['lr'], 6)
            trainProgressbar.set_description(
                "[{}][Train] avg_loss: {} loss: {} Dice_avg: {} Dice_score: {} lr:{}".format(currentEpoch + 1, round(trainLossAverager.avg, 4),
                round(trainLossAverager.val, 4), round(trainDiceAverage.avg, 4), round(trainDiceAverage.val, 4), currentLR))


        if currentEpoch % testEveryXEpoch == 0:
            net.eval()
            # Do eval here!
            testPixelAccuracyAverager = AverageMeter()
            testDiceAverage = AverageMeter()
            testProgressbar = tqdm(enumerate(testDatasetLoader), total=len(testDatasetLoader), dynamic_ncols=True)

            for testBatchIdx, testBatchData in testProgressbar:
                with torch.cuda.amp.autocast():  # Use mixed precision to acclerate training speed
                    with torch.no_grad():  # disable grad tracing in eval
                        sliceImagesT, sliceLabelsT, sliceImagesOriginal, sliceLabelsOriginal, imagePaths = testBatchData
                        sliceImagesT, sliceLabelsT = sliceImagesT.cuda(non_blocking=True), sliceLabelsT.cuda(
                            non_blocking=True)
                        testPreds = net(sliceImagesT)

                        predLabels = torch.argmax(testPreds, dim=1)


                        totalPixel = predLabels.nelement()
                        batchPixelAccuracy = torch.sum(torch.eq(predLabels,
                                                                sliceLabelsT)).item() / totalPixel  
                        testPixelAccuracyAverager.update(batchPixelAccuracy)

                        testDice = Dice_score(testPreds, sliceLabelsT)
                        testDiceAverage.update(testDice.item())

                        testProgressbar.set_description(
                            "[{}][Test] best_pixelacc:{} cur_pixelacc:{} Dice_avg: {} Dice_score: {}".format(currentEpoch + 1, round(bestPixelAccuracy, 5),
                            round(testPixelAccuracyAverager.val, 5), round(testDiceAverage.avg,4), round(testDiceAverage.val, 4)))




                        if testBatchIdx == 0:
                            testPreds= F.softmax(testPreds, dim=1)
                            testPreds = torch.max(testPreds, dim=1)
                            testPreds = testPreds[1]

                            predLabelsNP = testPreds.cpu().numpy()

                            predLabelsNP[np.where(predLabelsNP == 1)] = 127
                            predLabelsNP[np.where(predLabelsNP == 2)] = 255

                            for imageIdx in range(sliceImagesOriginal.shape[0]):
                                sliceImageOriginalNP = sliceImagesOriginal[imageIdx].numpy()

                                sliceLabelOriginalNP = sliceLabelsOriginal[imageIdx].numpy()
                                sliceLabelOriginalNP = np.stack([sliceLabelOriginalNP] * 3, axis=2)

                                slicePred = predLabelsNP[imageIdx, :, :]
                                slicePred = np.stack([slicePred] * 3, axis=2)

                                outputImage = np.concatenate([sliceImageOriginalNP, sliceLabelOriginalNP, slicePred],
                                                             axis=1)

                                cv2.imwrite(outputEvalSampleDirectory + str(currentEpoch+1).zfill(3) + '_' + str(imageIdx).zfill(2) + '.jpg', outputImage)


            output.append([currentEpoch + 1, round(trainLossAverager.avg, 4), round(trainDiceAverage.avg, 4), round(testDiceAverage.avg,4), currentLR])


            if testDiceAverage.avg > bestDice:  # best weight save criterion
                bestDice = testDiceAverage.avg
                # do something here when new best is found
                torch.save(
                    {
                        'state_dict': net.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'acuracy': bestPixelAccuracy,
                        'epoch': currentEpoch
                    }, outputWeightDirectory + 'best.pth')

            # output string
            outputString = '[{}][Test] best_pixelacc:{} cur_pixelacc:{} Dice_avg: {} Dice_score: {}'.format(currentEpoch + 1, round(bestPixelAccuracy, 5),
                            round(testPixelAccuracyAverager.avg, 5), round(testDiceAverage.avg,4), round(testDiceAverage.val, 4))
            print(outputString)


            # save current epoch weight
            torch.save(
                {
                    'state_dict': net.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'acuracy': bestPixelAccuracy,
                    'epoch': currentEpoch
                }, outputWeightDirectory + str(currentEpoch) + '.pth')

            # Save data to excel file
            if (currentEpoch+1) % outputEveryXEpoch == 0:
                OutputData = pd.DataFrame(output, columns = ['Epoch', 'avg_loss', 'Train: avg_Dice', 'Test: avg_Dice', 'Learning Rate'])

                writer = pd.ExcelWriter(writerDirectory)
                OutputData.to_excel(writer, sheet_name='Data', index=False)
                writer.save()

    OutputData = pd.DataFrame(output, columns = ['Epoch', 'avg_loss', 'Train: avg_Dice', 'Test: avg_Dice', 'Learning Rate'])

    writer = pd.ExcelWriter(writerDirectory)
    OutputData.to_excel(writer, sheet_name='Data', index=False)
    writer.save()






if __name__ == '__main__':

    modelRoot = 'D:/Thesis Research Experiments/Liver_Segmentation/Decathlon Dataset (Liver)/SampleCodeData/modelOutput/'
    dataRoot = 'E:/Datasets/Liver Dataset/LiTS17/prepro_liverData'
    # dataRoot = 'D:/Thesis Research Experiments/Liver_Segmentation/Decathlon Dataset (Liver)/SampleCodeData/prepro_liverData'

    transforms = A.Compose([
        A.VerticalFlip(p=0.4),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=20, interpolation=0, p=0.3),
    ])


    # liverTrainDataset = DatasetSort(datasetRoot=dataRoot, setName="train", isTraining=True,
    #                                 usedIndices=list(range(0, 104)), liver=False, tumor=True, dataAug=False, transform=transforms)
    #
    # trainDatasetLoader = DataLoader(liverTrainDataset, batch_size=4, num_workers=0, pin_memory=True, shuffle=True)
    #
    # liverTestDataset = DatasetSort(datasetRoot=dataRoot, setName="test", isTraining=False,
    #                                usedIndices=list(range(0, 26)), liver=False, tumor=True, dataAug=False, transform=transforms)
    #
    # testDatasetLoader = DataLoader(liverTestDataset, batch_size=8, num_workers=0, pin_memory=True, shuffle=True)





    liverTrainDataset = DatasetSortTumor(datasetRoot=dataRoot, setName="train", isTraining=True,
                                    usedIndices=list(range(0, 117)), liver=False, tumor=True, transform=transforms)       

    trainDatasetLoader = DataLoader(liverTrainDataset, batch_size=4, num_workers=0, pin_memory=True, shuffle=True)

    liverTestDataset = DatasetSortTumor(datasetRoot=dataRoot, setName="test", isTraining=False,
                                   usedIndices=list(range(0, 117)), liver=False, tumor=True, transform=transforms)

    testDatasetLoader = DataLoader(liverTestDataset, batch_size=8, num_workers=0, pin_memory=True, shuffle=True)

    print(len(trainDatasetLoader))



    lr = 1e-4
    totalEpoch = 50

    name = 'DefED_Net'
    model = DefED_Net(in_channels=3, num_classes=3)
    train(model, name, trainDatasetLoader, testDatasetLoader, lr, totalEpoch)
    print(f"\nCompleted :{name}\n\n")

    # name = 'AbModel_4'
    # model = AbModel_4(in_channels=3, n_classes=3)
    # train(model, name, trainDatasetLoader, testDatasetLoader, lr, totalEpoch)
    # print(f"\nCompleted :{name}\n\n")
    #
    # name = 'AbModel_5'
    # model = AbModel_5(in_channels=3, n_classes=3)
    # train(model, name, trainDatasetLoader, testDatasetLoader, lr, totalEpoch)
    # print(f"\nCompleted :{name}\n\n")

    # name = 'AbModel_7'
    # model = AbModel_7(in_channels=3, n_classes=3)
    # train(model, name, trainDatasetLoader, testDatasetLoader, lr, totalEpoch)
    # print(f"\nCompleted :{name}\n\n")


