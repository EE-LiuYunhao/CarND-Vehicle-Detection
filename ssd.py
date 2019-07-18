"""Python implementation of SSD."""

from ssd_layers import Normalize
from ssd_layers import PriorBox

import numpy as np
import torch
import torch.nn as nn

NUM_PIROR = 3
NUM_CLASS = 21

class SSD300(nn.Module):
    def __init__(self, img_size):
        super(SSD300, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3,64,3,1,1),
            nn.ReLU(),
            nn.Conv2d(64,64,3,1,1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64,128,3,1,1),
            nn.ReLU(),
            nn.Conv2d(128,128,3,1,1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(128,256,3,1,1),
            nn.ReLU(),
            nn.Conv2d(256,256,3,1,1),
            nn.ReLU(),
            nn.Conv2d(256,256,3,1,1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(256,512,3,1,1),
            nn.ReLU(),
            nn.Conv2d(512,512,3,1,1),
            nn.ReLU(),
            nn.Conv2d(512,512,3,1,1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(512,512,3,1,1),
            nn.ReLU(),
            nn.Conv2d(512,512,3,1,1),
            nn.ReLU(),
            nn.Conv2d(512,512,3,1,1),
            nn.ReLU(),
            nn.MaxPool2d(3)
        )
        self.atro6 = nn.Sequential(
            nn.Conv2d(512,1024,3,1,1,dilation=6,bias=False),
            nn.ReLU()
        )
        self.onek7 = nn.Sequential(
            nn.Conv2d(1024,1024,1,1,0),
            nn.ReLU()
        )
        self.conv6 = nn.Squential(
            nn.Conv2d(1024,512,1,1,0),
            nn.ReLU(),
            nn.Conv2d(256,512,3,2,1),
            nn.ReLU()
        )
        self.conv7 = nn.Sequential(
            nn.Conv2d(512,128,1,1,0),
            nn.ReLU(),
            nn.Conv2d(128,256,3,2,1),
            nn.ReLU()
        )
        self.conv8 = nn.Sequential(
            nn.Conv2d(256,128,1,1,0),
            nn.ReLU(),
            nn.Conv2d(128,256,3,2,1),
            nn.ReLU()
        )
        self.lastPool = nn.AvgPool2d(2)
        self.conv4_mbox = nn.Sequential(
            nn.BatchNorm2d(),
            nn.Conv2d(512,NUM_PIROR*4,3,1,1)
        )
        self.conv4_conf = nn.Sequential(
            nn.BatchNorm2d(),
            nn.Conv2d(512,NUM_CLASS*NUM_PIROR,3,1,1)
        )
        self.conv4_prior = nn.Sequential(
            nn.BatchNorm2d(),
            PriorBox(img_size, 30.0,aspect_ratios=[2],
                    variance=[0.1,0.1,0.2,0.2])
        )
        self.onek7_mbox = nn.Conv2d(1024,2*NUM_PIROR*4,3,1,1)
        self.onek7_conf = nn.Conv2d(1024,2*NUM_PIROR*NUM_CLASS,3,1,1)
        self.onek7_prior = PriorBox(img_size, 60.0, 114.0, aspect_ratios=[2, 3],
                                    variance=[0.1,0.1,0.2,0.2])
        self.conv6_mbox = nn.Conv2d(256,2*NUM_PIROR*4,3,1,1)
        self.conv6_conf = nn.Conv2d(256,2*NUM_PIROR*NUM_CLASS,3,1,1)
        self.conv6_prior = PriorBox(img_size, 114.0, 168.0, aspect_ratios=[2, 3],
                                    variance=[0.1,0.1,0.2,0.2])
        self.conv7_mbox = nn.Conv2d(128,2*NUM_PIROR*4,3,1,1)
        self.conv7_conf = nn.Conv2d(128,2*NUM_PIROR*NUM_CLASS,3,1,1)
        self.conv7_prior = PriorBox(img_size, 168.0, 222.0, aspect_ratios=[2, 3],
                                    variance=[0.1,0.1,0.2,0.2])
        self.conv8_mbox = nn.Conv2d(128,2*NUM_PIROR*4,3,1,1)
        self.conv8_conf = nn.Conv2d(128,2*NUM_PIROR*NUM_CLASS,3,1,1)
        self.conv8_prior = PriorBox(img_size, 222.0, 276.0, aspect_ratios=[2, 3],
                                    variance=[0.1,0.1,0.2,0.2])
        self.lastPool_mbox = nn.Linear(256,NUM_PIROR*2*4)
        self.lastPool_conf = nn.Linear(256,2*NUM_CLASS*NUM_PIROR)
        self.lastPool_prior = PriorBox(img_size, 276.0, max_size=330.0, aspect_ratios=[2, 3],
                                        variances=[0.1, 0.1, 0.2, 0.2])

        
        
    def forward(self,x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        fm4 = self.conv4(x)
        fm5 = self.conv5(fm4)
        atr6 = atro6(fm5)
        fmo7 = onek7(atr6)
        fm6 = conv6(fmo7)
        fm7 = conv7(fm6)
        fm8 = conv8(fm7)
        pl8 = self.lastPool(fm8)
        #Predict from conv4
        conv4_mbox_loc = conv4_mbox(fm4)
        conv4_mbox_loc = conv4_mbox_loc.view(conv4_mbox_loc.size()[0],-1)
        conv4_mbox_conf = conv4_conf(fm4)
        conv4_mbox_conf = conv4_mbox_conf.view(conv4_mbox_conf.size()[0],-1)
        conv4_mbox_prio = conv4_prior(fm4)
        #Predict from onek7
        onek7_mbox_loc = onek7_mbox(fmo7)
        onek7_mbox_loc = onek7_mbox_loc.view(onek7_mbox_loc.size()[0],-1)
        onek7_mbox_conf = onek7_conf(fmo7)
        onek7_mbox_conf = onek7_mbox_conf.view(onek7_mbox_conf.size()[0],-1)
        onek7_mbox_prio = onek7_prior(fmo7)
        #Predict from conv6
        conv6_mbox_loc = conv6_mbox(fm6)
        conv6_mbox_loc = conv6_mbox_loc.view(conv6_mbox_loc.size()[0],-1)
        conv6_mbox_conf = conv6_conf(fm6)
        conv6_mbox_conf = conv6_mbox_conf.view(conv6_mbox_conf.size()[0],-1)
        conv6_mbox_prio = conv6_prior(fm6)
        #Predict from conv7
        conv7_mbox_loc = conv7_mbox(fm7)
        conv7_mbox_loc = conv7_mbox_loc.view(conv7_mbox_loc.size()[0],-1)
        conv7_mbox_conf = conv7_conf(fm7)
        conv7_mbox_conf = conv7_mbox_conf.view(conv7_mbox_conf.size()[0],-1)
        conv7_mbox_prio = conv7_prior(fm7)
        #Predict from conv8
        conv8_mbox_loc = conv8_mbox(fm8)
        conv8_mbox_loc = conv8_mbox_loc.view(conv8_mbox_loc.size()[0],-1)
        conv8_mbox_conf = conv8_conf(fm8)
        conv8_mbox_conf = conv8_mbox_conf.view(conv8_mbox_conf.size()[0],-1)
        conv8_mbox_prio = conv8_prior(fm8)
        #Predict from lastPool
        lastPool_mbox_loc = lastPool_mbox(pl8)
        lastPool_mbox_loc = lastPool_mbox_loc.view(lastPool_mbox_loc.size()[0],-1)
        lastPool_mbox_conf = lastPool_conf(pl8)
        lastPool_mbox_conf = lastPool_mbox_conf.view(lastPool_mbox_conf.size()[0],-1)
        pl8_reshaped = pl8.view((256,1,1))
        lastPool_mbox_prio = lastPool_prior(pl8_reshaped)
        #Merge the box arguments together
        out_loc = torch.cat([conv4_mbox_loc,onek7_mbox_loc,conv6_mbox_loc, conv7_mbox_loc, conv8_mbox_loc, lastPool_mbox_loc],dim=1)
        out_conf = torch.cat([conv4_mbox_conf,onek7_mbox_conf,conv6_mbox_conf, conv7_mbox_conf, conv8_mbox_conf, lastPool_mbox_conf],dim=1)
        out_prior = torch.cat([conv4_mbox_prio,onek7_mbox_prio,conv6_mbox_prio, conv7_mbox_prio, conv8_mbox_prio, lastPool_mbox_prio],dim=1)

        num_boxs = out_loc.size()[-1] // 4

        out_loc.view(num_boxs, 4)
        out_conf.view(num_boxs, NUM_CLASS)
        out_conf = nn.Softmax(out_conf)
        prediction = torch.cat([out_loc, out_conf, out_prior], dim=2)

        return prediction

