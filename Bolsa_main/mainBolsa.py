#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main function
Check help for argpaser: $python mainBolsa.py -h 
OR check README.txt 

"""
import argparse
import trainModels as tm
import loadModels_and_test as lmat

class Train_Model:
    def __init__(self, model,  input_image = None, savefile = None):
        self.model = model
        self.img = input_image
        self.savefile=savefile
        
    def train(self):
        
        if self.model=="default":
            tm.train_default_mnist(self.savefile, self.img)
        elif self.model=="CNN":
            tm.train_CNN_mnist(self.savefile,self.img)
        else: #TODO: PUT FOR LOADING GOOGLE NET (this should probably be on the other class...)
            print("Choose another model, this one is not available yet...")
        
        
        
        
    



class Load_then_Classify:
    def __init__(self, model,  input_image = None):
        self.model = model
        self.img = input_image
        
    def loadAndTest(self):
        if self.model == None:
            print("No model path given...So we're using default model...")
            lmat.loadModel('Pretrained_models/default_model.h5', self.img)
        else:
            lmat.loadModel(self.model, self.img)


###############################################################################


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='mainBolsa.py')
    subparsers = parser.add_subparsers(help='sub-command help',dest='subparser_name')

########################################################################################################################

    # creation of parser for the training action...
    parser_train = subparsers.add_parser("train", help = "Choose your classifier to be trained.")
    
    parser_train.add_argument("--model", choices = ["default","CNN","Google"],
                        default = None, required=True, help = "Choose your classifier to be trained.")
    
    parser_train.add_argument("--save", metavar = "FILE", 
                              required=False,
                              default = None,
                              help = "Enter the path and name of the model (without .h5 in the end) to be saved, example: modeltest")
    
    parser_train.add_argument("--inputIMG", metavar="FILE",required=False,
                                 default=None,
                                 help="Enter path of the image file")
########################################################################################################################
    
    # creation of parser for the classifying&testing action...
    parser_classify = subparsers.add_parser("classify", help = "Choose the classifier to be loaded and tested.")
    
    parser_classify.add_argument("--model", metavar="FILE",
                        default = None, help = "Enter path of the classifier to be trained.")
    
    parser_classify.add_argument("--inputIMG", metavar="FILE",required=False,
                                 default=None,
                                 help="Enter path of the image file")
    
    

#########################################################################################################################
    print(parser.parse_args())
    args = parser.parse_args()
    
    #print(args.input_image)
    
    
    if( args.subparser_name=='train'):
        print("YOU CHOSE TRAINING")
        training = Train_Model(model = args.model , savefile= args.save,input_image=args.inputIMG)
        training.train()
        
    else:
        loadclassify = Load_then_Classify(model = args.model , input_image = args.inputIMG)
        loadclassify.loadAndTest()
    