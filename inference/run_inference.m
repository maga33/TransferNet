function run_inference()
  clear all; close all; clc;
  addpath(genpath('./ext'));
  addpath(genpath('./util'));

  config.imageset = 'val';
  config.Path.CNN.caffe_root = '../caffe'; % caffe root path
  config.save_root = './results';         % result will be save in this directory
  addpath(genpath(fullfile(config.Path.CNN.caffe_root, '/matlab/')));

  %% configuration
  config.gpuNum = 0;
  config.cmap = './util/voc_gt_cmap.mat';
  config.write_file = 1;
  config.model_name = 'TransferNet_demo';
  config.Path.CNN.model_data = '../models/transferNet.caffemodel';
  config.Path.CNN.model_proto = '../training/2_train_segmentation/deploy.prototxt';
  config.batch_size = 1;
  config.im_sz = 320;

  % configuration for classifier
  clsNet_path = '../models';
  config.Path.CNN.cls_model_data = [clsNet_path, '/classificationNet.caffemodel'];
  config.Path.CNN.cls_model_proto = [clsNet_path, '/classificationNet.prototxt'];

  TransferNet_inference(config);
  caffe.reset_all
end
