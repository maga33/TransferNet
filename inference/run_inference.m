function run_inference(model_name, model_data, model_proto)
  clear all; close all; clc;
  if nargin < 1 
    model_name = 'TransferNet_demo';
    model_data = '../models/transferNet.caffemodel';
    model_proto = '../training/2_train_segmentation/deploy.prototxt';
  end
  assert(exist(model_data)~=0, sprintf('there is no model data in %s', model_data));
  assert(exist(model_proto)~=0, sprintf('there is no model definition file in %s', model_proto));
  
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
  config.model_name = model_name
  config.Path.CNN.model_data = model_data
  config.Path.CNN.model_proto = model_proto
  config.batch_size = 1;
  config.im_sz = 320;

  % configuration for classifier
  clsNet_path = '../models';
  config.Path.CNN.cls_model_data = [clsNet_path, '/classificationNet.caffemodel'];
  config.Path.CNN.cls_model_proto = [clsNet_path, '/classificationNet.prototxt'];

  TransferNet_inference(config);
  caffe.reset_all
end
