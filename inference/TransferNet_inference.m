function TransferNet_inference(config) % only validation
fprintf('start inference [%s]\n', config.model_name);

%% initialization
load(config.cmap);

% configure voc dataset
VOCopts.seg.imgsetpath = '../data/VOC2012_SEG_AUG/ImageSets/%s.txt'; 
VOCopts.imgpath = '../data/VOC2012_SEG_AUG/images/%s.png';
VOCopts.seg.clsimgpath = '../data/VOC2012_SEG_AUG/segmentations/%s.png';

% load models
addpath(fullfile(config.Path.CNN.caffe_root, 'matlab/caffe'));
fprintf('load caffe models..\n');
caffe.set_mode_gpu();
caffe.set_device(config.gpuNum);
net = caffe.Net(config.Path.CNN.model_proto, config.Path.CNN.model_data, 'test');
cls_net = caffe.Net(config.Path.CNN.cls_model_proto, config.Path.CNN.cls_model_data, 'test');
fprintf('done\n');

% set class information
cls_list = 1:80; % it is becuase our model is trained on coco+voc categories.
numCls = length(cls_list);

%% initialize paths
save_res_dir = sprintf('%s/%s',config.save_root ,config.model_name);
save_res_path = [save_res_dir '/%s.png'];

%% create directory
if config.write_file
    if ~exist(save_res_dir), mkdir(save_res_dir), end
end

fprintf('start generating result\n');
fprintf('caffe model: %s\n', config.Path.CNN.model_proto);
fprintf('caffe weight: %s\n', config.Path.CNN.model_data);

%% read VOC2012 TEST image set
ids=textread(sprintf(VOCopts.seg.imgsetpath, config.imageset), '%s');

for i=1:length(ids)
    fprintf('progress: %d/%d [%s]...', i, length(ids), ids{i});  
    tic;
    
    % read image
    I=imread(sprintf(VOCopts.imgpath,ids{i}));
    % read cls and inst segmentation ground truth    
    [cls_seg,cmap_]=imread(sprintf(VOCopts.seg.clsimgpath,ids{i}));
    cls_ids = setxor([unique(cls_seg(:)); 0; 255]', [0,255]);
    
    im_sz = max(size(I,1),size(I,2));
    caffe_im_prepare = padarray(I,[im_sz - size(I,1), im_sz - size(I,2)],'post');
    caffe_im = preprocess_image(caffe_im_prepare, config.im_sz);
    
    tic;

    %% apply classification
    caffe_im_cls = preprocess_image(caffe_im_prepare, 320);
    cls_net.forward({caffe_im_cls});
    prediced_labels = cls_net.blobs('cls-score-sigmoid').get_data();
    
    cls_ids = find(squeeze(prediced_labels)>=0.6);
    if isempty(cls_ids)
       [~,cls_ids] = max(squeeze(prediced_labels));
    end
    if size(cls_ids,1) > 1
        cls_ids = cls_ids';
    end
    score_map = zeros([size(I,1),size(I,2), numCls]);
    
    %% compute segmentation     
    for j = cls_ids
        
        label = zeros([1,1,numCls]);
        label(j) = 1;
        label = single(label);
        tic;
        if isfield(config, 'batch_size')
            batch_size = config.batch_size;
            caffe_im_batch = zeros([size(caffe_im), batch_size], 'single');
            label_batch = zeros([size(label), batch_size], 'single');
            for bidx=1:batch_size
                caffe_im_batch(:,:,:,bidx) = caffe_im;
                label_batch(:,:,:,bidx) = label;
            end        
        else
            caffe_im_batch = cat(4, caffe_im, caffe_im, caffe_im, caffe_im);
            label_batch = cat(4, label, label, label, label);            
        end

        net.forward({caffe_im_batch;label_batch});
        fprintf('[%d:%f]',j,toc);
        seg_score = net.blobs('seg-score').get_data();
        
        %% do CRF
        prob = exp(seg_score(:,:,:,1));
        prob(:,:,1) = prob(:,:,1)/2; % give more importance to foreground probs
        prob = prob ./ repmat(sum(prob,3), [1,1,2]);
        
        resized_prob = imresize(prob, [im_sz, im_sz]); 
        resized_prob = permute(resized_prob, [2,1,3]);
        resized_prob = single(resized_prob(1:size(I,1),1:size(I,2),:));
        unary = -log(resized_prob);

        D = Densecrf(I,single(unary)); 

        D.gaussian_x_stddev = 3;
        D.gaussian_y_stddev = 3;
        D.gaussian_weight = 3; 
        D.bilateral_x_stddev = 20;
        D.bilateral_y_stddev = 20;
        D.bilateral_r_stddev = 3;
        D.bilateral_g_stddev = 3;
        D.bilateral_b_stddev = 3;
        D.bilateral_weight = 5;     
        D.iterations = 10;
        D.mean_field;
        score_map(:,:,j) = resized_prob(:,:,2).*(D.segmentation==2);
    end
    
    [segval, segmask] = max(score_map, [], 3);
    segmask(segval==0)=0;
    segmask = uint8(segmask);
                
    if config.write_file
        imwrite(segmask,cmap,sprintf(save_res_path, ids{i}));        
    else
        subplot(1,2,1);
        imshow(I);
        subplot(1,2,2);
        resulting_seg_im = reshape(cmap(int32(segmask)+1,:),[size(segmask,1),size(segmask,2),3]);
        imshow(resulting_seg_im);
        waitforbuttonpress;
    end
    fprintf(' done\n');
end

%%function end
end


function [ softmax_probs ] = softmax_( seg_output )
% compute softmax value of segmentation output
N = size(seg_output,4);
softmax_probs = cell(N,1);
for i=1:N
    probs = seg_output(:,:,:,i);
    probs = exp(probs);
    softmax_probs{i} = probs(:,:,2) ./ sum(probs,3);
end
end

