# Parallel Encoder
This implementation assumes that the network is divided into two parts (encoder/decoder) and the parameter of the encoder is fixed. <br>
The encoder and decoder is processed by two different GPUs. <br>
Since the encoder parameter is fixed, forward propagation of the encoder is computed in parallel with forward/backward/update of the decoder. 
# Usage
* Define encoder.prototxt decoder.prototxt separately
  * The output blobs of the encoder should match with the input blobs of the decoder. 
* Specify `encoder_net` in solver.prototxt
* `caffe train2 -solver=[solver.prototxt] -weights2=[encoder.caffemodel] -gpu=[decoder_gpu],[encoder_gpu]` 

# Example
An example is located in `examples/parallel_encoder/`. <br>
You should define encoder/decoder prototxt separately as follows.
```
name: "Encoder"
layer {
  name: "data"
  type: "DummyData"
  top: "data"
  dummy_data_param {
    data_filler {
      type: "xavier"
    }
    shape {
      dim: 64
      dim: 3
      dim: 128
      dim: 128
    }
  }
}
layer {
  name: "label"
  type: "DummyData"
  top: "label"
  dummy_data_param {
    data_filler {
      type: "constant"
      value: 2.0
    }
    shape {
      dim: 64
      dim: 1
    }
  }
}
layer {
  name: "ip"
  type: "InnerProduct"
  bottom: "data"
  top: "ip"
  inner_product_param {
    num_output: 20
    weight_filler {
      type: "xavier"
    }
    bias_filler {
      type: "constant"
    }
  }
}
```

The decoder network should have input blobs that exactly correspond to the output blobs of the encoder as follows.
```
name: "Decoder"
# input blobs should be exactly same (name, dim) as the output blobs of the encoder
input: "ip"
input_shape { dim: 64 dim: 20 }
input: "label"
input_shape { dim: 64 dim: 1 }
layer {
  name: "ip2"
  type: "InnerProduct"
  bottom: "ip"
  top: "ip2"
  inner_product_param {
    num_output: 1
    weight_filler {
      type: "xavier"
    }
    bias_filler {
      type: "constant"
    }
  }
}
layer {
  name: "loss"
  type: "EuclideanLoss"
  bottom: "ip2"
  bottom: "label"
  top: "loss"
}
```
Finally, you should specify each network in the solver parameter as follows.
```
net: "decoder.prototxt"
encoder_net: "encoder.prototxt"
encoder_device_id: 2
device_id: 0
```
This solver parameter will put the encoder network on GPU 2 and the decoder network on GPU 0. <br>
Alternatively, the equivalent form in the command line is `caffe train2 --gpu=0,2`.
