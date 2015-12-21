#include <algorithm>
#include <cfloat>
#include <cmath>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void AggregateProbabilityLossLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);
  LayerParameter softmax_param(this->layer_param_);
  softmax_param.set_type("Softmax");
  softmax_layer_ = LayerRegistry<Dtype>::CreateLayer(softmax_param);
  softmax_bottom_vec_.clear();
  softmax_bottom_vec_.push_back(bottom[0]);
  softmax_top_vec_.clear();
  softmax_top_vec_.push_back(&prob_);
  softmax_layer_->SetUp(softmax_bottom_vec_, softmax_top_vec_);

  has_ignore_label_ =
    this->layer_param_.loss_param().has_ignore_label();
  if (has_ignore_label_) {
    ignore_label_ = this->layer_param_.loss_param().ignore_label();
  }
  normalize_ = this->layer_param_.loss_param().normalize();
  
  
  if (bottom.size() < 3) {
    CHECK(this->layer_param_.infogain_loss_param().has_source())
        << "Infogain matrix source must be specified.";
    BlobProto blob_proto;
    ReadProtoFromBinaryFile(
      this->layer_param_.infogain_loss_param().source(), &blob_proto);
    infogain_.FromProto(blob_proto);
  }
}

template <typename Dtype>
void AggregateProbabilityLossLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  Blob<Dtype>* infogain = NULL;
  if (bottom.size() < 3) {
    infogain = &infogain_;
  } else {
    infogain = bottom[2];
  }
  CHECK_EQ(bottom[1]->channels(), 1);
  CHECK_EQ(bottom[1]->height(), 1);
  CHECK_EQ(bottom[1]->width(), 1);
  const int num = bottom[0]->num();
  const int dim = bottom[0]->count() / num;
  CHECK_EQ(infogain->num(), 1);
  CHECK_EQ(infogain->channels(), 1);
  CHECK_EQ(infogain->height(), dim);
  CHECK_EQ(infogain->width(), dim);
  
  softmax_layer_->Reshape(softmax_bottom_vec_, softmax_top_vec_);
  softmax_axis_ =
      bottom[0]->CanonicalAxisIndex(this->layer_param_.softmax_param().axis());
  outer_num_ = bottom[0]->count(0, softmax_axis_);
  inner_num_ = bottom[0]->count(softmax_axis_ + 1);
  CHECK_EQ(outer_num_ * inner_num_, bottom[1]->count())
      << "Number of labels must match number of predictions; "
      << "e.g., if softmax axis == 1 and prediction shape is (N, C, H, W), "
      << "label count (number of labels) must be N*H*W, "
      << "with integer values in {0, 1, ..., C-1}.";
  if (top.size() >= 2) {
    // softmax output
    top[1]->ReshapeLike(*bottom[0]);
  }
}


template <typename Dtype>
void AggregateProbabilityLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  
  // The forward pass computes the softmax prob values.
  softmax_layer_->Forward(softmax_bottom_vec_, softmax_top_vec_);
  const Dtype* prob_data = prob_.cpu_data();      
  const Dtype* label = bottom[1]->cpu_data();
  int dim = prob_.count() / outer_num_;
  int count = 0;
  Dtype loss = 0;
  for (int i = 0; i < outer_num_; ++i) {
    for (int j = 0; j < inner_num_; j++) {
      const int label_value = static_cast<int>(label[i * inner_num_ + j]);
      if (has_ignore_label_ && label_value == ignore_label_) {
        continue;
      }
      DCHECK_GE(label_value, 0);
      DCHECK_LT(label_value, prob_.shape(softmax_axis_));
      loss -= log(std::max(prob_data[i * dim + label_value * inner_num_ + j],
                           Dtype(FLT_MIN)));
    ++count;
  }
  }
  
 
  if (normalize_) {
    top[0]->mutable_cpu_data()[0] = loss / count;
  } else {
    top[0]->mutable_cpu_data()[0] = loss / outer_num_;
  }
  if (top.size() == 2) {
    top[1]->ShareData(prob_);
  }
}

static bool printed = false;

template <typename Dtype>
void AggregateProbabilityLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to label inputs.";
  }
  if (propagate_down.size() > 2 && propagate_down[2]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to infogain inputs.";
  }
  if (propagate_down[0]) {
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    const Dtype* prob_data = prob_.cpu_data();
    caffe_copy(prob_.count(), prob_data, bottom_diff);
    const Dtype* label = bottom[1]->cpu_data();
    int dim = prob_.count() / outer_num_;
    int count = 0;
    
    const Dtype* infogain_mat = NULL;
    if (bottom.size() < 3) {
      infogain_mat = infogain_.cpu_data();
    } else {
      infogain_mat = bottom[2]->cpu_data();
    }
    
    if(!printed)
    {
        for(int i=0;i<dim;i++)
        {
            LOG(ERROR) << infogain_mat[i*dim] << " " << infogain_mat[i*dim+1] << " " << infogain_mat[i*dim+2] << " " << infogain_mat[i*dim+3] << " "<< infogain_mat[i*dim+4] << " "<< infogain_mat[i*dim+5] << " "<< infogain_mat[i*dim+6];
        }
        printed = true;
    }

   for (int i = 0; i < outer_num_; ++i) {
      for (int j = 0; j < inner_num_; ++j) {
        const int label_value = static_cast<int>(label[i * inner_num_ + j]);
        if (has_ignore_label_ && label_value == ignore_label_) {
          for (int c = 0; c < bottom[0]->shape(softmax_axis_); ++c) {
            bottom_diff[i * dim + c * inner_num_ + j] = 0;
          }
        } else {
  /*          if (label_value == (dim - 1)) {
                int high_label = 0;
                Dtype high_prob = bottom_diff[i * dim + 0 * inner_num_ + j];
                for (int k = 1; k < dim; k++) {
                    if (bottom_diff[i * dim + k * inner_num_ + j] > high_prob) {
                        high_label = k;
                        high_prob = bottom_diff[i * dim + k * inner_num_ + j];
                    }
                }
                if (high_label != label_value && high_prob <= (Dtype)0.5) {
                    //Dtype t_prob = bottom_diff[i * dim + label_value * inner_num_ + j];
                    bottom_diff[i * dim + label_value * inner_num_ + j] = 1;//1 - high_prob;
                    bottom_diff[i * dim + high_label * inner_num_ + j] = 0;//t_prob;
                }
            }
            bottom_diff[i * dim + label_value * inner_num_ + j] -= 1;
            ++count;*/
            int high_label = 0;
            Dtype high_prob = bottom_diff[i * dim + 0 * inner_num_ + j];
            for (int k = 1; k < dim; k++) {
                        if (bottom_diff[i * dim + k * inner_num_ + j] > high_prob) {
                    high_label = k;
                            high_prob = bottom_diff[i * dim + k * inner_num_ + j];
                }
            }
            
            if(label_value == high_label)
            {
                if(infogain_mat[label_value * dim + label_value] > 0 && high_prob >= infogain_mat[label_value * dim + label_value])
                {
                    for (int k = 0; k < dim; k++) {
                      if(k != label_value && infogain_mat[label_value * dim + k] != 0) {
                        bottom_diff[i * dim + label_value * inner_num_ + j] += prob_data[i * dim + k * inner_num_ + j];
                        bottom_diff[i * dim + k * inner_num_ + j] = 0;
                      }
                      
                    }                
                }    
            }
            //used for the unknown class
            else if(infogain_mat[label_value * dim + label_value] < 0 && high_prob <= (Dtype)0.5) {
                bottom_diff[i * dim + label_value * inner_num_ + j] = 1;//1 - high_prob;
                bottom_diff[i * dim + high_label * inner_num_ + j] = 0;//t_prob;
            }

            bottom_diff[i * dim + label_value * inner_num_ + j] -= 1;
            ++count;
             
        }
      }
    }
    // Scale gradient
    const Dtype loss_weight = top[0]->cpu_diff()[0];
 
    if (normalize_) {
      caffe_scal(prob_.count(), loss_weight / count, bottom_diff);
    } else {
      caffe_scal(prob_.count(), loss_weight / outer_num_, bottom_diff);
    }
  }
}

INSTANTIATE_CLASS(AggregateProbabilityLossLayer);
REGISTER_LAYER_CLASS(AggregateProbabilityLoss);
}  // namespace caffe
