/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * Copyright (c) 2015 by Contributors
 * \file batch_norm-inl_v1.h
 * \brief
 * \author Bing Xu
*/
#ifndef MXNET_OPERATOR_BATCH_NORM_V1_INL_H_
#define MXNET_OPERATOR_BATCH_NORM_V1_INL_H_

#include <dmlc/logging.h>
#include <dmlc/parameter.h>
#include <mxnet/operator.h>
#include <map>
#include <vector>
#include <string>
#include <utility>
#include "./operator_common.h"
#include "./mshadow_op.h"
#include <nccl.h>
#include <pthread.h>
#include <cuda_runtime.h>
#include <cstdlib>
#include <unistd.h>


namespace mxnet {
namespace op {

#define NCCLCHECK(cmd) do {                             \
    ncclResult_t r=cmd;                                 \
    if(r!=ncclSuccess){                                 \
        printf("failed:NCCL error %s:%d '%s'\n",        \
                __FILE__,__LINE__,ncclGetErrorString(r)); \
        exit(EXIT_FAILURE);                             \
    }                                                   \
}while(0)

#define CUDACHECK(cmd) do {                             \
    cudaError_t e=cmd;                                 \
    if(e!=cudaSuccess){                                 \
        printf("failed:Cuda error %s:%d '%s'\n",        \
                __FILE__,__LINE__,cudaGetErrorString(e)); \
        exit(EXIT_FAILURE);                             \
    }                                                   \
}while(0)


//#define MAX_GPU_NUM 16
//#define nGPUs 1
//static ncclComm_t commsFor[MAX_GPU_NUM];
//static ncclComm_t commsBack[MAX_GPU_NUM];
static ncclUniqueId uidForward;
static ncclUniqueId uidBackward;
//static bool flagForward = true;
//static bool flagBackward = true;
//static int initForward[MAX_GPU_NUM];
//static int initBackward[MAX_GPU_NUM];
static pthread_mutex_t mm = PTHREAD_MUTEX_INITIALIZER;
static pthread_barrier_t  barrierFor,barrierBack;
static bool flagBarrierFor = false;
static bool flagBarrierBack = false;
static int rankFor = 0;
static int rankBack = 0;
//static bool flagPrint = false;

namespace batchnorm_v1 {
enum BatchNormOpInputs {kData, kGamma, kBeta};
enum BatchNormOpOutputs {kOut, kMean, kVar};
enum BatchNormOpAuxiliary {kMovingMean, kMovingVar};
enum BatchNormBackResource {kTempSpace};
}  // namespace batchnorm_v1

struct BatchNormV1Param : public dmlc::Parameter<BatchNormV1Param> {
  float eps;
  float momentum;
  bool fix_gamma;
  bool use_global_stats;
  bool output_mean_var;
  bool global_bn;
  int nGPUs;
  DMLC_DECLARE_PARAMETER(BatchNormV1Param) {
    DMLC_DECLARE_FIELD(eps).set_default(1e-3f)
    .describe("Epsilon to prevent div 0");
    DMLC_DECLARE_FIELD(momentum).set_default(0.9f)
    .describe("Momentum for moving average");
    DMLC_DECLARE_FIELD(fix_gamma).set_default(true)
    .describe("Fix gamma while training");
    DMLC_DECLARE_FIELD(use_global_stats).set_default(false)
    .describe("Whether use global moving statistics instead of local batch-norm. "
              "This will force change batch-norm into a scale shift operator.");
    DMLC_DECLARE_FIELD(output_mean_var).set_default(false)
    .describe("Output All,normal mean and var");
    DMLC_DECLARE_FIELD(global_bn).set_default(false)
    .describe("Whether use global bn");
    DMLC_DECLARE_FIELD(nGPUs).set_default(1)
    .describe("The count of GPUs");

  }
};

template<typename xpu>
class BatchNormV1Op : public Operator {
 public:
  explicit BatchNormV1Op(BatchNormV1Param param) {
    this->param_ = param;
  }
//  ~BatchNormV1Op()
//  {
//      if(!initForward){
//          CUDACHECK(cudaStreamDestroy(sFor));
//          ncclCommDestroy(commFor);
//      std::cout<<"Destroy"<<std::endl;
//      }
//      if(!initBackward){
//          CUDACHECK(cudaStreamDestroy(sBack));
//          ncclCommDestroy(commBack);
//      }
//  }

  virtual void Forward(const OpContext &ctx,
                       const std::vector<TBlob> &in_data,
                       const std::vector<OpReqType> &req,
                       const std::vector<TBlob> &out_data,
                       const std::vector<TBlob> &aux_states) {
    pthread_mutex_lock(&mm);
    if(flagBarrierFor == false){
        pthread_barrier_init(&barrierFor,NULL,param_.nGPUs);
        flagBarrierFor = true;
    }
    pthread_mutex_unlock(&mm);
    pthread_barrier_wait(&barrierFor);
    using namespace mshadow;
    using namespace mshadow::expr;
    CHECK_EQ(in_data.size(), 3U);
    CHECK_EQ(aux_states.size(), 2U);
    if (ctx.is_train) {
      CHECK_EQ(out_data.size(), 3U);
      CHECK_EQ(req.size(), 3U);
    } else {
      CHECK_GE(out_data.size(), 1U);
      CHECK_GE(req.size(), 1U);
      CHECK_EQ(req[batchnorm_v1::kOut], kWriteTo);
    }

    Stream<xpu> *s = ctx.get_stream<xpu>();
    const real_t scale = static_cast<real_t>(in_data[batchnorm_v1::kData].shape_[1]) /
                         static_cast<real_t>(in_data[batchnorm_v1::kData].shape_.Size());
    Tensor<xpu, 4> data;
    Tensor<xpu, 4> out;
    if (in_data[batchnorm_v1::kData].ndim() == 2) {
      Shape<4> dshape = Shape4(in_data[batchnorm_v1::kData].shape_[0],
                               in_data[batchnorm_v1::kData].shape_[1], 1, 1);
      data = in_data[batchnorm_v1::kData].get_with_shape<xpu, 4, real_t>(dshape, s);
      out = out_data[batchnorm_v1::kOut].get_with_shape<xpu, 4, real_t>(dshape, s);
    } else {
      data = in_data[batchnorm_v1::kData].get<xpu, 4, real_t>(s);
      out = out_data[batchnorm_v1::kOut].get<xpu, 4, real_t>(s);
    }
    Tensor<xpu, 1> slope = in_data[batchnorm_v1::kGamma].get<xpu, 1, real_t>(s);
    Tensor<xpu, 1> bias = in_data[batchnorm_v1::kBeta].get<xpu, 1, real_t>(s);
    Tensor<xpu, 1> moving_mean = aux_states[batchnorm_v1::kMovingMean].get<xpu, 1, real_t>(s);
    Tensor<xpu, 1> moving_var = aux_states[batchnorm_v1::kMovingVar].get<xpu, 1, real_t>(s);

    if (param_.fix_gamma) slope = 1.f;

    // whether use global statistics
    if (ctx.is_train && !param_.use_global_stats) {
      Tensor<xpu, 1> mean = out_data[batchnorm_v1::kMean].get<xpu, 1, real_t>(s);
      Tensor<xpu, 1> var = out_data[batchnorm_v1::kVar].get<xpu, 1, real_t>(s);
      CHECK(req[batchnorm_v1::kMean] == kNullOp || req[batchnorm_v1::kMean] == kWriteTo);
      CHECK(req[batchnorm_v1::kVar] == kNullOp || req[batchnorm_v1::kVar] == kWriteTo);
      // The first three steps must be enforced.
      int devid = ctx.run_ctx.ctx.dev_id;

      ncclUniqueId commUid;
      mean = scale * sumall_except_dim<1>(data);
//      var = scale * sumall_except_dim<1>(F<mshadow_op::square>(
//          data - broadcast<1>(mean, data.shape_)));
      var = scale * sumall_except_dim<1>(F<mshadow_op::square>(data));

      //Global BN
      if(param_.global_bn){
//          if(initForward){
//          pthread_mutex_lock(&mm);
//          if(flagForward){
//              std::cout<<"Forward: "<<"I am device "<<devid<<std::endl;
//              NCCLCHECK(ncclGetUniqueId(&uidForward));
//              flagForward = false;
//          }
//          myRank = rankFor;
//          rankFor += 1;
//          pthread_mutex_unlock(&mm);
//          pthread_barrier_wait(&barrierFor);
//          commUid = uidForward;
//          NCCLCHECK(ncclCommInitRank(&commFor,param_.nGPUs,commUid,devid));
//          pthread_barrier_wait(&barrierFor);
//          rankFor = 0;
//          CUDACHECK(cudaStreamCreate(&sFor));
//          initForward = false;
//          }
//
//          CUDACHECK(cudaDeviceSynchronize());
//          NCCLCHECK(ncclAllReduce((const void*)mean.dptr_,(void*)mean.dptr_,mean.shape_[0],ncclFloat,ncclSum,commFor,sFor));
//          cudaDeviceSynchronize();
//          NCCLCHECK(ncclAllReduce((const void*)var.dptr_,(void*)var.dptr_,var.shape_[0],ncclFloat,ncclSum,commFor,sFor));
//          cudaDeviceSynchronize();
//          mean = mean * 1.0f /param_.nGPUs;
//          var  = var  * 1.0f /param_.nGPUs;
//          std::cout<<"finish Forward"<<std::endl;
      }

      var = var-F<mshadow_op::square>(mean);
      Assign(out, req[batchnorm_v1::kOut], broadcast<1>(slope, out.shape_) *
             (data - broadcast<1>(mean, data.shape_)) /
             F<mshadow_op::square_root>(broadcast<1>(var + param_.eps, data.shape_)) +
             broadcast<1>(bias, out.shape_));
    } else {
      Assign(out, req[batchnorm_v1::kOut], broadcast<1>(slope /
                                          F<mshadow_op::square_root>(moving_var + param_.eps),
                                          data.shape_) * data +
             broadcast<1>(bias - (slope * moving_mean) /
                          F<mshadow_op::square_root>(moving_var + param_.eps), data.shape_));
      // Set mean and var tensors to their moving values
      Tensor<xpu, 1> mean = out_data[batchnorm_v1::kMean].get<xpu, 1, real_t>(s);
      Tensor<xpu, 1> var = out_data[batchnorm_v1::kVar].get<xpu, 1, real_t>(s);
      mean = F<mshadow_op::identity>(moving_mean);
      var  = F<mshadow_op::identity>(moving_var);
    }
  }

  virtual void Backward(const OpContext &ctx,
                        const std::vector<TBlob> &out_grad,
                        const std::vector<TBlob> &in_data,
                        const std::vector<TBlob> &out_data,
                        const std::vector<OpReqType> &req,
                        const std::vector<TBlob> &in_grad,
                        const std::vector<TBlob> &aux_states) {
    pthread_mutex_lock(&mm);
    if(flagBarrierBack == false){
        pthread_barrier_init(&barrierBack,NULL,param_.nGPUs);
        flagBarrierBack = true;
    }
    pthread_mutex_unlock(&mm);
    pthread_barrier_wait(&barrierBack);
    using namespace mshadow;
    using namespace mshadow::expr;
    CHECK_EQ(out_grad.size(), param_.output_mean_var ? 3U : 1U);
    CHECK_EQ(in_data.size(), 3U);
    CHECK_EQ(out_data.size(), 3U);
    CHECK_EQ(in_grad.size(), 3U);
    Stream<xpu> *s = ctx.get_stream<xpu>();
    Tensor<xpu, 4> data, grad, grad_in;
    const real_t scale = static_cast<real_t>(out_grad[batchnorm_v1::kOut].shape_[1]) /
                         static_cast<real_t>(out_grad[batchnorm_v1::kOut].shape_.Size());
    if (in_data[batchnorm_v1::kData].ndim() == 2) {
      Shape<4> dshape = Shape4(out_grad[batchnorm_v1::kOut].shape_[0],
                               out_grad[batchnorm_v1::kOut].shape_[1], 1, 1);
      data = in_data[batchnorm_v1::kData].get_with_shape<xpu, 4, real_t>(dshape, s);
      grad = out_grad[batchnorm_v1::kOut].get_with_shape<xpu, 4, real_t>(dshape, s);
      grad_in = in_grad[batchnorm_v1::kData].get_with_shape<xpu, 4, real_t>(dshape, s);
    } else {
      data = in_data[batchnorm_v1::kData].get<xpu, 4, real_t>(s);
      grad = out_grad[batchnorm_v1::kOut].get<xpu, 4, real_t>(s);
      grad_in = in_grad[batchnorm_v1::kData].get<xpu, 4, real_t>(s);
    }

    Tensor<xpu, 1> mean = out_data[batchnorm_v1::kMean].get<xpu, 1, real_t>(s);
    Tensor<xpu, 1> var = out_data[batchnorm_v1::kVar].get<xpu, 1, real_t>(s);
    Tensor<xpu, 1> slope = in_data[batchnorm_v1::kGamma].get<xpu, 1, real_t>(s);
    // Tensor<xpu, 1> bias = in_data[kBeta].get<xpu, 1, real_t>(s);
    Tensor<xpu, 1> gslope = in_grad[batchnorm_v1::kGamma].get<xpu, 1, real_t>(s);
    Tensor<xpu, 1> gbias = in_grad[batchnorm_v1::kBeta].get<xpu, 1, real_t>(s);
    // update moving avg
    Tensor<xpu, 1> moving_mean = aux_states[batchnorm_v1::kMovingMean].get<xpu, 1, real_t>(s);
    Tensor<xpu, 1> moving_var = aux_states[batchnorm_v1::kMovingVar].get<xpu, 1, real_t>(s);

    if (param_.fix_gamma) slope = 1.f;

    if (ctx.is_train && !param_.use_global_stats) {
      // get requested temp space
      Tensor<xpu, 2> workspace = ctx.requested[batchnorm_v1::kTempSpace].get_space<xpu>(
          mshadow::Shape2(5, mean.shape_[0]), s);
      Tensor<xpu, 1> gmean = workspace[0];
      Tensor<xpu, 1> gvar = workspace[1];
//      Tensor<xpu, 1> tmp = workspace[2];
      Tensor<xpu, 1> sumGrad = workspace[3];
      Tensor<xpu, 1> sumProd = workspace[4];
      

      moving_mean = moving_mean * param_.momentum + mean * (1 - param_.momentum);
      moving_var = moving_var * param_.momentum + var * (1 - param_.momentum);

      int devid = ctx.run_ctx.ctx.dev_id;

      sumGrad = sumall_except_dim<1>(grad);
      sumProd = sumall_except_dim<1>(grad*data);
      if(param_.global_bn){
//          test+=1;
//          if(initBackward){
//          pthread_mutex_lock(&mm);
//          if(flagBackward){
//          //    std::cout<<"Backward "<<test<<std::endl;
//              NCCLCHECK(ncclGetUniqueId(&uidBackward));
//              flagBackward = false;
//          }
//          myRank = rankBack;
//          rankBack += 1;
//          pthread_mutex_unlock(&mm);
//          pthread_barrier_wait(&barrierBack);
//          ncclUniqueId commUid = uidBackward;
//          NCCLCHECK(ncclCommInitRank(&commBack,param_.nGPUs,commUid,devid));
//          pthread_barrier_wait(&barrierBack);
//          rankBack = 0;
//          cudaStreamCreate(&sBack);
//          initBackward = false;
//          }
//
//          cudaDeviceSynchronize();
//          NCCLCHECK(ncclAllReduce((const void*)sumGrad.dptr_,(void*)sumGrad.dptr_,sumGrad.shape_[0],ncclFloat,ncclSum,commBack,sBack));
//          cudaDeviceSynchronize();
//          NCCLCHECK(ncclAllReduce((const void*)sumProd.dptr_,(void*)sumProd.dptr_,sumProd.shape_[0],ncclFloat,ncclSum,commBack,sBack));
//          cudaDeviceSynchronize();
//          sumGrad = sumGrad * 1.0f /param_.nGPUs;
//          sumProd = sumProd * 1.0f /param_.nGPUs;
//          std::cout<<"finish Backward"<<std::endl;
      }

      gvar = (sumProd-sumGrad*mean)*slope*(-0.5f)*F<mshadow_op::power>(var+param_.eps,-1.5f);
      gmean =  sumGrad*slope;
      gmean *= -1.0f/F<mshadow_op::square_root>(var+param_.eps);


      // cal
//      gvar = sumall_except_dim<1>((grad * broadcast<1>(slope, data.shape_)) *
//                                  (data - broadcast<1>(mean, data.shape_)) *
//                                  -0.5f *
//                                  F<mshadow_op::power>(broadcast<1>(var + param_.eps, data.shape_),
//                                                       -1.5f));
//      gmean = sumall_except_dim<1>(grad * broadcast<1>(slope, data.shape_));
//      gmean *= -1.0f / F<mshadow_op::square_root>(var + param_.eps);
//      tmp = scale * sumall_except_dim<1>(-2.0f * (data - broadcast<1>(mean, data.shape_)));
//      tmp *= gvar;
//      gmean += tmp;
      // assign
      if (!param_.fix_gamma) {
        Assign(gslope, req[batchnorm_v1::kGamma],
               sumall_except_dim<1>(
                   grad * (data - broadcast<1>(mean, data.shape_)) /
                   F<mshadow_op::square_root>(broadcast<1>(var + param_.eps, data.shape_))));
      } else {
        Assign(gslope, req[batchnorm_v1::kGamma], 0.0f);
      }
      Assign(grad_in, req[batchnorm_v1::kData],
             (grad * broadcast<1>(slope, data.shape_)) *
             broadcast<1>(1.0f / F<mshadow_op::square_root>(var + param_.eps), data.shape_) +
             broadcast<1>(gvar, data.shape_) * scale * 2.0f * (data - broadcast<1>(mean,
                                                                                   data.shape_)) +
             broadcast<1>(gmean, data.shape_) * scale);
      Assign(gbias, req[batchnorm_v1::kBeta], sumall_except_dim<1>(grad));
    } else {
      // use global statistics with freeze moving mean and var.
      //  std::cout<<"I am using global statistics"<<std::endl;
      if (!param_.fix_gamma) {
        Assign(gslope, req[batchnorm_v1::kGamma],
               sumall_except_dim<1>(
                   grad * (data - broadcast<1>(moving_mean, data.shape_)) /
                   F<mshadow_op::square_root>(broadcast<1>(moving_var + param_.eps, data.shape_))));
      } else {
        Assign(gslope, req[batchnorm_v1::kGamma], 0.0f);
      }
      Assign(gbias, req[batchnorm_v1::kBeta], sumall_except_dim<1>(grad));
      Assign(grad_in, req[batchnorm_v1::kData], (grad * broadcast<1>(slope, data.shape_)) *
             broadcast<1>(
                 1.0f / F<mshadow_op::square_root>(moving_var + param_.eps), data.shape_));
    }
  }

 private:
  BatchNormV1Param param_;
  ncclComm_t commFor;
  ncclComm_t commBack;
  cudaStream_t sFor;
  cudaStream_t sBack;
  int myRank;
  int test = 0;
  bool flagForward = true;
  bool flagBackward = true;
  bool initForward = true;
  bool initBackward = true;
};  // class BatchNormV1Op

template<typename xpu>
Operator *CreateOp(BatchNormV1Param param, int dtype);


#if DMLC_USE_CXX11
class BatchNormV1Prop : public OperatorProperty {
 public:
  void Init(const std::vector<std::pair<std::string, std::string> >& kwargs) override {
    param_.Init(kwargs);
  }

  std::map<std::string, std::string> GetParams() const override {
    return param_.__DICT__();
  }

  bool InferShape(std::vector<TShape> *in_shape,
                  std::vector<TShape> *out_shape,
                  std::vector<TShape> *aux_shape) const override {
    using namespace mshadow;
    CHECK_EQ(in_shape->size(), 3U) << "Input:[data, gamma, beta]";
    const TShape &dshape = in_shape->at(0);
    if (dshape.ndim() == 0) return false;
    in_shape->at(1) = TShape(Shape1(dshape[1]));
    in_shape->at(2) = TShape(Shape1(dshape[1]));
    out_shape->clear();
    out_shape->push_back(dshape);
    out_shape->push_back(Shape1(dshape[1]));
    out_shape->push_back(Shape1(dshape[1]));

    aux_shape->clear();
    aux_shape->push_back(Shape1(dshape[1]));
    aux_shape->push_back(Shape1(dshape[1]));
    return true;
  }

  bool InferType(std::vector<int> *in_type,
                 std::vector<int> *out_type,
                 std::vector<int> *aux_type) const override {
    using namespace mshadow;
    CHECK_GE(in_type->size(), 1U);
    int dtype = (*in_type)[0];
    CHECK_NE(dtype, -1) << "First input must have specified type";
    // For float16 input type beta, gamma, mean, and average are stored in float32.
    // For other input types, these parameters have the same type as input
    // NOTE: This requirement is from cuDNN (v. 4 and 5)
    int dtype_param = (dtype == kFloat16) ? kFloat32 : dtype;
    for (index_t i = 1; i < in_type->size(); ++i) {
      if ((*in_type)[i] == -1) {
        (*in_type)[i] = dtype_param;
      } else {
        UNIFORM_TYPE_CHECK((*in_type)[i], dtype_param, ListArguments()[i]);
      }
    }
    for (index_t i = 0; i < aux_type->size(); ++i) {
      if ((*aux_type)[i] != -1) {
        UNIFORM_TYPE_CHECK((*aux_type)[i], dtype_param, ListArguments()[i]);
      }
    }
    int n_aux = this->ListAuxiliaryStates().size();
    aux_type->clear();
    for (int i = 0; i < n_aux; ++i ) aux_type->push_back(dtype_param);
    int n_out = this->ListOutputs().size();
    out_type->clear();
    out_type->push_back(dtype);
    for (int i = 1; i < n_out; ++i ) out_type->push_back(dtype_param);
    return true;
  }

  OperatorProperty* Copy() const override {
    auto ptr = new BatchNormV1Prop();
    ptr->param_ = param_;
    return ptr;
  }

  std::string TypeString() const override {
    return "BatchNorm_v1";
  }

  std::vector<int> DeclareBackwardDependency(
    const std::vector<int> &out_grad,
    const std::vector<int> &in_data,
    const std::vector<int> &out_data) const override {
    return {out_grad[batchnorm_v1::kOut],
            out_data[batchnorm_v1::kMean],
            out_data[batchnorm_v1::kVar],
            in_data[batchnorm_v1::kData],
            in_data[batchnorm_v1::kGamma]
           };
  }

  std::vector<ResourceRequest> BackwardResource(
      const std::vector<TShape> &in_shape) const override {
    return {ResourceRequest::kTempSpace};
  }

  int NumVisibleOutputs() const override {
    if (param_.output_mean_var) {
      return 3;
    }
    return 1;
  }

  int NumOutputs() const override {
    return 3;
  }

  std::vector<std::string> ListArguments() const override {
    return {"data", "gamma", "beta"};
  }

  std::vector<std::string> ListOutputs() const override {
    return {"output", "mean", "var"};
  }

  std::vector<std::string> ListAuxiliaryStates() const override {
    return {"moving_mean", "moving_var"};
  }

  Operator* CreateOperator(Context ctx) const override {
      LOG(FATAL) << "Not Implemented.";
      return NULL;
  }

  Operator* CreateOperatorEx(Context ctx, std::vector<TShape> *in_shape,
      std::vector<int> *in_type) const override;

  inline const BatchNormV1Param& getParam() const {
    return param_;
  }

 private:
  BatchNormV1Param param_;
};  // class BatchNormV1Prop

#endif  // DMLC_USE_CXX11
}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_BATCH_NORM_V1_INL_H_
