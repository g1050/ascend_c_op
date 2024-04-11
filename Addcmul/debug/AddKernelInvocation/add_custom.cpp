/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2022-2023. All rights reserved.
 *
 * Function : z = x + y
 * This sample is a very basic sample that implements vector add on Ascend plaform.
 */
#include "kernel_operator.h"
using namespace AscendC;

constexpr int32_t TOTAL_LENGTH = 8 * 2048;                            // total length of data
constexpr int32_t USE_CORE_NUM = 8;                                   // num of core used
constexpr int32_t BLOCK_LENGTH = TOTAL_LENGTH / USE_CORE_NUM;         // length computed of each core
constexpr int32_t TILE_NUM = 8;                                       // split data into 8 tiles for each core
constexpr int32_t BUFFER_NUM = 2;                                     // tensor num for each queue
constexpr int32_t TILE_LENGTH = BLOCK_LENGTH / TILE_NUM / BUFFER_NUM; // seperate to 2 parts, due to double buffer

class KernelAdd {
public:
    __aicore__ inline KernelAdd() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z)
    {
        xGm.SetGlobalBuffer((__gm__ half*)x + BLOCK_LENGTH * GetBlockIdx(), BLOCK_LENGTH);
        yGm.SetGlobalBuffer((__gm__ half*)y + BLOCK_LENGTH * GetBlockIdx(), BLOCK_LENGTH);
        zGm.SetGlobalBuffer((__gm__ half*)z + BLOCK_LENGTH * GetBlockIdx(), BLOCK_LENGTH);
        pipe.InitBuffer(inQueueX, BUFFER_NUM, TILE_LENGTH * sizeof(half));
        pipe.InitBuffer(inQueueY, BUFFER_NUM, TILE_LENGTH * sizeof(half));
        pipe.InitBuffer(outQueueZ, BUFFER_NUM, TILE_LENGTH * sizeof(half));
    }
    __aicore__ inline void Process()
    {
        int32_t loopCount = TILE_NUM * BUFFER_NUM;
        for (int32_t i = 0; i < loopCount; i++) {
            CopyIn(i);
            Compute(i);
            CopyOut(i);
        }
    }

private:
    __aicore__ inline void CopyIn(int32_t progress)
    {
        LocalTensor<half> xLocal = inQueueX.AllocTensor<half>();
        LocalTensor<half> yLocal = inQueueY.AllocTensor<half>();
        DataCopy(xLocal, xGm[progress * TILE_LENGTH], TILE_LENGTH);
        DataCopy(yLocal, yGm[progress * TILE_LENGTH], TILE_LENGTH);
        inQueueX.EnQue(xLocal);
        inQueueY.EnQue(yLocal);
    }
    __aicore__ inline void Compute(int32_t progress)
    {
        LocalTensor<half> xLocal = inQueueX.DeQue<half>();
        LocalTensor<half> yLocal = inQueueY.DeQue<half>();
        LocalTensor<half> zLocal = outQueueZ.AllocTensor<half>();
        Add(zLocal, xLocal, yLocal, TILE_LENGTH);
        outQueueZ.EnQue<half>(zLocal);
        inQueueX.FreeTensor(xLocal);
        inQueueY.FreeTensor(yLocal);
    }
    __aicore__ inline void CopyOut(int32_t progress)
    {
        LocalTensor<half> zLocal = outQueueZ.DeQue<half>();
        DataCopy(zGm[progress * TILE_LENGTH], zLocal, TILE_LENGTH);
        outQueueZ.FreeTensor(zLocal);
    }

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX, inQueueY;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueZ;
    GlobalTensor<half> xGm;
    GlobalTensor<half> yGm;
    GlobalTensor<half> zGm;
};

extern "C" __global__ __aicore__ void add_custom(GM_ADDR x, GM_ADDR y, GM_ADDR z)
{
    KernelAdd op;
    op.Init(x, y, z);
    op.Process();
}

#ifndef __CCE_KT_TEST__
// call of kernel function
void add_custom_do(uint32_t blockDim, void* l2ctrl, void* stream, uint8_t* x, uint8_t* y, uint8_t* z)
{
    add_custom<<<blockDim, l2ctrl, stream>>>(x, y, z);
}
#endif

template<typename dataType>
class KernelAddcmulBase{
    public:
        __aicore__ inline KernelAddcmulBase(){}
        __aicore__ inline void Init(GM_ADDR input_data, GM_ADDR x1, GM_ADDR x2, GM_ADDR value, GM_ADDR y,uint32_t totalLength,uint32_t tileNum){
            this->blockNum = GetBlockNum();
            ASSERT(this->blockNum != 0 && "block dim can not be zero");
            this->blockLength = totalLength / this->blockNum;
            this->tileNum = tileNum;
            ASSERT(this->tileNum != 0 && "tile num can not be zero");
            this->tileLength = this->blockLength / this->tileNum / BUFFER_NUM;
            // address to Tensor
            x1Gm.SetGlobalBuffer((__gm__ dataType*)x1 + this->blockLength * GetBlockIdx(), this->blockLength); //x1  
            x2Gm.SetGlobalBuffer((__gm__ dataType*)x2 + this->blockLength * GetBlockIdx(), this->blockLength); //x2  
            inputDataGm.SetGlobalBuffer((__gm__ dataType*)input_data + this->blockLength * GetBlockIdx(), this->blockLength); //input_data  
            valueGm.SetGlobalBuffer((__gm__ dataType*)value + this->blockLength * GetBlockIdx(), this->blockLength); //value  
            yGm.SetGlobalBuffer((__gm__ dataType*)y + this->blockLength * GetBlockIdx(), this->blockLength); //output  
            
            pipe.InitBuffer(inQueueX1,BUFFER_NUM,this->tileLength*sizeof(dataType)); // (分配内存的对象，内存块的个数，每块的大小)
            pipe.InitBuffer(inQueueX2,BUFFER_NUM,this->tileLength*sizeof(dataType));
            pipe.InitBuffer(inQueueInputData,BUFFER_NUM,this->tileLength*sizeof(dataType));
            pipe.InitBuffer(inQueueValue,BUFFER_NUM,this->tileLength*sizeof(dataType));
            pipe.InitBuffer(outQueueY,BUFFER_NUM,this->tileLength*sizeof(dataType));
        }
        __aicore__ inline void process(){
            int32_t loopCount = this->tileNum*BUFFER_NUM; // 每个core上执行的次数
            for(auto i = 0;i<loopCount;i++){
                CopyIn(i);
                Compute(i);
                CopyOut(i);
            }
        }
    public:
        __aicore__ inline void CopyIn(int32_t progress){
            Alloc(this->inQueueX1,x1Gm,progress);
            Alloc(this->inQueueX2,x2Gm,progress);
            Alloc(this->inQueueInputData,inputDataGm,progress);
            Alloc(this->inQueueValue,valueGm,progress);
        }
        __aicore__ inline void Alloc(TQue<QuePosition::VECIN, BUFFER_NUM> &que,GlobalTensor<dataType> gm,int32_t progress){
            LocalTensor<dataType> local = que.AllocTensor<dataType>(); // 从输入队列中申请local tensor,alloc enqueue, dequeue, free
            DataCopy(local,gm[progress*this->tileLength],this->tileLength);
            que.EnQue(local);
        }
        virtual __aicore__ inline void Compute(int32_t progress)
        {
            // LocalTensor<dataType> y = outQueueY.AllocTensor<dataType>();
            // LocalTensor<dataType> x1 = inQueueX1.DeQue<dataType>();
            // LocalTensor<dataType> x2 = inQueueX2.DeQue<dataType>();            
            // LocalTensor<dataType> inputData = inQueueInputData.DeQue<dataType>();
            // LocalTensor<dataType> value = inQueueValue.DeQue<dataType>();
            // x1 = x1 * x2;
            // x1 = x1 * value;
            // y = inputData + x1;
            
            // outQueueY.EnQue<dataType>(y);
            // inQueueX1.FreeTensor(x1);
            // inQueueX2.FreeTensor(x2);
            // inQueueInputData.FreeTensor(inputData);
            // inQueueValue.FreeTensor(value);
        }
        __aicore__ inline void CopyOut(int32_t progress){
            LocalTensor<dataType> y = outQueueY.DeQue<dataType>();
            DataCopy(yGm[progress*this->tileLength],y,this->tileLength);
            outQueueY.FreeTensor(y);
        }
    public:
        TPipe pipe; //管理内存
        TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX1; // 输入队列
        TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX2; // 输入队列
        TQue<QuePosition::VECIN, BUFFER_NUM> inQueueInputData; // 输入队列
        TQue<QuePosition::VECIN, BUFFER_NUM> inQueueValue; // 输入队列
        TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueY; // 输出队列
        // TBuf<QuePosition::VECCALC> tmpBuffer1, tmpBuffer2;

        GlobalTensor<dataType> x1Gm;        
        GlobalTensor<dataType> x2Gm;
        GlobalTensor<dataType> inputDataGm;
        GlobalTensor<dataType> valueGm;        
        GlobalTensor<dataType> yGm;


        uint32_t coreNum;
        uint32_t totalLength;
        uint32_t tileLength;
        uint32_t blockLength;
        uint32_t blockNum;
        uint32_t tileNum ;

};

template <typename dataType>
class KernelAddcmul: public KernelAddcmulBase<dataType>
{
public:
    __aicore__ inline KernelAddcmul():KernelAddcmulBase<dataType>(){}
public:
    __aicore__ inline void Compute(int32_t progress) override
    {
        LocalTensor<dataType> y = reinterpret_cast<TQue<QuePosition::VECOUT, BUFFER_NUM>*>(&(this->outQueueY))->template AllocTensor<dataType>();
        // LocalTensor<dataType> x1 = this->template inQueueX1.template DeQue<dataType>();
        // LocalTensor<dataType> x2 = this->template inQueueX2.template DeQue<dataType>();            
        // LocalTensor<dataType> inputData = this->template inQueueInputData.template DeQue<dataType>();
        // LocalTensor<dataType> value = this->template inQueueValue.template DeQue<dataType>();

        // x1 = x1 * x2;
        // x1 = x1 * value;
        // y = inputData + x1;

        // this->template outQueueY.template EnQue<dataType>(y);
        // this->inQueueX1.FreeTensor(x1);
        // this->inQueueX2.FreeTensor(x2);
        // this->inQueueInputData.FreeTensor(inputData);
        // this->inQueueValue.FreeTensor(value);
    }
};

// template<typename dataType,typename castDataType>
// class KernelAddcmulAdapter:public KernelAddcmulBase<dataType>
// {
//     public:
//     __aicore__ inline KernelAddcmulAdapter(){}
//     __aicore__ inline void Init(GM_ADDR input_data, GM_ADDR x1, GM_ADDR x2, GM_ADDR value, GM_ADDR y,uint32_t totalLength,uint32_t tileNum){
//         KernelAddcmulBase<dataType>::Init(input_data,x1,x2,value,y,totalLength,tileNum);
//         KernelAddcmulBase<dataType>::pipe.InitBuffer(tmpX1Queue, this->tileLength*sizeof(castDataType));
//         this->pipe.InitBuffer(tmpX2Queue, this->tileLength*sizeof(castDataType));
//         KernelAddcmulBase<dataType>::pipe.InitBuffer(tmpInputDataQueue, this->tileLength*sizeof(castDataType));
//         KernelAddcmulBase<dataType>::pipe.InitBuffer(tmpValue, this->tileLength*sizeof(castDataType));
//         KernelAddcmulBase<dataType>::pipe.InitBuffer(tmpYQueue, this->tileLength*sizeof(castDataType));
//     }

// private:
//     __aicore__ inline void Compute(int32_t progress){
//         LocalTensor<dataType> y = KernelAddcmulBase<dataType>::outQueueY::template AllocTensor<dataType>();
//         LocalTensor<dataType> x1 = KernelAddcmulBase<dataType>::inQueueX1::template DeQue<dataType>();
//         LocalTensor<dataType> x2 = KernelAddcmulBase<dataType>::inQueueX2::template DeQue<dataType>();            
//         LocalTensor<dataType> inputData = KernelAddcmulBase<dataType>::inQueueInputData::template DeQue<dataType>();
//         LocalTensor<dataType> value = KernelAddcmulBase<dataType>::inQueueValue::template DeQue<dataType>();
        
//         LocalTensor<castDataType> tmpx1 = KernelAddcmulBase<dataType>::tmpX1Queue::template Get<castDataType>();
//         LocalTensor<castDataType> tmpx2 = KernelAddcmulBase<dataType>::tmpX2Queue::template Get<castDataType>();
//         LocalTensor<castDataType> tmpInputData = KernelAddcmulBase<dataType>::tmpInputDataQueue::template Get<castDataType>();
//         LocalTensor<castDataType> tmpValue = KernelAddcmulBase<dataType>::tmpValue::template Get<castDataType>();
//         LocalTensor<castDataType> tmpY = KernelAddcmulBase<dataType>::tmpYQueue::template Get<castDataType>();

//         // Only support int8
//         Cast(tmpx1, x1, RoundMode::CAST_NONE, KernelAddcmulBase<dataType>::tileLength);
//         Cast(tmpx2, x2, RoundMode::CAST_NONE, KernelAddcmulBase<dataType>::tileLength);
//         Cast(tmpInputData, inputData, RoundMode::CAST_NONE, KernelAddcmulBase<dataType>::tileLength);
//         Cast(tmpValue, value, RoundMode::CAST_NONE, KernelAddcmulBase<dataType>::tileLength);

//         // x1 = x1 * x2;
//         // x1 = x1 * value;
//         // y = inputData + x1;
//         tmpx1 = tmpx1*tmpx2;
//         tmpx1 = tmpx1*tmpValue;
//         tmpY = tmpInputData + tmpx1;

//         Cast(y, tmpY, RoundMode::CAST_NONE, this->tileLength);
//         KernelAddcmulBase<dataType>::outQueueY::template EnQue<dataType>(y);
//         KernelAddcmulBase<dataType>::inQueueX1.FreeTensor(x1);
//         KernelAddcmulBase<dataType>::inQueueX2.FreeTensor(x2);
//         KernelAddcmulBase<dataType>::inQueueInputData.FreeTensor(inputData);
//         KernelAddcmulBase<dataType>::inQueueValue.FreeTensor(value);
//     }
//     TBuf<QuePosition::VECCALC> tmpX1Queue;
//     TBuf<QuePosition::VECCALC> tmpX2Queue;
//     TBuf<QuePosition::VECCALC> tmpInputDataQueue;
//     TBuf<QuePosition::VECCALC> tmpValue;
//     TBuf<QuePosition::VECCALC> tmpYQueue;
// };

extern "C" __global__ __aicore__ void addcmul_custom(GM_ADDR input_data, GM_ADDR x1, GM_ADDR x2, GM_ADDR value, GM_ADDR y) {
    // TODO: user kernel impl
    KernelAddcmul<half> sinh;
    sinh.Init(input_data,x1,x2,value,y,8*2048,8);
    sinh.process();
}


#ifndef __CCE_KT_TEST__
// call of kernel function
void addcmul_custom_do(uint32_t blockDim, void* l2ctrl, void* stream, uint8_t* input_data, uint8_t* x1, uint8_t* x2,uint8_t* value,uint8_t* y)
{
    addcmul_custom<<<blockDim, l2ctrl, stream>>>(input_data,x1,x2,value,y);
}
#endif