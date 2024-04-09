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

// Sinh------
typedef LocalTensor<half> Half;
class KernelSinh{
    public:
        __aicore__ inline KernelSinh(){}
        __aicore__ inline void Init(GM_ADDR x,GM_ADDR z,uint32_t totalLength,uint32_t tileNum){
            this->blockNum = GetBlockNum();
            ASSERT(this->blockNum != 0 && "block dim can not be zero");
            this->blockLength = totalLength / this->blockNum;
            this->tileNum = tileNum;
            ASSERT(this->tileNum != 0 && "tile num can not be zero");
            this->tileLength = this->blockLength / this->tileNum / BUFFER_NUM;
            #ifdef __CCE_KT_TEST__
            std::cout << "blockNum " << this->blockNum << std::endl;
            std::cout << "blockLength " << this->blockLength << std::endl;
            std::cout << "tileNum " << this->tileNum << std::endl;
            std::cout << "tileLength " << this->tileLength << std::endl;
            std::cout << "----" << std::endl;
            #endif
            // address to Tensor
            xGm.SetGlobalBuffer((__gm__ half*)x + this->blockLength * GetBlockIdx(), this->blockLength); //input  
            zGm.SetGlobalBuffer((__gm__ half*)z + this->blockLength * GetBlockIdx(),this->blockLength); // output
            pipe.InitBuffer(inQueueX,BUFFER_NUM,this->tileLength*sizeof(half)); // (分配内存的对象，内存块的个数，每块的大小)
            pipe.InitBuffer(outQueueZ,BUFFER_NUM,this->tileLength*sizeof(half));

        }
        __aicore__ inline void process(){
            int32_t loopCount = this->tileNum*BUFFER_NUM; // 每个core上执行的次数
            for(auto i = 0;i<loopCount;i++){
                CopyIn(i);
                Compute(i);
                CopyOut(i);
            }
        }
    private:
        __aicore__ inline void CopyIn(int32_t progress){
            Half xLocal = inQueueX.AllocTensor<half>(); // 从输入队列中申请local tensor,alloc enqueue, dequeue, free
            DataCopy(xLocal,xGm[progress*this->tileLength],this->tileLength);
            inQueueX.EnQue(xLocal);
        }
        __aicore__ inline void Compute(int32_t progress){
            Half xLocal = inQueueX.DeQue<half>();
            Half zLocal = outQueueZ.AllocTensor<half>();
            #ifdef __CCE_KT_TEST__
            std::cout << "xLocal " << float(xLocal.GetValue(0)) << std::endl;
            #endif
            // compute 
            Exp(xLocal,xLocal,this->tileLength); // e^x
            Reciprocal(zLocal,xLocal,this->tileLength); // e^(-x)
            Sub(zLocal,xLocal,zLocal,this->tileLength);
            half scalar = 0.5;
            Muls(zLocal,zLocal,scalar,this->tileLength);
            #ifdef __CCE_KT_TEST__
            std::cout << "zLocal " << float(zLocal.GetValue(0)) << std::endl;
            #endif
            outQueueZ.EnQue<half>(zLocal);
            inQueueX.FreeTensor(xLocal);
        }
        __aicore__ inline void CopyOut(int32_t progress){
            Half zLocal =  outQueueZ.DeQue<half>();
            DataCopy(zGm[progress*this->tileLength],zLocal,this->tileLength);
            outQueueZ.FreeTensor(zLocal);
        }
    private:
        TPipe pipe; //管理内存
        TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX; // 输入队列
        TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueZ; // 输出队列
        GlobalTensor<half> xGm;
        GlobalTensor<half> zGm;
        uint32_t coreNum;
        uint32_t totalLength;
        uint32_t tileLength;
        uint32_t blockLength;
        uint32_t blockNum;
        uint32_t tileNum ;

};

extern "C" __global__ __aicore__ void sinh_custom(GM_ADDR x, GM_ADDR y) {
    // GET_TILING_DATA(tiling_data, tiling);
    // TODO: user kernel impl
    // #ifdef __CCE_KT_TEST__
    // std::cout << "Hello" << std::endl;
    // #endif

    KernelSinh sinh;
    sinh.Init(x,y,TOTAL_LENGTH,8);
    sinh.process();
}

#ifndef __CCE_KT_TEST__
// call of kernel function
void sinh_custom_do(uint32_t blockDim, void* l2ctrl, void* stream, uint8_t* x, uint8_t* y)
{
    sinh_custom<<<blockDim, l2ctrl, stream>>>(x, y);
}
#endif

