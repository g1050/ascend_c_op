#include "kernel_operator.h"
using namespace AscendC;

// constexpr int32_t TOTAL_LENGTH = 8 * 2048;                            // total length of data 数据总长度
// constexpr int32_t USE_CORE_NUM = 8;                                   // num of core used 核心数
// constexpr int32_t BLOCK_LENGTH = TOTAL_LENGTH / USE_CORE_NUM;         // length computed of each core 每核分配的计算任务，2048
// constexpr int32_t TILE_NUM = 8;                                       // split data into 8 tiles for each core 每核分 8tiles
constexpr int32_t BUFFER_NUM = 2;                                     // tensor num for each queue double buffer技术
// constexpr int32_t TILE_LENGTH = BLOCK_LENGTH / TILE_NUM / BUFFER_NUM; // seperate to 2 parts, due to double buffer 2048/2/8=128
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
        int a;
        __aicore__ inline void CopyIn(int32_t progress){
            Half xLocal = inQueueX.AllocTensor<half>(); // 从输入队列中申请local tensor,alloc enqueue, dequeue, free
            DataCopy(xLocal,xGm[progress*this->tileLength],this->tileLength);
            inQueueX.EnQue(xLocal);
        }
        __aicore__ inline void Compute(int32_t progress){
            Half xLocal = inQueueX.DeQue<half>();
            Half zLocal = outQueueZ.AllocTensor<half>();
            // compute 
            Exp(xLocal,xLocal,this->tileLength); // e^x
            Reciprocal(zLocal,xLocal,this->tileLength); // e^(-x)
            Sub(zLocal,xLocal,zLocal,this->tileLength);
            half scalar = 0.5;
            Muls(zLocal,zLocal,scalar,this->tileLength);
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

extern "C" __global__ __aicore__ void sinh_custom(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    // TODO: user kernel impl
    KernelSinh sinh;
    sinh.Init(x,y,tiling_data.totalLength,tiling_data.tileNum);
    sinh.process();
}