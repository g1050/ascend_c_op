#include "kernel_operator.h"
using namespace AscendC;
constexpr int32_t TOTAL_LENGTH = 8 * 2048;                            // total length of data 数据总长度
constexpr int32_t USE_CORE_NUM = 8;                                   // num of core used 核心数
constexpr int32_t BLOCK_LENGTH = TOTAL_LENGTH / USE_CORE_NUM;         // length computed of each core 每核分配的计算任务，2048
constexpr int32_t TILE_NUM = 8;                                       // split data into 8 tiles for each core 每核分 8tiles
constexpr int32_t BUFFER_NUM = 2;                                     // tensor num for each queue double buffer技术
constexpr int32_t TILE_LENGTH = BLOCK_LENGTH / TILE_NUM / BUFFER_NUM; // seperate to 2 parts, due to double buffer 2048/2/8=128
typedef LocalTensor<half> Half;
class KernelSinh{
    public:
        __aicore__ inline KernelSinh(){}
        __aicore__ inline void Init(GM_ADDR x,GM_ADDR z){
            // address to Tensor
            xGm.SetGlobalBuffer((__gm__ half*)x + BLOCK_LENGTH * GetBlockIdx(), BLOCK_LENGTH); //input  
            zGm.SetGlobalBuffer((__gm__ half*)z + BLOCK_LENGTH * GetBlockIdx(),BLOCK_LENGTH); // output
            pipe.InitBuffer(inQueueX,BUFFER_NUM,TILE_LENGTH*sizeof(half)); // (分配内存的对象，内存块的个数，每块的大小)
            pipe.InitBuffer(outQueueZ,BUFFER_NUM,TILE_LENGTH*sizeof(half));

        }
        __aicore__ inline void process(){
            int32_t loopCount = TILE_NUM*BUFFER_NUM; // 每个core上执行的次数
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
            DataCopy(xLocal,xGm[progress*TILE_LENGTH],TILE_LENGTH);
            inQueueX.EnQue(xLocal);
        }
        __aicore__ inline void Compute(int32_t progress){
            Half xLocal = inQueueX.DeQue<half>();
            Half zLocal = outQueueZ.AllocTensor<half>();
            // compute 
            Exp(xLocal,xLocal,TILE_LENGTH); // e^x
            Reciprocal(zLocal,xLocal,TILE_LENGTH); // e^(-x)
            Sub(zLocal,xLocal,zLocal,TILE_LENGTH);
            half scalar = 0.5;
            Muls(zLocal,zLocal,scalar,TILE_LENGTH);
            outQueueZ.EnQue<half>(zLocal);
            inQueueX.FreeTensor(xLocal);
        }
        __aicore__ inline void CopyOut(int32_t progress){
            Half zLocal =  outQueueZ.DeQue<half>();
            DataCopy(zLocal,zGm[progress*TILE_LENGTH],TILE_LENGTH);
            outQueueZ.FreeTensor(zLocal);
        }
    private:
        TPipe pipe; //管理内存
        TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX; // 输入队列
        TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueZ; // 输出队列
        GlobalTensor<half> xGm;
        GlobalTensor<half> zGm;
};

extern "C" __global__ __aicore__ void sinh_custom(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    // TODO: user kernel impl
    KernelSinh sinh;
}