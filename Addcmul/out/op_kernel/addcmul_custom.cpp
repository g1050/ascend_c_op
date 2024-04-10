#include "kernel_operator.h"
using namespace AscendC;
// constexpr int32_t TOTAL_LENGTH = 8 * 2048;                            // total length of data 数据总长度
// constexpr int32_t USE_CORE_NUM = 8;                                   // num of core used 核心数
// constexpr int32_t BLOCK_LENGTH = TOTAL_LENGTH / USE_CORE_NUM;         // length computed of each core 每核分配的计算任务，2048
// constexpr int32_t TILE_NUM = 8;                                       // split data into 8 tiles for each core 每核分 8tiles
constexpr int32_t BUFFER_NUM = 2;                                     // tensor num for each queue double buffer技术
// constexpr int32_t TILE_LENGTH = BLOCK_LENGTH / TILE_NUM / BUFFER_NUM; // seperate to 2 parts, due to double buffer 2048/2/8=128
typedef LocalTensor<half> Half;
class KernelAddcmul{
    public:
        __aicore__ inline KernelAddcmul(){}
        __aicore__ inline void Init(GM_ADDR input_data, GM_ADDR x1, GM_ADDR x2, GM_ADDR value, GM_ADDR y,uint32_t totalLength,uint32_t tileNum){
            this->blockNum = GetBlockNum();
            ASSERT(this->blockNum != 0 && "block dim can not be zero");
            this->blockLength = totalLength / this->blockNum;
            this->tileNum = tileNum;
            ASSERT(this->tileNum != 0 && "tile num can not be zero");
            this->tileLength = this->blockLength / this->tileNum / BUFFER_NUM;
            // address to Tensor
            x1Gm.SetGlobalBuffer((__gm__ half*)x1 + this->blockLength * GetBlockIdx(), this->blockLength); //x1  
            x2Gm.SetGlobalBuffer((__gm__ half*)x2 + this->blockLength * GetBlockIdx(), this->blockLength); //x2  
            inputDataGm.SetGlobalBuffer((__gm__ half*)input_data + this->blockLength * GetBlockIdx(), this->blockLength); //input_data  
            valueGm.SetGlobalBuffer((__gm__ half*)value + this->blockLength * GetBlockIdx(), this->blockLength); //value  
            yGm.SetGlobalBuffer((__gm__ half*)y + this->blockLength * GetBlockIdx(), this->blockLength); //output  
            
            pipe.InitBuffer(inQueueX1,BUFFER_NUM,this->tileLength*sizeof(half)); // (分配内存的对象，内存块的个数，每块的大小)
            pipe.InitBuffer(inQueueX2,BUFFER_NUM,this->tileLength*sizeof(half));
            pipe.InitBuffer(inQueueInputData,BUFFER_NUM,this->tileLength*sizeof(half));
            pipe.InitBuffer(inQueueValue,BUFFER_NUM,this->tileLength*sizeof(half));
            pipe.InitBuffer(outQueueY,BUFFER_NUM,this->tileLength*sizeof(half));

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
            Alloc(this->inQueueX1,x1Gm,progress);
            Alloc(this->inQueueX2,x2Gm,progress);
            Alloc(this->inQueueInputData,inputDataGm,progress);
            Alloc(this->inQueueValue,valueGm,progress);
        }
        __aicore__ inline void Alloc(TQue<QuePosition::VECIN, BUFFER_NUM> &que,GlobalTensor<half> gm,int32_t progress){
            Half local = que.AllocTensor<half>(); // 从输入队列中申请local tensor,alloc enqueue, dequeue, free
            DataCopy(local,gm[progress*this->tileLength],this->tileLength);
            que.EnQue(local);
        }
        __aicore__ inline void Compute(int32_t progress){
            Half y = outQueueY.AllocTensor<half>();

            Half x1 = inQueueX1.DeQue<half>();
            Half x2 = inQueueX2.DeQue<half>();            
            Half inputData = inQueueInputData.DeQue<half>();
            Half value = inQueueValue.DeQue<half>();
            x1 = x1 * x2;
            x1 = x1 * value;
            y = inputData + x1;
            
            outQueueY.EnQue<half>(y);
            inQueueX1.FreeTensor(x1);
            inQueueX2.FreeTensor(x2);
            inQueueInputData.FreeTensor(inputData);
            inQueueValue.FreeTensor(value);


            // Half zLocal = outQueueZ.AllocTensor<half>();
            // // compute 
            // Exp(xLocal,xLocal,this->tileLength); // e^x
            // Reciprocal(zLocal,xLocal,this->tileLength); // e^(-x)
            // Sub(zLocal,xLocal,zLocal,this->tileLength);
            // half scalar = 0.5;
            // Muls(zLocal,zLocal,scalar,this->tileLength);
            // outQueueZ.EnQue<half>(zLocal);
            // inQueueX.FreeTensor(xLocal);
        }
        __aicore__ inline void CopyOut(int32_t progress){
            Half y = outQueueY.DeQue<half>();
            DataCopy(yGm[progress*this->tileLength],y,this->tileLength);
            outQueueY.FreeTensor(y);

            // Half zLocal =  outQueueZ.DeQue<half>();
            // DataCopy(zGm[progress*this->tileLength],zLocal,this->tileLength);
            // outQueueZ.FreeTensor(zLocal);
        }
    private:
        TPipe pipe; //管理内存
        TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX1; // 输入队列
        TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX2; // 输入队列
        TQue<QuePosition::VECIN, BUFFER_NUM> inQueueInputData; // 输入队列
        TQue<QuePosition::VECIN, BUFFER_NUM> inQueueValue; // 输入队列
        TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueY; // 输出队列

        GlobalTensor<half> x1Gm;        
        GlobalTensor<half> x2Gm;
        GlobalTensor<half> inputDataGm;
        GlobalTensor<half> valueGm;        
        GlobalTensor<half> yGm;

        uint32_t coreNum;
        uint32_t totalLength;
        uint32_t tileLength;
        uint32_t blockLength;
        uint32_t blockNum;
        uint32_t tileNum ;

};

extern "C" __global__ __aicore__ void addcmul_custom(GM_ADDR input_data, GM_ADDR x1, GM_ADDR x2, GM_ADDR value, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    // TODO: user kernel impl
    KernelAddcmul sinh;
    sinh.Init(input_data,x1,x2,value,y,tiling_data.totalLength,tiling_data.tileNum);
    sinh.process();
}