#include "kernel_operator.h"
using namespace AscendC;
constexpr int32_t BUFFER_NUM = 2;                                     

template<typename dataType>
class KernelMseLoss{
    public:
        __aicore__ inline KernelMseLoss(){}
        __aicore__ inline void Init(GM_ADDR x1, GM_ADDR x2,GM_ADDR y, uint32_t totalLength,uint32_t tileNum){
            this->blockNum = GetBlockNum();
            ASSERT(this->blockNum != 0 && "block dim can not be zero");
            this->blockLength = totalLength / this->blockNum;
            this->tileNum = tileNum;
            ASSERT(this->tileNum != 0 && "tile num can not be zero");
            this->tileLength = this->blockLength / this->tileNum / BUFFER_NUM;            
            this->selDataSize = this->tileLength / 8;
            x2Gm.SetGlobalBuffer((__gm__ dataType*)x2 + this->blockLength * GetBlockIdx(), this->blockLength);   
            x1Gm.SetGlobalBuffer((__gm__ dataType*)x1 + this->blockLength * GetBlockIdx(), this->blockLength); 
            yGm.SetGlobalBuffer((__gm__ dataType*)y + this->blockLength * GetBlockIdx(), this->blockLength);
            // std::cout << "tileLength " << tileLength << " block Length " << blockLength << std::endl;
            pipe.InitBuffer(inQueueX2,BUFFER_NUM,this->tileLength*sizeof(dataType)); 
            pipe.InitBuffer(inQueueX1,BUFFER_NUM,this->tileLength*sizeof(dataType)); 
            pipe.InitBuffer(outQueueY,BUFFER_NUM,this->tileLength*sizeof(dataType)); 

        }
        __aicore__ inline void process(){
            int32_t loopCount = this->tileNum*BUFFER_NUM; 
            for(auto i = 0;i<loopCount;i++){
                CopyIn(i);
                Compute(i);
                CopyOut(i);
            }
        }
    private:
        __aicore__ inline void CopyIn(int32_t progress){
            Alloc(this->inQueueX2,x2Gm,progress);
            Alloc(this->inQueueX1,x1Gm,progress);
        }
        __aicore__ inline void Alloc(TQue<QuePosition::VECIN, BUFFER_NUM> &que,GlobalTensor<dataType> &gm,int32_t progress){
            LocalTensor<dataType> local = que.AllocTensor<dataType>(); 
            DataCopy(local,gm[progress*this->tileLength],this->tileLength);
            que.EnQue(local);
        }
        __aicore__ inline void Compute(int32_t progress){
            LocalTensor<dataType> x1 = inQueueX1.DeQue<dataType>();
            LocalTensor<dataType> x2 = inQueueX2.DeQue<dataType>();
            LocalTensor<dataType> y = outQueueY.AllocTensor<dataType>();
            x1 = x1 - x2;
            x1 = x1 * x1;
            y = x1;
            // 8片，每片16个float,32Bytes
            outQueueY.EnQue<dataType>(y);
            inQueueX1.FreeTensor(x1);
            inQueueX2.FreeTensor(x2);
        }
        __aicore__ inline void CopyOut(int32_t progress){
            LocalTensor<dataType> y = outQueueY.DeQue<dataType>();
            DataCopy(yGm[progress*this->tileLength],y,this->tileLength);
            outQueueY.FreeTensor(y);
        }
    protected:
        TPipe pipe; 
        TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX1; 
        TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX2; 
        TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueY; 

        GlobalTensor<dataType> x1Gm;      
        GlobalTensor<dataType> x2Gm;      
        GlobalTensor<dataType> yGm;

        uint32_t coreNum;
        uint32_t totalLength;
        uint32_t selDataSize;
        uint32_t tileLength;
        uint32_t blockLength;
        uint32_t blockNum;
        uint32_t tileNum ;

};

enum class Reduction {
    Mean,
    Green,
    Blue
};

extern "C" __global__ __aicore__ void mse_loss(GM_ADDR predict, GM_ADDR label, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    Reduction reduction = static_cast<Reduction>(tiling_data.reduction);
    if (TILING_KEY_IS(DT_FLOAT16)) {
        KernelMseLoss<half> op;
        op.Init(predict,label,y,tiling_data.totalLength,tiling_data.tileNum);
        op.process();
    }else if (TILING_KEY_IS(DT_FLOAT)) {
        KernelMseLoss<float> op;
        op.Init(predict,label,y,tiling_data.totalLength,tiling_data.tileNum);
        op.process();
    }
}