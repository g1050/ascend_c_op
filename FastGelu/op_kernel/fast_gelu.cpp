#include "kernel_operator.h"
using namespace AscendC;
constexpr int32_t BUFFER_NUM = 2;                                     

template<typename dataType>
class KernelFastGelu{
    public:
        __aicore__ inline KernelFastGelu(){}
        __aicore__ inline void Init(GM_ADDR x1, GM_ADDR y, uint32_t totalLength,uint32_t tileNum){
            this->blockNum = GetBlockNum();
            ASSERT(this->blockNum != 0 && "block dim can not be zero");
            this->blockLength = totalLength / this->blockNum;
            this->tileNum = tileNum;
            ASSERT(this->tileNum != 0 && "tile num can not be zero");
            this->tileLength = this->blockLength / this->tileNum / BUFFER_NUM;
            // address to Tensor
            x1Gm.SetGlobalBuffer((__gm__ dataType*)x1 + this->blockLength * GetBlockIdx(), this->blockLength); 
            yGm.SetGlobalBuffer((__gm__ dataType*)y + this->blockLength * GetBlockIdx(), this->blockLength);   
            
            pipe.InitBuffer(inQueueX1,BUFFER_NUM,this->tileLength*sizeof(dataType)); 
            pipe.InitBuffer(outQueueY,BUFFER_NUM,this->tileLength*sizeof(dataType));            
            pipe.InitBuffer(tmpAbsX, this->tileLength*sizeof(dataType));
            pipe.InitBuffer(tmpDivDown, this->tileLength*sizeof(dataType));

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
            Alloc(this->inQueueX1,x1Gm,progress);
        }
        __aicore__ inline void Alloc(TQue<QuePosition::VECIN, BUFFER_NUM> &que,GlobalTensor<dataType> gm,int32_t progress){
            LocalTensor<dataType> local = que.AllocTensor<dataType>(); 
            DataCopy(local,gm[progress*this->tileLength],this->tileLength);
            que.EnQue(local);
        }
        __aicore__ inline void Compute(int32_t progress){
            LocalTensor<dataType> y = outQueueY.AllocTensor<dataType>();
            LocalTensor<dataType> x1 = inQueueX1.DeQue<dataType>();
            LocalTensor<dataType> absX = tmpAbsX.template Get<dataType>();
            LocalTensor<dataType> divDown = tmpDivDown.template Get<dataType>();
            dataType scalarDown = -1.702;            
            dataType scalarUp = 0.851;
            dataType one = 1;
            // div down
            Abs(absX,x1,this->tileLength);
            Muls(divDown,absX,scalarDown,this->tileLength);
            Exp(divDown,divDown,this->tileLength);
            Adds(divDown,divDown,one,this->tileLength);
            // div up
            LocalTensor<dataType> &divUp = absX;
            divUp = x1 - absX;
            Muls(divUp,divUp,scalarUp,this->tileLength);
            Exp(divUp,divUp,this->tileLength);
            divUp = x1 * divUp;
            // y
            y = divUp / divDown;
            outQueueY.EnQue<dataType>(y);
            inQueueX1.FreeTensor(x1);
        }
        __aicore__ inline void CopyOut(int32_t progress){
            LocalTensor<dataType> y = outQueueY.DeQue<dataType>();
            DataCopy(yGm[progress*this->tileLength],y,this->tileLength);
            outQueueY.FreeTensor(y);
        }
    protected:
        TPipe pipe; 
        TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX1; 
        TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueY; 

        GlobalTensor<dataType> x1Gm;      
        GlobalTensor<dataType> yGm;

        TBuf<QuePosition::VECCALC> tmpAbsX;
        TBuf<QuePosition::VECCALC> tmpDivDown;

        dataType scalarValue;
        uint32_t valueLength;
        uint32_t coreNum;
        uint32_t totalLength;
        uint32_t tileLength;
        uint32_t blockLength;
        uint32_t blockNum;
        uint32_t tileNum ;

};
extern "C" __global__ __aicore__ void fast_gelu(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    if (TILING_KEY_IS(DT_FLOAT16)) {
        KernelFastGelu<half> op;
        op.Init(x,y,tiling_data.totalLength,tiling_data.tileNum);
        op.process();
    }else if (TILING_KEY_IS(DT_FLOAT)) {
        KernelFastGelu<float> op;
        op.Init(x,y,tiling_data.totalLength,tiling_data.tileNum);
        op.process();
    }
}