#include "kernel_operator.h"
using namespace AscendC;
constexpr int32_t BUFFER_NUM = 2;                                     

template<typename dataType>
class KernelLessClipByValue{
    public:
        __aicore__ inline KernelLessClipByValue(){}
        __aicore__ inline void Init(GM_ADDR x1, GM_ADDR clip_value_min, GM_ADDR clip_value_max, GM_ADDR y, uint32_t totalLength,uint32_t tileNum){
            this->blockNum = GetBlockNum();
            ASSERT(this->blockNum != 0 && "block dim can not be zero");
            this->blockLength = totalLength / this->blockNum;
            this->tileNum = tileNum;
            ASSERT(this->tileNum != 0 && "tile num can not be zero");
            this->tileLength = this->blockLength / this->tileNum / BUFFER_NUM;            
            this->selDataSize = this->tileLength / 8;
            maxGm.SetGlobalBuffer((__gm__ dataType*)clip_value_max, this->blockLength);   
            minGm.SetGlobalBuffer((__gm__ dataType*)clip_value_min, this->blockLength);   
            x1Gm.SetGlobalBuffer((__gm__ dataType*)x1 + this->blockLength * GetBlockIdx(), this->blockLength); 
            yGm.SetGlobalBuffer((__gm__ dataType*)y + this->blockLength * GetBlockIdx(), this->blockLength);

            pipe.InitBuffer(inQueueMin,BUFFER_NUM,this->tileLength*sizeof(dataType)); 
            pipe.InitBuffer(inQueueMax,BUFFER_NUM,this->tileLength*sizeof(dataType)); 
            pipe.InitBuffer(inQueueX1,BUFFER_NUM,this->tileLength*sizeof(dataType)); 
            pipe.InitBuffer(outQueueY,BUFFER_NUM,this->tileLength*sizeof(dataType)); 
            pipe.InitBuffer(tmpMask,this->tileLength*sizeof(dataType)); // selDataSize can't allign to 32


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
            Alloc(this->inQueueMin,minGm,0);
            Alloc(this->inQueueMax,maxGm,0);
            Alloc(this->inQueueX1,x1Gm,progress);
        }
        __aicore__ inline void Alloc(TQue<QuePosition::VECIN, BUFFER_NUM> &que,GlobalTensor<dataType> &gm,int32_t progress){
            LocalTensor<dataType> local = que.AllocTensor<dataType>(); 
            DataCopy(local,gm[progress*this->tileLength],this->tileLength);
            que.EnQue(local);
        }
        __aicore__ inline void Compute(int32_t progress){
            LocalTensor<dataType> x1 = inQueueX1.DeQue<dataType>();
            LocalTensor<dataType> y = outQueueY.AllocTensor<dataType>();
            LocalTensor<dataType> mask = tmpMask.Get<dataType>();
            // workspace ?
            LocalTensor<dataType> min = inQueueMin.DeQue<dataType>();
            LocalTensor<dataType> max = inQueueMax.DeQue<dataType>();
            minValue = min.GetValue(0);
            maxValue = max.GetValue(0);
            Muls(min,min,(dataType)0,this->tileLength);
            Adds(min,min,(dataType)minValue,this->tileLength);
            Muls(max,max,(dataType)0,this->tileLength);
            Adds(max,max,(dataType)maxValue,this->tileLength);
            // x1 <= max ? 1 : 0
            Compare(mask, x1, max, CMPMODE::LE, this->tileLength);
            // mask == 1 ? : src0 : src1
            Select(y, mask, x1 , max, SELMODE::VSEL_TENSOR_TENSOR_MODE, this->tileLength,1, { 1, 1, 1, 8, 8, 8 });
            // y <= min ? 1 : 0
            Compare(mask, y, min, CMPMODE::LE, this->tileLength);
            Select(y, mask, min , y, SELMODE::VSEL_TENSOR_TENSOR_MODE, this->tileLength,1, { 1, 1, 1, 8, 8, 8 });

            outQueueY.EnQue<dataType>(y);
            inQueueX1.FreeTensor(x1);
            inQueueMin.FreeTensor(min);
            inQueueMax.FreeTensor(max);
        }
        __aicore__ inline void CopyOut(int32_t progress){
            LocalTensor<dataType> y = outQueueY.DeQue<dataType>();
            DataCopy(yGm[progress*this->tileLength],y,this->tileLength);
            outQueueY.FreeTensor(y);
        }
    protected:
        TPipe pipe; 
        TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX1; 
        TQue<QuePosition::VECIN, BUFFER_NUM> inQueueMin; 
        TQue<QuePosition::VECIN, BUFFER_NUM> inQueueMax; 
        TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueY; 

        dataType maxValue;
        dataType minValue;
        GlobalTensor<dataType> x1Gm;      
        GlobalTensor<dataType> maxGm;     
        GlobalTensor<dataType> minGm;       
        GlobalTensor<dataType> yGm;

        TBuf<QuePosition::VECCALC> tmpMask;

        uint32_t coreNum;
        uint32_t totalLength;
        uint32_t selDataSize;
        uint32_t tileLength;
        uint32_t blockLength;
        uint32_t blockNum;
        uint32_t tileNum ;

};

extern "C" __global__ __aicore__ void clip_by_value(GM_ADDR x, GM_ADDR clip_value_min, GM_ADDR clip_value_max, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    if (TILING_KEY_IS(DT_FLOAT16)) {
        KernelLessClipByValue<half> op;
        op.Init(x,clip_value_min,clip_value_max,y,tiling_data.totalLength,tiling_data.tileNum);
        op.process();
    }
    else if (TILING_KEY_IS(DT_FLOAT)) {
        KernelLessClipByValue<float> op;
        op.Init(x,clip_value_min,clip_value_max,y,tiling_data.totalLength,tiling_data.tileNum);
        op.process();
    }
    // else if (TILING_KEY_IS(DT_INT32)) {
    //     KernelAddcmul<int32_t> op;
    //     op.Init(input_data,x1,x2,value,y,tiling_data.totalLength,tiling_data.tileNum);
    //     op.process();
    // }
}