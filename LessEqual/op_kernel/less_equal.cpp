#include "kernel_operator.h"
using namespace AscendC;
constexpr int32_t BUFFER_NUM = 2;                                     
#define ALIGN_LENGTH(length, align_num) (((length) + (align_num) - 1) / (align_num) * (align_num))

template<typename dataType>
class KernelLessEqual{
    public:
        __aicore__ inline KernelLessEqual(){}
        __aicore__ inline void Init(GM_ADDR x1, GM_ADDR x2,GM_ADDR y, uint32_t totalLength,uint32_t tileNum,LessEqualTilingData &tilingData){
            uint32_t offset = 0;
            if (GetBlockIdx() < tilingData.formerNum) { 
                this->blockLength = tilingData.formerLength;
                offset = tilingData.formerLength * GetBlockIdx();
            } else { 
                this->blockLength = tilingData.tailLength;
                offset = tilingData.formerLength * tilingData.formerNum + tilingData.tailLength * (GetBlockIdx() - tilingData.formerNum);
            }
            this->tileLength = this->blockLength / tilingData.tileNum / BUFFER_NUM;
            this->tileNum = tilingData.tileNum;

            // align to 32
            this->tileLength = ALIGN_LENGTH(this->tileLength,32);

            x2Gm.SetGlobalBuffer((__gm__ dataType*)x2 + offset, this->blockLength);   
            x1Gm.SetGlobalBuffer((__gm__ dataType*)x1 + offset, this->blockLength); 
            yGm.SetGlobalBuffer((__gm__ uint8_t*)y + offset, this->blockLength);
            
            pipe.InitBuffer(inQueueX2,BUFFER_NUM,this->tileLength*sizeof(dataType)); 
            pipe.InitBuffer(inQueueX1,BUFFER_NUM,this->tileLength*sizeof(dataType)); 
            pipe.InitBuffer(outQueueY,BUFFER_NUM,this->tileLength*sizeof(uint8_t)); 

            pipe.InitBuffer(tmpMask,this->tileLength*sizeof(uint8_t)); // todo: selDataSize can't allign to 32
            pipe.InitBuffer(tmpOneTensor,this->tileLength*sizeof(half)); 
            pipe.InitBuffer(tmpZeroTensor,this->tileLength*sizeof(half)); 
            pipe.InitBuffer(tmpSelRes,this->tileLength*sizeof(half)); 

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
            LocalTensor<uint8_t> y = outQueueY.AllocTensor<uint8_t>();
            LocalTensor<uint8_t> mask = tmpMask.Get<uint8_t>();
            LocalTensor<half> selRes = tmpSelRes.Get<half>();
            one = tmpOneTensor.Get<half>();
            zero = tmpZeroTensor.Get<half>();
            Muls(zero,zero,(half)0,this->tileLength);
            Adds(one,zero,(half)1,this->tileLength);
            // Atlas 200/500 A2推理产品，支持的数据类型为：half/float
            Compare(mask, x1, x2, CMPMODE::LE, this->tileLength);
            // half/float
            Select(selRes, mask, one, zero, SELMODE::VSEL_TENSOR_TENSOR_MODE, this->tileLength,1, { 1, 1, 1, 8, 8, 8 });
            Cast(y,selRes,RoundMode::CAST_NONE,this->tileLength);

            outQueueY.EnQue<uint8_t>(y);
            inQueueX1.FreeTensor(x1);
            inQueueX2.FreeTensor(x2);
        }
        __aicore__ inline void CopyOut(int32_t progress){
            LocalTensor<uint8_t> y = outQueueY.DeQue<uint8_t>();
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
        GlobalTensor<uint8_t> yGm;

        TBuf<QuePosition::VECCALC> tmpMask;
        TBuf<QuePosition::VECCALC> tmpOneTensor;
        TBuf<QuePosition::VECCALC> tmpZeroTensor;
        TBuf<QuePosition::VECCALC> tmpSelRes;
        LocalTensor<half> one;
        LocalTensor<half> zero;

        uint32_t coreNum;
        uint32_t totalLength;
        uint32_t selDataSize;
        uint32_t tileLength;
        uint32_t blockLength;
        uint32_t blockNum;
        uint32_t tileNum ;

};


extern "C" __global__ __aicore__ void less_equal(GM_ADDR x1, GM_ADDR x2, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    if (TILING_KEY_IS(DT_FLOAT16)) {
        KernelLessEqual<half> op;
        op.Init(x1,x2,y,tiling_data.totalLength,tiling_data.tileNum,tiling_data);
        op.process();
    }else if (TILING_KEY_IS(DT_FLOAT)) {
        KernelLessEqual<float> op;
        op.Init(x1,x2,y,tiling_data.totalLength,tiling_data.tileNum,tiling_data);
        op.process();
    }
    // else if (TILING_KEY_IS(DT_INT32)) {
    //     KernelAddcmul<int32_t> op;
    //     op.Init(input_data,x1,x2,value,y,tiling_data.totalLength,tiling_data.tileNum,tiling_data);
    //     op.process();
    // }
    // else if (TILING_KEY_IS(DT_INT8)) {    // need cast
    //     KernelAddcmulAdapter<int8_t,half> op;
    //     op.Init(input_data,x1,x2,value,y,tiling_data.totalLength,tiling_data.tileNum,tiling_data);
    //     op.process();
    // }
}