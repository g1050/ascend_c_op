#include "kernel_operator.h"
#define ALIGN_LENGTH(length, align_num) (((length) + (align_num) - 1) / (align_num) * (align_num))

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;                                     

template<typename dataType>
class KernelAddcmul{
    public:
        __aicore__ inline KernelAddcmul(){}
        __aicore__ inline void Init(GM_ADDR input_data, GM_ADDR x1, GM_ADDR x2, GM_ADDR value, GM_ADDR y,uint32_t totalLength,uint32_t tileNum,AddcmulTilingData &tilingData){
            this->valueLength = 1;

            // this->blockNum = GetBlockNum();
            // ASSERT(this->blockNum != 0 && "block dim can not be zero");
            // this->blockLength = totalLength / this->blockNum;
            // this->tileNum = tileNum;
            // ASSERT(this->tileNum != 0 && "tile num can not be zero");
            // this->tileLength = this->blockLength / this->tileNum / BUFFER_NUM;            
            
            // x1Gm.SetGlobalBuffer((__gm__ dataType*)x1 + this->blockLength * GetBlockIdx(), this->blockLength); 
            // x2Gm.SetGlobalBuffer((__gm__ dataType*)x2 + this->blockLength * GetBlockIdx(), this->blockLength); 
            // inputDataGm.SetGlobalBuffer((__gm__ dataType*)input_data + this->blockLength * GetBlockIdx(), this->blockLength); 
            // valueGm.SetGlobalBuffer((__gm__ dataType*)value, this->valueLength);   
            // yGm.SetGlobalBuffer((__gm__ dataType*)y + this->blockLength * GetBlockIdx(), this->blockLength);   

            uint32_t offset = 0;
            if (GetBlockIdx() < tilingData.formerNum) { // x < 5     0,1,2,3,4 5个大块
                this->blockLength = tilingData.formerLength;
                offset = tilingData.formerLength * GetBlockIdx();
            } else { // 小块 5,6,7 三个小块
                this->blockLength = tilingData.tailLength;
                offset = tilingData.formerLength * tilingData.formerNum + tilingData.tailLength * (GetBlockIdx() - tilingData.formerNum);
            }
            this->tileLength = this->blockLength / tilingData.tileNum / BUFFER_NUM;
            this->tileNum = tilingData.tileNum;

            // align to 32
            this->tileLength = ALIGN_LENGTH(this->tileLength,32);
            this->valueLength = ALIGN_LENGTH(this->valueLength,32); // value's real length equals to 1

            x1Gm.SetGlobalBuffer((__gm__ dataType*)x1 + offset, this->blockLength); 
            x2Gm.SetGlobalBuffer((__gm__ dataType*)x2 + offset, this->blockLength); 
            inputDataGm.SetGlobalBuffer((__gm__ dataType*)input_data + offset, this->blockLength); 
            valueGm.SetGlobalBuffer((__gm__ dataType*)value, this->valueLength);   
            yGm.SetGlobalBuffer((__gm__ dataType*)y + offset, this->blockLength);   


            pipe.InitBuffer(inQueueX1,BUFFER_NUM,this->tileLength*sizeof(dataType)); 
            pipe.InitBuffer(inQueueX2,BUFFER_NUM,this->tileLength*sizeof(dataType));
            pipe.InitBuffer(inQueueInputData,BUFFER_NUM,this->tileLength*sizeof(dataType));
            pipe.InitBuffer(inQueueValue,BUFFER_NUM,this->valueLength*sizeof(dataType));
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
            Alloc(this->inQueueX1,x1Gm,progress);
            Alloc(this->inQueueX2,x2Gm,progress);
            Alloc(this->inQueueInputData,inputDataGm,progress);
            // Alloc(this->inQueueValue,valueGm,progress);
            LocalTensor<dataType> local = this->inQueueValue.template AllocTensor<dataType>();
            DataCopy(local,valueGm[0],this->valueLength);
            this->inQueueValue.EnQue(local);
        }
        __aicore__ inline void Alloc(TQue<QuePosition::VECIN, BUFFER_NUM> &que,GlobalTensor<dataType> gm,int32_t progress){
            LocalTensor<dataType> local = que.AllocTensor<dataType>(); 
            DataCopy(local,gm[progress*this->tileLength],this->tileLength);
            que.EnQue(local);
        }
        __aicore__ inline void Compute(int32_t progress){
            LocalTensor<dataType> y = outQueueY.AllocTensor<dataType>();
            LocalTensor<dataType> x1 = inQueueX1.DeQue<dataType>();
            LocalTensor<dataType> x2 = inQueueX2.DeQue<dataType>();            
            LocalTensor<dataType> inputData = inQueueInputData.DeQue<dataType>();
            LocalTensor<dataType> value = inQueueValue.DeQue<dataType>();
            dataType scalar = value.GetValue(0);
            x1 = x1 * x2;
            // x1 = x1 * value;
            Muls(x1, x1, scalar, this->tileLength);
            y = inputData + x1;

            outQueueY.EnQue<dataType>(y);
            inQueueX1.FreeTensor(x1);
            inQueueX2.FreeTensor(x2);
            inQueueInputData.FreeTensor(inputData);
            inQueueValue.FreeTensor(value);
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
        TQue<QuePosition::VECIN, BUFFER_NUM> inQueueInputData; 
        TQue<QuePosition::VECIN, BUFFER_NUM> inQueueValue; 
        TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueY; 
        
        LocalTensor<dataType> localValue;
        GlobalTensor<dataType> x1Gm;        
        GlobalTensor<dataType> x2Gm;
        GlobalTensor<dataType> inputDataGm;
        GlobalTensor<dataType> valueGm;        
        GlobalTensor<dataType> yGm;

        dataType scalarValue;
        uint32_t valueLength;
        uint32_t tileLength;
        uint32_t blockLength;
        uint32_t blockNum;
        uint32_t tileNum ;

};

template<typename dataType,typename castDataType>
class KernelAddcmulAdapter{
    public:
        __aicore__ inline KernelAddcmulAdapter(){}
        __aicore__ inline void Init(GM_ADDR input_data, GM_ADDR x1, GM_ADDR x2, GM_ADDR value, GM_ADDR y,uint32_t totalLength,uint32_t tileNum,AddcmulTilingData tilingData){
            uint32_t offset = 0;
            if (GetBlockIdx() < tilingData.formerNum) { // x < 5     0,1,2,3,4 5个大块
                this->blockLength = tilingData.formerLength;
                offset = tilingData.formerLength * GetBlockIdx();
            } else { // 小块 5,6,7 三个小块
                this->blockLength = tilingData.tailLength;
                offset = tilingData.formerLength * tilingData.formerNum + tilingData.tailLength * (GetBlockIdx() - tilingData.formerNum);
            }
            this->tileLength = this->blockLength / tilingData.tileNum / BUFFER_NUM;
            this->tileNum = tilingData.tileNum;

            // align to 32
            this->tileLength = ALIGN_LENGTH(this->tileLength,32);
            this->valueLength = ALIGN_LENGTH(this->valueLength,32); // value's real length equals to 1

            x1Gm.SetGlobalBuffer((__gm__ dataType*)x1 + offset, this->blockLength); 
            x2Gm.SetGlobalBuffer((__gm__ dataType*)x2 + offset, this->blockLength); 
            inputDataGm.SetGlobalBuffer((__gm__ dataType*)input_data + offset, this->blockLength); 
            valueGm.SetGlobalBuffer((__gm__ dataType*)value, this->valueLength);   
            yGm.SetGlobalBuffer((__gm__ dataType*)y + offset, this->blockLength);   


            pipe.InitBuffer(inQueueX1,BUFFER_NUM,this->tileLength*sizeof(dataType)); 
            pipe.InitBuffer(inQueueX2,BUFFER_NUM,this->tileLength*sizeof(dataType));
            pipe.InitBuffer(inQueueInputData,BUFFER_NUM,this->tileLength*sizeof(dataType));
            pipe.InitBuffer(inQueueValue,BUFFER_NUM,this->valueLength*sizeof(dataType));
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
            Alloc(this->inQueueX1,x1Gm,progress);
            Alloc(this->inQueueX2,x2Gm,progress);
            Alloc(this->inQueueInputData,inputDataGm,progress);
            // Alloc(this->inQueueValue,valueGm,progress);
            LocalTensor<dataType> local = this->inQueueValue.template AllocTensor<dataType>(); 
            DataCopy(local,valueGm[0],this->valueLength);
            this->inQueueValue.EnQue(local);
        }
        __aicore__ inline void Alloc(TQue<QuePosition::VECIN, BUFFER_NUM> &que,GlobalTensor<dataType> gm,int32_t progress){
            LocalTensor<dataType> local = que.AllocTensor<dataType>(); 
            DataCopy(local,gm[progress*this->tileLength],this->tileLength);
            que.EnQue(local);
        }
        __aicore__ inline void Compute(int32_t progress){
            LocalTensor<dataType> y =  this->outQueueY.template AllocTensor<dataType>();
            LocalTensor<dataType> x1 = inQueueX1.template DeQue<dataType>();
            LocalTensor<dataType> x2 = inQueueX2.template DeQue<dataType>();            
            LocalTensor<dataType> inputData = inQueueInputData.template DeQue<dataType>();
            LocalTensor<dataType> value = inQueueValue.template DeQue<dataType>();
            
            LocalTensor<castDataType> tmpx1 = tmpX1Queue.template Get<castDataType>();
            LocalTensor<castDataType> tmpx2 = tmpX2Queue.template Get<castDataType>();
            LocalTensor<castDataType> tmpInputData = tmpInputDataQueue.template Get<castDataType>();
            LocalTensor<castDataType> tmpValue = tmpValueQueue.template Get<castDataType>();
            LocalTensor<castDataType> tmpY = tmpYQueue.template Get<castDataType>();

            // Only support int8
            Cast(tmpx1, x1, RoundMode::CAST_NONE, this->tileLength);
            Cast(tmpx2, x2, RoundMode::CAST_NONE, this->tileLength);
            Cast(tmpInputData, inputData, RoundMode::CAST_NONE, this->tileLength);
            Cast(tmpValue, value, RoundMode::CAST_NONE, this->valueLength);
            castDataType scalar = value.GetValue(0);

            tmpx1 = tmpx1*tmpx2;
            // tmpx1 = tmpx1*tmpValue;
            Muls(tmpx1,tmpx1,scalar,this->tileLength);
            tmpY = tmpInputData + tmpx1;

            Cast(y, tmpY, RoundMode::CAST_NONE, this->tileLength);

            this->outQueueY.template EnQue<dataType>(y);
            this->inQueueX1.FreeTensor(x1);
            this->inQueueX2.FreeTensor(x2);
            this->inQueueInputData.FreeTensor(inputData);
            this->inQueueValue.FreeTensor(value);

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
        TQue<QuePosition::VECIN, BUFFER_NUM> inQueueInputData; 
        TQue<QuePosition::VECIN, BUFFER_NUM> inQueueValue; 
        TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueY; 

        TBuf<QuePosition::VECCALC> tmpX1Queue;
        TBuf<QuePosition::VECCALC> tmpX2Queue;
        TBuf<QuePosition::VECCALC> tmpInputDataQueue;
        TBuf<QuePosition::VECCALC> tmpValueQueue;
        TBuf<QuePosition::VECCALC> tmpYQueue;

        GlobalTensor<dataType> x1Gm;        
        GlobalTensor<dataType> x2Gm;
        GlobalTensor<dataType> inputDataGm;
        GlobalTensor<dataType> valueGm;        
        GlobalTensor<dataType> yGm;

        uint32_t valueLength;
        uint32_t coreNum;
        uint32_t totalLength;
        uint32_t tileLength;
        uint32_t blockLength;
        uint32_t blockNum;
        uint32_t tileNum ;

};

extern "C" __global__ __aicore__ void addcmul(GM_ADDR input_data, GM_ADDR x1, GM_ADDR x2, GM_ADDR value, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    if (TILING_KEY_IS(DT_FLOAT16)) {
        KernelAddcmul<half> op;
        op.Init(input_data,x1,x2,value,y,tiling_data.totalLength,tiling_data.tileNum,tiling_data);
        op.process();
    }else if (TILING_KEY_IS(DT_FLOAT)) {
        KernelAddcmul<float> op;
        op.Init(input_data,x1,x2,value,y,tiling_data.totalLength,tiling_data.tileNum,tiling_data);
        op.process();
    }else if (TILING_KEY_IS(DT_INT32)) {
        KernelAddcmul<int32_t> op;
        op.Init(input_data,x1,x2,value,y,tiling_data.totalLength,tiling_data.tileNum,tiling_data);
        op.process();
    }
    else if (TILING_KEY_IS(DT_INT8)) {    // need cast
        KernelAddcmulAdapter<int8_t,half> op;
        op.Init(input_data,x1,x2,value,y,tiling_data.totalLength,tiling_data.tileNum,tiling_data);
        op.process();
    }
}