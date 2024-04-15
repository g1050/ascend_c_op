#include "kernel_operator.h"
using namespace AscendC;
constexpr int32_t BUFFER_NUM = 2;                                     

template<typename dataType>
class KernelLessEqual{
    public:
        __aicore__ inline KernelLessEqual(){}
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
            yGm.SetGlobalBuffer((__gm__ uint8_t*)y + this->blockLength * GetBlockIdx(), this->blockLength);
            // std::cout << "tileLength " << tileLength << " block Length " << blockLength << std::endl;
            pipe.InitBuffer(inQueueX2,BUFFER_NUM,this->tileLength*sizeof(dataType)); 
            pipe.InitBuffer(inQueueX1,BUFFER_NUM,this->tileLength*sizeof(dataType)); 
            pipe.InitBuffer(outQueueY,BUFFER_NUM,this->tileLength*sizeof(uint8_t)); 
            pipe.InitBuffer(tmpMask,this->tileLength*sizeof(uint8_t)); // selDataSize can't allign to 32
            pipe.InitBuffer(tmpOneTensor,this->tileLength*sizeof(half)); 
            pipe.InitBuffer(tmpZeroTensor,this->tileLength*sizeof(half)); 
            pipe.InitBuffer(tmpSelRes,this->tileLength*sizeof(half)); 
            one = tmpOneTensor.Get<half>();
            zero = tmpZeroTensor.Get<half>();
            Muls(zero,zero,(half)0,this->tileLength);
            Adds(one,zero,(half)1,this->tileLength);
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
            // if(GetBlockIdx() == 0 && progress == 0 ){
            //     std::cout << "x1 " << (float)x1.GetValue(0) << " " <<  (float)x1.GetValue(1)<<  " "<< (float)x1.GetValue(2) << std::endl;
            //     std::cout << "x2 " << (float)x2.GetValue(0) << " "<< (float)x2.GetValue(1)<< " " <<(float)x2.GetValue(2) << std::endl;
            // }
            Compare(mask, x1, x2, CMPMODE::LE, this->tileLength);
            // mask = x1 <= x2;
            // std::cout << " sizeof() " << sizeof(y.GetValue(0)) << std::endl;
            // if(GetBlockIdx() == 0 && progress == 0 ){
            //     std::cout << "y " << (uint8_t)y.GetValue(0) << " " <<  (uint8_t)y.GetValue(1)<<  " "<< (uint8_t)y.GetValue(2) << std::endl;
            // }
            // Muls(y,y,(uint8_t)0,this->tileLength);
            // Adds(one,y,(uint8_t)1,this->tileLength);
            // Cast(y,zero,RoundMode::CAST_NONE,this->tileLength);
            // mask =1 ? src0 : src1
            Select(selRes, mask, one, zero, SELMODE::VSEL_TENSOR_TENSOR_MODE, this->tileLength);
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
        
        TBuf<QuePosition::VECCALC> tmpMulAbsX;
        TBuf<QuePosition::VECCALC> tmpExpX;
        TBuf<QuePosition::VECCALC> tmpAdd2;
        TBuf<QuePosition::VECCALC> tmpAdd2H; // high precision
        TBuf<QuePosition::VECCALC> tmpExpXH;

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
        op.Init(x1,x2,y,tiling_data.totalLength,tiling_data.tileNum);
        op.process();
    }else if (TILING_KEY_IS(DT_FLOAT)) {
        KernelLessEqual<float> op;
        op.Init(x1,x2,y,tiling_data.totalLength,tiling_data.tileNum);
        op.process();
    }
    // else if (TILING_KEY_IS(DT_INT32)) {
    //     KernelAddcmul<int32_t> op;
    //     op.Init(input_data,x1,x2,value,y,tiling_data.totalLength,tiling_data.tileNum);
    //     op.process();
    // }
    // else if (TILING_KEY_IS(DT_INT8)) {    // need cast
    //     KernelAddcmulAdapter<int8_t,half> op;
    //     op.Init(input_data,x1,x2,value,y,tiling_data.totalLength,tiling_data.tileNum);
    //     op.process();
    // }
}