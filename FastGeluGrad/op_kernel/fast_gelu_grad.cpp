#include "kernel_operator.h"
using namespace AscendC;
constexpr int32_t BUFFER_NUM = 2;                                     

template<typename dataType>
class KernelFastGeluGrad{
    public:
        __aicore__ inline KernelFastGeluGrad(){}
        __aicore__ inline void Init(GM_ADDR dy, GM_ADDR x,GM_ADDR z, uint32_t totalLength,uint32_t tileNum){
            this->blockNum = GetBlockNum();
            ASSERT(this->blockNum != 0 && "block dim can not be zero");
            this->blockLength = totalLength / this->blockNum;
            this->tileNum = tileNum;
            ASSERT(this->tileNum != 0 && "tile num can not be zero");
            this->tileLength = this->blockLength / this->tileNum / BUFFER_NUM;
            // address to Tensor
            dyGm.SetGlobalBuffer((__gm__ dataType*)dy + this->blockLength * GetBlockIdx(), this->blockLength);   
            xGm.SetGlobalBuffer((__gm__ dataType*)x + this->blockLength * GetBlockIdx(), this->blockLength); 
            zGm.SetGlobalBuffer((__gm__ dataType*)z + this->blockLength * GetBlockIdx(),this->blockLength);

            pipe.InitBuffer(inQueueDy,BUFFER_NUM,this->tileLength*sizeof(dataType)); 
            pipe.InitBuffer(inQueueX,BUFFER_NUM,this->tileLength*sizeof(dataType)); 
            pipe.InitBuffer(outQueueZ,BUFFER_NUM,this->tileLength*sizeof(dataType)); 

            pipe.InitBuffer(tmpAbsX, this->tileLength*sizeof(dataType));
            pipe.InitBuffer(tmpDivDown, this->tileLength*sizeof(dataType));
            pipe.InitBuffer(tmpDivUp, this->tileLength*sizeof(dataType));
            pipe.InitBuffer(tmpMulAbsX, this->tileLength*sizeof(dataType));
            pipe.InitBuffer(tmpExpX, this->tileLength*sizeof(dataType));
            pipe.InitBuffer(tmpExpPnX, this->tileLength*sizeof(dataType));
            pipe.InitBuffer(tmpAdd2, this->tileLength*sizeof(dataType));


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
            Alloc(this->inQueueDy,dyGm,progress);
            Alloc(this->inQueueX,xGm,progress);
        }
        __aicore__ inline void Alloc(TQue<QuePosition::VECIN, BUFFER_NUM> &que,GlobalTensor<dataType> gm,int32_t progress){
            LocalTensor<dataType> local = que.AllocTensor<dataType>(); 
            DataCopy(local,gm[progress*this->tileLength],this->tileLength);
            que.EnQue(local);
        }
        __aicore__ inline void Compute(int32_t progress){
            LocalTensor<dataType> z = outQueueZ.AllocTensor<dataType>();
            LocalTensor<dataType> x = inQueueX.DeQue<dataType>();
            LocalTensor<dataType> dy = inQueueDy.DeQue<dataType>();

            LocalTensor<dataType> absX = tmpAbsX.template Get<dataType>();
            LocalTensor<dataType> &divDown = z;
            LocalTensor<dataType> &divUp = x;
            LocalTensor<dataType> mulAbsX = tmpMulAbsX.template Get<dataType>();
            LocalTensor<dataType> expX = tmpExpX.template Get<dataType>();
            LocalTensor<dataType> &expPnX = divUp;
            LocalTensor<dataType> add2 = tmpAdd2.template Get<dataType>();

            dataType attr = 1.702;
            dataType attrOpp = -1.702;            
            dataType attrHalf = 0.851;
            dataType one = 1;
            

            // attr = 1.702
            // attr_opp = 0 - attr
            // attr_half = attr / 2

            // abs_x = np.abs(input_x)
            // mul_abs_x = abs_x * attr_opp # -1.702 * |x|
            // exp_x = np.exp(mul_abs_x) # e ^ (-1.702 * |x|)
            // add_2 = input_x * exp_x * attr # 1.702*x*e ^ (-1.702 * |x|)
            // exp_pn_x = np.exp((input_x - abs_x) * attr) # e^(1.702(x-x|x|))
            // div_up = exp_x + add_2 +exp_pn_x
            // div_down = (exp_x + 1) ** 2 # (e^(-1.702*|x|))^2
            // res = div_up / div_down
            // golden = dy * res

            Abs(absX,x,this->tileLength);
            Muls(mulAbsX,absX,attrOpp,this->tileLength);
            Exp(expX,mulAbsX,this->tileLength);
            add2 = x * expX; // 可以先把add2加到结果上,add2就可以重复利用了
            Muls(add2,add2,attr,this->tileLength);
            expPnX = x - absX;
            Muls(expPnX,expPnX,attr,this->tileLength);
            Exp(divUp,expPnX,this->tileLength);
            divUp = divUp + add2;
            divUp = divUp + expX;
            
            Adds(divDown,expX,one,this->tileLength);
            divDown = divDown * divDown;
            
            z = divUp / divDown;
            z = z * dy;
            
            outQueueZ.EnQue<dataType>(z);
            inQueueX.FreeTensor(x);
            inQueueDy.FreeTensor(dy);
        }
        __aicore__ inline void CopyOut(int32_t progress){
            LocalTensor<dataType> z = outQueueZ.DeQue<dataType>();
            DataCopy(z[progress*this->tileLength],z,this->tileLength);
            outQueueZ.FreeTensor(z);
        }
    protected:
        TPipe pipe; 
        TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX; 
        TQue<QuePosition::VECIN, BUFFER_NUM> inQueueDy; 
        TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueZ; 

        GlobalTensor<dataType> xGm;      
        GlobalTensor<dataType> dyGm;      
        GlobalTensor<dataType> zGm;

        TBuf<QuePosition::VECCALC> tmpAbsX;
        TBuf<QuePosition::VECCALC> tmpDivDown;
        TBuf<QuePosition::VECCALC> tmpDivUp;
        TBuf<QuePosition::VECCALC> tmpMulAbsX;
        TBuf<QuePosition::VECCALC> tmpExpX;
        TBuf<QuePosition::VECCALC> tmpExpPnX;
        TBuf<QuePosition::VECCALC> tmpAdd2;

        dataType scalarValue;
        uint32_t coreNum;
        uint32_t totalLength;
        uint32_t tileLength;
        uint32_t blockLength;
        uint32_t blockNum;
        uint32_t tileNum ;

};

extern "C" __global__ __aicore__ void fast_gelu_grad(GM_ADDR dy, GM_ADDR x, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    // TODO: user kernel impl
    if (TILING_KEY_IS(DT_FLOAT16)) {
        KernelFastGeluGrad<half> op;
        op.Init(dy,x,z,tiling_data.totalLength,tiling_data.tileNum);
        op.process();
    }else if (TILING_KEY_IS(DT_FLOAT)) {
        KernelFastGeluGrad<float> op;
        op.Init(dy,x,z,tiling_data.totalLength,tiling_data.tileNum);
        op.process();
    }
}