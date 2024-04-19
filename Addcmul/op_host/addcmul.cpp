
#include "addcmul_tiling.h"
#include "register/op_def_registry.h"
#define ALIGN_LENGTH(length, align_num) (((length) + (align_num) - 1) / (align_num) * (align_num))


namespace optiling {
constexpr uint32_t BLOCK_DIM = 8;
uint32_t BLOCK_SIZE = 256;
constexpr uint32_t TILE_NUM = 8;

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
  std::cout << " BLOCK_DIM " << BLOCK_DIM << std::endl;
  AddcmulTilingData tiling;
  const gert::StorageShape* x1_shape = context->GetInputShape(0);
  const gert::Tensor *x1 = context->GetInputTensor(0);
  ge::DataType x1_type = x1->GetDataType();
  uint32_t SIZE_OF_TYPE = 0;
  uint32_t ALIGN_NUM = 0;
  if (x1_type == ge::DT_FLOAT16){
    SIZE_OF_TYPE = sizeof(uint16_t);
    context->SetTilingKey(ge::DT_FLOAT16);
  }else if (x1_type == ge::DT_FLOAT){
    SIZE_OF_TYPE = sizeof(float);
    context->SetTilingKey(ge::DT_FLOAT);
  }else if (x1_type == ge::DT_INT8){
    SIZE_OF_TYPE = sizeof(int8_t);
    // BLOCK_SIZE = 32;
    context->SetTilingKey(ge::DT_INT8);
  }else if (x1_type == ge::DT_INT32){
    SIZE_OF_TYPE = sizeof(int32_t);
    context->SetTilingKey(ge::DT_INT32);
  }
  ALIGN_NUM = BLOCK_SIZE / SIZE_OF_TYPE;
  std::cout << "xkgao: SIZE_OF_TYPE " << SIZE_OF_TYPE << std::endl;
  int32_t data_sz = 1;
  for (int i = 0; i < x1_shape->GetStorageShape().GetDimNum(); i++)
    data_sz *= x1_shape->GetStorageShape().GetDim(i);
  
  uint32_t totalLengthAligned = ALIGN_LENGTH(data_sz,ALIGN_NUM);
  uint32_t formerNum = (totalLengthAligned / ALIGN_NUM) % BLOCK_DIM; 
  uint32_t tailNum = BLOCK_DIM - formerNum; // 3
  uint32_t formerLength = ((totalLengthAligned / BLOCK_DIM + ALIGN_NUM - 1) / ALIGN_NUM) * ALIGN_NUM; 
  uint32_t tailLength = (totalLengthAligned / BLOCK_DIM / ALIGN_NUM) * ALIGN_NUM; 
  tiling.set_formerNum(formerNum);
  tiling.set_tailNum(tailNum);
  tiling.set_formerLength(formerLength);
  tiling.set_tailLength(tailLength);
  tiling.set_alignNum(ALIGN_NUM);
  
  tiling.set_size(data_sz);
  tiling.set_totalLength(data_sz);
  tiling.set_tileNum(TILE_NUM);
  context->SetBlockDim(BLOCK_DIM);
  std::cout << "formerNum " << formerNum << " tailNum " << tailNum << " formerLength " << formerLength << " tailLength " << tailLength
  << " ALIGN_NUM " << ALIGN_NUM << " SIZE_OF_TYPE " << SIZE_OF_TYPE << std::endl << " data_sz " << data_sz << " totalLengthAligned " << totalLengthAligned;
  tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
  context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

  return ge::GRAPH_SUCCESS;
}
}


namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    const gert::Shape* x1_shape = context->GetInputShape(0);
    gert::Shape* y_shape = context->GetOutputShape(0);
    *y_shape = *x1_shape;
    return GRAPH_SUCCESS;
}
}


namespace ops {
class Addcmul : public OpDef {
public:
    explicit Addcmul(const char* name) : OpDef(name)
    {
        this->Input("input_data")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_INT32, ge::DT_INT8})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("x1")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_INT32, ge::DT_INT8})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("x2")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_INT32, ge::DT_INT8})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("value")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_INT32, ge::DT_INT8})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_INT32, ge::DT_INT8})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});

        this->SetInferShape(ge::InferShape);

        this->AICore()
            .SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend310b");

    }
};

OP_ADD(Addcmul);
}
