
#include "mse_loss_tiling.h"
#include "register/op_def_registry.h"


namespace optiling {
enum class Reduction {
    Mean,
    Green,
    Blue
};
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    MseLossTilingData tiling;
    const gert::Tensor *x1 = context->GetInputTensor(0);
    ge::DataType x1_type = x1->GetDataType();
    if (x1_type == ge::DT_FLOAT16){
        context->SetTilingKey(ge::DT_FLOAT16);
    }else if (x1_type == ge::DT_FLOAT){
        context->SetTilingKey(ge::DT_FLOAT);
    }else if (x1_type == ge::DT_INT8){
        context->SetTilingKey(ge::DT_INT8);
    }else if (x1_type == ge::DT_INT32){
        context->SetTilingKey(ge::DT_INT32);
    }
    //
    const gert::StorageShape* x1_shape = context->GetInputShape(0);
    int32_t data_sz = 1;
    for (int i = 0; i < x1_shape->GetStorageShape().GetDimNum(); i++)
    data_sz *= x1_shape->GetStorageShape().GetDim(i);
    tiling.set_size(data_sz);    
    tiling.set_totalLength(data_sz);
    tiling.set_tileNum(4);
    const std::string reduction = context->GetAttrs()->GetStr(0);
    std::cout << "reduction " << reduction << std::endl;
    Reduction reduct = Reduction::Mean;
    if(reduction == "mean"){
        reduct = Reduction::Mean;
    }
    tiling.set_reduction(static_cast<uint32_t>(reduct));
    context->SetBlockDim(8);
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
class MseLoss : public OpDef {
public:
    explicit MseLoss(const char* name) : OpDef(name)
    {
        this->Input("predict")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("label")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Attr("reduction").AttrType(OPTIONAL).String("mean");

        this->SetInferShape(ge::InferShape);

        this->AICore()
            .SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend310b");

    }
};

OP_ADD(MseLoss);
}
