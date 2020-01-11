import flatbuffers
from tflite.Model import Model
from tflite.TensorType import TensorType
from tflite.BuiltinOperator import BuiltinOperator
from tflite.ActivationFunctionType import ActivationFunctionType
from tflite.BuiltinOptions import BuiltinOptions

MODEL_LOCATION = r"C:\Users\erezi\Downloads\mnist_lenet5.tflite"


def _add_map_to_const_class(cls):
    map = {cls.__dict__[name]: name for name in cls.__dict__ if not name.startswith('__')}
    cls.__new__ = lambda cls, x: map[x]
    return map


class IndentPrinter:

    def __init__(self, indengtation='\t'):
        self.PRINT = print
        self.indengtation = indengtation
        self.indent_count = 0

    def __enter__(self):
        self.indent_count += 1

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.indent_count -= 1

    def __call__(self, msg, *args, **kwargs):
        self.PRINT(f"{self.indengtation * self.indent_count}{msg}", *args, **kwargs)


if __name__ == '__main__':
    buf = open(MODEL_LOCATION, 'rb').read()
    buf = bytearray(buf)

    tabbing = 0


    def print_(s):
        print('\t' * tabbing + s)


    for cls in [TensorType, BuiltinOptions, BuiltinOperator, ActivationFunctionType]:
        _add_map_to_const_class(cls)

    print = IndentPrinter()
    model = Model.GetRootAsModel(buf, 0)
    print(f'The model contains {model.SubgraphsLength()} subgraphs, and {model.BuffersLength()} buffers:')

    tensor_name_mapping = dict()

    with print:
        for subgraph_idx in range(model.SubgraphsLength()):
            subgraph = model.Subgraphs(subgraph_idx)
            print(f'[{subgraph_idx}] Subgraph "{subgraph.Name()}" contains {subgraph.TensorsLength()} tensors:')
            with print:
                for tensor_idx in range(subgraph.TensorsLength()):
                    tensor = subgraph.Tensors(tensor_idx)
                    shape = tuple(tensor.ShapeAsNumpy())
                    name = "".join(map(chr, tensor.Name()))
                    print(f'[{tensor_idx}] Tensor "{name}"({TensorType(tensor.Type())}) has shape {shape}.')
                    tensor_name_mapping[tensor_idx] = name
                    buffer_idx = tensor.Buffer()
                    buffer = model.Buffers(buffer_idx)
                    with print:
                        print(f"The tensor's buffer has length {buffer.DataLength()}.")
            print(f'The graph has {subgraph.OperatorsLength()} operators.')
            with print:
                for op_idx in range(subgraph.OperatorsLength()):
                    op = subgraph.Operators(op_idx)
                    print(f'[{op_idx}] Operator of type "{BuiltinOperator(op.OpcodeIndex())}" has:')
                    # BuiltinOptions(op.BuiltinOptionsType())
                    with print:
                        for (name, item) in [('Inputs', op.InputsAsNumpy()), ('Outputs', op.OutputsAsNumpy())]:
                            print(f'{name}: [{" -> ".join(f"{tensor_name_mapping[node]}({node})" for node in item)}]')
