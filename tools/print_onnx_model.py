import argparse
import collections
import onnx
from google.protobuf.json_format import MessageToDict


def _shorten(s, limit):
    return f"{s[:limit - 3]}.." if len(s) > limit - 3 else s


def _print_summary(onnx_model, title):
    print(f"{'Model ':-<120}")
    print(f"Name: {title}")
    print(f"IR version: {onnx_model.ir_version}")
    opset_ids = ', '.join(sorted(f"{x.domain or '<default>'}:{x.version}" for x in onnx_model.opset_import))
    print(f"Opsets: {opset_ids}")
    for _input in onnx_model.graph.input:
        print("Input:", MessageToDict(_input))
    for _output in onnx_model.graph.output:
        print("Output:", MessageToDict(_output))


def print_model_info(onnx_model_path):
    model = onnx.load(onnx_model_path)
    # onnx.checker.check_model(model, full_check=True)
    _print_summary(model, onnx_model_path)

    graph = model.graph
    op_counter = collections.Counter()
    functions = {fn.name for fn in model.functions}

    if model.functions:
        print(f"{'Functions ':-<180}")
        print(f"{'':30}{'Op':30}{'Name':30}{'Input':30}{'Output':30}{'Attributes':30}")
        for fn in model.functions:
            fqdn = f"{fn.domain}:{fn.name}"
            print(f"{fqdn:30}")
            for node in fn.node:
                if node.op_type not in functions:
                    op_counter[node.op_type] += 1
                node_name = _shorten(node.name, 30)
                node_inputs = _shorten(', '.join(str(x) for x in list(node.input)), 30)
                node_outputs = _shorten(', '.join(str(x) for x in list(node.output)), 30)
                node_attributes = ', '.join(f"{x.name}:{x.type}" for x in list(node.attribute))
                print(f"{'':30}{node.op_type:30}{node_name:30}{node_inputs:30}{node_outputs:30}{node_attributes:30}")

    print(f"{'Graph ':-<180}")
    print(f"{'Op/Function':60}{'Name':30}{'Input':30}{'Output':30}{'Attributes':30}")
    for node in graph.node:
        if node.op_type not in functions:
            op_counter[node.op_type] += 1
        node_op_type = _shorten(node.op_type, 60)
        node_name = _shorten(node.name, 30)
        node_inputs = _shorten(', '.join(str(x) for x in list(node.input)), 30)
        node_outputs = _shorten(', '.join(str(x) for x in list(node.output)), 30)
        node_attributes = ', '.join(f"{x.name}:{x.type}" for x in list(node.attribute))
        print(f"{node_op_type:60}{node_name:30}{node_inputs:30}{node_outputs:30}{node_attributes:30}")
        
    print(f"{'Operator summary ':-<180}")
    print(f"{'Operator':30}{'Count':30}")
    for op in sorted(op_counter.items(), key=lambda x: -x[1]):
        print(f"{op[0]:30}{op[1]:<30}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prints ONNX model info, graph and functions")
    parser.add_argument("onnx_model_path", type=str, help="Path to ONNX model file.")
    args = parser.parse_args()

    print_model_info(args.onnx_model_path)
