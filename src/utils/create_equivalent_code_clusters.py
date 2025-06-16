import json
import time
import argparse
import ast
from tqdm import tqdm


def get_bytecode(code_str):
    """Compile code string to bytecode."""
    code = ast.parse(code_str)
    
    # Bytecode Object
    bytecode = compile(code, '<string>', 'exec')

    # Bytecode bytes
    bytecode = bytecode.co_code  

    return bytecode


def read_code_samples_lazy(filename):
    """
    Generator to read code blocks one by one from a large file.
    Each block is separated by an empty line.
    """
    with open(filename, 'r') as f:
        current_block = []

        for line in f:
            stripped = line.strip()

            if stripped == '':
                if current_block:
                    yield '\n'.join(current_block)
                    current_block = []
            else:
                current_block.append(line.rstrip())

        if current_block:
            yield '\n'.join(current_block)


class CanonicalizeCommutativeOps(ast.NodeTransformer):
    def visit_BinOp(self, node):
        self.generic_visit(node)

        if isinstance(node.op, (ast.Add, ast.Mult)):
            operands = self.flatten_commutative_chain(node, type(node.op))
            sorted_operands = sorted(operands, key=self.sort_key)

            return self.build_commutative_chain(sorted_operands, node.op)

        return node

    def flatten_commutative_chain(self, node, op_type):
        if isinstance(node, ast.BinOp) and isinstance(node.op, op_type):

            return self.flatten_commutative_chain(node.left, op_type) + \
                   self.flatten_commutative_chain(node.right, op_type)
        else:
            return [node]

    def build_commutative_chain(self, operands, op):
        if not operands:
            return ast.Constant(value=0 if isinstance(op, ast.Add) else 1)

        expr = operands[0]

        for operand in operands[1:]:
            expr = ast.BinOp(left=expr, op=op, right=operand)

        return expr

    def sort_key(self, node):
        if isinstance(node, ast.Constant):
            return (0, repr(node.value))
        elif isinstance(node, ast.Name):
            return (1, node.id)
        else:
            return (2, ast.dump(node))


def normalize_code(code):
    tree = ast.parse(code)
    normalized = CanonicalizeCommutativeOps().visit(tree)
    ast.fix_missing_locations(normalized)

    return ast.unparse(normalized)


def read_code_samples(input_path):
    with open(input_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Split by double newlines, assuming samples are separated by empty lines
    samples = [sample.strip() for sample in content.strip().split('\n\n') if sample.strip()]

    return samples

def deserialize_bytecode(s):
    try:
        return eval(s)  # Be cautious: `eval` is only safe if data is trusted
    except:
        print("Defective byte code", s)
        return None


def main():
    parser = argparse.ArgumentParser(description='N/A')
    parser.add_argument('--subset_path', type=str, help='Path to the exhaustive generated file')
    parser.add_argument('--bytecodes_path', type=str, help='Path to the bytecodes')
    parser.add_argument('--output_path', type=str, help='Path to the output of the dict')
    args = parser.parse_args()
    t = time.time()

    subset_codes = read_code_samples(args.subset_path)
    subset_bytecodes = read_code_samples(args.bytecodes_path)

    subset_bytecodes = [deserialize_bytecode(b) for b in subset_bytecodes]
    subset_bytecodes = [b for b in subset_bytecodes if b is not None]

    # Create initial bytecodes
    clusters = {str(b): [] for b in subset_bytecodes}

    for line_block in tqdm(subset_codes):
        # Important: use Python 3.11.7 !!!!!! (Molgen conda)
        normalized_line = normalize_code(line_block)
        bytecode = get_bytecode(str(normalized_line))
        if str(bytecode) in clusters:
            clusters[str(bytecode)].append(line_block)

    # Save to JSON
    with open(args.output_path, 'w') as f:
        json.dump(clusters, f)

    print(f"Time: {time.time() - t :.2f}s")    

if __name__ == "__main__":
    main()
