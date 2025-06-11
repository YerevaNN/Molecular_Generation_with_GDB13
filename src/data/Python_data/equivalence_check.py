import os
import ast
import argparse
from tqdm import tqdm
from  pathlib import Path


def get_bytecode(code_str):
    """Compile code string to bytecode."""
    code = ast.parse(code_str)
    
    # Bytecode Object
    bytecode = compile(code, '<string>', 'exec')

    # Bytecode bytes
    bytecode = bytecode.co_code

    return bytecode


def read_code_blocks_lazy(filename):
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


def main():
    # Parsing arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--path")
    args = parser.parse_args()

    # Path handling
    # output_folder = os.path.dirname(args.path)
    output_folder = "/nfs/dgx/raid/molgen/exhaustive_generations"
    file_name = Path(args.path).stem
    bytecode_path = os.path.join(output_folder, f"{file_name}_bytecodes.txt")
    canon_forms_path = os.path.join(output_folder, f"{file_name}_canon_forms.txt")

    seen_bytcodes = set()
    code_sample_count = 0
    buffer_size = 1000
    canonical_forms_buffer = []
    bytecodes_buffer = []

    with (
        open(bytecode_path, "w") as f_bytecode, 
        open(canon_forms_path, "w") as f_canon,
        ):
        for block_line in tqdm(read_code_blocks_lazy(args.path)):
            # normalized_block = normalize_code(block_line)
            bytecode_line = get_bytecode(block_line)

            # Find different code samples
            if bytecode_line not in seen_bytcodes:
                canonical_forms_buffer.append(block_line)
                bytecodes_buffer.append(str(bytecode_line))
                seen_bytcodes.add(bytecode_line)

                # Flush buffers if full
                if len(canonical_forms_buffer) > buffer_size:
                    f_bytecode.write("\n\n".join(bytecodes_buffer) + "\n\n")
                    f_canon.write("\n\n".join(canonical_forms_buffer) + "\n\n")
                bytecodes_buffer.clear()
                canonical_forms_buffer.clear()

            code_sample_count += 1

         # Flush remaining buffers
        if canonical_forms_buffer:
            f_canon.write("\n\n".join(canonical_forms_buffer) + "\n\n")
            f_bytecode.write("\n\n".join(bytecodes_buffer) + "\n\n")        

        print(f"Code count is {code_sample_count}, AST count is {len(seen_bytcodes)}")

if __name__ == "__main__":
    main()