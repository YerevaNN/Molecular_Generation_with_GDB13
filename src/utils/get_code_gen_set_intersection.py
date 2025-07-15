import json
import time
import re
import csv
import argparse
import ast
from tqdm import tqdm
from io import StringIO
import builtins
from typing import Iterator
from contextlib import redirect_stdout


def is_valid(code_line):
    try:
        code = ast.parse(code_line)
        bytecode = compile(code, '<string>', 'exec')
        return True
    except:
        return False    


def process_code_block(code_line):
    normalized_line = normalize_code(code_line)
    bytecode_line = get_bytecode(normalized_line)

    return (normalized_line, bytecode_line)


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


def check_validenss(prog):
    is_valid = True

    try:
        SIO = StringIO()
        with redirect_stdout(SIO):
            exec(prog)
    except:
        is_valid = False 

    return is_valid 


def find_undeclared_vars(source):
    tree = ast.parse(source)
    assigned = set()
    used = set()

    for node in ast.walk(tree):
        # any assignment target
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            assigned.add(node.id)
        # any usage
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            used.add(node.id)

    # drop builtins (print, len, etc.)
    builtins_names = set(dir(builtins))
    undeclared = used - assigned - builtins_names

    return undeclared


def main():
    parser = argparse.ArgumentParser(description='N/A')
    parser.add_argument('--gen_path', type=str, help='Path to the generation file')
    parser.add_argument('--bytecodes_path', type=str, help='Path to the training file / full subset')
    parser.add_argument('--subset_path', type=str, help='Path to the training file / full subset')
    parser.add_argument('--output_path', type=str, help='Path to the training file / full subset')
    parser.add_argument('--get_first', type=int, default=None, help='Path to the training file / full subset')
    args = parser.parse_args()

    gen_data = read_code_samples(args.gen_path)
    # gen_data = []

    # with open(args.gen_path, newline='') as csvfile:
    #     reader = csv.reader(csvfile)
    #     next(reader)  # Skip header

    #     for row in reader:
    #         indx = row[0].find("<>")
    #         code_line = row[0][:indx].strip()
    #         gen_data.append(code_line)

    if args.get_first:
        gen_data = gen_data[:args.get_first]

    subset_codes = read_code_samples(args.subset_path)
    subset_bytecodes = read_code_samples(args.bytecodes_path)
    
    all_codes = []
    gen_bytecodes = []
    invalid_count = 0
    valid_codes_outside_subset = []

    subset_bytecodes = [deserialize_bytecode(b) for b in subset_bytecodes]
    subset_bytecodes = [b for b in subset_bytecodes if b is not None]

    subset_codes = set(subset_codes)
    precision = 0

    for line_block in tqdm(gen_data):
        all_codes.append(line_block)

        if is_valid(line_block):
            line_block_norm, bytecode = process_code_block(line_block)
            gen_bytecodes.append(bytecode)

            if bytecode not in subset_bytecodes:
                precision += 1

            if bytecode in subset_bytecodes:
                if line_block not in subset_codes:
                    valid_codes_outside_subset.append(line_block)
        else:
            invalid_count += 1
            precision += 1

    print(args.gen_path)
    print("Unique bytecodes count", len(set(gen_bytecodes)))
    print("Invalid count", invalid_count)

    inside_subset = set(gen_bytecodes) & set(subset_bytecodes)

    ##########################################################
    # Let's calculate the precision
    # precision = 0

    # for line_block in tqdm(gen_data):
    #     if is_valid(line_block):
    #         line_block_norm, bytecode = process_code_block(line_block)
    #         if bytecode not in subset_bytecodes:
    #             precision += 1
    #     else:
    #         precision += 1
    # print(precision)        
    # exit()        
    ##########################################################

    outside_subset = set(gen_bytecodes) - set(subset_bytecodes)

    print("Precision", precision)
    print("Intersection", len(inside_subset))
    print("Not belonging", len(outside_subset))
    print("Valid but outside subset", len(valid_codes_outside_subset))

    with open("valid_outside_subset.txt", "w", encoding="utf-8") as f:
        for line in valid_codes_outside_subset:
            f.write(line + "\n\n")

    remained_valid_codes_outside_subset = set()

    # check with my filters
    with open("remained_valid_outside_subset.txt", "w", encoding="utf-8") as f:
        for prog in valid_codes_outside_subset:
            prog = f"""{prog}"""
            is_valid_prog = check_validenss(prog)
            errors = find_undeclared_vars(prog)

            if is_valid_prog and not errors and (prog not in remained_valid_codes_outside_subset):
            
                remained_valid_codes_outside_subset.add(prog)
                f.write(prog + "\n\n")

    print(f"Remained {len(remained_valid_codes_outside_subset)} / {len(valid_codes_outside_subset)}")            


if __name__ == "__main__":
    main()
