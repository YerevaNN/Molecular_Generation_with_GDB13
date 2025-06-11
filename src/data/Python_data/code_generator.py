# Based on original code from https://github.com/MarwaNair/TinyPy-Generator 

import ast
import time
import copy
import builtins
import argparse
import itertools
from tqdm import tqdm
from io import StringIO
from typing import Iterator
from contextlib import redirect_stdout


class CodeGenerator:
    def __init__(self):
        """
        Initialize the CodeGenerator object with the given context-free grammar rules.

        """
        self.init_count = 0
        
        # Dictionary containing context-free grammar rules.
        self.cfg_rules = {
                # Variables and digits
                "VARIABLE": ["a", "b", "c"],
                "DIGIT": ["7", "8"],

                # Operators
                "ARITHMETIC_OPERATOR": ["+", "-", "*", "/"],
                "RELATIONAL_OPERATOR": ["<", ">", "<=", ">=", "!=", "=="],
                "LOGICAL_OPERATOR_INFIX": ["and", "or"],
                "LOGICAL_OPERATOR_PREFIX": ["not"],
                "OPERATOR": ["ARITHMETIC_OPERATOR"], 

                # Formatting
                "NEW_LINE": ["\n"],
                "TAB_INDENT": ["\t"],
                "BRACKET_OPEN": ['('],
                "BRACKET_CLOSE": [')'],
                "EQUALS": ["="],
                "COLON": [":"],
                "COMMA": [","],


                # Keywords
                "IF": ["if"],
                "ELIF": ["elif"],
                "ELSE": ["else"],
                "FOR": ["for"],
                "IN": ["in"],
                "RANGE": ["range"],
                "WHILE": ["while"],
                "PRINT": ["print"],

                # Terms and expressions
                "TERM": ["VARIABLE", "DIGIT"],
                "EXPRESSION": ["TERM SPACE OPERATOR SPACE TERM"],
                "ENCLOSED_EXPRESSION": ["BRACKET_OPEN EXPRESSION BRACKET_CLOSE"],
                "DISPLAY_EXPRESSION": ["VARIABLE SPACE OPERATOR SPACE VARIABLE" ,
                                        "VARIABLE SPACE OPERATOR SPACE DIGIT"],
  
                # Initializations and assignments
                "IDENTIFIER_INITIALIZATION": ["IDENTIFIER_INITIALIZATION INITIALIZATION", 
                                              "INITIALIZATION"],

                "INITIALIZATION": ["VARIABLE SPACE EQUALS SPACE DIGIT NEW_LINE"],

                "SIMPLE_ASSIGNMENTS": ["VARIABLE SPACE EQUALS SPACE EXPRESSION NEW_LINE" , ""],
                "ADVANCED_ASSIGNMENTS": ["VARIABLE SPACE EQUALS SPACE SIMPLE_ARITHMETIC_EVALUATION NEW_LINE", 
                                         "VARIABLE SPACE EQUALS SPACE EXPRESSION NEW_LINE" , 
                                         ""],

                "SIMPLE_ARITHMETIC_EVALUATION": ["SIMPLE_ARITHMETIC_EVALUATION ARITHMETIC_OPERATOR ENCLOSED_EXPRESSION", 
                                                 "ENCLOSED_EXPRESSION",
                                                ], 

                # Conditions
                "SIMPLE_IF_STATEMENT": ["IF SPACE CONDITION SPACE COLON NEW_LINE"],
                "ADVANCED_IF_STATEMENT": ["IF SPACE CHAIN_CONDITION SPACE COLON NEW_LINE"],
                "SIMPLE_ELIF_STATEMENT": ["ELIF SPACE CONDITION SPACE COLON NEW_LINE"],
                "ADVANCED_ELIF_STATEMENT": ["ELIF SPACE CHAIN_CONDITION SPACE COLON NEW_LINE"],
                "ELSE_STATEMENT": ["ELSE SPACE COLON NEW_LINE"],

                "CHAIN_CONDITION": ["CHAIN_CONDITION SPACE LOGICAL_OPERATOR_INFIX SPACE ENCLOSED_CONDITION", 
                                    "LOGICAL_OPERATOR_PREFIX SPACE ENCLOSED_CONDITION", 
                                    "ENCLOSED_CONDITION"],
                "ENCLOSED_CONDITION": ["BRACKET_OPEN CONDITION BRACKET_CLOSE"],
                "CONDITION": ["OPTIONAL_NOT CONDITION_EXPRESSION", "CONDITION_EXPRESSION"],
                "CONDITION_EXPRESSION": ["VARIABLE SPACE RELATIONAL_OPERATOR SPACE VARIABLE", 
                                         "VARIABLE SPACE RELATIONAL_OPERATOR SPACE DIGIT"],
                "OPTIONAL_NOT": ["LOGICAL_OPERATOR_PREFIX SPACE", "SPACE"], 

                # Loops
                "FOR_HEADER": ["FOR SPACE VARIABLE SPACE IN SPACE RANGE BRACKET_OPEN INITIAL COMMA SPACE FINAL COMMA SPACE STEP BRACKET_CLOSE SPACE COLON", 
                               "FOR SPACE VARIABLE SPACE IN SPACE RANGE BRACKET_OPEN INITIAL COMMA SPACE FINAL BRACKET_CLOSE SPACE COLON"],
                "INITIAL": ["DIGIT"],
            
                "FOR_LOOP": ["FOR_HEADER NEW_LINE TAB_INDENT DISPLAY"],
                "ADVANCED_FOR_LOOP": ["FOR_LOOP",
                                      "FOR_HEADER NEW_LINE TAB_INDENT ADVANCED_DISPLAY"],


                # While Loops

                # Definitions for relational operators
                "RELATIONAL_OPERATOR_LESS": [ "<", "<="],
                "RELATIONAL_OPERATOR_GREATER": [">", ">="],

                # Less than or equal conditions
                "CONDITION_EXPRESSION_LESS": [
                    "VARIABLE SPACE RELATIONAL_OPERATOR_LESS SPACE FINAL_LESS"
                ],
                
                # Greater than or equal conditions
                "CONDITION_EXPRESSION_GREATER": [
                    "VARIABLE SPACE RELATIONAL_OPERATOR_GREATER SPACE FINAL_GREATER"
                ],
            
                # While 
                "WHILE_HEADER_LESS": ["WHILE SPACE CONDITION_EXPRESSION_LESS SPACE COLON NEW_LINE"],
                "WHILE_LOOP_LESS": ["WHILE_HEADER_LESS TAB_INDENT DISPLAY NEW_LINE TAB_INDENT UPDATE_LESS"],
                "UPDATE_LESS": ["VARIABLE SPACE EQUALS SPACE VARIABLE SPACE + SPACE STEP"],
                
                "WHILE_HEADER_GREATER": ["WHILE SPACE CONDITION_EXPRESSION_GREATER SPACE COLON NEW_LINE"],
                "WHILE_LOOP_GREATER": ["WHILE_HEADER_GREATER TAB_INDENT DISPLAY NEW_LINE TAB_INDENT UPDATE_GREATER"],
                "UPDATE_GREATER": ["VARIABLE SPACE EQUALS SPACE VARIABLE SPACE - SPACE STEP"],
    
                # Displaying 
                "DISPLAY" : ["PRINT BRACKET_OPEN VARIABLE BRACKET_CLOSE"],
                "ADVANCED_DISPLAY" : ["DISPLAY",
                                      "PRINT BRACKET_OPEN DISPLAY_EXPRESSION BRACKET_CLOSE"],


                "LEVEL1.1": ["IDENTIFIER_INITIALIZATION SIMPLE_ASSIGNMENTS ADVANCED_DISPLAY"],
                "LEVEL1.2": ["IDENTIFIER_INITIALIZATION ADVANCED_ASSIGNMENTS ADVANCED_DISPLAY"],
                "LEVEL2.1": ["IDENTIFIER_INITIALIZATION SIMPLE_IF_STATEMENT TAB_INDENT DISPLAY", 
                            "IDENTIFIER_INITIALIZATION SIMPLE_IF_STATEMENT TAB_INDENT DISPLAY NEW_LINE SIMPLE_ELIF_STATEMENT TAB_INDENT DISPLAY NEW_LINE ELSE_STATEMENT TAB_INDENT DISPLAY", 
                            "IDENTIFIER_INITIALIZATION SIMPLE_IF_STATEMENT TAB_INDENT DISPLAY NEW_LINE ELSE_STATEMENT TAB_INDENT DISPLAY"],
                "LEVEL2.2": ["IDENTIFIER_INITIALIZATION ADVANCED_ASSIGNMENTS ADVANCED_IF_STATEMENT TAB_INDENT ADVANCED_DISPLAY"],
                            # "IDENTIFIER_INITIALIZATION ADVANCED_ASSIGNMENTS ADVANCED_IF_STATEMENT TAB_INDENT ADVANCED_DISPLAY NEW_LINE ADVANCED_ELIF_STATEMENT TAB_INDENT ADVANCED_DISPLAY NEW_LINE ELSE_STATEMENT TAB_INDENT ADVANCED_DISPLAY"],
                            # "IDENTIFIER_INITIALIZATION ADVANCED_ASSIGNMENTS ADVANCED_IF_STATEMENT TAB_INDENT ADVANCED_DISPLAY NEW_LINE ELSE_STATEMENT TAB_INDENT ADVANCED_DISPLAY"],
                "LEVEL3.1": ["IDENTIFIER_INITIALIZATION FOR_LOOP"],
                "LEVEL3.2": ["IDENTIFIER_INITIALIZATION ADVANCED_ASSIGNMENTS ADVANCED_FOR_LOOP"],

                "LEVEL4.1": ["IDENTIFIER_INITIALIZATION WHILE_LOOP_LESS", "IDENTIFIER_INITIALIZATION WHILE_LOOP_GREATER"],
            
                "ALL": ["LEVEL1.1", "LEVEL1.2", "LEVEL2.1", "LEVEL2.2", "LEVEL3.1", "LEVEL3.2", "LEVEL4.1"],
        }
  

    def generate_all_codes(self,
                            symbol: str,
                            for_init_step: dict = None,
                            depth: int = 0) -> Iterator[str]:
        """
        Exhaustively generate all code strings derivable from `symbol` up to `depth` levels deep.

        Parameters:
        - symbol: nonterminal or terminal to expand
        - for_init_step: dict carrying 'initial_value', 'step', 'initial_var'
        - depth: remaining recursion depth

        Yields:
        - str: one fully expanded code fragment for this symbol
        """ 
        def child_step():
            return dict(for_init_step)
        
        # Initialize state containers if first call
        if for_init_step is None:
            for_init_step = {}

        # 1) If it's a true terminal, resolve and emit it
        if symbol not in self.cfg_rules:
            yield self._resolve_terminal(symbol, for_init_step)
            
            return

        # 2) If we've hit max depth, drop this branch entirely
        if depth <= 0:
            return
        
        if symbol == "IDENTIFIER_INITIALIZATION":
                # if self.init_count < len(self.cfg_rules["VARIABLE"])-1:
                if self.init_count < 0:
                    self.init_count += 1
                else:
                    symbol = "INITIALIZATION"

        # 3) Otherwise, expand each production systematically
        for rule in self.cfg_rules[symbol]:

            rhs = rule.split(" ")
            expansion_lists = []

            for part in rhs:
                subexpansions = list(
                    self.generate_all_codes(
                        part,
                        for_init_step=child_step(),
                        depth=depth - 1
                    )
                )
                
                expansion_lists.append(subexpansions)

            if symbol == "INITIAL":
                for_init_step["initial_value"] = expansion_lists[0][0]

            # Cartesian product: stitch together one choice from each slot
            # print(depth, "Expansion list", expansion_lists)
            for combo in itertools.product(*expansion_lists):
                result = ''.join(combo)
                yield result

    def _resolve_terminal(self, symbol, for_init_step):
        """
        Handle leaf symbols and stateful productions (INITIAL, FINAL, STEP, identifiers).
        """
        if symbol == "SPACE":        
            return " "
        
        if symbol == "TAB_INDENT":   
            return "\t"
        
        if symbol == "NEW_LINE":     
                return "\n"
    
        # Your original base-case logic:
        if symbol == "INITIAL":
            # initialization digit from DIGIT rule
            # return random.choice(self.cfg_rules['DIGIT'])
            return self.cfg_rules['DIGIT'][0]
        
        elif symbol == 'FINAL' or symbol == 'FINAL_LESS':
            init = int(for_init_step.get('initial_value', '0'))
            # step, exec_count = random.choice([(1,2),(2,1),(2,2),(2,3),(3,2)])
            step, exec_count = (1,2)

            for_init_step['step'] = str(step)
            val = step * exec_count + init - (1 if symbol.endswith('LESS') else -1)
            return str(val)
        
        elif symbol == 'FINAL_GREATER':
            init = int(for_init_step.get('initial_value', '0'))
            # step, exec_count = random.choice([(1,2),(2,1),(2,2),(2,3),(3,2)])
            step, exec_count = (1,2)

            for_init_step['step'] = str(step)
            return str(init - step * exec_count + 1)
        
        elif symbol == 'STEP':
            return for_init_step.get('step', '0')
        
        # Default: literal or symbol name maps to itself
        return symbol


    def generate_programs_exhaustive(self, level: str, max_depth: int):
        """
        Wrapper that sets up initial state and walks the 'ALL' nonterminal.
        """
        for_init = {}
        start_symbol = "LEVEL" + level if level != "ALL" else "ALL"

        for prog in tqdm(self.generate_all_codes(start_symbol,
                                            for_init_step=for_init,
                                            depth=max_depth)):
            text = prog.replace("SPACE", " ")
            text = text.replace("NEW_LINE", "\n")
            text = text.replace("TAB_INDENT", "\t")
            yield text
            

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


def main(args): 
    all_programs = set()
    start_time = time.time()
    cg = CodeGenerator()

    with open(args.outfile, 'w') as f:
        for prog in cg.generate_programs_exhaustive(level=args.level, max_depth=args.depth):
            prog = f"""{prog}"""
            is_valid = check_validenss(prog)
            errors = find_undeclared_vars(prog)

            if is_valid and not errors and (prog not in all_programs):
                f.write(prog + '\n\n')
                all_programs.add(prog)

    print(f"Generated {len(all_programs)} unique programs in {time.time() - start_time:.2f}s")  


if __name__ == "__main__":
    # levels = ["1.1", "1.2", "2.1", "2.2", "3.1", "3.2"]
    levels = ["3.2"]
    depth = 8
    vars = 3

    for level in levels:
        print("Generating from level:", level, "depth:", depth, "vars:", vars)

        parser = argparse.ArgumentParser(description="Exhaustive CFG code generator.")
        parser.add_argument('--level', default=level)
        parser.add_argument('--depth', type=int, default=depth)
        parser.add_argument('--outfile', default=f"/nfs/dgx/raid/molgen/code_recall/data/data_level_{level}_depth_{depth}_{vars}_vars_new.txt")
        # parser.add_argument('--outfile', default=f"./exhaustive_generations/data_level_{level}_depth_{depth}_{vars}_vars_check.txt")
        # parser.add_argument('--outfile', default=f"data_level_{level}_depth_{depth}_{vars}_vars_check.txt")
        args = parser.parse_args()

        main(args)
