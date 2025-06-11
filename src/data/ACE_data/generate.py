import os
import time
import nltk
import argparse
from nltk import CFG
from nltk.parse.generate import generate
from tqdm import tqdm


def main(output_dir, depth):
    # Example grammar. Replace the grammar text with ACEWiki rules or another grammar.
    grammar_text = """
      S             -> NP VP
      NP            -> HumanNP | NonHumanNP
      HumanNP       -> Det HumanN | Det Adj HumanN | HumanNP2 RelClauseWho
      HumanNP2      -> Det HumanN | Det Adj HumanN
      NonHumanNP    -> Det NonHumanN | Det Adj NonHumanN | NonHumanNP2 RelClauseThat
      NonHumanNP2   -> Det NonHumanN | Det Adj NonHumanN
      RelClauseWho  -> 'who' VP
      RelClauseThat -> 'that' VP
      RelClauseThat -> 'that' PassiveVP
      PassiveVP     -> 'is' PastPart 'by' NP
      VP            -> V NP
      Det           -> 'a'
      Adj           -> 'big'
      HumanN        -> 'customer'
      NonHumanN     -> 'card'
      V             -> 'inserts' | 'uses'
      PastPart      -> 'inserted' | 'used'
    """
    grammar_text_simple = """
      S -> 'a' S 'b'
      S -> 'c'
    """
    print(grammar_text_simple)

    grammar = CFG.fromstring(grammar_text_simple)

    # Settings
    chunk_size = 10_000_000
    iteration = 0
    sentences_set = set()
    start_time = time.time()

    # Output directory and base filename
    output_file = os.path.join("./", f"all_sentences_{depth}_test.txt")

    # Checking dataset lengths till the certain depth
    # for l in range(1, d+1):
    #     print(f"Depth={l}, sentence count=", len(list(generate(grammar, depth=l))))
    # exit()

    with open(output_file, 'a') as file:
        # Generate sentences and flush to file in chunks
        for production in tqdm(generate(grammar, depth=depth), desc="Generating sentences"):
            iteration += 1
            sentence_text = " ".join(production)
            sentences_set.add(sentence_text)
            
            if iteration % chunk_size == 0:
                # Sort and write the current batch to file
                sorted_sentences = sorted(sentences_set)

                for s in sorted_sentences:
                    file.write(f"{s}.\n")

                print(f"Saving {iteration // chunk_size} chunk.")

                # Reset for the next chunk
                sentences_set.clear()

        # Write any remaining sentences
        if sentences_set:
            sorted_sentences = sorted(sentences_set)
            for s in sorted_sentences:
                file.write(f"{s}.\n")

    print(f"Generated {iteration} sentences up to depth {depth}. Time: {time.time() - start_time}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create dataset with the given grammar."
    )
    parser.add_argument("--output_dir", default="./", type=str,
                        help="Path to a folder where the txt file be created.")
    parser.add_argument("--depth", default=6, type=int,
                        help="Depth of the tree.")
    
    # Parsing args
    args = parser.parse_args()

    # log_args(args)
    main(args.output_dir, args.depth)