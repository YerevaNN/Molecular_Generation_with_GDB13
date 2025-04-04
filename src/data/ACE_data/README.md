# Creation of ACE dataset

Step 1. Generate txt sentences with NLTK Grammar with specified depth
```shellsession
    $ bash run_generate_ace_dataset.sh
```
Step 2. Convert ACE sentences to TPTP (one line per sentence) using APE   
```shellsession
    $ bash run_to_tptp.sh
```  

Step 3. Group sentences with BOW in different files for easy equivalence search
```shellsession
    $ bash run_group_tptp.sh
```  

Step 4. Run Vampire theorem prover for checking equivalence and creating equivalence clusters
```shellsession
    $ bash run_theorem_prover.sh
```