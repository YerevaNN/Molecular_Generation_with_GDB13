export CUDA_VISIBLE_DEVICES="0"
export CUDA_LAUNCH_BLOCKING=1


python  ../metaseq/metaseq/cli/interactive_cli_simple.py --task language_modeling 
               