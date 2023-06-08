export CUDA_VISIBLE_DEVICES="1"
export CUDA_LAUNCH_BLOCKING=1


python  ../metaseq/metaseq/cli/interactive_cli.py --task language_modeling 
               