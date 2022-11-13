from datetime import datetime
import argparse
import tensorflow as tf

from tensorflow_asr.utils import app_util
from tensorflow_asr.helpers import exec_helpers

logger = tf.get_logger()

#def evaluate_file(file:str):

    

def main(result_file: str):

    print("calling evaluate file %s" % result_file) 
    app_util.evaluate_results(result_file)
    

if __name__ == "__main__":
      
    time_s = datetime.now()

    parser = argparse.ArgumentParser(description = "control the functions for conformer")
    parser.add_argument("--result_file", action='store', type=str, default = "/home/liuyi/TensorFlowASR/examples/conformer/test_outputs/librispeech_testems.tsv", help="get the wer and cer")

    args = parser.parse_args()

    main(result_file=args.result_file)
    
    time_t = datetime.now() - time_s
    print("This run takes %s" % time_t)

