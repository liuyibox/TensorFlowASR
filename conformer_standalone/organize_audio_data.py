
import Utils as util
import natsort
from pydub import AudioSegment
import os
#import tensorflow as tf
import random
import numpy as np
import librosa
import soundfile as sf

def read_text(file_path):
    
    f = open(file_path, 'r')
    return f.read().splitlines()

#def quick_sort():
#
#    y_pred = [ 
#               0.0154717, 0.0000527, 0.0001319, 0.0000039, 0.0000931, 
#               0.0001134, 0.0001246, 0.0001975, 0.0001810, 0.0007910, 
#               0.0001413, 0.0007680, 0.0025058, 0.0000438, 0.0000772, 
#               0.0000350, 0.0030968, 0.0000798, 0.0000198, 0.0000801, 
#               0.0000082, 0.0003388, 0.0000575, 0.0002570, 0.0002034, 
#               0.9361117, 0.0006198, 0.0011428, 0.0003388, 0.0001523, 
#               0.0012686, 0.0079135, 0.0007651, 0.0020957, 0.0222392, 
#               0.0003920, 0.0008601, 0.0001843, 0.0001299, 0.0000160, 
#               0.0000152, 0.0000022, 0.0000864, 0.0004262, 0.0003072, 
#               0.0000597
#            ]
#    labels = util.readFile("/home/liuyi/tflite_experimental/emsBert/data/fitted_label_names.txt")
#    topk_prob, topk_indices = tf.math.top_k(y_pred, k=5)
#    
#    for i in range(5):
#        topk_idx = topk_indices[i]
#        print(topk_prob[i], labels[topk_idx])

def organize_sample100():


    # organizing signs and symptoms as train dataset of fine-tuning
    transcript_dict = { "ps":"sampled_primary_symptom_transcripts.txt",
                        "pi":"sampled_primary_impression_transcripts.txt",
                        "as":"sampled_additional_symptom_transcripts.txt",
                        "si":"sampled_secondary_impression_transcripts.txt",
                      }

    
    signs_symptoms_audio_dir = "/home/liuyi/audio_data/sample100/signs_symptoms_audio"
    ps_transcript_path = os.path.join(signs_symptoms_audio_dir, "sampled_primary_symptom_transcripts.txt")
    pi_transcript_path = os.path.join(signs_symptoms_audio_dir, "sampled_primary_impression_transcripts.txt")
    as_transcript_path = os.path.join(signs_symptoms_audio_dir, "sampled_additional_symptom_transcripts.txt")
    si_transcript_path = os.path.join(signs_symptoms_audio_dir, "sampled_secondary_impression_transcripts.txt")

    ps_lines = util.readFile(ps_transcript_path)
    pi_lines = util.readFile(pi_transcript_path)
    as_lines = util.readFile(as_transcript_path)
    si_lines = util.readFile(si_transcript_path)

    all_signs_symptoms_audios = set(os.listdir(signs_symptoms_audio_dir))

    def get_transcript_records(lines, ss_type):
    
        ss_records = []
        for idx, line in enumerate(lines):
            audio_f = ss_type + str(idx+1) + ".m4a"
            audio_path = os.path.join(signs_symptoms_audio_dir, audio_f)
            track = AudioSegment.from_file(audio_path, "m4a", sr=16000)
            wav_f_path = audio_path.replace("m4a", "wav")
            wav_file_handle = track.export(wav_f_path, format='wav')
            time_dur = track.duration_seconds
    #        print(type(time_dur))
            transcript = line.strip()
            record = wav_f_path + "\t" + str(round(time_dur, 2)) + "\t" + transcript
            ss_records.append(record)
        return ss_records

    all_records = get_transcript_records(ps_lines, "ps") + get_transcript_records(pi_lines, "pi") + get_transcript_records(as_lines, "as") + get_transcript_records(si_lines, "si")
   
    # print(all_records)

    # we are trying to organize populated template transcripts as test dataset of fine-tuning
    dataset_header = "PATH" + "\t" + "DURATION" + "\t" + "TRANSCRIPT"
    template_audio_dir = "/home/liuyi/audio_data/sample100/ems_templates_audio"
    sample_num = 100
    poped_template_records = []

    template_count_dict = dict()

    for i in range(sample_num):
        transcript_file_path = os.path.join(template_audio_dir, "sampled_test" + str(i) + ".txt")
        transcripts = util.readFile(transcript_file_path)
        assert len(transcripts) == 1
        transcript = transcripts[0].split("\t")[1]
        
        template_num = transcripts[0].split("\t")[0]
        if template_num in template_count_dict:
            template_count_dict[template_num] += 1
        else:
            template_count_dict[template_num] = 1

        audio_f = "tmp" + str(i) + ".m4a"
        audio_path = os.path.join(template_audio_dir, audio_f)
        track = AudioSegment.from_file(audio_path, "m4a", sr=16000)
        wav_f_path = audio_path.replace("m4a", "wav")
        wav_file_handle = track.export(wav_f_path, format='wav')
        time_dur = track.duration_seconds
        record = wav_f_path + "\t" + str(round(time_dur, 2)) + "\t" + transcript

        poped_template_records.append(record)


    all_records.insert(0, dataset_header)
    util.writeListFile("finetune-train_transcripts.tsv", all_records)
    print("finetune-train_transcripts.tsv",len(all_records))

#    print(poped_template_records)
    util.writeListFile("finetune-test_transcripts.tsv", poped_template_records)
    print("finetune-test_transcripts.tsv", len(poped_template_records))

#    print(template_count_dict)

    # how many test transcripts we want to move to train
    # this is to teach the conformer to recognize the template transcripts
    np.random.seed(1993)
    random.seed(1993)
    train_poped_tmp_test_poped_tmp = False
    test_percentage = 0.8
    if train_poped_tmp_test_poped_tmp:
        np.random.shuffle(poped_template_records)
        train_poped_template_records = random.sample(poped_template_records, int(test_percentage * len(poped_template_records)))
        test_poped_template_records = [x for x in poped_template_records if x not in train_poped_template_records]
        #test_poped_template_records = np.random.choice(poped_template_records, size=int(test_percentage * len(poped_template_records)), replace = False)
        #train_poped_template_records = poped_template_records

        train_poped_template_records.insert(0, dataset_header)
        util.writeListFile("finetune-train_poped_tmp_transcripts.tsv", train_poped_template_records)
        print("finetune-train_transcripts.tsv",len(train_poped_template_records))
    
        test_poped_template_records.insert(0, dataset_header)
    #    print(poped_template_records)
        util.writeListFile("finetune-test_poped_tmp_transcripts.tsv", test_poped_template_records)
        print("finetune-test_transcripts.tsv", len(test_poped_template_records))

    train_standalone_ss_test_concatenated_ss = True
    if train_standalone_ss_test_concatenated_ss:
#                                        "/home/liuyi/audio_data/sample100/signs_symptoms_audio_concatenated "

        concatenated_records = []
        concatenated_ss_audios_path = "/home/liuyi/audio_data/sample100/signs_symptoms_audio_concatenated/"
#        all_concatenated_ss_audios = os.listdir(concatenated_ss_audios_path)
        #print(len(all_concatenated_ss_audios))
#        all_concatenated_ss_audios = natsort.natsorted(all_concatenated_ss_audios)

        #print(len(all_concatenated_ss_audios))
#        assert len(all_concatenated_ss_audios) == 200
        #print(all_concatenated_ss_audios[-1])
#        assert all_concatenated_ss_audios[-1] == "sss100.wav"

        transcript_path = "/home/liuyi/audio_data/sample100/sampled_signs_symptoms.txt"
        transcript_lines = util.readFile(transcript_path)

        for idx, line in enumerate(transcript_lines):
            audio_f = "sss" + str(idx+1) + ".m4a"
            audio_path = os.path.join(concatenated_ss_audios_path, audio_f)
            track = AudioSegment.from_file(audio_path, "m4a", sr=16000)
            wav_f_path = audio_path.replace("m4a", "wav")
            wav_file_handle = track.export(wav_f_path, format='wav')
            y, sr = librosa.load(wav_f_path, sr=16000)
            assert sr == 16000
            sf.write(wav_f_path, y, 16000, 'pcm_24' )
            time_dur = track.duration_seconds

            #line = transcript_lines[idx]
            transcript = line.strip().replace("~|~", " ")
            concatenated_record = wav_f_path + "\t" + str(round(time_dur, 2)) + "\t" + transcript
            concatenated_records.append(concatenated_record)

#        for idx, audio in enumerate(all_concatenated_ss_audios):
#            if "m4a" in audio:
#                continue
        concatenated_records.insert(0, dataset_header)
        util.writeListFile("finetune-test_concatenated_ss_transcripts.tsv", concatenated_records)
        print("finetune-test_concatenated_ss_transcripts.tsv", len(concatenated_records))

def organize_transcribed_audio_text():

    true_text_path = "/home/liuyi/tflite_experimental/emsBert/eval_pretrain/fitted_desc_sampled100e2e_test.tsv"
    transcbribed_text_path = "/home/liuyi/TensorFlowASR/examples/conformer/pretrained_librispeech_train_ss_test_concatenated_h5_models_testout.txt"

    true_lines = util.readFile(true_text_path)
    pred_lines = util.readFile(transcribed_text_path)

    assert len(true_lines) == len(pred_lines)
    assert len(true_lines) == 101
    
    #transcribed_e2e_lines = []
    for idx, (true_line, pred_line) in enumerate(zip(true_lines, pred_lines)):
        if idx == 0:
            continue
   
        #transcribed_line = []

        true_text = true_line.split("\t")[0]
        true_label = true_line.split("\t")[1]
        true_text_in_pred = pred_line.split("\t")[2]
        assert true_text == true_text_in_pred

        pred_text = pred_line.split("\t")[3]
        #transcribed_line.append(pred_text, true_label)
        transcribed_e2e_lines.append(pred_text + "\t" + true_label)

    concatenated_records.insert(0, dataset_header)
    util.writeListFile("finetune-test_concatenated_ss_transcripts.tsv", concatenated_records)
    print("finetune-test_concatenated_ss_transcripts.tsv", len(concatenated_records))
 
def organize_simple_test():

    audio_path = "/home/liuyi/audio_data/radu0.wav"
    transcript = "dispatch priority one for chest pain unspecified arrived on scene to find a patient suffering from chest pain unspecified after the treatment patient complains of nausea other chest pain shortness of breath asthenia not otherwise specified and shortness of breath"

    track = AudioSegment.from_file(audio_path, "wav", sr=16000)
    time_dur = track.duration_seconds
    record = audio_path + "\t" + str(round(time_dur, 2)) + "\t" + transcript

    records = [record]
    util.writeListFile("simple-test_transcripts.tsv", records)

if __name__ == "__main__":

    organize_sample100()
#    quick_sort()
#    organize_simple_test()
