
echo "Need to pass mel in the shape of (features,mel-bins)"

file_path=/Users/samarasimhareddygujjula/Desktop/GameChanger/predicted_mels/test/bengali_male/train_bengalimale_03255/test_speaker2_train_bengalimale_03255.pt

python /Users/samarasimhareddygujjula/Desktop/GameChanger/Vococers/waveglow/infer_waveglow.py $file_path