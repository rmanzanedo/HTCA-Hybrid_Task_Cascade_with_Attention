#create the dir
mkdir saved_model
mkdir outputs
#create graph
python model_inspect.py --runmode=saved_model --model_name=efficientdet-d7 --ckpt_path=efficientdet-d7 --hparams="image_size=1920x1280" --saved_model_dir=saved_model --batch_size=$1

#create features
python model_inspect.py --runmode=saved_model_infer --model_name=efficientdet-d7 --saved_model_dir=saved_model/efficientdet-d7_frozen.pb --input_image=seg --output_image_dir=outputs --batch_size=$1
