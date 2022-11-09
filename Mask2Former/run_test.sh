#python train_net.py --config-file configs/coco/instance-segmentation/effi2former.yaml --eval-only MODEL.WEIGHTS ./output_d3_taiwan/model_final.pth
#python train_net.py --config-file configs/coco/instance-segmentation/effi2former.yaml --saved_model_dir=../efficientdet/saved_model_d3 --model_name=efficientdet-d3 --eval-only MODEL.WEIGHTS ./output_d3/model_0004999.pth
#python train_net.py --config-file configs/coco/instance-segmentation/effi2former.yaml --saved_model_dir=../efficientdet/saved_model_d3 --model_name=efficientdet-d3 --eval-only MODEL.WEIGHTS ./output_d3/old/model_0004999.pth
python3 train_net.py --config-file configs/coco/instance-segmentation/effi2former.yaml --saved_model_dir=../efficientdet/saved_model_d3 --model_name=efficientdet-d3 --eval-only MODEL.WEIGHTS ./output_d3_taiwan/test3/model_final.pth
#python train_net.py --config-file configs/coco/instance-segmentation/effi2former.yaml --eval-only MODEL.WEIGHTS ./output_d7/model_0089999.pth
#python train_net.py --config-file configs/coco/instance-segmentation/effi2former.yaml --eval-only MODEL.WEIGHTS ./output_d7/model_0094999.pth


#Effiformer
#python3 train_net.py --config-file configs/coco/instance-segmentation/effiformer.yaml --saved_model_dir=../efficientdet/saved_model_d3 --model_name=efficientdet-d3 --eval-only MODEL.WEIGHTS ./output_d3_effiformer/model_0209999.pth