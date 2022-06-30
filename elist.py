import os 
# Evaluate
# list_of_commands = [
#     r'python .\evaluate.py Rotterdam_1000 0 segmentation\models\raw\20220602-1258\model-20.hdf5 15',
#     r'python .\evaluate.py Rotterdam_1000 0 segmentation\models\synthetic\20220625-0058\model-11.hdf5 15',
#     r'python .\evaluate.py Rotterdam_1000 1 segmentation\models\raw\20220610-0051\model-25.hdf5 15',
#     r'python .\evaluate.py Rotterdam_1000 1 segmentation\models\synthetic\20220625-0105\model-25.hdf5 15',
#     r'python .\evaluate.py Rotterdam_1000 2 segmentation\models\raw\20220610-0102\model-13.hdf5 15',
#     r'python .\evaluate.py Rotterdam_1000 2 segmentation\models\synthetic\20220625-0111\model-12.hdf5 15',
    
    
#     r'python .\evaluate.py Alizarine 0 segmentation\models\raw\20220602-1413\model-15.hdf5 7',
#     r'python .\evaluate.py Alizarine 0 segmentation\models\synthetic\20220628-0026\model-25.hdf5 7',
#     r'python .\evaluate.py Alizarine 1 segmentation\models\raw\20220628-1221\model-11.hdf5 7',
#     r'python .\evaluate.py Alizarine 1 segmentation\models\synthetic\20220628-1037\model-11.hdf5 7',
#     r'python .\evaluate.py Alizarine 2 segmentation\models\raw\20220610-0021\model-15.hdf5 7',
#     r'python .\evaluate.py Alizarine 2 segmentation\models\synthetic\20220628-1043\model-16.hdf5 7',
    
    
#     r'python .\evaluate.py Gavet 0 segmentation\models\raw\20220603-0826\model-23.hdf5 7',
#     r'python .\evaluate.py Gavet 0 segmentation\models\synthetic\20220625-0118\model-23.hdf5 7',
#     r'python .\evaluate.py Gavet 1 segmentation\models\raw\20220610-0031\model-25.hdf5 7',
#     r'python .\evaluate.py Gavet 1 segmentation\models\synthetic\20220625-0124\model-19.hdf5 7',
#     r'python .\evaluate.py Gavet 2 segmentation\models\raw\20220610-0038\model-19.hdf5 7',
#     r'python .\evaluate.py Gavet 2 segmentation\models\synthetic\20220625-0131\model-25.hdf5 7',
# ]

# Predict
list_of_commands = [
    r'python .\predict.py Rotterdam_1000 0 segmentation\models\raw\20220602-1258\model-20.hdf5 16',
    r'python .\predict.py Rotterdam_1000 0 segmentation\models\synthetic\20220625-0058\model-11.hdf5 16',
    r'python .\predict.py Rotterdam_1000 1 segmentation\models\raw\20220610-0051\model-25.hdf5 16',
    r'python .\predict.py Rotterdam_1000 1 segmentation\models\synthetic\20220625-0105\model-25.hdf5 16',
    r'python .\predict.py Rotterdam_1000 2 segmentation\models\raw\20220610-0102\model-13.hdf5 16',
    r'python .\predict.py Rotterdam_1000 2 segmentation\models\synthetic\20220625-0111\model-12.hdf5 16',
    
    
    r'python .\predict.py Alizarine 0 segmentation\models\raw\20220602-1413\model-15.hdf5 16',
    r'python .\predict.py Alizarine 0 segmentation\models\synthetic\20220628-0026\model-25.hdf5 16',
    r'python .\predict.py Alizarine 1 segmentation\models\raw\20220628-1221\model-11.hdf5 16',
    r'python .\predict.py Alizarine 1 segmentation\models\synthetic\20220628-1037\model-11.hdf5 16',
    r'python .\predict.py Alizarine 2 segmentation\models\raw\20220610-0021\model-15.hdf5 16',
    r'python .\predict.py Alizarine 2 segmentation\models\synthetic\20220628-1043\model-16.hdf5 16',
    
    
    r'python .\predict.py Gavet 0 segmentation\models\raw\20220603-0826\model-23.hdf5 16',
    r'python .\predict.py Gavet 0 segmentation\models\synthetic\20220625-0118\model-23.hdf5 16',
    r'python .\predict.py Gavet 1 segmentation\models\raw\20220610-0031\model-25.hdf5 16',
    r'python .\predict.py Gavet 1 segmentation\models\synthetic\20220625-0124\model-19.hdf5 16',
    r'python .\predict.py Gavet 2 segmentation\models\raw\20220610-0038\model-19.hdf5 16',
    r'python .\predict.py Gavet 2 segmentation\models\synthetic\20220625-0131\model-25.hdf5 16',
]

for command in list_of_commands:
    os.system(command)

# os.system("shutdown /s /t 60")
print('Exit')
