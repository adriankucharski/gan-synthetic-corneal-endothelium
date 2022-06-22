import os 

list_of_commands = [
    # r'python .\evaluate.py Alizarine 0 segmentation\models\raw\20220602-1413 7',
    # r'python .\evaluate.py Alizarine 1 segmentation\models\raw\20220603-2030 7',
    # r'python .\evaluate.py Alizarine 2 segmentation\models\raw\20220610-0021 7',
    # r'python .\evaluate.py Alizarine 0 segmentation\models\synthetic\20220603-0015 7',
    # r'python .\evaluate.py Alizarine 1 segmentation\models\synthetic\20220610-0115 7',
    # r'python .\evaluate.py Alizarine 2 segmentation\models\synthetic\20220610-0123 7',
    # r'python .\evaluate.py Rotterdam_1000 0 segmentation\models\raw\20220602-1258 15',
    # r'python .\evaluate.py Rotterdam_1000 1 segmentation\models\raw\20220610-0051 15',
    # r'python .\evaluate.py Rotterdam_1000 2 segmentation\models\raw\20220610-0102 15',
    # r'python .\evaluate.py Rotterdam_1000 0 segmentation\models\synthetic\20220610-0135 15',
    # r'python .\evaluate.py Rotterdam_1000 1 segmentation\models\synthetic\20220610-0145 15',
    # r'python .\evaluate.py Rotterdam_1000 2 segmentation\models\synthetic\20220610-0154 15',

    r'python .\evaluate.py Gavet 0 segmentation\models\raw\20220603-0826 9',
    r'python .\evaluate.py Gavet 1 segmentation\models\raw\20220610-0031 9',
    r'python .\evaluate.py Gavet 2 segmentation\models\raw\20220610-0038 9',
    
    r'python .\evaluate.py Gavet 0 segmentation\models\synthetic\20220610-1033 9',
    r'python .\evaluate.py Gavet 1 segmentation\models\synthetic\20220610-1110 9',
    r'python .\evaluate.py Gavet 2 segmentation\models\synthetic\20220610-1309 9',
]

for command in list_of_commands:
    os.system(command)

# os.system("shutdown /s /t 60")
print('Exit')
    