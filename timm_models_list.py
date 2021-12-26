import timm
import os
#print(os.path.dirname(timm.__file__))
print(timm.list_models(pretrained=True))
#m = timm.create_model('mobilenetv3_large_100', pretrained=True)
#m.eval()

#print(m)