# models/models.py

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image


class CustomAlexNetFeatureExtractor(nn.Module):
    def __init__(self, model_path, num_classes=2):
        super(CustomAlexNetFeatureExtractor, self).__init__()
        self.alexnet = models.alexnet(pretrained=False)
        in_features = self.alexnet.classifier[6].in_features
        self.alexnet.classifier[6] = nn.Linear(in_features, num_classes)
        self.feature_extractor_part = self.alexnet.classifier[:6]
        self.load_weights(model_path)

    def forward(self, x, layer='penultimate'):
        x = self.alexnet.features(x)
        x = self.alexnet.avgpool(x)
        x = torch.flatten(x, 1)
        if layer == 'penultimate':
            x = self.feature_extractor_part(x)
            return x
        else:
            x = self.alexnet.classifier(x)
            return x

    def load_weights(self, model_path):
        'Load checkpoint weights into this feature extractor.'
        try:
            state_dict = torch.load(model_path)
            if all(k.startswith('alexnet.') for k in state_dict.keys()):
                new_state_dict = {k[8:]: v for k, v in state_dict.items()}
                self.alexnet.load_state_dict(new_state_dict, strict=False)
                print("AlexNet weights loaded successfully (stripped 'alexnet.' prefix).")
            else:
                self.alexnet.load_state_dict(state_dict, strict=False)
                print("AlexNet weights loaded successfully.")
        except Exception as e:
            print(f"Error loading AlexNet weights: {e}")
            raise RuntimeError(f"Failed to load AlexNet weights: {e}")


class CustomResNeXtFeatureExtractor(nn.Module):
    def __init__(self, model_path, num_classes=2):
        super(CustomResNeXtFeatureExtractor, self).__init__()
        self.resnext = models.resnext50_32x4d(pretrained=False)
        in_features = self.resnext.fc.in_features
        self.resnext.fc = nn.Linear(in_features, num_classes)
        self.load_weights(model_path)
        self.penultimate_layer = nn.Sequential(*list(self.resnext.children())[:-1])

    def forward(self, x, layer='penultimate'):
        if layer == 'penultimate':
            x = self.penultimate_layer(x)
            return torch.flatten(x, 1)
        else:
            raise ValueError(f"Unsupported layer: {layer}")

    def load_weights(self, model_path):
        try:
            state_dict = torch.load(model_path, map_location='cpu')
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith('resnext.'):
                    new_state_dict[k[8:]] = v
                else:
                    new_state_dict[k] = v
            self.resnext.load_state_dict(new_state_dict, strict=False)
            print("ResNeXt weights loaded successfully.")
        except Exception as e:
            print(f"Error loading ResNeXt weights: {e}")
            raise RuntimeError(f"Failed to load ResNeXt weights: {e}")


class CustomConvNextLargeFeatureExtractor(nn.Module):
    def __init__(self, model_path, num_classes=2):
        super(CustomConvNextLargeFeatureExtractor, self).__init__()
        self.convnext = models.convnext_large(pretrained=False)
        in_features = self.convnext.classifier[2].in_features
        self.convnext.classifier[2] = nn.Linear(in_features, num_classes)
        self.load_weights(model_path)
        self.classifier_head = self.convnext.classifier[2]
        self.convnext.classifier[2] = nn.Identity()

    def forward(self, x, layer='penultimate'):
        features = self.convnext(x)
        if layer == 'penultimate':
            return features
        else:
            return self.classifier_head(features)

    def load_weights(self, model_path):
        try:
            state_dict = torch.load(model_path, map_location='cpu')
            if all(k.startswith('convnext.') for k in state_dict.keys()):
                new_state_dict = {k[9:]: v for k, v in state_dict.items()}
                self.convnext.load_state_dict(new_state_dict, strict=False)
            else:
                self.convnext.load_state_dict(state_dict, strict=False)
            print("ConvNeXt weights loaded successfully.")
        except Exception as e:
            print(f"Error loading ConvNeXt weights: {e}")
            raise RuntimeError(f"Failed to load ConvNeXt weights: {e}")


class CustomGoogLeNetFeatureExtractor(nn.Module):
    def __init__(self, model_path, num_classes=2):
        super(CustomGoogLeNetFeatureExtractor, self).__init__()
        self.googlenet = models.googlenet(pretrained=False, aux_logits=False)
        in_features = self.googlenet.fc.in_features
        self.googlenet.fc = nn.Linear(in_features, num_classes)
        self.load_weights(model_path)
        self.classifier_head = self.googlenet.fc
        self.googlenet.fc = nn.Identity()

    def forward(self, x, layer='penultimate'):
        features = self.googlenet(x)
        if layer == 'penultimate':
            return features
        else:
            return self.classifier_head(features)

    def load_weights(self, model_path):
        try:
            state_dict = torch.load(model_path, map_location='cpu')
            if all(k.startswith('googlenet.') for k in state_dict.keys()):
                new_state_dict = {k[10:]: v for k, v in state_dict.items()}
                self.googlenet.load_state_dict(new_state_dict, strict=False)
                print("GoogLeNet weights loaded successfully (stripped 'googlenet.' prefix).")
            else:
                self.googlenet.load_state_dict(state_dict, strict=False)
                print("GoogLeNet weights loaded successfully.")
        except Exception as e:
            print(f"Error loading GoogLeNet weights: {e}")
            raise RuntimeError(f"Failed to load GoogLeNet weights: {e}")


class CustomResNet50FeatureExtractor(nn.Module):
    def __init__(self, model_path, num_classes=2):
        super(CustomResNet50FeatureExtractor, self).__init__()
        self.resnet = models.resnet50(pretrained=False)
        in_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(in_features, num_classes)
        self.features = nn.Sequential(*list(self.resnet.children())[:-1])
        self.load_weights(model_path)

    def forward(self, x, layer='penultimate'):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return x

    def load_weights(self, model_path):
        'Load checkpoint weights into this feature extractor.'
        try:
            state_dict = torch.load(model_path)
            if all(k.startswith('resnet.') for k in state_dict.keys()):
                new_state_dict = {k[7:]: v for k, v in state_dict.items()}
                self.resnet.load_state_dict(new_state_dict, strict=False)
                print("ResNet50 weights loaded successfully (stripped 'resnet.' prefix).")
            else:
                self.resnet.load_state_dict(state_dict, strict=False)
                print("ResNet50 weights loaded successfully.")
        except Exception as e:
            print(f"Error loading ResNet50 weights: {e}")
            raise RuntimeError(f"Failed to load ResNet50 weights: {e}")


class CustomResNet101FeatureExtractor(nn.Module):
    def __init__(self, model_path, num_classes=2):
        super(CustomResNet101FeatureExtractor, self).__init__()
        self.resnet = models.resnet101(pretrained=False)
        in_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(in_features, num_classes)
        self.features = nn.Sequential(*list(self.resnet.children())[:-1])
        self.load_weights(model_path)

    def forward(self, x, layer='penultimate'):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return x

    def load_weights(self, model_path):
        'Load checkpoint weights into this feature extractor.'
        try:
            state_dict = torch.load(model_path)
            if all(k.startswith('resnet.') for k in state_dict.keys()):
                new_state_dict = {k[7:]: v for k, v in state_dict.items()}
                self.resnet.load_state_dict(new_state_dict, strict=False)
                print("ResNet101 weights loaded successfully (stripped 'resnet.' prefix).")
            else:
                self.resnet.load_state_dict(state_dict, strict=False)
                print("ResNet101 weights loaded successfully.")
        except Exception as e:
            print(f"Error loading ResNet101 weights: {e}")
            raise RuntimeError(f"Failed to load ResNet101 weights: {e}")


class CustomDenseNet121FeatureExtractor(nn.Module):
    def __init__(self, model_path, num_classes=2):
        super(CustomDenseNet121FeatureExtractor, self).__init__()
        self.densenet = models.densenet121(pretrained=False)
        in_features = self.densenet.classifier.in_features
        self.densenet.classifier = nn.Linear(in_features, num_classes)
        self.load_weights(model_path)
        self.classifier_head = self.densenet.classifier
        self.densenet.classifier = nn.Identity()

    def forward(self, x, layer='penultimate'):
        features = self.densenet(x)
        if layer == 'penultimate':
            return features
        else:
            return self.classifier_head(features)

    def load_weights(self, model_path):
        try:
            state_dict = torch.load(model_path, map_location='cpu')
            if all(k.startswith('densenet.') for k in state_dict.keys()):
                new_state_dict = {k[9:]: v for k, v in state_dict.items()}
                self.densenet.load_state_dict(new_state_dict, strict=False)
            else:
                self.densenet.load_state_dict(state_dict, strict=False)
            print("DenseNet121 weights loaded successfully.")
        except Exception as e:
            print(f"Error loading DenseNet121 weights: {e}")
            raise RuntimeError(f"Failed to load DenseNet121 weights: {e}")


class CustomDenseNet161FeatureExtractor(nn.Module):
    def __init__(self, model_path, num_classes=2):
        super(CustomDenseNet161FeatureExtractor, self).__init__()
        self.densenet = models.densenet161(pretrained=False)
        in_features = self.densenet.classifier.in_features
        self.densenet.classifier = nn.Linear(in_features, num_classes)
        self.load_weights(model_path)
        self.classifier_head = self.densenet.classifier
        self.densenet.classifier = nn.Identity()

    def forward(self, x, layer='penultimate'):
        features = self.densenet(x)
        if layer == 'penultimate':
            return features
        else:
            return self.classifier_head(features)

    def load_weights(self, model_path):
        try:
            state_dict = torch.load(model_path, map_location='cpu')
            if all(k.startswith('densenet.') for k in state_dict.keys()):
                new_state_dict = {k[9:]: v for k, v in state_dict.items()}
                self.densenet.load_state_dict(new_state_dict, strict=False)
            else:
                self.densenet.load_state_dict(state_dict, strict=False)
            print("DenseNet161 weights loaded successfully.")
        except Exception as e:
            print(f"Error loading DenseNet161 weights: {e}")
            raise RuntimeError(f"Failed to load DenseNet161 weights: {e}")


class CustomVGG16FeatureExtractor(nn.Module):
    def __init__(self, model_path, num_classes=2):
        super(CustomVGG16FeatureExtractor, self).__init__()
        self.vgg = models.vgg16(pretrained=False)
        in_features = self.vgg.classifier[6].in_features
        self.vgg.classifier[6] = nn.Linear(in_features, num_classes)
        self.load_weights(model_path)
        self.classifier_head = self.vgg.classifier[6]
        self.vgg.classifier = nn.Sequential(*list(self.vgg.classifier.children())[:6])

    def forward(self, x, layer='penultimate'):
        features = self.vgg(x)
        if layer == 'penultimate':
            return features
        else:
            return self.classifier_head(features)

    def load_weights(self, model_path):
        try:
            state_dict = torch.load(model_path, map_location='cpu')
            if all(k.startswith('vgg.') for k in state_dict.keys()):
                new_state_dict = {k[4:]: v for k, v in state_dict.items()}
                self.vgg.load_state_dict(new_state_dict, strict=False)
            else:
                self.vgg.load_state_dict(state_dict, strict=False)
            print("VGG16 weights loaded successfully.")
        except Exception as e:
            print(f"Error loading VGG16 weights: {e}")
            raise RuntimeError(f"Failed to load VGG16 weights: {e}")


class CustomInceptionV3FeatureExtractor(nn.Module):
    def __init__(self, model_path, num_classes=2):
        super(CustomInceptionV3FeatureExtractor, self).__init__()
        self.inception = models.inception_v3(pretrained=False, aux_logits=False)
        in_features = self.inception.fc.in_features
        self.inception.fc = nn.Linear(in_features, num_classes)
        self.load_weights(model_path)
        self.classifier_head = self.inception.fc
        self.inception.fc = nn.Identity()

    def forward(self, x, layer='penultimate'):
        features = self.inception(x)
        if layer == 'penultimate':
            return features
        else:
            return self.classifier_head(features)

    def load_weights(self, model_path):
        try:
            state_dict = torch.load(model_path, map_location='cpu')
            if all(k.startswith('inception.') for k in state_dict.keys()):
                new_state_dict = {k[10:]: v for k, v in state_dict.items()}
                self.inception.load_state_dict(new_state_dict, strict=False)
            else:
                self.inception.load_state_dict(state_dict, strict=False)
            print("InceptionV3 weights loaded successfully.")
        except Exception as e:
            print(f"Error loading InceptionV3 weights: {e}")
            raise RuntimeError(f"Failed to load InceptionV3 weights: {e}")


class CustomEfficientNetB0FeatureExtractor(nn.Module):
    def __init__(self, model_path, num_classes=2):
        super(CustomEfficientNetB0FeatureExtractor, self).__init__()
        self.efficientnet = models.efficientnet_b0(pretrained=False)
        in_features = self.efficientnet.classifier[1].in_features
        self.efficientnet.classifier[1] = nn.Linear(in_features, num_classes)
        self.features = self.efficientnet.features
        self.avgpool = self.efficientnet.avgpool
        self.load_weights(model_path)

    def forward(self, x, layer='penultimate'):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x

    def load_weights(self, model_path):
        'Load checkpoint weights into this feature extractor.'
        try:
            state_dict = torch.load(model_path)
            if all(k.startswith('efficientnet.') for k in state_dict.keys()):
                new_state_dict = {k[13:]: v for k, v in state_dict.items()}
                self.efficientnet.load_state_dict(new_state_dict, strict=False)
                print("EfficientNetB0 weights loaded successfully (stripped 'efficientnet.' prefix).")
            else:
                self.efficientnet.load_state_dict(state_dict, strict=False)
                print("EfficientNetB0 weights loaded successfully.")
        except Exception as e:
            print(f"Error loading EfficientNetB0 weights: {e}")
            raise RuntimeError(f"Failed to load EfficientNetB0 weights: {e}")


class CustomEfficientNetV2LargeFeatureExtractor(nn.Module):
    def __init__(self, model_path, num_classes=2):
        super(CustomEfficientNetV2LargeFeatureExtractor, self).__init__()
        self.efficientnet = models.efficientnet_v2_l(pretrained=False)
        in_features = self.efficientnet.classifier[1].in_features
        self.efficientnet.classifier[1] = nn.Linear(in_features, num_classes)
        self.features = self.efficientnet.features
        self.avgpool = self.efficientnet.avgpool
        self.load_weights(model_path)

    def forward(self, x, layer='penultimate'):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x

    def load_weights(self, model_path):
        'Load checkpoint weights into this feature extractor.'
        try:
            state_dict = torch.load(model_path)
            if all(k.startswith('efficientnet.') for k in state_dict.keys()):
                new_state_dict = {k[13:]: v for k, v in state_dict.items()}
                self.efficientnet.load_state_dict(new_state_dict, strict=False)
                print("EfficientNetV2L weights loaded successfully (stripped 'efficientnet.' prefix).")
            else:
                self.efficientnet.load_state_dict(state_dict, strict=False)
                print("EfficientNetV2L weights loaded successfully.")
        except Exception as e:
            print(f"Error loading EfficientNetV2L weights: {e}")
            raise RuntimeError(f"Failed to load EfficientNetV2L weights: {e}")


class CustomMobileNetV2FeatureExtractor(nn.Module):
    def __init__(self, model_path, num_classes=2):
        super(CustomMobileNetV2FeatureExtractor, self).__init__()
        self.mobilenet = models.mobilenet_v2(pretrained=False)
        in_features = self.mobilenet.classifier[1].in_features
        self.mobilenet.classifier[1] = nn.Linear(in_features, num_classes)
        self.load_weights(model_path)
        self.classifier_head = self.mobilenet.classifier
        self.mobilenet.classifier = nn.Identity()

    def forward(self, x, layer='penultimate'):
        features = self.mobilenet(x)
        if layer == 'penultimate':
            return features
        else:
            return self.classifier_head(features)

    def load_weights(self, model_path):
        try:
            state_dict = torch.load(model_path, map_location='cpu')
            if all(k.startswith('mobilenet.') for k in state_dict.keys()):
                new_state_dict = {k[10:]: v for k, v in state_dict.items()}
                self.mobilenet.load_state_dict(new_state_dict, strict=False)
            else:
                self.mobilenet.load_state_dict(state_dict, strict=False)
            print("MobileNetV2 weights loaded successfully.")
        except Exception as e:
            print(f"Error loading MobileNetV2 weights: {e}")
            raise RuntimeError(f"Failed to load MobileNetV2 weights: {e}")


class CustomMobileNetV3FeatureExtractor(nn.Module):
    def __init__(self, model_path, num_classes=2):
        super(CustomMobileNetV3FeatureExtractor, self).__init__()
        self.mobilenet = models.mobilenet_v3_large(pretrained=False)

        original_classifier = self.mobilenet.classifier
        if isinstance(original_classifier, nn.Sequential):
            layers = list(original_classifier.children())
            if isinstance(layers[-1], nn.Linear):
                in_features = layers[-1].in_features
                new_layers = layers[:-1] + [nn.Linear(in_features, num_classes)]
                self.mobilenet.classifier = nn.Sequential(*new_layers)
            else:
                self.mobilenet.classifier = nn.Sequential(
                    nn.Linear(1280, 1024),
                    nn.Hardswish(inplace=True),
                    nn.Dropout(p=0.2, inplace=True),
                    nn.Linear(1024, num_classes),
                )
        else:
            self.mobilenet.classifier = nn.Sequential(
                nn.Linear(1280, 1024),
                nn.Hardswish(inplace=True),
                nn.Dropout(p=0.2, inplace=True),
                nn.Linear(1024, num_classes),
            )

        self.features = self.mobilenet.features
        self.load_weights(model_path)

    def forward(self, x, layer='penultimate'):
        x = self.features(x)
        x = torch.nn.functional.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        return x

    def load_weights(self, model_path):
        'Load checkpoint weights into this feature extractor.'
        try:
            state_dict = torch.load(model_path)
            if all(k.startswith('mobilenet.') for k in state_dict.keys()):
                new_state_dict = {k[10:]: v for k, v in state_dict.items()}
                self.mobilenet.load_state_dict(new_state_dict, strict=False)
                print("MobileNetV3 weights loaded successfully (stripped 'mobilenet.' prefix).")
            else:
                self.mobilenet.load_state_dict(state_dict, strict=False)
                print("MobileNetV3 weights loaded successfully.")
        except Exception as e:
            print(f"Error loading MobileNetV3 weights: {e}")
            raise RuntimeError(f"Failed to load MobileNetV3 weights: {e}")


class CustomRegNetFeatureExtractor(nn.Module):
    def __init__(self, model_path, num_classes=2):
        super(CustomRegNetFeatureExtractor, self).__init__()
        self.regnet = models.regnet_y_400mf(pretrained=False)
        in_features = self.regnet.fc.in_features
        self.regnet.fc = nn.Linear(in_features, num_classes)
        self.load_weights(model_path)
        self.classifier_head = self.regnet.fc
        self.regnet.fc = nn.Identity()

    def forward(self, x, layer='penultimate'):
        features = self.regnet(x)
        if layer == 'penultimate':
            return features
        else:
            return self.classifier_head(features)

    def load_weights(self, model_path):
        try:
            state_dict = torch.load(model_path, map_location='cpu')
            if all(k.startswith('regnet.') for k in state_dict.keys()):
                new_state_dict = {k[7:]: v for k, v in state_dict.items()}
                self.regnet.load_state_dict(new_state_dict, strict=False)
            else:
                self.regnet.load_state_dict(state_dict, strict=False)
            print("RegNet weights loaded successfully.")
        except Exception as e:
            print(f"Error loading RegNet weights: {e}")
            raise RuntimeError(f"Failed to load RegNet weights: {e}")


class CustomShuffleNetV2FeatureExtractor(nn.Module):
    def __init__(self, model_path, num_classes=2):
        super(CustomShuffleNetV2FeatureExtractor, self).__init__()
        self.shufflenet = models.shufflenet_v2_x1_0(pretrained=False)
        in_features = self.shufflenet.fc.in_features
        self.shufflenet.fc = nn.Linear(in_features, num_classes)
        self.load_weights(model_path)
        self.classifier_head = self.shufflenet.fc
        self.shufflenet.fc = nn.Identity()

    def forward(self, x, layer='penultimate'):
        features = self.shufflenet(x)
        if layer == 'penultimate':
            return features
        else:
            return self.classifier_head(features)

    def load_weights(self, model_path):
        try:
            state_dict = torch.load(model_path, map_location='cpu')
            if all(k.startswith('shufflenet.') for k in state_dict.keys()):
                new_state_dict = {k[11:]: v for k, v in state_dict.items()}
                self.shufflenet.load_state_dict(new_state_dict, strict=False)
            else:
                self.shufflenet.load_state_dict(state_dict, strict=False)
            print("ShuffleNetV2 weights loaded successfully.")
        except Exception as e:
            print(f"Error loading ShuffleNetV2 weights: {e}")
            raise RuntimeError(f"Failed to load ShuffleNetV2 weights: {e}")


class CustomSqueezeNetFeatureExtractor(nn.Module):
    def __init__(self, model_path, num_classes=2):
        super(CustomSqueezeNetFeatureExtractor, self).__init__()

        state_dict, version = self._detect_version_and_load_dict(model_path)

        if version == '1.1':
            print("Detected SqueezeNet 1.1 weights, initialising accordingly.")
            self.squeezenet = models.squeezenet1_1(weights=None)
        else:
            print("Detected SqueezeNet 1.0 weights, initialising accordingly.")
            self.squeezenet = models.squeezenet1_0(weights=None)

        self.squeezenet.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=1)
        self.load_weights(model_path)
        self.features_block = self.squeezenet.features
        self.classifier_head = self.squeezenet.classifier

    def _detect_version_and_load_dict(self, model_path):
        'Load a SqueezeNet checkpoint and identify its architecture version.'
        try:
            state_dict = torch.load(model_path, map_location='cpu')
            if all(k.startswith('squeezenet.') for k in state_dict.keys()):
                state_dict = {k[11:]: v for k, v in state_dict.items()}
            # Kernel size of the first conv layer distinguishes v1.1 (3) from v1.0 (other)
            first_layer_shape = state_dict['features.0.weight'].shape
            version = '1.1' if first_layer_shape[2] == 3 else '1.0'
            return state_dict, version
        except Exception as e:
            raise RuntimeError(f"Failed to parse checkpoint; check path or file format: {e}")

    def forward(self, x, layer='penultimate'):
        x = self.features_block(x)
        x = torch.nn.functional.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        if layer == 'penultimate':
            return x
        else:
            return self.classifier_head(x.unsqueeze(2).unsqueeze(3)).flatten(1)

    def load_weights(self, model_path):
        'Load checkpoint weights into this feature extractor.'
        try:
            state_dict = torch.load(model_path, map_location='cpu')
            if all(k.startswith('squeezenet.') for k in state_dict.keys()):
                new_state_dict = {k[11:]: v for k, v in state_dict.items()}
            else:
                new_state_dict = state_dict
            self.squeezenet.load_state_dict(new_state_dict, strict=False)
            print("SqueezeNet weights loaded successfully.")
        except Exception as e:
            print(f"Error loading SqueezeNet weights: {e}")
            raise RuntimeError(f"Failed to load SqueezeNet weights: {e}")


class CustomViT_B16FeatureExtractor(nn.Module):
    def __init__(self, model_path, num_classes=2):
        super(CustomViT_B16FeatureExtractor, self).__init__()
        self.vit = models.vit_b_16(pretrained=False)
        in_features = self.vit.heads.head.in_features
        self.vit.heads.head = nn.Linear(in_features, num_classes)
        self.load_weights(model_path)
        self.classifier_head = self.vit.heads
        self.vit.heads = nn.Identity()

    def forward(self, x, layer='penultimate'):
        features = self.vit(x)
        if layer == 'penultimate':
            return features
        else:
            return self.classifier_head(features)

    def load_weights(self, model_path):
        try:
            state_dict = torch.load(model_path, map_location='cpu')
            if all(k.startswith('vit.') for k in state_dict.keys()):
                new_state_dict = {k[4:]: v for k, v in state_dict.items()}
                self.vit.load_state_dict(new_state_dict, strict=False)
            else:
                self.vit.load_state_dict(state_dict, strict=False)
            print("ViT-B16 weights loaded successfully.")
        except Exception as e:
            print(f"Error loading ViT-B16 weights: {e}")
            raise RuntimeError(f"Failed to load ViT-B16 weights: {e}")


class CustomViT_L16FeatureExtractor(nn.Module):
    def __init__(self, model_path, num_classes=2):
        super(CustomViT_L16FeatureExtractor, self).__init__()
        self.vit = models.vit_l_16(pretrained=False)
        in_features = self.vit.heads.head.in_features
        self.vit.heads.head = nn.Linear(in_features, num_classes)
        self.load_weights(model_path)

    def forward(self, x, layer=None):
        'Return the CLS token feature from the last encoder layer.'
        class_token_feature = None

        def hook_fn(module, input, output):
            nonlocal class_token_feature
            class_token_feature = output[:, 0]

        hook = self.vit.encoder.layers[-1].register_forward_hook(hook_fn)
        _ = self.vit(x)
        hook.remove()
        return class_token_feature

    def load_weights(self, model_path):
        'Load checkpoint weights into this feature extractor.'
        try:
            state_dict = torch.load(model_path)
            if all(k.startswith('vit.') for k in state_dict.keys()):
                new_state_dict = {k[4:]: v for k, v in state_dict.items()}
                self.vit.load_state_dict(new_state_dict)
                print("ViT-L16 weights loaded successfully (stripped 'vit.' prefix).")
            else:
                self.vit.load_state_dict(state_dict)
                print("ViT-L16 weights loaded successfully.")
        except Exception as e:
            print(f"Error loading ViT-L16 weights: {e}")
            raise RuntimeError(f"Failed to load ViT-L16 weights: {e}")


def pad_to_square_transform(img):
    try:
        width, height = img.size
        if width != height:
            max_side = max(width, height)
            new_image = Image.new('RGB', (max_side, max_side), (255, 255, 255))
            new_image.paste(img, ((max_side - width) // 2, (max_side - height) // 2))
            return new_image.resize((224, 224), Image.LANCZOS)
        else:
            return img.resize((224, 224), Image.LANCZOS)
    except Exception as e:
        print(f"Error in pad_to_square_transform: {e}")
        return Image.new('RGB', (224, 224), (255, 255, 255))


def pad_to_square_299_transform(img):
    try:
        width, height = img.size
        max_side = max(width, height)
        new_image = Image.new('RGB', (max_side, max_side), (255, 255, 255))
        new_image.paste(img, ((max_side - width) // 2, (max_side - height) // 2))
        return new_image.resize((299, 299), Image.LANCZOS)
    except Exception as e:
        print(f"Error in pad_to_square_299_transform: {e}")
        return Image.new('RGB', (299, 299), (255, 255, 255))


SUPPORTED_MODELS = {
    'AlexNet':             {'class': CustomAlexNetFeatureExtractor,            'input_size': 224, 'pad_func': pad_to_square_transform},
    'ResNeXt':             {'class': CustomResNeXtFeatureExtractor,            'input_size': 224, 'pad_func': pad_to_square_transform},
    'ConvNextLarge':       {'class': CustomConvNextLargeFeatureExtractor,      'input_size': 224, 'pad_func': pad_to_square_transform},
    'GoogLeNet':           {'class': CustomGoogLeNetFeatureExtractor,          'input_size': 224, 'pad_func': pad_to_square_transform},
    'ResNet50':            {'class': CustomResNet50FeatureExtractor,           'input_size': 224, 'pad_func': pad_to_square_transform},
    'ResNet101':           {'class': CustomResNet101FeatureExtractor,          'input_size': 224, 'pad_func': pad_to_square_transform},
    'DenseNet121':         {'class': CustomDenseNet121FeatureExtractor,        'input_size': 224, 'pad_func': pad_to_square_transform},
    'DenseNet161':         {'class': CustomDenseNet161FeatureExtractor,        'input_size': 224, 'pad_func': pad_to_square_transform},
    'VGG16':               {'class': CustomVGG16FeatureExtractor,              'input_size': 224, 'pad_func': pad_to_square_transform},
    'InceptionV3':         {'class': CustomInceptionV3FeatureExtractor,        'input_size': 299, 'pad_func': pad_to_square_299_transform},
    'EfficientNetB0':      {'class': CustomEfficientNetB0FeatureExtractor,     'input_size': 224, 'pad_func': pad_to_square_transform},
    'EfficientNetV2Large': {'class': CustomEfficientNetV2LargeFeatureExtractor,'input_size': 224, 'pad_func': pad_to_square_transform},
    'MobileNetV2':         {'class': CustomMobileNetV2FeatureExtractor,        'input_size': 224, 'pad_func': pad_to_square_transform},
    'MobileNetV3':         {'class': CustomMobileNetV3FeatureExtractor,        'input_size': 224, 'pad_func': pad_to_square_transform},
    'RegNet':              {'class': CustomRegNetFeatureExtractor,             'input_size': 224, 'pad_func': pad_to_square_transform},
    'ShuffleNetV2':        {'class': CustomShuffleNetV2FeatureExtractor,       'input_size': 224, 'pad_func': pad_to_square_transform},
    'SqueezeNet':          {'class': CustomSqueezeNetFeatureExtractor,         'input_size': 224, 'pad_func': pad_to_square_transform},
    'ViT_B16':             {'class': CustomViT_B16FeatureExtractor,            'input_size': 224, 'pad_func': pad_to_square_transform},
    'ViT_L16':             {'class': CustomViT_L16FeatureExtractor,            'input_size': 224, 'pad_func': pad_to_square_transform},
}


def get_model_and_transform(model_type, device, model_path=None, mean=None, std=None):
    'Build a feature-extraction model and its preprocessing transform.'
    if model_path is None or (mean is None and std is None):
        mean = [0.485, 0.456, 0.406]
        std  = [0.229, 0.224, 0.225]

    if model_type not in SUPPORTED_MODELS:
        raise ValueError(f"Unsupported model type: {model_type}. Supported types: {list(SUPPORTED_MODELS.keys())}")

    model_info  = SUPPORTED_MODELS[model_type]
    model_class = model_info['class']
    input_size  = model_info['input_size']
    pad_func    = model_info['pad_func']

    model = model_class(model_path=model_path).to(device)

    transform = transforms.Compose([
        transforms.Lambda(pad_func),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    return model, transform, input_size