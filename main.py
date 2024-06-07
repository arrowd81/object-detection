import os
import xml.etree.ElementTree as ET

import torch
import torch.optim as optim
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models.detection import ssdlite320_mobilenet_v3_large, SSDLite320_MobileNet_V3_Large_Weights


# Function to parse PASCAL VOC annotations
def parse_voc_annotation(ann_dir, img_dir, labels=[]):
    all_imgs = []
    seen_labels = {}

    for ann in sorted(os.listdir(ann_dir)):
        img = {'object': []}
        tree = ET.parse(os.path.join(ann_dir, ann))
        for elem in tree.iter():
            if 'filename' in elem.tag:
                img['filename'] = os.path.join(img_dir, elem.text)
            if 'width' in elem.tag:
                img['width'] = int(elem.text)
            if 'height' in elem.tag:
                img['height'] = int(elem.text)
            if 'object' in elem.tag or 'part' in elem.tag:
                obj = {}
                for attr in list(elem):
                    if 'name' in attr.tag:
                        obj['name'] = attr.text
                        if obj['name'] in seen_labels:
                            seen_labels[obj['name']] += 1
                        else:
                            seen_labels[obj['name']] = 1
                        if len(labels) > 0 and obj['name'] not in labels:
                            break
                        else:
                            img['object'] += [obj]
                    if 'bndbox' in attr.tag:
                        for dim in list(attr):
                            if 'xmin' in dim.tag:
                                obj['xmin'] = int(float(dim.text))
                            if 'ymin' in dim.tag:
                                obj['ymin'] = int(float(dim.text))
                            if 'xmax' in dim.tag:
                                obj['xmax'] = int(float(dim.text))
                            if 'ymax' in dim.tag:
                                obj['ymax'] = int(float(dim.text))
        if len(img['object']) > 0:
            all_imgs += [img]

    return all_imgs, seen_labels


# Paths to the dataset
ann_dir = './data/voc2007/VOCdevkit/VOC2007/Annotations/'
img_dir = './data/voc2007/VOCdevkit/VOC2007/JPEGImages/'
labels = ['person', 'car']  # Add other labels as needed
all_imgs, seen_labels = parse_voc_annotation(ann_dir, img_dir, labels)

# Create a label map
label_map = {label: idx for idx, label in enumerate(seen_labels.keys())}


# Data generator class
class VOCDataset(Dataset):
    def __init__(self, data, label_map, transforms=None):
        self.data = data
        self.label_map = label_map
        self.transforms = transforms

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_data = self.data[idx]
        image = Image.open(img_data['filename']).convert("RGB")
        boxes = []
        labels = []
        for obj in img_data['object']:
            boxes.append([obj['xmin'], obj['ymin'], obj['xmax'], obj['ymax']])
            labels.append(self.label_map[obj['name']])
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        target = {"boxes": boxes, "labels": labels}

        if self.transforms:
            image = self.transforms(image)

        return image, target


transform = transforms.Compose([
    transforms.ToTensor()
])

# Create dataset and dataloader
dataset = VOCDataset(all_imgs, label_map, transforms=transform)
data_loader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))

# Load pre-trained SSD model
model = ssdlite320_mobilenet_v3_large(weights=SSDLite320_MobileNet_V3_Large_Weights.DEFAULT)
model.train()

# Define optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)


# Training step function
def train_step(images, targets):
    images = list(image for image in images)
    targets = [{k: v for k, v in t.items()} for t in targets]
    loss_dict = model(images, targets)
    losses = sum(loss for loss in loss_dict.values())

    optimizer.zero_grad()
    losses.backward()
    optimizer.step()

    return losses.item()


# Training loop
epochs = 5
for epoch in range(epochs):
    for step, (images, targets) in enumerate(data_loader):
        loss = train_step(images, targets)
        if step % 10 == 0:
            print(f"Epoch {epoch + 1}, Step {step}, Loss: {loss}")
    print(f"Epoch {epoch + 1} completed.")

print("Training complete.")

# Evaluation and inference
model.eval()


def predict(image):
    image = transform(image).unsqueeze(0)
    with torch.no_grad():
        prediction = model(image)
    return prediction


# Example inference
test_image = Image.open(all_imgs[0]['filename']).convert("RGB")
predictions = predict(test_image)
print(predictions)

# save the model
torch.save(model.state_dict(), 'ssd_mobilenet_v3.pth')
print("Model saved successfully.")
