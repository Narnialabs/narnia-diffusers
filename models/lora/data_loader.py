from datasets import load_dataset
import torch
from torchvision import transforms
import matplotlib.pyplot as plt
from PIL import Image
from datasets import Dataset
import os

def get_data_loader(dataset_folder, 
                   tokenizer,
                   train_batch_size = 4,
                   resolution=512,
                   center_crop = True,
                   random_flip = True,
                   ):
    
    def load_text_file(file_path):
        with open(file_path, 'r') as file:
            text = file.read()
        return text

    def load_image_file(file_path):
        image = Image.open(file_path)
        return image

    def dataset_loader(dataset_folder):
        images_folder = os.path.join(dataset_folder, 'images')
        texts_folder = os.path.join(dataset_folder, 'texts')

        image_paths = sorted([os.path.join(images_folder, file) for file in os.listdir(images_folder) if (file.endswith('.png')) or (file.endswith('.jpg'))])
        text_paths = sorted([os.path.join(texts_folder, file) for file in os.listdir(texts_folder) if file.endswith('.txt')])

        dataset = []
        for image_path, text_path in zip(image_paths, text_paths):
            image = load_image_file(image_path)
            text = load_text_file(text_path)
            dataset.append({'image': image, 'text': text})

        return dataset    
    
  
    def get_captions(examples, is_train=True):
        captions = []
        for caption in examples['text']:
            if isinstance(caption, str):
                captions.append(caption)
            elif isinstance(caption, (list, np.ndarray)):
                # take a random caption if there are multiple
                captions.append(random.choice(caption) if is_train else caption[0])
            else:
                raise ValueError(
                    f"Caption column 'text' should contain either strings or lists of strings."
                )

        return captions


    def tokenize_captions(captions):
        inputs = tokenizer(
            captions, max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
        )
        return inputs.input_ids


    def preprocess_train(examples):
        images = [image.convert("RGB") for image in examples['image']]
        examples["pixel_values"] = [pp(image) for image in images]
        examples["captions"] = get_captions(examples)
        examples["tokens"] = tokenize_captions(examples["captions"])
        return examples

    def collate_fn(examples):
        pixel_values = torch.stack([example["pixel_values"] for example in examples])
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
        tokens = torch.stack([example["tokens"] for example in examples])

        return {"pixel_values": pixel_values, "tokens": tokens, "captions": [[example["captions"]] for example in examples]}


    pp = transforms.Compose(
        [
          transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
          transforms.CenterCrop(resolution) if center_crop else transforms.RandomCrop(config.resolution),
          transforms.RandomHorizontalFlip() if random_flip else transforms.Lambda(lambda x: x),
          transforms.ToTensor(),
          transforms.Normalize([0.5], [0.5]),
        ]
    )

    # load ds
    data = dataset_loader(dataset_folder)

    # Prepare dataset for datasets library
    dataset_dict = {
        'image': [sample['image'] for sample in data],
        'text': [sample['text'] for sample in data],
    }

    # Create a datasets.Dataset object
    dataset = Dataset.from_dict(dataset_dict)


    # transform
    train_dataset = dataset.with_transform(preprocess_train)

    # make loader
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=train_batch_size,
    )

    return train_dataloader

