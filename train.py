#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Russell Izadi 2023
"""

from vocalset import VOCALSET
import argparse
import utils
import torchaudio
import torch
import models
from torch.nn.utils.rnn import pad_sequence


def get_loaders(args):

    def resample(x, orig_freq):
        x = torchaudio.transforms.Resample(
            orig_freq=orig_freq, new_freq=args.sample_rate)(x)
        return x

    def mel_spec(x):
        x = torchaudio.transforms.MelSpectrogram(**args.mel_spec)(x)
        return x

    def feature_fn(path_wav):
        waveform, sample_rate = torchaudio.load(path_wav)
        waveform = torch.mean(waveform, dim=0, keepdim=True)
        waveform = resample(waveform, sample_rate)
        feature = mel_spec(waveform)
        return feature

    def prep_fn(sample):
        path_wav, path_wav_, attr = sample

        feature = feature_fn(path_wav)
        feature_ = feature_fn(path_wav_)
        return feature, feature_, attr

    def collate_fn(batch):
        features, features_, attrs = zip(*batch)
        features += features_
        attrs += attrs

        features = [feature.squeeze(0).transpose(0, 1) for feature in features]
        lengths = [len(feature) for feature in features]
        features = pad_sequence(features, batch_first=True)
        duration_range = torch.arange(features.size(1))
        masks = torch.stack([duration_range > length for length in lengths])
        return features, masks, attrs

    # Get the train
    train_ds = VOCALSET(split="train", **args.vocalset)
    train_ds_list = []
    for attr in args.attrs:
        train_ds = utils.Pair(train_ds, attr)
        train_ds = utils.Transform(train_ds, prep_fn)
        train_ds_list.append(train_ds)
    train_ds = torch.utils.data.ConcatDataset(train_ds_list)
    train_loader = torch.utils.data.DataLoader(
        dataset=train_ds,
        collate_fn=collate_fn,
        **args.loader)

    # Get the valid
    valid_ds = VOCALSET(split="valid", **args.vocalset)
    valid_ds_list = []
    for attr in args.attrs:
        valid_ds = utils.Pair(valid_ds, attr)
        valid_ds = utils.Transform(valid_ds, prep_fn)
        valid_ds_list.append(valid_ds)
    valid_ds = torch.utils.data.ConcatDataset(valid_ds_list)
    valid_loader = torch.utils.data.DataLoader(
        dataset=valid_ds,
        collate_fn=collate_fn,
        **args.loader)

    return train_loader, valid_loader


def get_model(args):
    model = models.Encoder(**args.encoder)

    return model


def main(args):

    # Get the dataset
    train_loader, valid_loader = get_loaders(args.data)

    model = get_model(args.model)

    for batch in train_loader:
        features, masks, attrs = batch
        print(features.shape, masks.shape, attrs)
        model(features, masks)
        break

    pass


if __name__ == "__main__":

    # Get the command line inputs
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default="config.yml", type=str,
                        metavar='c', help='Path to the config file.')
    args = parser.parse_args()

    # Load the config file
    args = utils.read_yml(args.config)

    main(args)
