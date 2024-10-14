device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using {device} device.")

hparams = Hparams()

train_loader, val_loader, test_loader = get_data_loaders(hparams)

early_stopping = EarlyStopping(patience=3, mode='max')
