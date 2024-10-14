model = CNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=hparams.learning_rate)
train_losses, train_accuracies, val_losses, val_accuracies = train_and_validate(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=hparams.num_epochs, early_stopping=early_stopping)
